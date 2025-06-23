import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, Model
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

# 关闭警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 配置参数
BATCH_SIZE = 4
EPOCHS = 50
CONTENT_WEIGHT = 0.04
STYLE_WEIGHT = 100
ADV_WEIGHT = 1
IMAGE_SIZE = (256, 256)
STYLE_DATASET_PATH = "path/to/style/images"  # 风格图像数据集路径
CONTENT_DATASET_PATH = "path/to/content/images"  # 内容图像数据集路径
CHECKPOINT_DIR = "checkpoints/"  # 检查点保存路径

# 定义生成器模型（U-Net结构）
def build_generator():
    # 内容图像输入
    content_input = layers.Input(shape=(*IMAGE_SIZE, 3), name="content_input")
    # 风格图像输入（或风格特征输入）
    style_input = layers.Input(shape=(*IMAGE_SIZE, 3), name="style_input")
    
    # 使用VGG提取风格特征
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=(*IMAGE_SIZE, 3))
    vgg.trainable = False
    style_features = vgg(style_input)
    
    # 内容图像编码（下采样）
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(content_input)
    x1 = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x1)
    x2 = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x2)
    x3 = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x3)
    x4 = layers.MaxPooling2D(2, padding='same')(x)
    
    # 风格特征融合（示例：将风格特征与内容编码拼接）
    x = layers.Concatenate()([x4, style_features])
    
    # 解码生成图像（上采样）
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Concatenate()([x, x3])  # 跳跃连接
    
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Concatenate()([x, x2])  # 跳跃连接
    
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Concatenate()([x, x1])  # 跳跃连接
    
    x = layers.UpSampling2D(2)(x)
    outputs = layers.Conv2D(3, 3, padding='same', activation='tanh')(x)  # tanh激活使输出在[-1, 1]
    
    return Model([content_input, style_input], outputs, name="generator")

# 定义判别器模型（PatchGAN）
def build_discriminator():
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3), name="discriminator_input")
    
    x = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(512, 4, strides=1, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)
    
    outputs = layers.Conv2D(1, 4, strides=1, padding='same')(x)  # 输出N×N的概率图
    
    return Model(inputs, outputs, name="discriminator")

# 加载VGG模型用于特征提取
def build_vgg_features():
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=(*IMAGE_SIZE, 3))
    vgg.trainable = False
    # 内容层（深层特征）
    content_layers = ['block4_conv2']
    # 风格层（浅层特征）
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    outputs = [vgg.get_layer(layer).output for layer in content_layers + style_layers]
    return Model(vgg.input, outputs, name="vgg_features")

# 计算格拉姆矩阵（风格损失用）
def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    a = tf.reshape(tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

# 内容损失
def content_loss(generated, content):
    vgg = build_vgg_features()
    gen_features = vgg(generated)
    content_features = vgg(content)
    # 假设content_layers在输出的前几位
    return tf.reduce_mean(tf.square(gen_features[0] - content_features[0]))

# 风格损失
def style_loss(generated, style):
    vgg = build_vgg_features()
    gen_features = vgg(generated)
    style_features = vgg(style)
    style_loss = 0.0
    # 风格层从content_layers之后开始
    for i, layer in enumerate(style_layers):
        gen_gram = gram_matrix(gen_features[i+1])
        style_gram = gram_matrix(style_features[i+1])
        style_loss += tf.reduce_mean(tf.square(gen_gram - style_gram))
    return style_loss / len(style_layers)  # 平均各层损失

# 判别器损失
def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# 生成器损失（对抗损失 + 内容损失 + 风格损失）
def generator_loss(fake_output, generated, content_image, style_image):
    adv_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(fake_output), fake_output)
    cont_loss = content_loss(generated, content_image)
    styl_loss = style_loss(generated, style_image)
    return (ADV_WEIGHT * adv_loss + 
            CONTENT_WEIGHT * cont_loss + 
            STYLE_WEIGHT * styl_loss)

# 数据加载与预处理
def load_and_preprocess_image(path, image_size=IMAGE_SIZE):
    img = load_img(path, target_size=image_size)
    img = img_to_array(img)
    img = preprocess_input(img)  # VGG预处理
    return tf.expand_dims(img, 0)  # 添加批次维度

def create_dataset(content_dir, style_dir, batch_size=BATCH_SIZE):
    # 简化示例：实际应使用tf.data.Dataset构建数据管道
    content_files = [os.path.join(content_dir, f) for f in os.listdir(content_dir) 
                     if f.endswith(('.jpg', '.png', '.jpeg'))]
    style_files = [os.path.join(style_dir, f) for f in os.listdir(style_dir) 
                   if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # 构建数据集（实际应用中应使用shuffle、batch等操作）
    def generate_pairs():
        for content_path in content_files:
            for style_path in style_files:
                content_img = load_and_preprocess_image(content_path)
                style_img = load_and_preprocess_image(style_path)
                yield content_img, style_img
    
    dataset = tf.data.Dataset.from_generator(
        generate_pairs, 
        (tf.float32, tf.float32), 
        (tf.TensorShape([1, *IMAGE_SIZE, 3]), tf.TensorShape([1, *IMAGE_SIZE, 3]))
    )
    return dataset.shuffle(100).batch(batch_size)

# 训练函数
def train():
    # 创建模型
    generator = build_generator()
    discriminator = build_discriminator()
    
    # 优化器
    generator_optimizer = optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    discriminator_optimizer = optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    
    # 加载数据集
    dataset = create_dataset(CONTENT_DATASET_PATH, STYLE_DATASET_PATH)
    
    # 检查点设置
    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator
    )
    manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=3)
    
    # 训练循环
    @tf.function
    def train_step(content_batch, style_batch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # 生成风格化图像
            generated_images = generator([content_batch, style_batch], training=True)
            
            # 判别器判断
            real_output = discriminator(style_batch, training=True)
            fake_output = discriminator(generated_images, training=True)
            
            # 计算损失
            gen_loss = generator_loss(fake_output, generated_images, content_batch, style_batch)
            disc_loss = discriminator_loss(real_output, fake_output)
        
        # 计算梯度并更新参数
        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        
        generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
        
        return gen_loss, disc_loss
    
    # 开始训练
    print("开始训练...")
    for epoch in range(EPOCHS):
        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0
        batch_count = 0
        
        for content_batch, style_batch in dataset:
            gen_loss, disc_loss = train_step(content_batch, style_batch)
            epoch_gen_loss += gen_loss
            epoch_disc_loss += disc_loss
            batch_count += 1
            
            if batch_count % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_count}, "
                      f"Gen Loss: {gen_loss.numpy():.4f}, Disc Loss: {disc_loss.numpy():.4f}")
        
        # 计算平均损失
        epoch_gen_loss /= batch_count
        epoch_disc_loss /= batch_count
        print(f"Epoch {epoch+1}, Avg Gen Loss: {epoch_gen_loss:.4f}, Avg Disc Loss: {epoch_disc_loss:.4f}")
        
        # 保存检查点
        if (epoch + 1) % 5 == 0:
            save_path = manager.save()
            print(f"检查点已保存到: {save_path}")
    
    print("训练完成！")
    return generator

# 可视化训练结果（示例）
def visualize_results(generator, content_path, style_path):
    content_img = load_and_preprocess_image(content_path)
    style_img = load_and_preprocess_image(style_path)
    generated_img = generator([content_img, style_img], training=False)
    generated_img = tf.squeeze(generated_img)
    generated_img = tf.cast((generated_img + 1) * 127.5, tf.uint8)  # 反归一化
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(tf.squeeze(content_img)[:, :, ::-1] + 103.939)  # 反VGG预处理
    plt.title("内容图像")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(tf.squeeze(style_img)[:, :, ::-1] + 103.939)  # 反VGG预处理
    plt.title("风格图像")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(generated_img)
    plt.title("生成图像")
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    # 确保检查点目录存在
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    
    # 开始训练
    trained_generator = train()
    
    # 可视化结果（替换为实际图像路径）
    # visualize_results(trained_generator, "test_content.jpg", "test_style.jpg")