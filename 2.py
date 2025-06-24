import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Model

# 加载预训练模型
hub_model = hub.load('./arbitrary-image-stylization-v1-tensorflow1-256-v2/')

def load_image(image_path, max_dim=512):
    """加载并调整图像大小"""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    
    img = tf.image.resize(img, new_shape)
    return img[tf.newaxis, :]

# 定义全局风格层（与论文一致）
STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
CONTENT_LAYERS = ['block4_conv2']  # 论文中内容层

def build_vgg_metrics():
    """构建VGG特征提取器并返回模型"""
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    # 合并内容层和风格层
    all_layers = CONTENT_LAYERS + STYLE_LAYERS
    outputs = [vgg.get_layer(layer).output for layer in all_layers]
    return Model(vgg.input, outputs)

def preprocess_vgg_input(img):
    """预处理图像为VGG输入格式"""
    img = tf.image.resize(img, (224, 224))
    return preprocess_input(img * 255)  # 还原为0-255再标准化

def calculate_content_loss(generated, content, vgg_model):
    """计算内容损失"""
    content_features = vgg_model(content)
    generated_features = vgg_model(generated)
    # 内容层对应索引为0（论文中block4_conv2）
    return tf.reduce_mean(tf.square(generated_features[0] - content_features[0]))

def gram_matrix(tensor):
    """计算格拉姆矩阵（论文公式）"""
    channels = int(tensor.shape[-1])
    a = tf.reshape(tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def calculate_style_loss(generated, style, vgg_model):
    """计算风格损失（遍历全局STYLE_LAYERS）"""
    style_features = vgg_model(style)
    generated_features = vgg_model(generated)
    style_loss = 0.0
    # 风格层对应索引从1开始（内容层在0）
    for i, layer_name in enumerate(STYLE_LAYERS):
        layer_idx = CONTENT_LAYERS.index(CONTENT_LAYERS[0]) + i + 1
        gen_gram = gram_matrix(generated_features[layer_idx])
        style_gram = gram_matrix(style_features[layer_idx])
        layer_loss = tf.reduce_mean(tf.square(gen_gram - style_gram))
        style_loss += layer_loss
    return style_loss / len(STYLE_LAYERS)  # 平均各层损失

# 加载图像
content_image = load_image('content.jpg')
style_image = load_image('style.jpg')

# 生成风格迁移图像
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

# 后处理结果
stylized_image = tf.squeeze(stylized_image)
stylized_image = tf.cast(stylized_image * 255, tf.uint8)

# 初始化VGG评价模型
vgg_model = build_vgg_metrics()

# 预处理图像用于评价
content_vgg = preprocess_vgg_input(content_image)
style_vgg = preprocess_vgg_input(style_image)
stylized_vgg = preprocess_vgg_input(tf.expand_dims(stylized_image, 0))

# 计算评价指标
content_loss = calculate_content_loss(stylized_vgg, content_vgg, vgg_model)
style_loss = calculate_style_loss(stylized_vgg, style_vgg, vgg_model)

# 显示结果与评价指标
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(tf.squeeze(content_image))
plt.title(f'Content\nContent Loss: {content_loss.numpy():.2e}')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(tf.squeeze(style_image))
plt.title(f'Style\nStyle Loss: {style_loss.numpy():.2e}')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(stylized_image)
plt.title(f'Stylized\nContent: {content_loss.numpy():.2e}\nStyle: {style_loss.numpy():.2e}')
plt.axis('off')
plt.tight_layout()
plt.show()

# 打印详细指标
print(f"内容损失 (Content Loss): {content_loss.numpy():.4e}")
print(f"风格损失 (Style Loss): {style_loss.numpy():.4e}")

# 保存结果
tf.keras.preprocessing.image.save_img('stylized_result.jpg', stylized_image)