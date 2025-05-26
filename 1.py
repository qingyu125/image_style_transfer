import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# 加载预训练模型
# hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
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

# 加载图像（示例路径，需替换为实际路径）
content_image = load_image('content.jpg')
style_image = load_image('style.jpg')

# 生成风格迁移图像
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

# 后处理并显示结果
stylized_image = tf.squeeze(stylized_image)
stylized_image = tf.cast(stylized_image * 255, tf.uint8)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(tf.squeeze(content_image))
plt.title('Content')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(tf.squeeze(style_image))
plt.title('Style')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(stylized_image)
plt.title('Stylized')
plt.axis('off')
plt.show()

# 保存结果
tf.keras.preprocessing.image.save_img('stylized_result.jpg', stylized_image)