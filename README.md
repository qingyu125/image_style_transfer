# image_style_transfer
智能综合实习:基于深度学习的图像风格迁移系统  
本仓库用于展示智能综合实习的内容  
使用Tensorflow Hub中的[Fast Style Transfer模型](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2)(谷歌开源的预训练模型，其训练逻辑基于GAN和VGG特征损失)。  
给定一张内容图片和一张风格图片，生成一张具备内容图片内容+风格图片风格的图片。  
其中，1.py是对现有文件下的content.jpg和style.jpg进行风格迁移；GUI.py是生成GUI界面，进行交互；trian.py是示例的训练部分代码（并未使用，因为使用了谷歌的模型）
