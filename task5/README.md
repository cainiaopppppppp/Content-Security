# 2021302181152 邓鹏 第五次实验

## 1.数据集

MNIST 数据集

运行每个代码文件都会自动下载数据集到对应文件夹下，无需另外下载

## 2.用到的预训练模型VGG-16

下载链接 https://www.kaggle.com/code/shreyasi2002/vgg16-on-mnist-and-fashion-mnist/input?select=vgg16_mnist_model.pth

## 3.代码文件介绍

### · 理解FGSM算法，在MNIST数据集上复现

参考论文： [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)

方法一： 对应文件 `fgsm-attack-cnn.ipynb` ,运行后会在同目录下生成 `model.pth`

方法二： 对应文件 `fgsm-attack-vgg16.ipynb` , 运行即可， 需要事先下载vgg16模型

### · 任意实现/复现一种图像对抗样本生成算法（除FGSM）

参考论文： [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)

对应`pgd-attack-vgg16.ipynb`, 运行即可， 需要事先下载vgg16模型

### · 基于实验1或实验2，将生成的对抗样本添加到训练数据集中，再次进行模型训练。观察重新训练的模型在原始正常的数据集上的表现，是否达到了预期的对抗样本防御目标？

参考论文： <br>
[Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models](https://arxiv.org/abs/1805.06605) <br>
[Defense Against Adversarial Attacks using Convolutional Auto-Encoders](https://arxiv.org/abs/2312.03520)

对应`defense.ipynb`, 运行即可， 需要事先下载vgg16模型

实现了 FGSM 和 PGD 的对抗样本防御
