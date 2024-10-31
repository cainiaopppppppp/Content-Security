# FGSM算法生成MNIST数据集的对抗样本

[![](https://img.shields.io/badge/%E4%B8%BB%E9%A1%B5-yzq/fgsm-orange)](https://github.com/Artistzq/fgsm-mnist)

一个使用FGSM对抗攻击算法，攻击MNIST手写数字识别，并生成对抗样本的小实验。不附带数据集，完整目录如下：

```
fgsm_mnist
├─ .gitignore
├─ LICENSE
├─ README.md
├─ data
│  └─ MNIST
│     ├─ processed
│     │  ├─ test.pt
│     │  └─ training.pt
│     └─ raw
│        ├─ t10k-images-idx3-ubyte
│        ├─ t10k-images-idx3-ubyte.gz
│        ├─ t10k-labels-idx1-ubyte
│        ├─ t10k-labels-idx1-ubyte.gz
│        ├─ train-images-idx3-ubyte
│        ├─ train-images-idx3-ubyte.gz
│        ├─ train-labels-idx1-ubyte
│        └─ train-labels-idx1-ubyte.gz
├─ examples.png
├─ fgsm.ipynb
└─ model.pth
```

## 训练
神经网络模型结构如下：
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #

================================================================
            Conv2d-1           [-1, 16, 24, 24]             416 
         MaxPool2d-2           [-1, 16, 12, 12]               0
              ReLU-3           [-1, 16, 12, 12]               0
            Conv2d-4           [-1, 32, 10, 10]           4,640
         Dropout2d-5           [-1, 32, 10, 10]               0
         MaxPool2d-6             [-1, 32, 5, 5]               0
              ReLU-7             [-1, 32, 5, 5]               0
            Linear-8                  [-1, 160]         128,160
              ReLU-9                  [-1, 160]               0
          Dropout-10                  [-1, 160]               0
           Linear-11                   [-1, 10]           1,610
       LogSoftmax-12                   [-1, 10]               0

================================================================

Total params: 134,826  
Trainable params: 134,826  
Non-trainable params: 0  

----------------------------------------------------------------

Input size (MB): 0.00  
Forward/backward pass size (MB): 0.17  
Params size (MB): 0.51  
Estimated Total Size (MB): 0.69  

----------------------------------------------------------------
```

简单训练了5个epoch后保存拿来实验，准确率 `98.38%`

## FGSM
epsilon选取 `0.2, 0.25, 0.3, 0.35, 0.4, 0.5`结果：
* total: 总样本数
* attack：被攻击的样本数，本来就判断错误的不被攻击
* success：攻击成功的样本数


### 无目标攻击
    args: epsilon 0.2	total / attack / success: 10000 / 9838 / 134 	 total accuracy: 0.9704
    args: epsilon 0.25	total / attack / success: 10000 / 9838 / 164 	 total accuracy: 0.9674
    args: epsilon 0.3	total / attack / success: 10000 / 9838 / 202 	 total accuracy: 0.9636
    args: epsilon 0.35	total / attack / success: 10000 / 9838 / 253 	 total accuracy: 0.9585
    args: epsilon 0.4	total / attack / success: 10000 / 9838 / 311 	 total accuracy: 0.9527
    args: epsilon 0.5	total / attack / success: 10000 / 9838 / 776 	 total accuracy: 0.9062


### 设定目标为【数字2】攻击

    args: epsilon 0.2	total / attack / success: 10000 / 9838 / 18 	 total accuracy: 0.982
    args: epsilon 0.25	total / attack / success: 10000 / 9838 / 22 	 total accuracy: 0.9816
    args: epsilon 0.3	total / attack / success: 10000 / 9838 / 29 	 total accuracy: 0.9809
    args: epsilon 0.35	total / attack / success: 10000 / 9838 / 36 	 total accuracy: 0.9802
    args: epsilon 0.4	total / attack / success: 10000 / 9838 / 47 	 total accuracy: 0.9791
    args: epsilon 0.5	total / attack / success: 10000 / 9838 / 129 	 total accuracy: 0.9709


几个生成的对抗样本，详见[笔记本](fgsm.ipynb)  
![avatar](/examples.png)

## 贡献人员

[@Yao-GitHub](https://github.com/Artistzq)([Gitee](https://gitee.com/devezq))

## 开源协议

[GPL](LICENSE) © Yao
