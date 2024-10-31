# 2021302181152 邓鹏 第四次实验

## 1.GPT-2 模型

### GPT-2 large 模型下载

https://huggingface.co/openai-community/gpt2-large/tree/main

模型放置到`gpt-large`目录下

## 2.GPT-2 Output Detector 模型

### GPT-2 Output Detector 模型下载

wget https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-base.pt

模型放置到根目录下

### GPT-2 Output Detector 数据集下载

运行`download_dataset.py`,数据集会下载到`data`目录中

## 3.GPT-2 Output Detector 中文模型

### GPT-2 Output Detector 中文数据集下载

http://10.201.201.213/#s/-UidRmcA&view=%E6%96%87%E6%9C%AC%E7%94%9F%E6%88%90%E4%B8%8E%E6%A3%80%E6%B5%8B%E6%95%B0%E6%8D%AE%E9%9B%86/WuDaoCorpus2.0_base_200G

在链接中下载任意六个文件，并分别命名`fake_data.train.json`、`fake_data.test.json`、`fake_data.valid.json`、`real_data.train.json`、`real_data.test.json`、`real_data.valid.json`,放置在`data`目录下

## 4.bert-base-chinese 模型

### bert-base-chinese 模型下载

https://huggingface.co/google-bert/bert-base-chinese/tree/main

模型放置到`bert-base-chinese`目录下

## 5.代码文件介绍

### · 以任意主题生成一段文本，用openAI/GPT2或者transformers的GPT2模型皆可。

对应`transformer-gpt2.ipynb`,可以直接看到相应的内容

### · 调整参数temperature、Top-K、Top-P并重新执行任务1，分析调参之后的影响，最好结合相关源代码进行分析，用openAI/GPT2或者transformers的GPT2模型皆可。

对应`transformer-gpt2.ipynb`,可以直接看到相应的内容

### · 结合openAI/GPT2源码分析GPT2的注意力机制和作用。

对应源代码中src目录下的`model.py`文件，这里是`gpt-2-attention/model.py`

### · 运行GPT-2 Output Detector进行文本检测，分析文本检测原理，分析文本长度对检测结果的影响及原因。

下载代码，调整后，运行`python -m detector.server detector-base.pt`,代码大致不变，故没给代码

### · 自选应用场景和数据集，基于transformers框架微调模型，并做相关评价分析。

使用的是之前的舆情分析课程的大作业，这里的`sentiment_argument.py`和`sentiment_train.py`分别是数据处理和训练模型的代码,完整的项目代码详见 https://github.com/cainiaopppppppp/NLP/tree/main

### · 基于GPT-2 Output Detector trainer，训练一个中文模型，并做相关评价分析。

修改了`detector`目录下的`dataset.py`、`train.py`、`server.py`，模型会生成在models目录下，`python -m detector.server models/我们的模型路径` 即可运行

这里代码在`gpt-2-output-dataset-chinese`目录下
