# Chinese-image-captioning

## 一、简介
基于PaddlePaddle框架搭建中文图像描述生成模型，目的是依托飞桨深度学习平台的优势，开拓基于飞桨框架的中文IC研究。

当前主流发展的还是生成英文的图像描述，基于COCO数据集训练提升指标分数，提供相应的离线和在线测试。英文IC的有关项目可参考如下经典的模型。

[Soft-Attention](https://aistudio.baidu.com/aistudio/projectdetail/2338889) 、
[Up-Down](https://aistudio.baidu.com/aistudio/projectdetail/2345929) 、
[AoANet](https://aistudio.baidu.com/aistudio/projectdetail/2879640)

中文和英文的语言区别比较大，中文语义更加丰富且灵活多变，而当前针对中文的图像描述生成研究相对较少，大多模型借鉴于英文的IC技术。在中文IC研究里，Flickr8k-cn、Flickr30k-cn、COCO-CN以及AI Challenger都是相对经典的数据集。

本项目基于Flickr8k-cn数据集展开复现和实验，并参考主流的图像描述生成代码，搭建了一个简易可复用，相对完备的中文图像描述模型项目。项目实现了CNN-LSTM-Attention结构，这是图像描述生成方向非常典型的基线模型，可基于此项目学习基础并展开研究。

| 信息 | 说明 |
| --- | --- |
| 框架版本 | Paddle 2.3.1 |
| 在线运行 | [notebook](https://aistudio.baidu.com/aistudio/projectdetail/4450898) |

## 二、数据集

Flick8k-CN 数据集共包含8000幅图像，其中每幅图像配有五句人工标注的中文描述，侧重于同一幅图像的多义表述。FIick8k-CN是雅虎英文数据集FIick8k的中文扩展。整个FIick8k-CN数据集分为三部分： 训练集、验证集和测试集。其中，训练集6000张图片，验证集和测试集各包含1000张图片。

[中文数据集参考](https://github.com/li-xirong/cross-lingual-cap)

[Flickr8k-cn下载](https://drive.google.com/file/d/1C6NfkfoSX42QER0n2tf1VcpuWhQelA3m/view)

## 三、预处理准备

项目是基于Flick8k-CN数据集训练测试，包括Flickr30k-cn、COCO-CN和AI Challenger数据集也是通用的，只是相应的预处理不太一样。本文使用了scripts/prepro_flick8k.py处理原数据文件生成filelists/flickr8k-cn/dataset_flickr8k_cn.json标签文件。

如果想使用其他数据集，可自行更改scripts/prepro_flick8k.py代码，参照dataset_flickr8k_cn.json生成对应格式的文件，其他代码文件不需要改动。

项目scripts/的prepro_labels.py、prepro_reference_json.py根据dataset_flickr8k_cn.json对中文标签进行预处理，生成flickr8k_label.h5和flickr8k.json。

全部训练数据[下载](https://aistudio.baidu.com/aistudio/datasetdetail/104803) 放在filelists/下

评估文件[下载](https://aistudio.baidu.com/aistudio/datasetdetail/108181) 放在本repo/下 

**Install dependencies**
```bash
pip install -r requestments.txt
```

## 四、模型运行

模型详细内容可参考
[Soft-Attention](https://aistudio.baidu.com/aistudio/projectdetail/2338889) 、
[AoANet](https://aistudio.baidu.com/aistudio/projectdetail/2879640)

训练评估结果如下:

|Bleu_1|Bleu_2|Bleu_3|Bleu_4|METEOR|ROUGE_L|CIDEr|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
|0.701 |0.555|0.439|0.337| 0.326|0.569|1.178|

### step1: 训练
训练过程过程分为两步：Cross-entropy Training和SCST(Self-critical Sequence Training)

第一步Cross-entropy Training：

```bash
python3 train.py --model_name vatt --sc_flag 0 --max_epochs 15 --learning_rate 2e-4 
```

第二步SCST(Self-critical Sequence Training)：

```bash
python3 train.py --model_name vatt --sc_flag 1 --max_epochs 30 --learning_rate 2e-5
```
训练的模型保存在本repo的saved_models/下

### step2: 评估

```bash
python3 test.py  --model_name vatt --beam_size 3
```
测试时会加载保存的训练模型，最终验证评估的是SCST优化模型。

### step3: 预测

加载训练好的模型，执行 predict.py，输入一张图像生成描述语句。读者可自己选择一张图像保存在images/中，执行预测命令，验证模型有效性。

```bash
python3 predict.py --model_name vatt --img_name 33108590_d685bfe51c.jpg --beam_size 3
```

## 五、致谢

This code is based on Ruotian Luo's image captioning repo [ruotianluo/self-critical.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch)