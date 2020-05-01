# [NON-AUTOREGRESSIVE NEURAL MACHINE TRANSLATION](https://arxiv.org/abs/1711.02281)

## 介绍
现存的翻译模型都是利用已生成的序列计算新的单词，本文提供一种**非递归**方法避免这种**递归**性质，并行地生成结果，能大幅度减少预测时地时间成本。利用knowledge distillation（利用输入信息作为潜在变量）和policy gradient fine-tuning，可以达到与**递归Transformer**同等地效果。利用三步训练策略，能有效优化模型。

本模型以[Transformer][transformer]为基础,针对encoder部分，添加了预测*fertilities*（肥料）的模块，这是传统翻译模型中的重要部分。这些肥料在训练过程中得到填充，在decoder预测的时候起效，在输出同步生成的时候起到一种全局约束的作用。

## 背景知识
### 递归翻译
给定原句X={x<sub>1</sub>,&hellip;,x<sub>T’</sub>}，翻译模型会给出一个译文Y={y<sub>1</sub>,&hellip;,y<sub>T</sub>}，而这个条件概率式子如下：\
![Imgur](https://i.imgur.com/7TgyxZO.png)\
其中y<sub>0</sub>（比如\<bos\>）和y<sub>T+1</sub>（比如\<eos\>）代表句子的起始符和终止符。

#### 最大似然训练 
这种递归性质的输出让训练可以直接通过最大似然法进行，利用cross-entropy误差能直接对每一步的条件概率预测进行矫正。\
![Imgur](https://i.imgur.com/ykhatGy.png)\

#### RNN之外的递归NMT
虽然预测时候必须递归进行，但训练时由于译文已知，就能够充分发挥这种优势。比如masked convolution layers。\
而最近的一项研究是[Transformer][transformer]，

[transformer]:(https://arxiv.org/abs/1706.03762)
