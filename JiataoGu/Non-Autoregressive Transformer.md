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
而最近的一项研究是[Transformer][transformer]，这种方法将序列的顺序计算进一步削减，通过mask让decoder的attention计算次数无关句长，保持不变。

### 非递归decoding
#### 递归decoding的优缺点
这种常规模型能够有效模拟人类词对词的翻译过程，在训练数据充足的情况下颇有成效，而且类似beam search的搜索策略能让结果更为优化。\
然而这种方法在decoder必须顺序执行，阻止了Transformer等模型在预测时候能发挥出训练时候的水平。beam search也会因为beam size而导致计算量激增。

#### 非递归decoding
最单纯的解决办法就是切断递归顺序。假定有独立的条件分布P<sub>L</sub>能求出target序列的长度T，就有下式：\
![Imgur](https://i.imgur.com/6birJyr.png)\
上式保留了似然函数，依旧能用cross-entropy误差得到分布。而且能够并行预测每个单词。

### 多峰问题(Multimodality Problem)
然而这种简单办法成效并不可观，因为这导致条件完全独立。每个token的分布p(y<sub>t</sub>)仅仅取决于原句X，而译文是在时间维度上强相关的，这就导致无法准确预测出译文的分布。直观上说，这就像是要人根据一段原文，在不知道前面几个译文单词是什么的情况下推断出下一个单词。\
比如英文“Thank you”可以翻译成法语“Danke”“Danke schon”和“Vielen Dank”，然而若是脱离了前后关系，就会出现不该出现的“Danke Dank”或“Vielen schon”。\
这种独立条件假设无法预测多峰分布的译文。后面将描述如何调整模型，并提出一种训练技巧解决这个问题。

## NAT（Non-Autoagressive Transformer）


[transformer]:(https://arxiv.org/abs/1706.03762)
