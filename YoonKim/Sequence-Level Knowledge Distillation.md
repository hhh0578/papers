# [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947)
[源代码](https://github.com/harvardnlp/seq2seq-attn)
## 已有研究
- [1000隐藏的4层LSTM](https://arxiv.org/abs/1409.3215)
- [512隐藏的16层LSTM](https://www.aclweb.org/anthology/Q16-1027/)

不过，虽然训练需要庞大的神经网络，但[训练过程中representation会冗余](https://arxiv.org/abs/1306.0543)，于是deep model的压缩就成了热门，现有压缩技术：
- pruning：[LeCun](https://papers.nips.cc/paper/250-optimal-brain-damage)和[Han](https://arxiv.org/abs/1510.00149)
- knowledge distillation：训练一个小的**student网络**来模仿**teacher网络**，误差通常用**L**<sub>2</sub>或者**cross-entropy**计算。
## 介绍
NMT在计算的时候，预测时会用到之前预测的结果。而本文先是在NMT上实验了standard knowledge distillation，随后提出了两种新的方式让其大致match教师的模型的sequence-level（另一个则是word-level）分布。这种模拟可以简化student模型在训练新数据集时的程序。

## 背景知识
### Attention的s2s模型。
s=\[s<sub>1</sub>,&hellip;,s<sub>I</sub>\]和t=\[t<sub>1,&hellip;</sub>,s<sub>J</sub>\]为source和target。I和J分别对应其长度。于是机器翻译就时要在给定s的情况下求出概率最大的t：\
![Imgur](https://i.imgur.com/YxuxpU0.png)\
&Tau;代表所有可能出现的序列。

本文模型采用[Luong的En&harr;De翻译模型](https://arxiv.org/abs/1508.04025)，用的是global-general attention模型和input-feeding方法。细节参考论文。
### 知识蒸馏
**知识蒸馏**代表一类利用一个更大的teacher网络来训练student以得到更好效果的方法。

本文假定teacher已经训练好，目标时训练student的参数。为此需要match两者的概率，或是用在[log尺度内L<sub>2</sub>](https://arxiv.org/abs/1312.6184)，或是用cross-entropy衡量。

假定要训练一个多分类器，通常训练目标时最小化NLL，这过程可以看作是在最小化**模型分布**和**degenerate data分布**之间的cross-entropy：\
![Imgur](https://i.imgur.com/VWIPoOo.png)\
大括号那个是indicator function，p表示模型给出的概率分布。

在知识蒸馏中，会在同一个数据集中用到teacher的分布q。然而需要最小化的是**模型分布**和**teacher分布**之间的cross-entropy：\
![Imgur](https://i.imgur.com/DGG0TM5.png)\
其中教师参数&theta;<sub>T</sub>保持固定。

用q训练能够得到比原数据更多的other classes信息，而且[梯度的方差更小](https://arxiv.org/abs/1503.02531)。由于上述公式没有直接用到原文，实际上更一般的方法是再两个误差间取插值：\
![Imgur](https://i.imgur.com/R49JUkH.png)\
&alpha;是mixture parameter。
## NMT的知识蒸馏
### Word-Level知识蒸馏

