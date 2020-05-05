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
![Imgur](https://i.imgur.com/9Mkz8DI.png)
### Word-Level知识蒸馏
NMT系统的训练通过最小化每个位置的word NLL，L<sub>WORD-NLL</sub>进行。于是传统的multi-class cross-entropy就能用在知识蒸馏上：\
![Imgur](https://i.imgur.com/eMTfpDz.png)\
V代表target vocabulary set。  
训练student的时候可以以L<sub>WORD-KD</sub>和L<sub>WORD-NLL</sub>的混合作为目标函数。这种方法极为图1中的word-level knowledge distillation。
### Sequence-Level知识蒸馏
**序列分布**在NMT中尤为重要，因为test时这种错误会向前传递。  
首先，在任意长度J下，模型可能生成的序列t&isin;&Tau;的概率是：  
![Imgur](https://i.imgur.com/rEPJMQC.png)  

而**序列分布**的negative log-likelihood的one-hot形式则为：  
![Imgur](https://i.imgur.com/B8cepyG.png)  
y为observed sequence。  

用q(t|s)表示teacher在所有可能生成的序列构成的sample space中的序列分布：  
![Imgur](https://i.imgur.com/koGdsE7.png)  
要注意L<sub>SEQ-KD</sub>的求和是指数级，但本文仍旧认为这种序列级别的知识蒸馏能获取更大范围的知识。

于是考虑如何近似这个函数，最简单办法就是取mode（从teacher训练的结果中取概率最大的序列：  
![Imgur](https://i.imgur.com/gjBJ4To.png)  
而这个过程本身极其困难，所以就利用beam search来获取一个近似，最后误差就为：  
![Imgur](https://i.imgur.com/HIsrKxN.png)  
y_hat代表teacher模型用beam search输出的结果。
> 其实还能在beam search的结果中取数学期望来近似：  
![Imgur](https://i.imgur.com/LzUCtFa.png)  
![Imgur](https://i.imgur.com/cRKp3hC.png)  
然而实验中发现取mode效果更好。

总结起来序列知识蒸馏步骤如下：  
1. 训练一个teacher model
2. 在训练集上运行beam search得到一个输出，构成新数据集。
3. 在这新的数据集上训练student。
步骤3和单词知识蒸馏一样，只不过换了数据集。见图1（center）。
### Sequence-Level Interpolation
最后考虑也利用原数据集进行训练，取一个L<sub>SEQ-KE</sub>和L<sub>SEQ-NLL</sub>的混合：  
![Imgur](https://i.imgur.com/k1HO7kN.png)  
其中y为gold target sequence。

由于第二项难以计算，依旧利用前述的mode approximation，得到：  
![Imgur](https://i.imgur.com/bgH5WXN.png)  

可惜这个过程并不理想：1.这把数据集加倍了；2.一个source会得到两个target。由于y和y_hat常常相去甚远，后者的问题尤为严重。所以single-sequence approximation更为有用。灵感来自于[local updating](https://www.aclweb.org/anthology/P06-1096/)，即用teacher模型中概率较高，且与y最相近的序列来训练：  
![Imgur](https://i.imgur.com/vmae0NV.png)  
sim是用来表示近似度的函数（如Jaccard similarity或BLEU）。按照local updating的做法，就可以从beam search近似选择序列：  
![Imgur](https://i.imgur.com/bzCL7fr.png)  
&Tau;<sub>K</sub>代表beam search中的K-best list。sim用的则是[sentence-level BLEU](https://www.aclweb.org/anthology/W14-3346/)。

定义用y_va训练知识蒸馏的过程可以如下描述：  
假定数据分布D中有一个**真实结果**t（无法观察到），再假定有一个受**噪音**影响的观察结果y：
1. t ~ D
2. y ~ &epsilon;(t)， 其中&epsilon;为独立的noise function，会小概率随机替换掉t中的单词。

此时理想的student分布应当为混合分布：  
![Imgur](https://i.imgur.com/GMBJPrh.png)  

由于噪音的影响，D导出的高概率会分布在y附近，因此最大化的结果并非y也并非y_hat。可以看出，对于这个混合分布，y_va是一个合理的近似，框架看图1（right），可视化结果见图2。

![Imgur](https://i.imgur.com/AEZgl6G.png)  
运行beam searrch之后，将final hidden state用t-SNE表示了出来。轮廓表示的是corresponding（smoothed） probability。本例中，beam search的最高答案（绿色）距离真实答案（red）很远。所以训练的时候用的是beam中存在的，和真实答案有highest sim的答案（purple）。

## 实验设置
### 数据集
- high resource，
  - 训练集：WMT2014（En-Ge）
  - dev set：newstest2012/newstest2013
  - test set：newstest2014
  - 用50k个高频词，其他作为UNK。
  - teacher model：4 x 1000 LSTM
  - student model：2 x 300和2 x 500
- low resource
  - 训练集：IWSLT2015（Thal-En）
  - dev set：2010/2011/2012
  - test set：2012/2013
  - 词汇25k
  - teacher model：2 x 500（performed better than 4 x 1000 和 2 x 750）
  - student model：2 x 100，
其他训练细节参考[Luong](https://arxiv.org/abs/1508.04025)
### 实验参数
用multi-bleu.perl计算BLEU
- Word-KD：用**原始数据**和**cross-entropy of the teacher distribution**训练  
&alpha;&isin;{0.5，0.9}，发现0.5最佳。
- Seq-KD：用**teacher generated data**训练  
beam size K=5（增加这个参数并没有效果）
- Seq-Inter：用teacher的beam search（K=35）中BLEU最高的作为数据训练。  

在**原始数据**和**Seq-KD数据**训练前都对预训练模型以rate 0.1实施了fine-tuning，在En-De中为了效率仅仅用了部分（~50%）。

## 实验结果
![Imgur](https://i.imgur.com/PUax1L5.png)

### decoding速度
![Imgur](https://i.imgur.com/t7q5N1R.png)

### Weight Pruning
学习时大部分的参数都在embedding，可是在rum-time时embedding layer只起到一个查询的作用，因此考虑是否能在student模型上剪枝。[See](https://www.aclweb.org/anthology/K16-1029/)研究过大型神经网络80%~90%的参数都可以剪枝，而且对结果影响不大。  
于是本文针对student模型，减去几个绝对值最小的x%个参数后，用0.2的训练rate重新训练。对Seq-KD data的fine-tuning则是0.1。

![Imgur](https://i.imgur.com/BH1R1rt.png)

且由此发现，Knowledge distillation和Weight pruning是一个正交关系。

## 数据压缩的相关研究
- Low rank factorizations of weight matrices
- sparsity-inducing regularizers
- binarization of weights
- [weight sharing](https://arxiv.org/abs/1510.00149)

本文用到的技巧
- [local updating](https://www.aclweb.org/anthology/P06-1096/)
- [hope/fear traning](http://jmlr.csail.mit.edu/papers/volume13/chiang12a/chiang12a.pdf)
- [SEARN](https://arxiv.org/abs/0907.0786)
- [Dagger](https://arxiv.org/abs/1011.0686)
- [minimum risk training](https://arxiv.org/abs/1512.02433)

