# [NON-AUTOREGRESSIVE NEURAL MACHINE TRANSLATION](https://arxiv.org/abs/1711.02281)

## 介绍
现存的翻译模型都是利用已生成的序列计算新的单词，本文提供一种**非递归**方法避免这种**递归**性质，并行地生成结果，能大幅度减少预测时地时间成本。利用knowledge distillation（利用输入信息作为潜在变量）和policy gradient fine-tuning，可以达到与**递归Transformer**同等地效果。利用三步训练策略，能有效优化模型。

本模型以[Transformer][transformer]为基础,针对encoder部分，添加了预测*fertilities*的模块，这是传统翻译模型中的重要部分。这些fertilities在训练过程中得到训练，在decoder预测的时候起效，在输出同步生成的时候起到一种全局约束的作用。

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
![Imgur](https://i.imgur.com/cNQekHu.png)
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
这里介绍一种模型——NAT，能完全并行地预测译文。看图2，模型由：Encoder stack，Decoder stack，Fertility predictor和Translation predictor构成。\
![Imgur](https://i.imgur.com/igqCvRH.png)

### Encoder Stack
在NAT中，encoder与传统Transformer保持不变。

### Decoder Stack
为了能够并行翻译，做了如下调整：

#### Decoder Inputs 
在NAT中，需要知道target序列会有多长。更关键的是，在训练和预测的时候不能输入已知的target。但是什么都不输入，或是只输入positional embeddings成效不佳。所以，就复制source序列来输入decoder，由于长度不一致，采用以下两个方法：

  - 均一复制source：给定要预测的target长度，将输入t复制到Round(T’t/T)的位置。
  - 利用fertilities复制source：参考图2，根据单词的fertility确定要复制几次，target的长度是fertility值的总和。
#### 无因果self-attention
由于输出的分布独立，不需要一步步喂target，decoder部分的self-attention可以不需要causal mask。相对的，会在query对自己的运算上加一个mask，这在实践中发现效果更好。

#### Positional attention
在multi-head attention的模块前加了一个multi-head的positional attention。其中*positional encoding*是Q和K，*decoder state*是V。这强化了位置关系。而且可以预测这对局部的顺序重组起到了好的作用。\
![Imgur](https://i.imgur.com/e9Dc6Kd.png)


### Modeling Fertility处理多峰问题
处理办法可以是在翻译步骤中导入潜在变量z：先从一个先验分布采样得到z，然后以z为条件进行非递归翻译。\
模仿[Martin](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2930890/)讨论过的一个方案，可以将这个潜在变量看作句子的“plan”，这隐藏变量应当由下列性质：

  - 给定输入输出，应当容易求出这个值以用于训练end-to-end模型。
  - 作为上下文条件出现越频繁越好，以表达不同输出的先后关联，这么一来输出本身包含的信息就可以更独立。
  - 变化不应当直接表现在输出上，如p(y | x,z)，这会导致训练繁杂，因为这个函数会用于近似计算。

公式3的举例满足了1和3。此处推荐一个*fertilities*，这种方法得到一个整数，代表原句与译文的序号关系，这曾用于[hard alignment algorithm](https://www.aclweb.org/anthology/J93-2003/)。\
本文的NAT一个重要特性就是能在利用predicted fertilities复制encoder的输入时自然而然地生成一个隐藏变量。严谨点说就是给定一个原句X，能够得到条件概率下地译文Y。\
![Imgur](https://i.imgur.com/loelHeD.png)\
其中F代表了fertility序列，综合就是Y的长度，x{f}决定了x复制f次。
> ![Imgur](https://i.imgur.com/yv4csS4.png)

#### Fertility Prediction


[transformer]:(https://arxiv.org/abs/1706.03762)
