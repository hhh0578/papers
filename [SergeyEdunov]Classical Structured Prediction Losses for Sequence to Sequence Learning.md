# [Classical Structured Prediction Losses for Sequence to Sequence Learning](https://www.aclweb.org/anthology/N18-1033/)
[参考代码](https://github.com/pytorch/fairseq/tree/classic_seqlevel)
## 介绍
s2s的训练通常利用token-level似然函数，然而test的时候模型生成的是整句。为了解决这个问题，近期有人着眼于sequence-level模型
- [REINFORCE](https://research.fb.com/publications/sequence-level-training-with-recurrent-neural-networks/)
- [actor-critic](https://arxiv.org/abs/1409.0473)
- [beam search optimization](Sequence-to-Sequence Learning as Beam-Search Optimization)
## S2S学习
本文用[conv网络](https://arxiv.org/abs/1705.03122)代替recurrent网络，这是出于计算和精度考虑。然而本文结论同样能用于普通recurrent模型。
### 符号
- 原句为**x**，译文为**t**。在某些目标函数中用u\*代替，如从BLEU或ROUGE打分的候补U中取u的时候。
- 原句x有m个单词，输出状态z长度为m，生成译文u有n个单词。decoder中隐藏状态h<sub>i</sub>由前一个状态h<sub>i-1</sub>，前一个单词的嵌入g<sub>i-1</sub>，以及从z得来的输入c<sub>i</sub>得到。attention context c<sub>i</sub>通过每一步对z取权重和得到。
- 本文的encoder-decoder是[Gated convolutional神经网络](https://arxiv.org/abs/1705.03122)。快速生成译文在本实验中尤为重要，因为求sequence-level losses需要在训练时生成译文。
- encoder和decoder共用一个简单block，能从固定长度的输入得到中间状态，随后本文在两边堆了几个不同的block，每个block包含一个一维convolution，从k个特征的向量输入得到一个输出向量。随后那一层处理这k个输出元素，然后输出到GLU模块。 在decoder，本文依靠causal convolution。
## 目标函数
本文通过训练上述模型比较了几个目标函数，或是利用单独token，或是利用整句，或是两者结合。图片1可以一览所有。  
![Imgur](https://i.imgur.com/osKZyDh.png)
### Token-Level objectives
#### Token Negative Log Likelihood（TokNLL）
#### Token NLL with Label Smoothing（TokLS）
### Sequence-Level Objectives
句子级别的objectives需要生成候补并且为多个候补打分，这允许我们能直接用BLEU或是ROUGE进行训练，但这在计算上十分昂贵。  
可是这种函数通常定义了整个输出空间中的可能性，这在本实验中无法计算。因此，本实验取了输出空间中的一个子集来计算，后续会讨论这个问题。
#### Sequence Negative Log Likelihood（SeqNLL）
类似TokNLL，我们可以最小化整句的NLL而非单词的。序列u的log似然是单词的log似然求和，并且通过长度归一化处理防止偏袒短句子。  
![Imgur](https://i.imgur.com/7ttp4qC.png)  
目标是从候补中选择一个pseudo reference最大化BLEU或ROUGE，其gold reference为  
![Imgur](https://i.imgur.com/It4FOHA.png)  
实际中通常计算BLEU，初始化counts设定为1（除了unigram counts），防止geometric mean被zero-valued n-gram match counts主导。
#### Expected Risk Minimization（Risk）
该目标函数最小化了候补序列的cost期待值。本实验使用task-specific cost function，如1-BLEU（t，u）。该方法在无法保证取得reference的时候非常有用，不过于传统短语模型相比，神经网络的s2s模型很少有这个问题。
Risk目标函数和REINFORCE目标函数有些类似，旨在最大化期待值，或说是回报。不同点在于1）强化学习得到的是单句的期待值，而Risk函数考虑的是多句子；2）强化学习需要baseline reward来确定当前序列的梯度方向，而Risk估求的是候补集合的期待值。3）强化学习中每个单词的baseline reward都不一样，而Risk之中每个单词的expected cost在是一样的。
#### Max-Margin
这是structured prediction的传统loss，该方法操作的是**候补译文中blue最高的选项u_hat**和**参考译文**之间模型打分的margin。而比起人类译文t，pseudo reference u\*效果更好。u\*代表有着最高BLUE值的候补。样本间的margin size由u\*和u_hat的cost得到。实验中对验证集用&beta;调整margin  
![Imgur](https://i.imgur.com/B6c4dwm.png)  
这个loss在最后取softmax之前计算非归一化的分数  
![Imgur](https://i.imgur.com/HJR9ihS.png)  
#### Multi-Margin
MaxMargin仅更新候补集合中两个元素。因此我们考虑MultiMargin，操作所有候补集合u和参考文之间的margin。类似MaxMargin，同样用pseudo refrence u\*替换了t。
#### Softmax-Margin
本方法利用cost扩张了SeqNLL中的exp。直观上就类似于高cost的输出需要高惩罚。
### Combined Objectives
首先，考虑weighted combination，如TokLS和Risk的话  
![Imgur](https://i.imgur.com/i7ihWzy.png)  
其中&alpha;是在held-out validation set上调整的scaling constant。

其次，考虑constrained combination，对于任何输入，或是采用词汇级或是采用句子级。目的是同时保持单词准确地同时达到整体优化效果。例如，当模型的token loss好过baseline模型的时候，切换成sequence loss。  
![Imgur](https://i.imgur.com/GX4m0LK.png)  
本实验中用的是一个固定的baseline模型，该模型事先用token-level训练至收敛。
## 候补生成策略
句子级目标函数都考虑的是可能输出的语句构成的空间，而这会导致无法计算，因此需要选择K个候补构成子集运算。本文采用两种策略。
- beam search  
  广度优先取前K个候补的搜索方式，该方法在机器学习中实际上是decoding步骤中采用的策略。
- sampling  
  从模型的条件分布中独立取K个。不同于仅仅取高概率输出的beam search，取样方法得到的候补更具有多样性。
- online  
  在online设定中，训练时每碰到一个输入x就重新生成候补。
- offline  
  在offline设定中，候补仅仅在训练前生成，且不再改变。
最后，有些研究把参考文加入候补，但由于模型学习时会给所有模型生成的候补低概率，却依旧给参考文高概率，我们发现这会导致学习不稳定。因此我们并不把译文加入候补集。
## 实验设定
### 翻译实验
- 训练集160K，IWSLT14 German to English。
- 验证集7K，从训练集随机取出。
- 测试集，tst2010，tst2011，tst2012，tst2013，dev2010
- 全部小写
- BPE，14,000 types
- 评价方式：忽视大小写的BLEU。
#### 其他实验设定
- 训练集35.5M，WMT14 English-French
- 验证集26,658，从训练集中取。
- 测试集，newstest2014
- 删除单词超过175个的长句，删除原句与译文长度比超过1.5的句子组合。
- 词汇量基于40K BPE types
#### 模型设定
- 调整fairseq-py实现目标函数。
- 4层conv encoder层，3层conv decoder层，核宽为3，隐藏层维宽为256
- 优化算法为[Nesterov's accelerated gradient](https://dl.acm.org/doi/10.5555/3042817.3043064)模型，训练速率为0.25，momentum为0.99。
- [Gradient vector归一化为norm 0.1](https://arxiv.org/abs/1211.5063)
- 单词级训练200 epochs，之后进行annealing，每个epoch学习速率10倍减小，直至低于0.0001。
- 句子级训练使用annealing之前的单词级模型，视情况而定额外训练10~20 epochs。
- 一个batch含有8K toknes，每个mini-batch按照非padding的数量进行归一化处理。本文对所有层进行[weight normalization](https://arxiv.org/abs/1602.07868)，除了查询表。
- 除了embedding和decoder output的dropout，convolution block也有0.3的dropout。
- 从验证集结果最好的设定来进行测试
- 在语句级loss中所有的打分和概率都针对长度归一化处理以便比较。
- 在训练过程中生成候补的时候，候补长度限制在200词以下。
- 候补数量通常为16，在ablations实验中为求效率设为5.
### 概括实验
- 训练集3.8M，Gigaword corpus，按照[Rush et al.](https://arxiv.org/abs/1509.00685)的方法预处理，
- 验证集190K，由Rush的预处理得到。
- 测试集2,000，同样预处理。
- 评价方式，ROUGE-1，ROUGE-2，ROUGE-L
- 类似[Ayana et al.](https://arxiv.org/abs/1604.01904)，利用一个30k词汇的source和target。
- 12层encoder layers和decoder layers，隐藏层维宽256，核宽3。
- 一个batch有8,000单词，学习速率0.25，训练20epoch，然后annealing。
