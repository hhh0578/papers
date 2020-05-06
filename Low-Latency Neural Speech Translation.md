# [Low-Latency Neural Speech Translation](https://arxiv.org/abs/1808.00491)
## 已有研究
- [探究在training data和test data间的domain mismatch问题](https://arxiv.org/abs/1612.06140)。

为了解决speech翻译的时候ASR system造成的错误：
- [在训练集中导入人工corrupted input。](https://www.aclweb.org/anthology/P18-1163/)
- [直接利用ASR系统生成的lattices来训练。](https://www.aclweb.org/anthology/D17-1145/)

Multi-task学习在NLP问题中被广泛利用：
- [用multi-task挖掘语义信息](https://www.aclweb.org/anthology/W17-4708.pdf)

关于low-latency speech translation：
- statistical phrash-based model
- Gu的神经网络强化学习
- [revision strategy](https://secondhands.eu/wp-content/uploads/2016/07/Niehues2016.pdf)
## 介绍
在revision strategy中，随着新的单词输入，输出会进行修改，而最后的正确句子只有在原句全部读取之后才能正确输出。这就导致了一个延迟，而且由于训练时候的数据集是完整句子，这就导致途中生成的句子可能出现巨大偏差。  
本文的目标就是在句子被全部读取之前，以尽可能少的原句单词数得到正确结果。

## 部分翻译
本文首先探究生成partial sentence的parallel corpora。然后用这个数据，调整学习process来构建一个能像翻译完整句子一样翻译部分句子的模型。
### 生成Partial Parallel Corpora
现有的数据集并没有能够训练**部分翻译**的数据，于是本文首先给出一个能利用普通数据集生成部分语句数据的模型。如此一来就不需要搜集新的数据，而且能适用于其他各种工作。

#### 生成原句 
给定一个原句S=s1&hellip;sI，可以取样得到由**前i个单词**组成的S<sup>(i)</sup>=s1&hellip;si。

#### 生成译文
为了实现low-latency speech translation，部分语句的译文有以下限制：为了降低**延迟**，句子理应越长越好；对所有i’&gt;i，译文S<sup>(i)</sup>应当为S<sup>(i’)</sup>的子集。  
一种方法是直接用reference translation的整个句子当译文，但是从一个单词的原文生成整个译文不现实。因此这本文介绍两种方法来从reference translation中生成可用的译文子集。  
  - 第一种模型想法为`得到越多原句segments，就应当生成更多的target`。于是就让**译文按原文等比例取**。虽然有些语言之间会有单词顺序的问题，但是大部分语言是类似的。  
  可是这就导入了一个**噪音**。如果单词顺序不同，会强制模型猜测原文还没读取的单词。
  - 于是第二种方法，首先用[Giza++](https://www.aclweb.org/anthology/J03-1002/)将译文和原文对齐，然后取reference中的最长前缀，如此一来就不会有原文还没出现译文却出现翻译的情况了：  
  ![Imgur](https://i.imgur.com/wDgEUJN.png)
  
### 训练步骤
#### Multi-task训练
首先利用**人造数据**训练模型，然后用在speech translation上。因为这两个task都很简单，就先预训练一个标准NMT，然后用**部分语句**fine-tune。  
这种方法缺点就在于完整句子的翻译水平可能会下降，因为NMT趋向于迅速忘记上一次学习过的东西。为了让完整句子和部分句子都保持高翻译水平，本文采用multi-task学习，把这当作是两个不同的task。

本文随机采样让**部分语句**的数据集总数和**原数据**保持一致，让两边数据占比相同。然后用两边的task对NMT系统进行fine-tune。

#### sequence level优化
本文还用了[policy gradient methods](https://arxiv.org/pdf/1511.06732.pdf)强化学习来最大化[GLEU score](https://www.semanticscholar.org/paper/Google's-Neural-Machine-Translation-System%3A-the-Gap-Wu-Schuster/dbde7dfa6cae81df8ac19ef500c42db96c3d1edd)。这使得生成的句子不会太长。而且由于这种方法高方差，也用了[Rennie](https://arxiv.org/abs/1612.00563)的方法，利用greedy search的baseline减少方差。
## 实验结果
### 数据集
用的**语言组合**有：EnSp，EnFr，DeEn
- 数据集：Europarl和WIT-TED corpora
- 测试集：IWSLT evaluation campaign
- 所有系统都用in-domain TED data微调过。
- 在En&rarr;Sp和En&rarr;Fr用强化学习优化了GLEU scores。
- 在部分语句翻译上只用了TED corpus
- 训练用的[OpenNMT-py toolkit](https://arxiv.org/abs/1701.02810)
- 原句和译文都用了BPE
### 评价方式
- Bleu：由于ASR会自动分段，于是需要重组分段以配合reference，因此用了[Matusov](https://pdfs.semanticscholar.org/6de4/f789b2d56f40105b79692fc42809991040c2.pdf)的方法。
- 本文还评价了该模型是如何减少SLT模型的**修正次数**的。为此，本文输出了**ASR模型**和**翻译模型**的所有输出，对于每次的updated translation，本文给出了相邻的译文s<sub>t</sub>和s<sub>t+1</sub>并且计算了这个过程中所需要的re-writing。其中，修正的单词数是用译文s<sub>t</sub>的长度减去s<sub>t</sub>和s<sub>t+1</sub>之间的共同前缀长度（Word Up）。也给出了修正过的语句数（Messg. Up.）。
### 实验
**第一组** 
初始化了EnSp的翻译结果，见表1。给出了**整句翻译**的dev和test结果（Valid，Test）；给出了**所有possible prefixes**的翻译结果（TEDTest Partial）；给出了**ASR output**的翻译结果（SLT）。初始化实验中reference是按照原文比例取的。baseline系统仅仅用了**整句**训练。“Partial”系统是用**部分句子**微调过的。  
**第二组**
训练集中用了alignment-based method生成reference的**部分语句**。由于差别不大，后续实验继续用length-ratio based method。  
**第三组**
用BLEU目标函数强化训练。模型先是用cross-entropy训练，然后用RL训练。此处同样先用**整句**得到了一个baseline，然后用**整句**和**部分句**混合的数据进行了multi-task训练。  
![Imgur](https://i.imgur.com/XOQiOLH.png)

**EnFr实验**
和之前一样训练了一个baseline，然后用baseline训练了一个RL。验证时时用了一个**整句**的数据集（tst2010 Fianl）、一个**混合**的数据集（tst2010 Mix）和一个**ASR**的数据集（SLT）。  
![Imgur](https://i.imgur.com/e3fG6J4.png)

**DeEn**
这个翻译效果不佳，也许时因为语序重组更严重。  
![Imgur](https://i.imgur.com/Jjk14WK.png)
