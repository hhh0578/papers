# [Simpler and Faster Learning of Adaptive Policies for Simultaneous Translation](https://arxiv.org/abs/1909.01559)
## 已有研究
- [rule-based decoding algorithm](https://arxiv.org/abs/1606.02012)：启发式decoder并不能或用hidden层中的信息。
- [MT moedel with extended vocabulary](https://arxiv.org/abs/1906.01135)：需要从restricted dynamic oracle学习adaptive policy，而其size过于巨大只能近似。
- separate policy：或是用到不稳定不高效的**强化学习**，或是训练只能autoregressive而低效。
而且以上方法都无法满足test时对不同延迟的需求。
## 介绍
本文提出了一种学习adaptive policy的简单有师学习框架，且可以控制延迟。这个框架建立在数据pair对应的action列基础上，而这READ/WRITE操作序列用预训练过的NMT生成。
## 生成操作序列
同步翻译policy有两种操作：READ（读入一个原文单词），WRITE（输出一个译文单词）。而(s,t)对应的一列**操作列**就表示了翻译过程。因此操作序列中必定有|t|个WRITE。表格1给出了一个例子。\
![Imgur](https://i.imgur.com/UsL2NUr.png)

其中有些**操作列**的翻译过程不好。比如没有READ的操作列得不到任何原文信息；而|s|个READ在所有WRITE前面的话会导致大延迟。因此理想的操作列应有以下两个特征：
- 在翻译过程中没有猜测。在执行WRITE的时候必须有足够的原文信息让MT模型给出正确译文。
- 延迟越低越好。WRITE越早越好。

假定预训练的NMT模型能够在不完全信息下给出正确译文，于是有一种简单的方法能对sentence pair (s,t)生成**操作列**：如果正确的译文单词在rank中排位够靠前，就假定此时的原文信息已经足够。具体看算法1。\
![Imgur](https://i.imgur.com/ilGBttf.png)\
r是正整数，M是预训练的NMT模型，s<sub>&le;</sub>代表前i个原文单词，rank<sub>M</sub>代表在原文为s<sub>&le;</sub>的情况下target单词t<sub>j</sub>在预测中的排名。

调节r参数能得到**质量**和**延迟**两个指标下想要的操作列。但生成操作列时仍需要两个约束：
- 语言的语序问题依旧可能造成巨大延迟：用Average Lagging（AL）进行过滤。
- 预训练的模型可能太aggressive，有些句子原文信息还没读完就翻译完了，也许是因为本文模型在同一个数据下进行训练的关系：只接受在最后一次WRITE的时候读入了所有原文的操作列。
## 同步policy的有师学习框架
以[Transformer](https://arxiv.org/abs/1706.03762)为基础，和[wait-k](https://arxiv.org/abs/1810.08398)模型一样对encoder的**隐藏层**进行处理。policy在i step的输入o<sub>i</sub>由以下三部分组成
- h<sup>s</sup><sub>i</sub>：i step时原文第一个单词的last-layer hidden state。
- h<sup>t</sup><sub>i</sub>：i step时译文第一个单词的last-layer hidden state。
- c<sub>i</sub>：i step时current input target word在所有decoder attention层上的cross-attention scores，取所有current source words的均值。
即o<sub>i</sub>=\[h<sup>s</sup><sub>i</sub>, h<sup>t</sup><sub>i</sub>,c<sub>i</sub>\]
a<sub>i</sub>代表操作列a中的第i项操作。pocily的决定取决于所有之前的输入o<sub>&le;i</sub>和所有之前的操作a<sub>&lt;i</sub>，于是需要最大化的概率公式如下：\
![Imgur](https://i.imgur.com/XENvANb.png)\
其中p<sub>&theta;</sub>代表参数&theta;组成的policy分布。
## 延迟可控的Decoding
这里介绍一种简单的延迟可控计算技巧，无需重新训练policy model。

设定&rho;为概率阈值。每一步中仅当READ的概率大于&rho;的时候踩选择READ，否则选择WRITE。这个阈值就权衡了**质量**和**延迟**。
## 实验
### 数据集
- 训练集：WMT15 （EN&harr;DN）
- validation set：newstest-2013
- test set：newstest-2015
- BPE处理
- 句子pair长度小于50。
### 模型设定
- Transformer（[PyTorch-based Open-NMT](https://arxiv.org/abs/1701.02810)）
- **原文**添加了<eos> token。
- recurrent policy model：维度64、512 units的GRU layer；维度2，fully-connected 的ReLU激发函数；softmax函数生成action分布。
- 质量评价：BLEU
- 延迟评价：Averaged Lagging
  
### 操作列生成的影响
本文结果设定rank r=50和filtering latency &alpha=3;图1展示了&rho;的影响，点出来的位置时&rho;=0.5。\
![Imgur](https://i.imgur.com/eLOIb1v.png)\
实验表明&alpha;影响较大，rank影响不大。
### 比较结果
- [Wait-If-Worse/Wait-it-Diff（WIW/WID）](https://arxiv.org/abs/1606.02012)
  只用于预训练model。预先读取s<sub>0</sub>个原文单词，仅当概率最大的target word概率下降时READ（WIW）；或仅当概率最大的target word变了的时候READ（WID）。
- [RL method](https://www.aclweb.org/anthology/E17-1099/)
  AP-based reward function效果更好，本文仅仅比较CW-based。
- [wait-k model和test-time wait-k method](https://www.aclweb.org/anthology/P19-1289/)
  test-time直接用预训练的model，wait-k的以相同模型构造重新训练。
  
图2展示了比较结果。\
![Imgur](https://i.imgur.com/Hiz4IGQ.png)
