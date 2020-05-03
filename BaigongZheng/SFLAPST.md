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

