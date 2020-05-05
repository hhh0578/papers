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
