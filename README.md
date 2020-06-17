# papers

[2016 Sequence-Level Knowledge Distillation](https://github.com/hhh0578/papers/blob/master/YoonKim/Sequence-Level%20Knowledge%20Distillation.md)
用teacher网络训练student网络的方法，利用teacher生成的结果作为target来训练student，以更好的模仿teacher网络。

[2017 Learning to Translate in Real-time with Neural Machine Translation](https://github.com/hhh0578/papers/blob/master/JiataoGu/Real-time%20NMT.md)
组合上下文向量、decoder状态和候补单词作为环境参数训练RL模型，选择是否采用候补单词（WRITE）或是不采用（READ）。

[2017 REGULARIZING NEURAL NETWORKS BY PENALIZING CONFIDENT OUTPUT DISTRIBUTIONS](https://arxiv.org/abs/1701.06548)
与label smoothing类似的regularizing算法。

[2018 Low-Latency Neural Speech Translation](https://github.com/hhh0578/papers/blob/master/%5BJanNiehues%5DLow-Latency%20Neural%20Speech%20Translation.md)
利用整句的数据集构造一个部分句的数据集混在一起训练，以减少翻译途中的句子错误。

[2018 NAT (Non-Autoregressive Transformer)](https://github.com/hhh0578/papers/blob/master/JiataoGu/Non-Autoregressive%20Transformer.md)
构造一个fertility作为译文蓝图，翻译时decoder可以不用递归按序生成译文，而是从fertility一步生成完整译文。

[2018 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://github.com/hhh0578/papers/blob/master/%5BJacobDevlin%5DBERT.md)
利用MLM方法，在无师数据上预训练transformer之后，仅需要微调模型即能适应大多数task。

[2019 OPTIMAL COMPLETION DISTILLATION FOR SEQUENCE LEARNING](https://github.com/hhh0578/papers/blob/master/%5BSaraSabour%5DOCD.md)
因为目标函数在训练时只能从真实target预测下一个目标（MLE），和实际测试时候用生成target预测下一个目标的过程有偏差，于是在训练时也利用生成的target前缀预测下一个目标（OCD）。其中构造了一个Q-values，以导出能用以模仿学习的函数。

[2019 LevT (Levenshtein Transformer)](https://github.com/hhh0578/papers/blob/master/JiataoGu/Levenshtein%20Transformer.md)

[2019 Simpler and Faster Learning of Adaptive Policies for Simultaneous Translation](https://github.com/hhh0578/papers/blob/master/BaigongZheng/SFLAPST.md)
先用预训练的模型生成一个读写的操作列，然后用另一个神经网络以这个操作列为参考训练何时读何时写。

[2019 STACL](https://github.com/hhh0578/papers/blob/master/%5BMingboMa%5DSTACL.md)
针对同声传译定义了prefix to prefix框架，并且提出了其中最简单的情况，即wait-k策略，还给出了transoformer的实现思路。并且提出了一种新的延迟计算方法Average Lagging。

[2020 Lite Transformer](https://github.com/hhh0578/papers/blob/master/%5BZhanghaoWu%5DLITE%20TRANSFORMER%20WITH%20LONG-SHORT%20RANGE%20ATTENTION.md)
为了减少计算量，将传统transformer的attention层替换为attention模块和convolution模块，分别处理全局信息和局部信息，强化transformer在NLP方面的专业性。

[2020 Monotonic Attention](https://github.com/hhh0578/papers/blob/master/%5BXutaiMa%5DMONOTONICMULTIHEADATTENTION.md)
在Transformer的en-de attention上运用了Monotonic手法，实现同声传译需要的效果。给出了MMA-H和MMA-IL两种方案，前者适用于streaming的场景，后者更加准确。
