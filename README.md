# papers

[2016 Sequence-Level Knowledge Distillation](https://github.com/hhh0578/papers/blob/master/YoonKim/Sequence-Level%20Knowledge%20Distillation.md)  
用teacher网络训练student网络的方法，利用teacher生成的结果作为target来训练student，以更好的模仿teacher网络。

[2017 Learning to Translate in Real-time with Neural Machine Translation](https://github.com/hhh0578/papers/blob/master/JiataoGu/Real-time%20NMT.md)  
组合上下文向量、decoder状态和候补单词作为环境参数训练RL模型，选择是否采用候补单词（WRITE）或是不采用（READ）。

[2018 Low-Latency Neural Speech Translation](https://github.com/hhh0578/papers/blob/master/Low-Latency%20Neural%20Speech%20Translation.md)  
利用整句的数据集构造一个部分句的数据集混在一起训练，以减少翻译途中的句子错误。

[2018 NAT (Non-Autoregressive Transformer)](https://github.com/hhh0578/papers/blob/master/JiataoGu/Non-Autoregressive%20Transformer.md)  
构造一个fertility，翻译时decoder可以不用递归按序生成译文，而是从fertility一步生成完整译文。

[2019 LevT (Levenshtein Transformer)](https://github.com/hhh0578/papers/blob/master/JiataoGu/Levenshtein%20Transformer.md)

[2019 Simpler and Faster Learning of Adaptive Policies for Simultaneous Translation](https://github.com/hhh0578/papers/blob/master/BaigongZheng/SFLAPST.md)  
先用预训练的模型生成一个读写的操作列，然后用另一个神经网络以这个操作列为参考训练何时读何时写。

