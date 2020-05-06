# [OPTIMAL COMPLETION DISTILLATION FOR SEQUENCE LEARNING](https://arxiv.org/abs/1810.01398)
## 已有研究
### 自然语言处理和语音识别
- S2S学习，encoder-decoder结构，传统的multi-state pipelines。
- 图像讲解，语法解析，内容总结，程序语法。
### 学习算法
- convolution，self attention，MLE，[label smoothing](https://arxiv.org/abs/1701.06548)，[scheduled sampling](https://arxiv.org/abs/1506.03099)。

本文从search-based structured prediction和[policy distillation](https://arxiv.org/abs/1509.00685)得到了灵感，找到了一种用edit distance优化s2s模型的算法。`给定任意序列前缀，可以高效得出与真实结果最为相近的所有后缀。`。总结如下：
- 用于训练的前缀，是从模型中取样生成的。
- 利用**动态编程**，可以求出每个前缀的最优后缀，即和真实答案的**编辑距离**最短。
- 本文最大化每个最优后缀第一个token的average log probability，来教模型扩张前缀。

本文的贡献如下：
- 提出了OCD，利用**编辑距离**优化s2s模型的独立算法，可用于长句子多词汇的实际数据，而且该方法在远优于MLE。
- 给定译文长度m，原文长度n，算法复杂度为O(nm)，而且能找出generated sequence的所有最优extension。
- 本文展示了OCD在e2e语音识别中的效用。
## 背景：用MLE进行序列学习
给定数据集D&equiv;{(x,y\*)<sub>i</sub>}<sup>N</sup><sub>i=1</sub>，我们想要知道一组从输入x到输出y\*&isin;Y的映射x&rarr;y。其中Y代表有限词汇表V中的单词所组成的可变长序列。要计算这种映射，通常使用的方法是优化条件分布的参数P<sub>&theta;</sub>。于是概率模型P<sub>&theta;</sub>下的sequence prediction就由exact or approximate inference表示为：  
![Imgur](https://i.imgur.com/dhRcSKA.png)  
于是标准的求解方法就类似有师学习的分类器，最大化conditional log-likelihood objective来计算![Imgur](https://i.imgur.com/5XTjxcT.png)。这种方法叫做`Maximum Likelihood Estimation`。

aoturegressive模型是给定原文，一个词一个词地求译文的概率，通常是left-to-right。用特殊的*end-of-sequence标志*来控制译文长度。其条件概率用chain rule可以写作：  
![Imgur](https://i.imgur.com/D7r6nBz.png)  
p<sub>D</sub>代表empirical data distribution，在数据集D上均一分布。

本文则是提出了可以用于任意autoregressive s2s模型结构的目标函数。

### MLE在autoregressive模型中的局限
为了最大化conditional log-likelihood，递归s2s模型从译文中取前缀y\*<sub>&lt;t</sub>，用于训练下一个token y\*<sub>t</sub>的概率。然而实际训练时，模型却只能根据自己生成的前缀y_hat<sub>&lt;t</sub>来预测下一个单词，这就导致了[exposure bias](https://arxiv.org/pdf/1511.06732.pdf)。

总结下来，MLE在autoregressive s2s模型上由两个局限：
- 学习时和预测时的前缀有所不同。
- 训练误差和评价方式之间有所不同。训练用的log概率，评价用的却是其他方式（比如语音识别用的时**编辑距离**）
### Optimal Completion Distillation


