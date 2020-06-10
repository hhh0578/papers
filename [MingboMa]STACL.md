# [STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency using Prefix-to-Prefix Framework](https://arxiv.org/abs/1810.08398)
[参考代码？](https://simultrans-demo.github.io/)

## 介绍
- prefix2prefix框架专为同步翻译定制，无需从整句翻译训练。
- 训练和预测都只需要在一个模型上进行，直接预测译文而无需给出具体的伪原文。
- 对于特殊情况，给定wait-k策略来确保延迟需求。
- 该方法适用于任何s2s模型。
## 整句翻译
## prefix2prefix和wait-k策略
### prefix2prefix结构
`定义1.  定义g（t）为单调非减函数，该函数表示生成第t个译文时读取的原文个数。`
> ![Imgur](https://i.imgur.com/g3W6Lki.png)

`定义2.  定义“中止”步骤，这表示读取全部原文时译文的个数。`
> ![Imgur](https://i.imgur.com/GZ79Q9p.png)
### Wait-k策略
我们提出prefix2prefix框架下最为简单的一种策略——wait-k策略——原文总是提前译文k个单词。  
![Imgur](https://i.imgur.com/qq3pAja.png)  
如图3，是k=2时的情况，用公式表示就是
![Imgur](https://i.imgur.com/FSUiOXF.png)  
此时&tau;（|x|）=|x|-k+1，这代表剩下的译文和整句翻译一样（包括眼下的单词），这部分成为“尾巴（tail）”，可以用beam-search，但之前的操作全部greedy-search。
## 新的延迟评价方法：Average Lagging
### 现有方法：CW和AP
### 新方法：AL
![Imgur](https://i.imgur.com/yrbZ3h8.png)  
其中 `r=译文长度/原文长度`。
## 实现细节
### 整句翻译transformer
### 训练同步翻译transformer
encoder调整mask，让每个原文单词只能看到它之前的单词（类似decoder的self-attention）。  
![Imgur](https://i.imgur.com/W6OH6I9.png)  
