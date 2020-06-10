# [Learning to Translate in Real-time with Neural Machine Translation](https://arxiv.org/abs/1610.00388)
[源代码](https://github.com/nyu-dl/dl4mt-simul-trans)
## 关于同步翻译的先行研究
- [用启发式decoder](https://arxiv.org/abs/1606.02012)
  无法学习合适的翻译时机。
- [训练一个独立segmentation network决定何时执行标准NMT模型](https://www.semanticscholar.org/paper/Simultaneous-Machine-Translation-using-Deep-Satija-Pineau/ee1eacd383ffaf0b4b00d7326dd4e6efc80dbb74)
  无法模拟同步翻译的学习。
## 介绍
本文给出一种框架，定义翻译过程分两个操作：**读**和**写**。以此为基础，把**NMT系统**与**R/W决策**联系起来。翻译过程可以用图1表现。而为了学习如何决策，采用了**强化学习**策略。也适用了beam-search方法。\
![Imgur](https://i.imgur.com/sGhahkH.png)
## 定义问题
假定有输入X={x<sub>1</sub>,&hellip;x<sub.Ts</sub>},**READ**操作从input buffer按时间顺序读取原文，**WRITE**操作则翻译y到output buffer，得到译文Y={y<sub>1</sub>,&hellip;,y<sub>Tt</sub>}，并有一个操作序列A={a<sub>1</sub>,&hellip;,a<sub>T</sub>}.其中时间T=T<sub>s</sub>+T<sub>t</sub>。\
评价标准：Q(Y)衡量翻译质量，如[BLEU](https://www.aclweb.org/anthology/P02-1040)；D(A)评价时间延迟。
## 用NMT实现同步翻译
框架如图2所示，分为**环境**和**机器人**两部分。\
![Imgur](https://i.imgur.com/6sFvGPp.png)
### 环境
- **Encoder：READ**
encoder将输入X={x<sub>1</sub>,&hellip;x<sub.Ts</sub>}转化w为上下文向量H={h<sub>1</sub>,&hellip;h<sub>Ts</sub>}。通常NMT会用到**双向RNN**，不过不适合同步翻译，因此用单向RNN：\
![Imgur](https://i.imgur.com/ZSIo9qi.png)
- **Decoder：WRITE**
参考MT实用attention-based decoder。不过，仅仅参考已经读取的input：\
![Imgur](https://i.imgur.com/Nf9horI.png)\
其中&tau;，z<sub>&tau;-1</sub>和y<sub>&tau;-1</sub>分别代表decoder的前一个状态和输出。H<sup>&eta;</sup>表示非完整的input states。\
WRITE操作要计算下一个单词的概率，用greedy decoding：\
![Imgur](https://i.imgur.com/SJyV7fg.png)\
注意：y<sup>&eta;</sup><sub>&tau;</sub>和z<sup>&eta;</sup><sub>&tau;</sub>对应H<sup>&eta;</sup>，而且是y<sub>&tau;</sub>和z<sub>&tau;</sub>的candidate。而**机器人**将会决定是是否采用这个candidate。
### 机器人
训练后的**机器人**要能够在ovservation O={o<sub>1</sub>,&hellip;,o<sub>T</sub>},下做出一系列决定A={a<sub>1</sub>,&hellip;,a<sub>T</sub>，a<sub>t</sub>&isin;{WRITE,READ}，以此推进环境的变化。

- **Ovservation**
如图2所示，组合**当前上下文向量**c<sup>&eta;</sup><sub>&tau;</sub>，**当前decoder状态**z<sup>&eta;</sup><sub>&tau;</sub>以及**候补单词**y<sup>&eta;</sup><sub>&tau;</sub>三项成为一个ovservation，即o<sub>&tau;+&eta;</sub>表示current state。
- **Action**
参考[Grissom的先行研究](https://www.aclweb.org/anthology/D14-1140/)，定义以下两个动作：

  - READ：机器人拒绝这个candidate，等待encoder从input buffer中读取下个单词。
  - WRITE：机器人接受这个candidate，并将这个预测输入output buffer。
- **Policy**
如何从ovserbation中决定**操作**。本实验采用从RNN神经网络参数化的stochastic plicy &pi;<sub>&theta;</sub>：\
![Imgur](https://i.imgur.com/Ioqrwin.png)\
其中s<sub>t</sub>是机器人的**内在状态**，根据操作a<sub>t</sub>的分布导出。由这个机器人的policy可以得到算法1，输出结果是**译文**和一列observation-action pairs。\
![Imgur](https://i.imgur.com/jYOyetG.png)
## 学习
改实验用**强化学习**，具体就是用policy gradient algorithm加上variance reduction和regularization techniques。
### 预训练
先正常训练NMT。
### 报酬函数
机器人每一步都会从(o<sub>t</sub>,a<sub>t</sub>)获得一个报酬r<sub>t</sub>。为了训练同步翻译，报酬需要同时考虑quality和delay。

- **Quality**
BLEU是对BLEU<sup>0</sup>添加了BP项以惩罚短句的结果。\
![Imgur](https://i.imgur.com/jX9FoZT.png)\
Y<sup>\*</sup>代表**参考结果**，Y代表**输出**。

  然而实际实验中光用BLEU效果不佳。于是在局部BLEU予以变化，**质量**的报酬函数：\
![Imgur](https://i.imgur.com/Nw0uqOw.png)
  > ![Imgur](https://i.imgur.com/dWZx7nO.png)

  即在READ的时候报酬为0。

- **Delay**
为简单起见，每个单词的延迟时间认定为一致。
  - [Average Proportion（AP）](https://arxiv.org/abs/1606.02012)\
  ![Imgur](https://i.imgur.com/4vVFgOt.png)\
  s(&tau;)表示输出单词y<sub>&tau;</sub>的时候读取的原文单词个数。该方法用来表示global delay。
  - Consecutive Wait length（CW）\
  ![Imgur](https://i.imgur.com/OZqmyFY.png)\
  该方法用来表示local delay。
  - Target Delay\
  针对d和c，定义d<sup>\*</sup>和c<sup>\*</sup>表示**目标延迟**，因为不同场景可能需要的延迟标准不同。于是**延迟**的报酬函数写作如下：\
  ![Imgur](https://i.imgur.com/IGQKPJC.png)
- 权衡**质量**和**延迟**
同步翻译要求权衡这两项指标，显然这两项是一对此消彼长的矛盾关系。本文定义报酬：\
![Imgur](https://i.imgur.com/9touHLs.png)\
调整的参数为公式9中的相关系数&alpha;和&beta;，以及**目标延迟**d<sup>\*</sup>和c<sup>\*</sup>。
### 强化学习
- [Policy Gradient](https://link.springer.com/article/10.1007/BF00992696)\
固定预训练好的NMT模型，用Policy gradient训练机器人。Policy gradient将未来的累计报酬的**数学期望**![Imgur](https://i.imgur.com/D0fdsUr.png)最大化，得到梯度：\
![Imgur](https://i.imgur.com/wwfzrie.png)\
其中![Imgur](https://i.imgur.com/jSaMXyy.png)代表**未来累计报酬**。实际计算时候会从当前策略&pi;<sub>&theta;</sub>采样数个action trajectories来计算相应的报酬。
- [Variance Reduction](https://arxiv.org/abs/1402.0030) \
直接计算policy gradient会引出巨大方差，导致学习不稳定不高效，于是本文采用了variance recution techniques。
  > ![Imgur](https://i.imgur.com/HD9jqlA.png)
  
于是整个算法如算法2所示：\
![Imgur](https://i.imgur.com/DHu2naQ.png)
## 同步beam search
仅在连续WRITE的时候使用Beam Search。\
![Imgur](https://i.imgur.com/q4bvpbh.png)
## 实验
### 数据集
- train data：[WMT15](http://www.statmt.org/wmt15/)，EN-DE双向，EN-RU双向
- validation data：newstest-2013
- [BPE](https://arxiv.org/abs/1508.07909)处理
- 句子长度小于50个subword
### 环境&机器人设定
- NMT环境：设定与[Cho](https://arxiv.org/abs/1606.02012)一致。
- 机器人：512GRUs + softmax function，Adam，mini-batch 10。
- 动作采样数：5列trajectories。测试时不采样，只取概率最高列。
### Baselines
- Wait-Until-End（WUE）：原句全部读取之后开始WRITE。
- Wait-One-Step（WOS）：每次READ之后就WRITE。
- Wait-If-Worse/Wait-if-Diff（WIW/WID）；[Cho](https://arxiv.org/abs/1606.02012)的方法
- Segmentation-based（SEG）：[Oda](https://www.aclweb.org/anthology/P14-2090.pdf)的方法。本文中对比了simple greedy method（SEG1）和greedy method with POS Constraint（SEG2）方法。
### 结果分析
![Imgur](https://i.imgur.com/q4bvpbh.png)

