# Learning to Translate in Real-time with Neural Machine Translation
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
评价标准：Q(Y)衡量翻译质量，如[BLEU][bleu]；D(A)评价时间延迟。


[bleu]:(https://www.aclweb.org/anthology/P02-1040)
