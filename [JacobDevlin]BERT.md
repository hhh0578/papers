# [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
[参考代码](https://github.com/google-research/bert)

## 介绍
本文介绍一种BERT（Bidirectional Encoder Representations from Transformers）模型。可以从非标签文本的正向逆向的jointly condition中*预训练*双向representations。训练出来的模型可以直接用一层layer微调实现大多task的state of the art模型，比如**问题解答**，**语言预测**，无需调整结构。

## 现有研究
- 语言处理方面的预训练模型
- sentence-level tasks，如natural language inference，paraphrasing。
- token-level tasks，如named entity recognition，question answering。
### 预训练的两种策略
- [feature-based](https://arxiv.org/abs/1802.05365)
- [fine-tuning](https://openai.com/blog/language-unsupervised/)
这些预训练都用的单向模型，而这限制了可选模型，而且这导致了sentence-level tasks会陷入局部最优解。
### 本文方案
- MLM（masked language model）：随机屏蔽输入的某些token，目标是仅利用上下文预测出原token是什么。

## 相关研究
### 无师feature-based方法
- non-neural和neural
- 预训练word embeddings from scratch，用正向模型或是预训练word embeddings vectors，或是在[正向上下文判断错词](https://arxiv.org/abs/1310.4546)。
- 粒度更大的sentence embeddings，排序candidate next sentences，利用上一个句子的representation正向生成下一个句子的单词；paragraph embeddings。[denoising auto-encoder derived objectives](https://arxiv.org/abs/1602.03483)。
- ELMo，将正向逆向的上下文特征串接以表示单词向量，当用于task-specific结构的时候，在某些NLP的benchmarks上创了新高，如问题回答，情感分析，词性识别。
- [Melamud](https://www.aclweb.org/anthology/K16-1006/)提出了用正向逆向LSTM得到上下文向量以预测单侧的方法。
- [Fedus](https://arxiv.org/abs/1801.07736)表示cloze task能提高文字生成模型的可靠性。
### 无师fine-tuning方法
- [首次从无标记文本训练单词embeddings](https://dl.acm.org/doi/10.1145/1390156.1390177)。
- 从无标记文本训练contextual token representations，然后用有师学习微调。
- 正向语言模型和auto-encoder objectives。
### Transfer Learning from Supervised Data
- netural language inference
- machine translation
- ImageNet

## BERT
#### 模型结构
多层双向transformer。具体参考Vaswani的论文或是“[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)”。
- BEART<sub>BASE</sub>（L=12，H=768，A=12，total parameters=110M）
- BERA<sub>LARGE</sub>（L=24，H=1024，A=16，total parameters=340M）
base主要用于OpenAI GPT的比较，然而本文是双向，GPT Transformer是单向的。

![Imgur](https://i.imgur.com/0ZWXwjK.png)
#### 输入形式
为保证能适用于多工作，输入能明确区分单句和双句。

本文利用30,000 token的[WordPiece embeddings](https://arxiv.org/abs/1609.08144)，其中首字母永远是特殊分类token（\[CLS\]）。该tokne的对应输出在多分类task中用作aggregate sequence representation。

双句打包成单句，通过两种方法区分：1 用特殊token（\[SEP\]）。2 给每个token添加一个learned embedding表示是属于句子A还是句子B。如图1中，输入的embedding为E，\[CLS\]的输出为C，其他的输出为T。

### 预训练BERT
本文不用传统正序或倒序训练，而是用两个无师学习训练。
#### Task1：Masked LM
直观上，双向学习必然比单向或是简单把前序后序模型拼接要有效。可惜，传统的条件语言模型只能单向训练，因为双向条线会让每个单词能间接“看到自己”，模型一下子就能从multi-layerd context预测target单词。  

为了训练深度双向representation，本文随机将输入token隐去部分，然后预测隐去的部分。这个步骤叫做“*masked LM（MLM）*”，这称呼通常指的是[Cloze task](https://www.semanticscholar.org/paper/%22Cloze-procedure%22%3A-a-new-tool-for-measuring-Taylor/766ce989b8b8b984f7a4691fd8c9af4bdb2b74cd)。在本例中，隐藏token的输出会被送到词汇output softmax，正如普通LM。本文所有的实验中隐藏比率为15%。与[denoising auto-encoder](https://dl.acm.org/doi/10.1145/1390156.1390294)不同，本文仅仅预测隐藏单词，而非调整所有输入。  

虽说这种方式得以实现了双向训练，但由于fine-tuning的时候没有\[mask\]，就导致**预训练**和**微调**不匹配。为解决这个问题，隐藏单词的时候并不完全利用\[mask\]token。训练集生成器以15%的概率随机选择预测哪个位置的单词，随后针对这第i个单词，80%的时间用\[mask\]替换，10%的时间随机用token替换，10%的时间不变。然后用T<sub>i</sub>预测原单词，取cross entropy loss。
#### Task2：Next Sentence Prediction（NSP）
许多工作如问题回答（QA）和自然语言预测（NLI）都是需要理解句子间关系的。为了训练这种关系，本文训练binarized next sentence prediction task，这数据能从任意单语料库轻松生成。

具体来说，选中句子A和句子B来训练，B有50%的概率是真实的下一句（标记为IsNext），50%的概率是语料库中随机选的句子（标记为NotNext）。如图1中就用C来标记NSP。这方法虽然简单，但对QA和NLI非常有效。
>用NSP训练的C在微调前并不具有实际意义。  
NSP工作与Jernite et al.和Logeswaran and Lee的representation-learning objectives很相似。然而他们仅仅将sentence embeddings转化成down-stream task，BERT则是将左右参数都初始化成了end-task模型的参数。
#### 预训练数据
- BooksCorpus（800M单词）
- English Wikipedia（2,500M单词）：仅仅提取文字信息，忽视列表，表格和标题。
为了获取连续长句子的信息，用文档级语料而非打乱的语句级语料，如Billion Word Benchmark，很关键。
### 微调BERT









