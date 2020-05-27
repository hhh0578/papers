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
