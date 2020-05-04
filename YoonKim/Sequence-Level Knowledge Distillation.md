# [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947)

## 已有研究
- [1000隐藏的4层LSTM](https://arxiv.org/abs/1409.3215)
- [512隐藏的16层LSTM](https://www.aclweb.org/anthology/Q16-1027/)

不过，虽然训练需要庞大的神经网络，但[训练过程中representation会冗余](https://arxiv.org/abs/1306.0543)，于是deep model的压缩就成了热门，现有压缩技术由：
- pruning
- knowledge distillation
