# [CONVOLUTIONAL RECURRENT NEURAL NETWORKS FOR MUSIC CLASSIFICATION](https://arxiv.org/abs/1609.04243)
[参考代码](https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/music_tagger_crnn.py)
## 本文介绍
本文提出一种CRNN模型来music tagging。该方法汲取了CNN模型局部特征抽取和RNN模型抽取特征时temporal summarisation的优点。

## 已有研究
- music classification tasks
- [user-item latent feature prediction for recommendation](https://papers.nips.cc/paper/5004-deep-content-based-music-recommendation)
- 用于文章分类的CRNN
- 用于图像分类的CRNN
- 用于音乐注释的CRNN

## 模型
本文将把CRNN模型与k1c2，k2c1，k2c2模型进行比较。如图1。  
![Imgur](https://i.imgur.com/2CEOMZF.png)  
所有的网络输入都为96x1366（mel-frequency band x time frame），并且是单频道。由于本task为多分类模型，输出用Sigmoid函数。
- batch normalization
- ELU activation function
- 在convolutional layers之间有dropout（0.1）
### CNN-k1c2
### CNN-k2c1
### CNN-k2c2
### CRNN
最后2层为RNN with GRU，前面4层为二维CNN，如图1。  
本CNN sizes和max-pooling layers为3x3和（2x2）-（3x3）-（4x4）-（4x4）。最终得到feature size（Nx1x15）（map x frequency x time）。然后输入2-layer RNN。最后的hidden state输出为结果。
### Scalling networks

## 实验
- Million Song Dataset
- 使用的是30~60秒预览片段，将其剪为中央的29秒并且利用Librosa从22.05kHz降到12kHz。
- 输入使用Log-amplitude mel-spectrogram，因为在早期研究中表明比STFT、MFCC和linear-amplitude mel-spectrogram效果好。
- 结果用AUC-ROC衡量，当validation集上AUC不再增长的时候停止学习。
