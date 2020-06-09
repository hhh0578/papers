# [LITE TRANSFORMER WITH LONG-SHORT RANGE ATTENTION](https://arxiv.org/abs/2004.11886)
[参考代码](https://github.com/mit-han-lab/lite-transformer)
## 介绍
如今Transformer在多方面有着优良表现，然而其计算量依旧是个难关。[Automatic neural architecture search](https://arxiv.org/abs/1901.11117)是个解决办法，然而庞大的搜索代价堪忧。  
![Imgur](https://i.imgur.com/drtpxpx.png)  
本文研究移动设备情景下的使用，这要求Mult-Adds操作压到500m以下。最直接的办法就是减少embedding size，但这也会减少模型的特征捕捉能力。而在系统分析之后，我们发现占据计算资源的主要部分是FFN，而非attention，于是现行的bottleneck-structured transformer block可能不够高效。于是我们提出Long-Short Range Attention（LSRA）结构。LSRA削减了FFN计算量扩展了attention层，它让attention层得以捕捉更多的关系特征，以此减少enbedding size从而减少总体计算量。LSRA抛弃了捕捉“普通”信息的结构，转而以specialized heads捕捉长短距离上下文。受[Wu et al](https://arxiv.org/abs/1901.10430)的影响，LSRA在并行branch中引入了convolution提取局部关系特征，使得attention branch可以专注远距离上下文。用结构，我们构建了Lite Transformer。

本论文由四个部分组成
1. 系统分析了计算量的瓶颈，指出在1-D attention中这个部分还不够优化。
2. 提出了specialized multi-branch特征提取器LSRA，用convolution帮助把握局部上下文，然后拼接attention得到全局上下文特征。
3. 以LSRA为基础，组装了满足计算资源限制的Lite Transformer，并且在3个翻译数据集上精度有所提升。
4. 本文的BLEU值甚至比AutoML-searched Evolved Transformer还高出0.5，这警示我们需要思考是否值得花费大量时间计算AutoML的结构。
## 相关研究
### RNNs和CNNs
RNN流行了很长一段时间，然而由于其利用的是temporal dependency，无法并行处理序列。近期甚至有些工作表明在state of art performance下RNN甚至并非必不可少。举例来说研究者们发表了convolution-based的高效模型。Convolution在获取局部上下文信息的时候很有用，然而它无法捕捉长距离关系，这在众多序列模型工程中至关重要。
### Transformer
attention可以捕捉到全局信息，Transformer表明了堆叠self-attention可以达到state of the art指标。近来更是出现了许多变种。其中，Ott et al.放大了batch size；Shaw et al.利用了相对位置representation；Ahmed et al.提出了加权多头attention；Sukhbaatar et al.在处理超长序列的character-level语言模型中利用adaptive maskes来捕捉长距离信息。这些工作都与我们处于不同维度，因为我们改进的是模型结构本身，而非一般技巧。
### Automated Model Design
由于结构的设计空间很大，最近兴起用neural architecture search（NAS）自动设计compact模型，并且在optimization loop中加入硬件资源条件，如MnasNet，ProxylessNAS和FBNet。在NLP圈内，evolved transformer针对基础block使用NAS，以寻找更好的参数。然而，AutoML模型需要昂贵的GPU资源，这对大多研究者而言难以承受。
### Model Acceleration
除了直接设计更有效的模型，还有一种方法是压缩或优化已有large模型。比如，有些研究剪枝separate neurons；有些研究quantize the network。近来，AutoML同样也被用于自动化压缩和优化。这些技巧都是不针对特定模型的，因此与我们研究无冲突，我们的研究针对NLP领域。
## Bottleneck在1-D attention中是否足够高效？
Attention结构在众多工程中被广泛应用，包括1-D（语言处理），2-D（图像识别），3-D（视频识别）。它点乘所有的输入元素来捕捉近关系和远关系，虽然有效，计算代价却也昂贵。假设输入attention层的元素数为N（可以是语言处理中的词汇个数，也可是图片中的像素个数），其维度为d（如channels），点乘的计算量就是N<sup>2</sup>d。在图片和视频中，N的值通常非常大。比如在视频网络中，中间特征有16帧，每一帧有112x112个像素，于是就有N=2x10<sup>5</sup>。convolution和全连接层的计算量也会随着N线性增加，attention层更是会二次增长。

为了解决这个冲突，一种共同处理法是在输入attention之前先用线性投影减少channels d的数量，之后再恢复（如图2a）。在Vaswani et al.的原设计中，attention模块内的chaneels维度是4x，小于FFN层。类似的，在non-local video network中，在利用non-local attention模块之前，channel数先被削减到一半。实践中计算量分别是16x和4x。不必说，这减少了模型的特征捕捉能力，在语言处理工程中甚至影响更大，因为特征模块主要就是attention（在图像和视频处理中主要是convolution）。  
![Imgur](https://i.imgur.com/FsVE2Sr.png)

在诸如翻译的工程中，输入长度N通常较小，在20到30之间。一个transformer block由一个attention（decoder中是2个），一个FFN构成。在attention层，Mult-Adds数有O（4Nd<sup>2</sup>+N<sup>2<sup/>d）；在FNN层，Mult-Add数有O（2x4Nd<sup>2</sup>）。在N很小的时候，1D attention上的这种瓶口结构是否在计算量和准确度上权衡足够值得商榷。为了验证这个问题，我们首先在图2b中绘出了transformer的计算量。惊人的是，在原始transformer中（标记为base），FFN层占据了大部分的计算，这并非理想情况，因为FNN本身并不处理上下文信息。结论，在N较小的时候，1D attention上的瓶口设计并不能减少计算量，这减少的部分会被FNN层所填补，同时还会由于减少了特征维度从而影响attention层的效用。
  
因此，我们认为1-D attention中瓶口设计并不是最优解。我们设计了“flattened”版，不增加也不减少channel 维度。在这设计下，attention部分占据了主要计算量（图2b中的Flat），留下了极大的优化空间。我们在IWSLT和WMT数据上测试了结果（表1）。  
![Imgur](https://i.imgur.com/smKtPZ5.png)
## Long-Short Range Attention（LSRA）
由于attention网络在NLP中颇有成效，研究者们尝试理解attention捕捉到了什么信息。Kovaleva et al.和Clark et al.将BERT的不同层attention可视化。如图3b，权重w表示了原文单词和译文单词间的关系（和self-attention同理）。权重越大（颜色越深）代表词间关系越大。而这些attention模块有一些strong pattern：稀疏在外的，聚集于对角线的。前者是长距离信息，后者是近处信息。我们称前者为全局，后者为局部。  
![Imgur](https://i.imgur.com/JZwhZFq.png)

为了在翻译中获取远近信息，attention模块需要以同样的结构提取不同的信息，这就需要足够强的能力。而这不够优化，常用的设备往往比专业设备冗余。当模型相对较大的时候，这种冗余可以容忍甚至发挥更好效果。然而放到移动端上，在计算量和耗能限制下，就需要更高效。于是我们将结构专业化，用LSRA分别捕捉远信息和近信息。

如图3a，LSRA模块分两拨设计，左边捕捉全局信息，右边捕捉局部信息。我们不把所有输入同时送入两边，而是按照channel次元分为两份，之后在用FFN混合。这减少了2x的计算量。左边是普通的attention模块，此时channel次元减半了。右边是局部关系，很自然地想到利用convolution。用滑动窗口，可以很轻松地捕捉到对角线信息。为了进一步减少计算，我们用lighter vertion代替正常convolution。

为了加强理解，我们把attention平均权重在图3表示了出来。可以很清楚地认识到LSRA的attention没有尝试获取全局和局部所有信息，而是集中于全局信息。
## 实验设定
