# Online and Linear-Time Attention by Enforcing Monotonic Alignments
[参考代码](https://github.com/craffel/mad/blob/master/example_decoder.py)

## 介绍
RNN将原文信息压缩到一个定长向量，这就导致它难以处理长句子。用Attention，可以让所有原文得到参考，这种方法在译文和原文之间建立一个soft alignment。

soft attention的一个批判点在于生成每个译文的时候都需要过一遍所有原文，这就导致decoding的时候复杂度O（TU），T代表原文长度，U代表译文长度。而且，这种方案必须先输入所有原文，这就意味着无法“online”操作。

本文目标就是给出一种attention机构，能用于online setting且时间复杂度为线性。我们发现在大多的问题下，输入输出之间的alignment大致为单调。比如有人说“good morning”，其对应“good”的发音必定会在“morning”之前。哪怕alignment不是严格单调，反序问题也常常只会出现在局部。  
而尽管attention能够关注到多个原文词汇，其结果往往会集中在单个原文词汇上。当然，并非所有问题都是如此，图像问题中的attention就会分散于输入图片的大片区域上。

鉴于这种现象，针对s2s问题我们给出hard monotonic alignment方法，这种方法能够用线性时间online计算attention，我们将解释这种attention算法能够用一个二次复杂度的算法计算expected output，这使得该方法能够用传统的后向传播计算误差，且test的时候仍旧能online decoding。我们发现这种方法比起softmax-based attention准确度下降极少。
## Online and Linear-Time Attention
我们首先会指出传统softmax-based attention用一种简单随机过程计算了expected output。随后我们给出一种能线性时间online计算的decoding。由于这样算出来的结果不可求导，我们给出一种算法求expected output，让我们能用传统反向传播进行训练。最后我们给出从monotonic attention和softmax-attention的不同点得到的energy function概念。
### Soft Attention
encoder处理输入x={x1，，，xT}，生成hidden序列h={h1，，，hT}。我们将h称为“memory”，以突出其和[memory-augmented神经网络](https://arxiv.org/abs/1503.08895)的关系。随后由decoder生成译文y={y1，，，yU}，直到输出终止符为止。

计算yi的时候，sotf attention-based decoder会在状态s<sub>i-1</sub>时用一个learnable nonlinear function，a（·）针对所有hj计算出一个常数e<sub>i，j</sub>。通常a（·）是一个单层网络，用非线性函数tanh，还有单纯把s<sub>i-1</sub>和hj点乘的。这些常数值会用softmax函数归一化，在memory上构造一个概率密度，用以对h取加权和得到上下文ci，由于memory和input一一对应，这个概率密度就构造了output和input之前的soft alignment。最后，decoder将其状态从s<sub>i-1</sub>更新到si。整个过程如下  
![Imgur](https://i.imgur.com/b4vJ7tt.png)  
f（·）代表RNN，g（·）代表将decoder状态投影到output空间上的learnable nonlinear function。

要运用monotonic alignment结构，我们发现公式2和公式3是用一个简单随机过程计算了expected output。公式化后如下
> ![Imgur](https://i.imgur.com/RuCy531.png)

我们将这个过程可视化为图1。很显然，公式3就是没有采样k，而是直接计算ci的期待值。  
![Imgur](https://i.imgur.com/XPQSGXS.png)
### A Hard Monotonic Attention Process
上面已经描述了计算&alpha;<sub>i,j</sub>的时候需要过一遍所有原文，导致生成译文的时候时间复杂度O（TU）。除此之外，尽管h是从一个序列转化而来（表面上有着先后关系），在计算attention概率的时候对局部顺序是独立的，每一步attention概率密度的计算也是独立的。

为了解决这个短板，我们首先公式化这个随机过程，从左至右处理memory。当译文timestep为i，我们从index t<sub>i-1</sub>开始处理memory，ti代表timestep i的时候选中的memory entry。（为简化，设t0=1）。我们顺序计算j=t<sub>i-1</sub>, t<sub>i-1</sub>+1, t<sub>i-1</sub>+2  
![Imgur](https://i.imgur.com/OnNHCw6.png)  
a（·）是learnable deterministic “entergy function”，&sigma;（·）代表logistic sigmoid function。一旦取样z<sub>i,j</sub>=1，便让ci=hj，ti=j。“选择”memory entry j为上下文向量。每个z代表一次独立的选择，选择是否从memory获取新item（z=0），或是生成一个输出（z=1）。对所有后序输出timesteps重复以上步骤。从t<sub>i-1</sub>（前一个timestep选择的memory index）开始。如果所有的timestep i我们都得到z=0，就把ci设为0向量。这个过程可视化为图2，具体算法见算法1.  
![Imgur](https://i.imgur.com/WHWJ13n.png)  
![Imgur](https://i.imgur.com/BQ3jssp.png)  
这个结构要计算p<sub>i，j</sub>只需要计算hk，k&isin;{1，，，j}。这让我们能够用online计算，而不需要得到所有原序列输入再开始输出。进一步说，由于我们舍去了前一个timestep之前的输入，这个计算最大只需要max（T，U）时间，能够满足线性运行时间。虽然这也强假定了原文和译文之间的alignment是严格单调的。
### Training in Expectation
上述取样方法使得标准反向传播无法使用。模仿softmax-based attention，我们训练的时候用ci的expected value，这可以用以下方式计算：先用公式6和公式7计算e<sub>i,j</sub>和p<sub>i,j</sub>，p<sub>i,j</sub>表示在timestep i选取memory j的概率。于是有memory上的attention概率密度公式如下，推导见附录C  
![Imgur](https://i.imgur.com/ZALdf1r.png)  
![Imgur](https://i.imgur.com/TT8yCyE.png)  
在附录C.1中，我们给出了一种利用求积求和来并行处理公式10中的recurrence关系的解决方法。定义q<sub>i,j</sub>=&alpha;/p，得到以下计算&alpha;的步骤  
![Imgur](https://i.imgur.com/g16kgMq.png)  
![Imgur](https://i.imgur.com/G9XmKC7.png)  
定义q<sub>i,0</sub>=0，p<sub>i,0</sub>=0以保证公式9的等价性。

在softmax-based attention中，&alpha;可以给memory加上一个权重，然后用公式3在每一步timestep计算上下文向量。然而，注意&alpha;i可能并非有效密度，因为对j求和小于等于0。直接利用&alpha;i能有效把所有额外概率分配到全零memory上。若是归一化&alpha;i会有两个问题：1）test的时候无法归一化处理；2）这会导致monotonic attention process生成的概率密度不匹配。

计算ci时候依旧有二次复杂度，然而，由于计算的是期待值，我们将用公式11和公式14训练。进一步说，如果p<sub>i,j</sub>&isin;{0,1}这些方法将等价，所以为了使模型在训练和测试的时候一致，需要p约等于0或1。
### Modified Energy Function
在“energy function”&alpha;（·）中，已知最通用的是以下公式  
![Imgur](https://i.imgur.com/6q3Wd7P.png)  
W和V代表weight matrices，b是bias vector，v是weight vector。我们在公式15上做两个调整：1）softmax是不受offset影响的，但logistic sigmoid不是。因此，我们在tanh函数外加一个scalar变量r，让模型能够学习这个偏移量。注意公式13中会有一个指数级别decay，因此初始化r为负数，使得1-p<sub>i,j</sub>趋向于1；2）公式12用了logistic意味着我们的方法会受到energy team e<sub>i,j</sub>的尺度影响，或者说是向量v的影响。解决办法是在v上进行weight normalization，g代表一个scalar参数，初始化为hidden次元大小的平方根倒数。最终得到公式  
![Imgur](https://i.imgur.com/G0BskX0.png)  
这个公式仅仅用两个参数，g和r，解决了所有实验中的上述问题。
### Encouraging Discreteness
为了让模型在训练和测试时保持一致，要求p<sub>i,j</sub>接近0或者1。有一个直接办法就是在公式12的sigmoid之前加入噪音。加入一个（0,1）高斯噪音很有效。这个方法与Gumbel-Softmax技巧相似，但是我们没发现需要退火temperature。

既然p<sub>i,j</sub>有着足够的方差，我们可以在采样哪一步用一个单纯的函数![Imgur](https://i.imgur.com/jSjPh4l.png)表示，&tau;代表阈值。我们所有的实验中阈值为0.5。在test中，我们不需要pre-sigmoid noise。组合以上方法，我们给出了monotonic alignment的可导训练算法，见算法2  
![Imgur](https://i.imgur.com/ihh0H50.png)
