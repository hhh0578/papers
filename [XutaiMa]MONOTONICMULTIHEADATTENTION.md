# [MONOTONIC MULTIHEAD ATTENTION](https://arxiv.org/abs/1909.12406)
[参考代码](https://github.com/pytorch/fairseq/blob/master/examples/simultaneous_translation/models/transformer_monotonic_attention.py)

## 介绍
- hard monotonic attention
- monotonic chunkwish attention（MoChA）
- [monotonic infinite lookback attention（MILK）](https://arxiv.org/abs/1906.05218)
- wait-k
- wait-if-\*

现有monotonic attention-based模型，包括MILk在内顶端都是RNN模型。

我们提出了monotonic multihead attention（MMA），这种方法将multihead attention的高翻译质量和monotonic attention的低延迟统合了起来。我们提出了两种方案，Hard MMA和Infinite Lookback MMA（MMA-IL）。前者中心放在了流水式作业上，此时attention span必须加以限制。后者则是把重心放在了翻译质量上。

我们还提出两个latency regularization methods。第一种直接最小化平均延迟来促使模型提高翻译速度。第二种让多个heads维系相似位置，以防止延迟被其中某个或少数几个head所主导。

本文内容
1. 标准monotonic attention机构，MMA。这让Transformer得以进行online decoding。
2. 和MILk相比有着更好的质量/延迟。
3. 分析模型如何控制attention span，分析了head的速度和它所属layer之间的关系。

## Monotonic Multihead Attention Model
### Monotonic Attention
Hard Monotonic Attention是Raffel et al.最先提出的，用于在RNN模型上实现online linear time decoding。定义输入为x={x<sub>1</sub>,&hellip;,x<sub>T</sub>}，其对应的encoder state是m={m<sub>1</sub>,&hellip;,m<sub>T</sub>}，T表示原文序列的长度。该模型会生成一个原文序列y={y<sub>1</sub>,&hellip;,y<sub>U</sub>}，U表示译文序列长度。在第i步，decoder仅仅参考encoder state *mt*，此时t<sub>i</sub>=j。当生成新的译文y<sub>i</sub>时，decoder遵从伯努利分布p<sub>i,j</sub>选择是否下一步，或是保持在当前位置，因此有t<sub>i</sub>&ge;t<sub>t-1</sub>。标记decoder的第i不状态，从j=t<sub>i-1</sub>,t<sub>i-1</sub>+1,t<sub>i-1</sub>+2,&hellip;开始，这个步骤计算如下  
![Imgur](https://i.imgur.com/LZdTYKs.png)  
当z<sub>i,j</sub>=1，设定t<sub>i</sub>=j，且生成译文y<sub>i</sub>。否则设定t<sub>i</sub>=j+1重复上述步骤。  

训练时，用expected alignment &alpha;替代softmax attention。这可以通过RNN方式计算，见公式4  
![Imgur](https://i.imgur.com/ksSMZUt.png)  

Raffel et al.也给出了一种closed-form parallel方法解决这种recurrence关系，见公式5  
![Imgur](https://i.imgur.com/G3D8W8X.png)  
![Imgur](https://i.imgur.com/KzVdhZ1.png)  

在实践中，公式5中的分母被限制在\[&epsilon;,1\]之间，以防止cumprod造成的数值不稳定。
虽然这种monotonic attention达到了online linear time decoding的效果，然而这仅仅用了一个encoder state。这就导致难以捕捉到反序信息，使得翻译质量下降。

进一步说，这种模型缺少了对延迟需求的满足。Chiu&Raffel给出了MoChA解决这个问题，该方法在decoder上操作softmax attention和encoder states定长的子序列。同样的，Arivazhagan et al.给出了MILk方法，该方法让decoder能从头获取原文序列。MILk模型的expected attention见公式6  
![Imgur](https://i.imgur.com/BjXDkm4.png)
### Monotonic Multihead Attention
给定Q，K，V，multihead attention公式如7  
![Imgur](https://i.imgur.com/PV90nr5.png)  
其中attention函数是dot-product attention，公式如8  
![Imgur](https://i.imgur.com/KPu1yi3.png)  

transformer中有三处multihead attention：Encoder的self-attention，Decoder的self-attention，Encoder-Decoder attention。在MMA方法中，我们在encoder-decoder每个head上分别运用monotonic attention。  
对于L层的decoder layers，每层的H heads attention，我们定义第l层的第h个attention如下  
![Imgur](https://i.imgur.com/TOHxhrV.png)

我们研究了两种MMA，MMA-H和MMA-IL。MMA-H中，我们在每层利用公式4计算expected alignment。MMA-IL中，我们对每个head用下式计算softmax energy  
![Imgur](https://i.imgur.com/fueYHts.png)  
然后用公式6计算expected attention。  
MMA-H的每个head进考虑一个encoder state，而MMA-IL可以考虑所有previous encoder state。所以MMA-IL能利用起更多的信息，而MMA-H更适合流水作业。

我们的模型利用单向encoder，encoder的self-attention仅仅能看到前面的状态。

预测时，decoding策略见算法1。每个l，h，decoding step i时，独立使用前面提到的sampling process，设置encoder step。然后通过公式13，生成第i个译文。仅当所有的attention都决定写的时候才生成下一个token。  
![Imgur](https://i.imgur.com/WQa7uMC.png)  

图1比较了我们的模型和单attention的monotonic模型。多头模型下，部分读取新输入，而部分保留在过去，可以获取更多的信息。而单头模型就容易丢失过去信息。
![Imgur](https://i.imgur.com/qbAxVef.png)  
![Imgur](https://i.imgur.com/Xzcoeds.png)

### 延迟控制
MMA模型每个头有着独立的读写进度，而延迟由最快的——即读取数最多的头决定。可能会有一个头光读不写，导致延迟达到最大值。  
注意MMA-H和MMA-IL中的attention行为是不一样的。MMA-IL中，抵达句末的head能提供最多的原文信息。而MMA-H中，抵达句末的head仅仅是给了句末token的对齐信息。进一步说，MMA-H的头可能就在句首不前进，这避免了延迟却也降低了准确度，而且这不适合流水系统。

为了解决这个问题，我们给出两种延迟控制模型，第一种是加权平均延迟，见公式14  
![Imgur](https://i.imgur.com/H0OJ0BZ.png)

和Arivazhagan et al.类似，我们用Differentiable Average Lagging。需要注意的是，与原本的延迟augmented training不同，公式15不是给定C的expected latency，而是所有attention上的weighted average C。真实的expected latency是用g的最大值取代g的平均值，可是这么做只能影响最快的头。公式15可以调整所有的头，并且自动调整快慢的权重。在MMA-H中，我们发现离群的延迟会跳过所有的token，这种加权平均延迟loss对于离群延迟就不够有效。因此我们给出head devergence loss，在每一步的average variance of expected delays，见公式16  
![Imgur](https://i.imgur.com/rJS95FP.png)  
其中两个&lambda;是超参数。直觉上，&lambda;<sub>avg</sub>控制整体速度，&lambda<sub>var</sub>控制头的收敛。在MMA-IL我们仅仅用L<sub>avg</sub>，在MMa-H中，我们仅仅用L<sub>var</sub>

### 实验设定
