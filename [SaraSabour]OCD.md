# [OPTIMAL COMPLETION DISTILLATION FOR SEQUENCE LEARNING](https://arxiv.org/abs/1810.01398)
## 已有研究
### 自然语言处理和语音识别
- S2S学习，encoder-decoder结构，传统的multi-state pipelines。
- 图像讲解，语法解析，内容总结，程序语法。
### 学习算法
- convolution，self attention，MLE，[label smoothing](https://arxiv.org/abs/1701.06548)，[scheduled sampling](https://arxiv.org/abs/1506.03099)。

本文从search-based structured prediction和[policy distillation](https://arxiv.org/abs/1509.00685)得到了灵感，找到了一种用edit distance优化s2s模型的算法。`给定任意序列前缀，可以高效得出与真实结果最为相近的所有后缀。`。总结如下：
- 用于训练的前缀，是从模型中取样生成的。
- 利用**动态编程**，可以求出每个前缀的最优后缀，即和真实答案的**编辑距离**最短。
- 本文最大化每个最优后缀第一个token的average log probability，来教模型扩张前缀。

本文的贡献如下：
- 提出了OCD，利用**编辑距离**优化s2s模型的独立算法，可用于长句子多词汇的实际数据，而且该方法在远优于MLE。
- 给定译文长度m，原文长度n，算法复杂度为O(nm)，而且能找出generated sequence的所有最优extension。
- 本文展示了OCD在e2e语音识别中的效用。
## 背景：用MLE进行序列学习
给定数据集D&equiv;{(x,y\*)<sub>i</sub>}<sup>N</sup><sub>i=1</sub>，我们想要知道一组从输入x到输出y\*&isin;Y的映射x&rarr;y。其中Y代表有限词汇表V中的单词所组成的可变长序列。要计算这种映射，通常使用的方法是优化条件分布的参数P<sub>&theta;</sub>。于是概率模型P<sub>&theta;</sub>下的sequence prediction就由exact or approximate inference表示为：  
![Imgur](https://i.imgur.com/dhRcSKA.png)  
于是标准的求解方法就类似有师学习的分类器，最大化conditional log-likelihood objective来计算![Imgur](https://i.imgur.com/5XTjxcT.png)。这种方法叫做`Maximum Likelihood Estimation`。

aoturegressive模型是给定原文，一个词一个词地求译文的概率，通常是left-to-right。用特殊的*end-of-sequence标志*来控制译文长度。其条件概率用chain rule可以写作：  
![Imgur](https://i.imgur.com/D7r6nBz.png)  
p<sub>D</sub>代表empirical data distribution，在数据集D上均一分布。

本文则是提出了可以用于任意autoregressive s2s模型结构的目标函数。

### MLE在autoregressive模型中的局限
为了最大化conditional log-likelihood，递归s2s模型从译文中取前缀y\*<sub>&lt;t</sub>，用于训练下一个token y\*<sub>t</sub>的概率。然而实际训练时，模型却只能根据自己生成的前缀y_hat<sub>&lt;t</sub>来预测下一个单词，这就导致了[exposure bias](https://arxiv.org/pdf/1511.06732.pdf)。

总结下来，MLE在autoregressive s2s模型上由两个局限：
- 学习时和预测时的前缀有所不同。
- 训练误差和评价方式之间有所不同。训练用的log概率，评价用的却是其他方式（比如语音识别用的时**编辑距离**）
## Optimal Completion Distillation
本文为了弥补这个缺陷，不用真实序列训练。而是用从模型生成的序列中取样。用y_tilde表示取样的序列，y\*表示真实结果序列。MLE将这个问题看作是处理映（x，y\*<sub>&lt;t</sub>）&rarr;y\*<sub>t</sub>。相对的，一个关键的问题就是，在用采样序列训练的时候next token应该取什么（x，y_tilde<sub>&lt;t</sub>）&rarr;？？。OCD利用task evaluation metric找到最佳的补全，解决这个prefix-specific问题。然后用这些最佳选择来扩展每一个前缀。

本文方法依赖于task evaluation metric，记为R(·，·)，用于比较两个完整序列之间的相似度。**编辑距离**是一种通用的方式。本文的序列学习目标是训练一个在方法R(y\*,y_tilde)上高得分的模型。

为了和强化学习的目标联系起来，先复习一个optimal Q-values的概念。对于**状态-操作pair（s，a）**，Optimal Q-values记作Q\*(s,a)，代表机器人在状态s下采取了行动a之后所能累积的，包括后续可能的行动在内的未来报酬。类似的，本文为前缀y_tilde<sub>&lt;t</sub>和将要扩展的token a定义Q-values，组合起来得到\[y_tilde<sub>&lt;t</sub>,a\]，于是未来可能得到全文score为\[y_tilde<sub>&lt;t</sub>,a,y\]，求其最大值，公式为：  
![Imgur](https://i.imgur.com/3pH1rK0.png)  
于是这个针对前缀y_tilde的优化过程就可以定义为是能最大化Q-values的token，即![Imgur](https://i.imgur.com/t8KfoPq.png)。这个公式中的前缀可以从模型p<sub>&theta;</sub> on-policy采样，或是用其他方法off-policy。图1列举了Wall Street Journal数据集的一例，展示了模型生成的sample，画出当某个前缀有多个**编辑距离**一样的extension时的情况。  
![Imgur](https://i.imgur.com/KbmoEap.png)

拿到prefix-token pairs的Q-values，用exponential transform正规化，表示扩展下一个token时的soft optimal policy：  
![Imgur](https://i.imgur.com/KEs4n01.png)  
&tau;&ge;0表示temperature parameter。要注意这个参数和MLE中的label smothing类似。本实验中对hard targets，&tau;会趋于0，而不需要调整额外参数。

给定一个训练列(x,y\*)，首先利用当前模型生成完整序列y_tilde ~ p<sub>&theta;</sub>(·|x)。然后再每一步t对next token最小化其optimal policy和model distribution之间的per-step KL divergence。 这个OCD目标函数可以表示为：  
![Imgur](https://i.imgur.com/caVWVlA.png)  
对每一步的前缀y_tilde<sub>&lt;t</sub>，计算Q-values然后用公式5组成optimal policy distribution &pi;\*。然后用KL误差将其整合到parametric model里。**编辑距离**是序列学习问题中的一项评价指标，本文使用动态编程算法计算y_tilde的所有前缀optimal Q-values。
### 编辑距离的Optimal Q-values
本文用动态编程算法计算reward metric，即编辑距离的负数![Imgur](https://i.imgur.com/66SXSVq.png)

给定序列y\*和y_tilde，可以用所有前缀y_tilde<sub>&lt;t</sub>的Q-values，并以复杂度O（|y\*|·|y_tilde|+|V|·|y_tilde|）扩展token a&isin;V。假定|y\*|&asymp;|y_tilde|&le;|V|，本算法的时间复杂度就不会超过MLE。而实际上wall clock time主要取决于神经网络的前向后向传播，OCD的cost往往可以忽略。

重温计算编辑距离用到的Levenshtein algorithm。  
![Imgur](https://i.imgur.com/OCMVYJa.png)  
表2举了一个例子，计算“Satrapy”和“Sunday”的编辑距离。本文的目标是列出所有最优后缀y&isin;Y得出和y\*距离最小的所有序列\[y_tilde<sub>&lt;i</sub>,y\]。  
![Imgur](https://i.imgur.com/zMXkXLE.png)

> Lemma 1. 接上任意后缀y&isin;Y后得到的编辑距离有下界m<sub>i</sub>。  
![Imgur](https://i.imgur.com/npwyLlp.png)  
证明：考虑从距离D逆推到D<sub>&lt;0</sub>的路径P，按公式7，继承了相邻父级的最小值。

接下来
1. 假定路径P经过第i行的（i，k）格，按照公式7，路径上的编辑距离不会减小，因此![Imgur](https://i.imgur.com/SzGPaBx.png)
2. 对于任意![Imgur](https://i.imgur.com/2xqwvSY.png)的k，用![Imgur](https://i.imgur.com/71cSQPs.png)代表y\*的后缀，可以得到结论![Imgur](https://i.imgur.com/sHt7Xc1.png)i。一方面这表示有编辑方法能让编辑距离为m<sub>i</sub>，另一方面按照Lemma.1表述，m<sub>i</sub>是下界，因此y\*<sub>&ge;k</sub>就是y_tilde<sub>&lt;i</sub>的最优后缀。
3. 进一步说，这反过来直接证明了最优后缀被限制为y\*<sub>&ge;k</sub>且![Imgur](https://i.imgur.com/YVo3qqg.png)。
既然最优后缀得到限制，就可以得出报酬最大的后缀从token y\*<sub>k</sub>开始。由于![Imgur](https://i.imgur.com/YVo3qqg.png)，可以通过计算y_tilde和y\*所有前缀的编辑距离计算最优extensions，这能用复杂度O（|y_tilde|,|y\*|）的动态编程计算。对于一个前缀y_tilde<sub>&lt;t</sub>，计算得到和所有y\*前缀的最小编辑距离m<sub>i</sub>后，对所有和y\*<sub>&lt;k</sub>编辑距离为m<sub>i</sub>的k，定义![Imgur](https://i.imgur.com/NiLmO0d.png)。其他token的Q\*为-m<sub>i</sub>-1。
## 相关研究
该研究出发点是[Learning to Search](https://arxiv.org/abs/0907.0809)和Imitation Learning techniques，其想法在于训练student policy去学习expert teacher。

与本文类似的一个方法[DAgger](https://arxiv.org/abs/1011.0686)从past student model中取样得到trajectories数据集，然后模仿一个既有的expert policy &pi;\*来优化policy。OCD与此类似，对所有的前缀学习策略policy &pi;\*。只不过OCD中策略直接从online student中获取。因为训练中不会提供oracle policy，因此通过找到Q-value最优解来得到最优策略。

[AggreVatTeD](https://arxiv.org/abs/1703.01030)方法试图建立一个无偏移的Q-value，利用variance reduction techniques和共轭矩阵解决policy optimization problem。OCD计算exact Q-values且用的是普通SGD优化。更重要的是，本文roll-in前缀都是从srudent model中获取的，而且不需要和ground truth samples混起来。虽然[Cheng和Boots](https://www.researchgate.net/publication/322674823_Convergence_of_Value_Aggregation_for_Imitation_Learning)研究过将其混合起来能够有效正则化模仿学习中的value aggregation convergence。

本文方法和[Policy Distillation](https://arxiv.org/abs/1511.06295)有较大关系，他将已训练好的[Deep Q-Network](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)机器人作为expert teacher，然后从teacher采样actiono sequence，用KL误差将Q-value蒸馏到student网络中去。OCD采用了类似的误差函数，但并非用bootstrapping估计Q-value，而是直接用动态编程计算了Q-value。而且sample取自student而不是teacher。

Learning to Search（L2S）与OCD类似，用类似[LOLS](https://arxiv.org/abs/1502.02206)等技巧试图对每个state-action pair估计Q-value。这种方式测试多个roll-out生成的前缀然后整合一个返回值。[SeaRNN](https://arxiv.org/abs/1706.04499)则是为每个token计算cost-to-go，每一步以词汇表大小roll-out，导致复杂度为O（VT）。这就难以用到实际生活中。OCD的结构每步就只有O（V+T）。而且不需要ground truth的前缀来让维持训练稳定。

强化学习比如[REINFORCE](https://arxiv.org/abs/1511.06732)，[Actor-Critic](https://arxiv.org/abs/1607.07086)和[Self-critical Sequence Training](https://arxiv.org/abs/1612.00563)也用来处理sequence prediction problem。这些方法从模型分布中得到sample sequence然后利用sequence-level task objective（比如编辑距离）来逆传播计算。[Beam Search Optimization](https://arxiv.org/abs/1606.02960)和[Edit-based Minimum Bayes Risk（EMBR）](https://arxiv.org/abs/1712.01818)相似，只不过取样用的是Beam search。这些模型都有high variance和credit assignment的问题。相对的，OCD将序列级目标函数分解成了token level optimal completion target。这就减少了梯度方差让模型更稳定。而且不同于大多数RL方法，本文完全没有利用MLE预训练或是在优化函数中加入log-likelihood。[Bahdanau](https://arxiv.org/abs/1511.06456v2)也给出过使用编辑距离的模型，但是他是将模型输出到编辑距离的值来体操suboptimal performance。本文最先提出的构建optimal policy然后用知识蒸馏训练。另外，[Karita](https://ieeexplore.ieee.org/document/8462245)也分解过编辑距离，分解为每个token的分布然后用于EMBR框架。也就是说，他并非基于理论分析做的这个选择，而且他的梯度报告了高方差。

[Reward Augmented Maximum Likelihood](https://arxiv.org/abs/1609.00150)极其各种衍生也和RL类似。RAML没有从模型分布取样，而是从true exponentiated reward distribution取样了。然而，这种方法往往难以实现。而且RAML也碰到了RL模型中会有的credit assignment问题。[SPG](https://arxiv.org/abs/1709.09346)改用了policy gradient formulation来从reward shaped model distribution中取样。这种取样方式比起RAML更接近**从模型取样**。为此SPG提供了一个启发式的ROUGE score。SPG达到了降低方差的效果，但也和RAML、RL同样有credit assignment的问题。

通常来说，OCD擅长从零开始的模型训练，能作为MLE的替代。因此OCD和需要MLE预训练，或是需要joint optimization的模型毫不冲突。

## 实验

## 附录A OCD算法
![Imgur](https://i.imgur.com/SyjyS8i.png)










