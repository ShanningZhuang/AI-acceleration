4
1
0
2

p
e
S
7
1

]

V
C
.
s
c
[

1
v
2
4
8
4
.
9
0
4
1
:
v
i
X
r
a

Going deeper with convolutions

Christian Szegedy

Wei Liu

Yangqing Jia

Google Inc.

University of North Carolina, Chapel Hill

Google Inc.

Pierre Sermanet

Scott Reed

Dragomir Anguelov

Dumitru Erhan

Google Inc.

University of Michigan

Google Inc.

Google Inc.

Vincent Vanhoucke

Andrew Rabinovich

Google Inc.

Google Inc.

Abstract

We propose a deep convolutional neural network architecture codenamed Incep-
tion, which was responsible for setting the new state of the art for classiﬁcation
and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014
(ILSVRC14). The main hallmark of this architecture is the improved utilization
of the computing resources inside the network. This was achieved by a carefully
crafted design that allows for increasing the depth and width of the network while
keeping the computational budget constant. To optimize quality, the architectural
decisions were based on the Hebbian principle and the intuition of multi-scale
processing. One particular incarnation used in our submission for ILSVRC14 is
called GoogLeNet, a 22 layers deep network, the quality of which is assessed in
the context of classiﬁcation and detection.

1

Introduction

In the last three years, mainly due to the advances of deep learning, more concretely convolutional
networks [10], the quality of image recognition and object detection has been progressing at a dra-
matic pace. One encouraging news is that most of this progress is not just the result of more powerful
hardware, larger datasets and bigger models, but mainly a consequence of new ideas, algorithms and
improved network architectures. No new data sources were used, for example, by the top entries in
the ILSVRC 2014 competition besides the classiﬁcation dataset of the same competition for detec-
tion purposes. Our GoogLeNet submission to ILSVRC 2014 actually uses 12× fewer parameters
than the winning architecture of Krizhevsky et al [9] from two years ago, while being signiﬁcantly
more accurate. The biggest gains in object-detection have not come from the utilization of deep
networks alone or bigger models, but from the synergy of deep architectures and classical computer
vision, like the R-CNN algorithm by Girshick et al [6].

Another notable factor is that with the ongoing traction of mobile and embedded computing, the
efﬁciency of our algorithms – especially their power and memory use – gains importance.
It is
noteworthy that the considerations leading to the design of the deep architecture presented in this
paper included this factor rather than having a sheer ﬁxation on accuracy numbers. For most of the
experiments, the models were designed to keep a computational budget of 1.5 billion multiply-adds
at inference time, so that the they do not end up to be a purely academic curiosity, but could be put
to real world use, even on large datasets, at a reasonable cost.

1

In this paper, we will focus on an efﬁcient deep neural network architecture for computer vision,
codenamed Inception, which derives its name from the Network in network paper by Lin et al [12]
in conjunction with the famous “we need to go deeper” internet meme [1]. In our case, the word
“deep” is used in two different meanings: ﬁrst of all, in the sense that we introduce a new level of
organization in the form of the “Inception module” and also in the more direct sense of increased
network depth.
[12]
while taking inspiration and guidance from the theoretical work by Arora et al [2]. The beneﬁts
of the architecture are experimentally veriﬁed on the ILSVRC 2014 classiﬁcation and detection
challenges, on which it signiﬁcantly outperforms the current state of the art.

In general, one can view the Inception model as a logical culmination of

2 Related Work

Starting with LeNet-5 [10], convolutional neural networks (CNN) have typically had a standard
structure – stacked convolutional layers (optionally followed by contrast normalization and max-
pooling) are followed by one or more fully-connected layers. Variants of this basic design are
prevalent in the image classiﬁcation literature and have yielded the best results to-date on MNIST,
CIFAR and most notably on the ImageNet classiﬁcation challenge [9, 21]. For larger datasets such
as Imagenet, the recent trend has been to increase the number of layers [12] and layer size [21, 14],
while using dropout [7] to address the problem of overﬁtting.

Despite concerns that max-pooling layers result in loss of accurate spatial information, the same
convolutional network architecture as [9] has also been successfully employed for localization [9,
14], object detection [6, 14, 18, 5] and human pose estimation [19]. Inspired by a neuroscience
model of the primate visual cortex, Serre et al. [15] use a series of ﬁxed Gabor ﬁlters of different sizes
in order to handle multiple scales, similarly to the Inception model. However, contrary to the ﬁxed
2-layer deep model of [15], all ﬁlters in the Inception model are learned. Furthermore, Inception
layers are repeated many times, leading to a 22-layer deep model in the case of the GoogLeNet
model.

Network-in-Network is an approach proposed by Lin et al. [12] in order to increase the representa-
tional power of neural networks. When applied to convolutional layers, the method could be viewed
as additional 1 × 1 convolutional layers followed typically by the rectiﬁed linear activation [9]. This
enables it to be easily integrated in the current CNN pipelines. We use this approach heavily in our
architecture. However, in our setting, 1 × 1 convolutions have dual purpose: most critically, they
are used mainly as dimension reduction modules to remove computational bottlenecks, that would
otherwise limit the size of our networks. This allows for not just increasing the depth, but also the
width of our networks without signiﬁcant performance penalty.

The current leading approach for object detection is the Regions with Convolutional Neural Net-
works (R-CNN) proposed by Girshick et al. [6]. R-CNN decomposes the overall detection problem
into two subproblems: to ﬁrst utilize low-level cues such as color and superpixel consistency for
potential object proposals in a category-agnostic fashion, and to then use CNN classiﬁers to identify
object categories at those locations. Such a two stage approach leverages the accuracy of bound-
ing box segmentation with low-level cues, as well as the highly powerful classiﬁcation power of
state-of-the-art CNNs. We adopted a similar pipeline in our detection submissions, but have ex-
plored enhancements in both stages, such as multi-box [5] prediction for higher object bounding
box recall, and ensemble approaches for better categorization of bounding box proposals.

3 Motivation and High Level Considerations

The most straightforward way of improving the performance of deep neural networks is by increas-
ing their size. This includes both increasing the depth – the number of levels – of the network and its
width: the number of units at each level. This is as an easy and safe way of training higher quality
models, especially given the availability of a large amount of labeled training data. However this
simple solution comes with two major drawbacks.

Bigger size typically means a larger number of parameters, which makes the enlarged network more
prone to overﬁtting, especially if the number of labeled examples in the training set is limited.
This can become a major bottleneck, since the creation of high quality training sets can be tricky

2

(a) Siberian husky

(b) Eskimo dog

Figure 1: Two distinct classes from the 1000 classes of the ILSVRC 2014 classiﬁcation challenge.

and expensive, especially if expert human raters are necessary to distinguish between ﬁne-grained
visual categories like those in ImageNet (even in the 1000-class ILSVRC subset) as demonstrated
by Figure 1.

Another drawback of uniformly increased network size is the dramatically increased use of compu-
tational resources. For example, in a deep vision network, if two convolutional layers are chained,
any uniform increase in the number of their ﬁlters results in a quadratic increase of computation. If
the added capacity is used inefﬁciently (for example, if most weights end up to be close to zero),
then a lot of computation is wasted. Since in practice the computational budget is always ﬁnite, an
efﬁcient distribution of computing resources is preferred to an indiscriminate increase of size, even
when the main objective is to increase the quality of results.

The fundamental way of solving both issues would be by ultimately moving from fully connected
to sparsely connected architectures, even inside the convolutions. Besides mimicking biological
systems, this would also have the advantage of ﬁrmer theoretical underpinnings due to the ground-
breaking work of Arora et al. [2]. Their main result states that if the probability distribution of
the data-set is representable by a large, very sparse deep neural network, then the optimal network
topology can be constructed layer by layer by analyzing the correlation statistics of the activations
of the last layer and clustering neurons with highly correlated outputs. Although the strict math-
ematical proof requires very strong conditions, the fact that this statement resonates with the well
known Hebbian principle – neurons that ﬁre together, wire together – suggests that the underlying
idea is applicable even under less strict conditions, in practice.

On the downside, todays computing infrastructures are very inefﬁcient when it comes to numerical
calculation on non-uniform sparse data structures. Even if the number of arithmetic operations is
reduced by 100×, the overhead of lookups and cache misses is so dominant that switching to sparse
matrices would not pay off. The gap is widened even further by the use of steadily improving,
highly tuned, numerical libraries that allow for extremely fast dense matrix multiplication, exploit-
ing the minute details of the underlying CPU or GPU hardware [16, 9]. Also, non-uniform sparse
models require more sophisticated engineering and computing infrastructure. Most current vision
oriented machine learning systems utilize sparsity in the spatial domain just by the virtue of em-
ploying convolutions. However, convolutions are implemented as collections of dense connections
to the patches in the earlier layer. ConvNets have traditionally used random and sparse connection
tables in the feature dimensions since [11] in order to break the symmetry and improve learning, the
trend changed back to full connections with [9] in order to better optimize parallel computing. The
uniformity of the structure and a large number of ﬁlters and greater batch size allow for utilizing
efﬁcient dense computation.

This raises the question whether there is any hope for a next, intermediate step: an architecture
that makes use of the extra sparsity, even at ﬁlter level, as suggested by the theory, but exploits our

3

current hardware by utilizing computations on dense matrices. The vast literature on sparse matrix
computations (e.g. [3]) suggests that clustering sparse matrices into relatively dense submatrices
tends to give state of the art practical performance for sparse matrix multiplication.
It does not
seem far-fetched to think that similar methods would be utilized for the automated construction of
non-uniform deep-learning architectures in the near future.

The Inception architecture started out as a case study of the ﬁrst author for assessing the hypothetical
output of a sophisticated network topology construction algorithm that tries to approximate a sparse
structure implied by [2] for vision networks and covering the hypothesized outcome by dense, read-
ily available components. Despite being a highly speculative undertaking, only after two iterations
on the exact choice of topology, we could already see modest gains against the reference architec-
ture based on [12]. After further tuning of learning rate, hyperparameters and improved training
methodology, we established that the resulting Inception architecture was especially useful in the
context of localization and object detection as the base network for [6] and [5]. Interestingly, while
most of the original architectural choices have been questioned and tested thoroughly, they turned
out to be at least locally optimal.

One must be cautious though: although the proposed architecture has become a success for computer
vision, it is still questionable whether its quality can be attributed to the guiding principles that have
lead to its construction. Making sure would require much more thorough analysis and veriﬁcation:
for example, if automated tools based on the principles described below would ﬁnd similar, but
better topology for the vision networks. The most convincing proof would be if an automated
system would create network topologies resulting in similar gains in other domains using the same
algorithm but with very differently looking global architecture. At very least, the initial success of
the Inception architecture yields ﬁrm motivation for exciting future work in this direction.

4 Architectural Details

The main idea of the Inception architecture is based on ﬁnding out how an optimal local sparse
structure in a convolutional vision network can be approximated and covered by readily available
dense components. Note that assuming translation invariance means that our network will be built
from convolutional building blocks. All we need is to ﬁnd the optimal local construction and to
repeat it spatially. Arora et al. [2] suggests a layer-by layer construction in which one should analyze
the correlation statistics of the last layer and cluster them into groups of units with high correlation.
These clusters form the units of the next layer and are connected to the units in the previous layer. We
assume that each unit from the earlier layer corresponds to some region of the input image and these
units are grouped into ﬁlter banks. In the lower layers (the ones close to the input) correlated units
would concentrate in local regions. This means, we would end up with a lot of clusters concentrated
in a single region and they can be covered by a layer of 1×1 convolutions in the next layer, as
suggested in [12]. However, one can also expect that there will be a smaller number of more
spatially spread out clusters that can be covered by convolutions over larger patches, and there
will be a decreasing number of patches over larger and larger regions.
In order to avoid patch-
alignment issues, current incarnations of the Inception architecture are restricted to ﬁlter sizes 1×1,
3×3 and 5×5, however this decision was based more on convenience rather than necessity. It also
means that the suggested architecture is a combination of all those layers with their output ﬁlter
banks concatenated into a single output vector forming the input of the next stage. Additionally,
since pooling operations have been essential for the success in current state of the art convolutional
networks, it suggests that adding an alternative parallel pooling path in each such stage should have
additional beneﬁcial effect, too (see Figure 2(a)).

As these “Inception modules” are stacked on top of each other, their output correlation statistics
are bound to vary: as features of higher abstraction are captured by higher layers, their spatial
concentration is expected to decrease suggesting that the ratio of 3×3 and 5×5 convolutions should
increase as we move to higher layers.

One big problem with the above modules, at least in this na¨ıve form, is that even a modest number of
5×5 convolutions can be prohibitively expensive on top of a convolutional layer with a large number
of ﬁlters. This problem becomes even more pronounced once pooling units are added to the mix:
their number of output ﬁlters equals to the number of ﬁlters in the previous stage. The merging of
the output of the pooling layer with the outputs of convolutional layers would lead to an inevitable

4

(a) Inception module, na¨ıve version

(b) Inception module with dimension reductions

Figure 2: Inception module

increase in the number of outputs from stage to stage. Even while this architecture might cover the
optimal sparse structure, it would do it very inefﬁciently, leading to a computational blow up within
a few stages.

This leads to the second idea of the proposed architecture: judiciously applying dimension reduc-
tions and projections wherever the computational requirements would increase too much otherwise.
This is based on the success of embeddings: even low dimensional embeddings might contain a lot
of information about a relatively large image patch. However, embeddings represent information in
a dense, compressed form and compressed information is harder to model. We would like to keep
our representation sparse at most places (as required by the conditions of
[2]) and compress the
signals only whenever they have to be aggregated en masse. That is, 1×1 convolutions are used to
compute reductions before the expensive 3×3 and 5×5 convolutions. Besides being used as reduc-
tions, they also include the use of rectiﬁed linear activation which makes them dual-purpose. The
ﬁnal result is depicted in Figure 2(b).

In general, an Inception network is a network consisting of modules of the above type stacked upon
each other, with occasional max-pooling layers with stride 2 to halve the resolution of the grid. For
technical reasons (memory efﬁciency during training), it seemed beneﬁcial to start using Inception
modules only at higher layers while keeping the lower layers in traditional convolutional fashion.
This is not strictly necessary, simply reﬂecting some infrastructural inefﬁciencies in our current
implementation.

One of the main beneﬁcial aspects of this architecture is that it allows for increasing the number of
units at each stage signiﬁcantly without an uncontrolled blow-up in computational complexity. The
ubiquitous use of dimension reduction allows for shielding the large number of input ﬁlters of the
last stage to the next layer, ﬁrst reducing their dimension before convolving over them with a large
patch size. Another practically useful aspect of this design is that it aligns with the intuition that
visual information should be processed at various scales and then aggregated so that the next stage
can abstract features from different scales simultaneously.

The improved use of computational resources allows for increasing both the width of each stage
as well as the number of stages without getting into computational difﬁculties. Another way to
utilize the inception architecture is to create slightly inferior, but computationally cheaper versions
of it. We have found that all the included the knobs and levers allow for a controlled balancing of
computational resources that can result in networks that are 2 − 3× faster than similarly performing
networks with non-Inception architecture, however this requires careful manual design at this point.

5 GoogLeNet

We chose GoogLeNet as our team-name in the ILSVRC14 competition. This name is an homage to
Yann LeCuns pioneering LeNet 5 network [10]. We also use GoogLeNet to refer to the particular
incarnation of the Inception architecture used in our submission for the competition. We have also
used a deeper and wider Inception network, the quality of which was slightly inferior, but adding it
to the ensemble seemed to improve the results marginally. We omit the details of that network, since
our experiments have shown that the inﬂuence of the exact architectural parameters is relatively

5

1x1 convolutions3x3 convolutions5x5 convolutionsFilter concatenationPrevious layer3x3 max pooling1x1 convolutions3x3 convolutions5x5 convolutionsFilter concatenationPrevious layer3x3 max pooling1x1 convolutions1x1 convolutions1x1 convolutionspatch size/
stride

output
size

depth

#1×1

#3×3
reduce

#3×3

#5×5
reduce

#5×5

pool
proj

type

convolution

max pool

convolution

max pool

inception (3a)

inception (3b)

7×7/2

3×3/2

3×3/1

3×3/2

112×112×64

56×56×64

56×56×192

28×28×192

28×28×256

28×28×480

max pool

3×3/2

14×14×480

inception (4a)

inception (4b)

inception (4c)

inception (4d)

inception (4e)

14×14×512

14×14×512

14×14×512

14×14×528

14×14×832

max pool

3×3/2

7×7×832

inception (5a)

inception (5b)

7×7×832

7×7×1024

avg pool

7×7/1

1×1×1024

dropout (40%)

linear

softmax

1×1×1024

1×1×1000

1×1×1000

64

96

128

96

112

128

144

160

160

192

192

128

192

208

224

256

288

320

320

384

64

128

192

160

128

112

256

256

384

16

32

16

24

24

32

32

32

48

1

0

2

0

2

2

0

2

2

2

2

2

0

2

2

0

0

1

0

params

ops

2.7K

34M

112K

360M

159K

380K

364K

437K

463K

580K

840K

128M

304M

73M

88M

100M

119M

170M

32

96

48

64

64

64

32

64

64

64

64

64

128

128

128

128

128

128

1072K

1388K

54M

71M

1000K

1M

Table 1: GoogLeNet incarnation of the Inception architecture

minor. Here, the most successful particular instance (named GoogLeNet) is described in Table 1 for
demonstrational purposes. The exact same topology (trained with different sampling methods) was
used for 6 out of the 7 models in our ensemble.

All the convolutions, including those inside the Inception modules, use rectiﬁed linear activation.
The size of the receptive ﬁeld in our network is 224×224 taking RGB color channels with mean sub-
traction. “#3×3 reduce” and “#5×5 reduce” stands for the number of 1×1 ﬁlters in the reduction
layer used before the 3×3 and 5×5 convolutions. One can see the number of 1×1 ﬁlters in the pro-
jection layer after the built-in max-pooling in the pool proj column. All these reduction/projection
layers use rectiﬁed linear activation as well.

The network was designed with computational efﬁciency and practicality in mind, so that inference
can be run on individual devices including even those with limited computational resources, espe-
cially with low-memory footprint. The network is 22 layers deep when counting only layers with
parameters (or 27 layers if we also count pooling). The overall number of layers (independent build-
ing blocks) used for the construction of the network is about 100. However this number depends on
the machine learning infrastructure system used. The use of average pooling before the classiﬁer is
based on [12], although our implementation differs in that we use an extra linear layer. This enables
adapting and ﬁne-tuning our networks for other label sets easily, but it is mostly convenience and
we do not expect it to have a major effect. It was found that a move from fully connected layers to
average pooling improved the top-1 accuracy by about 0.6%, however the use of dropout remained
essential even after removing the fully connected layers.

Given the relatively large depth of the network, the ability to propagate gradients back through all the
layers in an effective manner was a concern. One interesting insight is that the strong performance
of relatively shallower networks on this task suggests that the features produced by the layers in the
middle of the network should be very discriminative. By adding auxiliary classiﬁers connected to
these intermediate layers, we would expect to encourage discrimination in the lower stages in the
classiﬁer, increase the gradient signal that gets propagated back, and provide additional regulariza-
tion. These classiﬁers take the form of smaller convolutional networks put on top of the output of
the Inception (4a) and (4d) modules. During training, their loss gets added to the total loss of the
network with a discount weight (the losses of the auxiliary classiﬁers were weighted by 0.3). At
inference time, these auxiliary networks are discarded.

The exact structure of the extra network on the side, including the auxiliary classiﬁer, is as follows:

• An average pooling layer with 5×5 ﬁlter size and stride 3, resulting in an 4×4×512 output

for the (4a), and 4×4×528 for the (4d) stage.

6

Figure 3: GoogLeNet network with all the bells and whistles

7

inputConv7x7+2(S)MaxPool3x3+2(S)LocalRespNormConv1x1+1(V)Conv3x3+1(S)LocalRespNormMaxPool3x3+2(S)Conv1x1+1(S)Conv1x1+1(S)Conv1x1+1(S)MaxPool3x3+1(S)DepthConcatConv3x3+1(S)Conv5x5+1(S)Conv1x1+1(S)Conv1x1+1(S)Conv1x1+1(S)Conv1x1+1(S)MaxPool3x3+1(S)DepthConcatConv3x3+1(S)Conv5x5+1(S)Conv1x1+1(S)MaxPool3x3+2(S)Conv1x1+1(S)Conv1x1+1(S)Conv1x1+1(S)MaxPool3x3+1(S)DepthConcatConv3x3+1(S)Conv5x5+1(S)Conv1x1+1(S)Conv1x1+1(S)Conv1x1+1(S)Conv1x1+1(S)MaxPool3x3+1(S)AveragePool5x5+3(V)DepthConcatConv3x3+1(S)Conv5x5+1(S)Conv1x1+1(S)Conv1x1+1(S)Conv1x1+1(S)Conv1x1+1(S)MaxPool3x3+1(S)DepthConcatConv3x3+1(S)Conv5x5+1(S)Conv1x1+1(S)Conv1x1+1(S)Conv1x1+1(S)Conv1x1+1(S)MaxPool3x3+1(S)DepthConcatConv3x3+1(S)Conv5x5+1(S)Conv1x1+1(S)Conv1x1+1(S)Conv1x1+1(S)Conv1x1+1(S)MaxPool3x3+1(S)AveragePool5x5+3(V)DepthConcatConv3x3+1(S)Conv5x5+1(S)Conv1x1+1(S)MaxPool3x3+2(S)Conv1x1+1(S)Conv1x1+1(S)Conv1x1+1(S)MaxPool3x3+1(S)DepthConcatConv3x3+1(S)Conv5x5+1(S)Conv1x1+1(S)Conv1x1+1(S)Conv1x1+1(S)Conv1x1+1(S)MaxPool3x3+1(S)DepthConcatConv3x3+1(S)Conv5x5+1(S)Conv1x1+1(S)AveragePool7x7+1(V)FCConv1x1+1(S)FCFCSoftmaxActivationsoftmax0Conv1x1+1(S)FCFCSoftmaxActivationsoftmax1SoftmaxActivationsoftmax2• A 1×1 convolution with 128 ﬁlters for dimension reduction and rectiﬁed linear activation.

• A fully connected layer with 1024 units and rectiﬁed linear activation.

• A dropout layer with 70% ratio of dropped outputs.

• A linear layer with softmax loss as the classiﬁer (predicting the same 1000 classes as the

main classiﬁer, but removed at inference time).

A schematic view of the resulting network is depicted in Figure 3.

6 Training Methodology

Our networks were trained using the DistBelief [4] distributed machine learning system using mod-
est amount of model and data-parallelism. Although we used CPU based implementation only, a
rough estimate suggests that the GoogLeNet network could be trained to convergence using few
high-end GPUs within a week, the main limitation being the memory usage. Our training used
asynchronous stochastic gradient descent with 0.9 momentum [17], ﬁxed learning rate schedule (de-
creasing the learning rate by 4% every 8 epochs). Polyak averaging [13] was used to create the ﬁnal
model used at inference time.

Our image sampling methods have changed substantially over the months leading to the competition,
and already converged models were trained on with other options, sometimes in conjunction with
changed hyperparameters, like dropout and learning rate, so it is hard to give a deﬁnitive guidance
to the most effective single way to train these networks. To complicate matters further, some of
the models were mainly trained on smaller relative crops, others on larger ones, inspired by [8].
Still, one prescription that was veriﬁed to work very well after the competition includes sampling
of various sized patches of the image whose size is distributed evenly between 8% and 100% of the
image area and whose aspect ratio is chosen randomly between 3/4 and 4/3. Also, we found that the
photometric distortions by Andrew Howard [8] were useful to combat overﬁtting to some extent. In
addition, we started to use random interpolation methods (bilinear, area, nearest neighbor and cubic,
with equal probability) for resizing relatively late and in conjunction with other hyperparameter
changes, so we could not tell deﬁnitely whether the ﬁnal results were affected positively by their
use.

7

ILSVRC 2014 Classiﬁcation Challenge Setup and Results

The ILSVRC 2014 classiﬁcation challenge involves the task of classifying the image into one of
1000 leaf-node categories in the Imagenet hierarchy. There are about 1.2 million images for training,
50,000 for validation and 100,000 images for testing. Each image is associated with one ground
truth category, and performance is measured based on the highest scoring classiﬁer predictions.
Two numbers are usually reported: the top-1 accuracy rate, which compares the ground truth against
the ﬁrst predicted class, and the top-5 error rate, which compares the ground truth against the ﬁrst
5 predicted classes: an image is deemed correctly classiﬁed if the ground truth is among the top-5,
regardless of its rank in them. The challenge uses the top-5 error rate for ranking purposes.

We participated in the challenge with no external data used for training. In addition to the training
techniques aforementioned in this paper, we adopted a set of techniques during testing to obtain a
higher performance, which we elaborate below.

1. We independently trained 7 versions of the same GoogLeNet model (including one wider
version), and performed ensemble prediction with them. These models were trained with
the same initialization (even with the same initial weights, mainly because of an oversight)
and learning rate policies, and they only differ in sampling methodologies and the random
order in which they see input images.

2. During testing, we adopted a more aggressive cropping approach than that of Krizhevsky et
al. [9]. Speciﬁcally, we resize the image to 4 scales where the shorter dimension (height or
width) is 256, 288, 320 and 352 respectively, take the left, center and right square of these
resized images (in the case of portrait images, we take the top, center and bottom squares).
For each square, we then take the 4 corners and the center 224×224 crop as well as the

8

Team

Year Place Error (top-5) Uses external data

SuperVision

SuperVision

Clarifai

Clarifai

MSRA

VGG

GoogLeNet

2012

2012

2013

2013

2014

2014

2014

1st

1st

1st

1st

3rd

2nd

1st

16.4%

15.3%

11.7%

11.2%

7.35%

7.32%

6.67%

no

Imagenet 22k

no

Imagenet 22k

no

no

no

Table 2: Classiﬁcation performance

Number of models Number of Crops Cost Top-5 error

compared to base

1

1

1

7

7

7

1

10

144

1

10

144

1

10

144

7

70

10.07%

9.15%

7.89%

8.09%

7.62%

1008

6.67%

base

-0.92%

-2.18%

-1.98%

-2.45%

-3.45%

Table 3: GoogLeNet classiﬁcation performance break down

square resized to 224×224, and their mirrored versions. This results in 4×3×6×2 = 144
crops per image. A similar approach was used by Andrew Howard [8] in the previous year’s
entry, which we empirically veriﬁed to perform slightly worse than the proposed scheme.
We note that such aggressive cropping may not be necessary in real applications, as the
beneﬁt of more crops becomes marginal after a reasonable number of crops are present (as
we will show later on).

3. The softmax probabilities are averaged over multiple crops and over all the individual clas-
siﬁers to obtain the ﬁnal prediction. In our experiments we analyzed alternative approaches
on the validation data, such as max pooling over crops and averaging over classiﬁers, but
they lead to inferior performance than the simple averaging.

In the remainder of this paper, we analyze the multiple factors that contribute to the overall perfor-
mance of the ﬁnal submission.

Our ﬁnal submission in the challenge obtains a top-5 error of 6.67% on both the validation and
testing data, ranking the ﬁrst among other participants. This is a 56.5% relative reduction compared
to the SuperVision approach in 2012, and about 40% relative reduction compared to the previous
year’s best approach (Clarifai), both of which used external data for training the classiﬁers. The
following table shows the statistics of some of the top-performing approaches.

We also analyze and report the performance of multiple testing choices, by varying the number of
models and the number of crops used when predicting an image in the following table. When we
use one model, we chose the one with the lowest top-1 error rate on the validation data. All numbers
are reported on the validation dataset in order to not overﬁt to the testing data statistics.

8

ILSVRC 2014 Detection Challenge Setup and Results

The ILSVRC detection task is to produce bounding boxes around objects in images among 200
possible classes. Detected objects count as correct if they match the class of the groundtruth and
their bounding boxes overlap by at least 50% (using the Jaccard index). Extraneous detections count
as false positives and are penalized. Contrary to the classiﬁcation task, each image may contain

9

Team

UvA-Euvision

Deep Insight

CUHK DeepID-Net

GoogLeNet

Year Place mAP

external data

ensemble

approach

2013

2014

2014

2014

1st

3rd

2nd

1st

22.6% none

40.5% ImageNet 1k

40.7% ImageNet 1k

43.9% ImageNet 1k

?

3

?

6

Fisher vectors

CNN

CNN

CNN

Table 4: Detection performance

Team

mAP

Contextual model Bounding box regression

Trimps-Soushen

Berkeley Vision

UvA-Euvision

CUHK DeepID-Net2

GoogLeNet

Deep Insight

31.6%

34.5%

35.4%

37.7%

38.02%

40.2%

no

no

?

no

no

yes

?

yes

?

?

no

yes

Table 5: Single model performance for detection

many objects or none, and their scale may vary from large to tiny. Results are reported using the
mean average precision (mAP).

The approach taken by GoogLeNet for detection is similar to the R-CNN by [6], but is augmented
with the Inception model as the region classiﬁer. Additionally, the region proposal step is improved
by combining the Selective Search [20] approach with multi-box [5] predictions for higher object
bounding box recall. In order to cut down the number of false positives, the superpixel size was
increased by 2×. This halves the proposals coming from the selective search algorithm. We added
back 200 region proposals coming from multi-box [5] resulting, in total, in about 60% of the pro-
posals used by [6], while increasing the coverage from 92% to 93%. The overall effect of cutting the
number of proposals with increased coverage is a 1% improvement of the mean average precision
for the single model case. Finally, we use an ensemble of 6 ConvNets when classifying each region
which improves results from 40% to 43.9% accuracy. Note that contrary to R-CNN, we did not use
bounding box regression due to lack of time.

We ﬁrst report the top detection results and show the progress since the ﬁrst edition of the detection
task. Compared to the 2013 result, the accuracy has almost doubled. The top performing teams all
use Convolutional Networks. We report the ofﬁcial scores in Table 4 and common strategies for each
team: the use of external data, ensemble models or contextual models. The external data is typically
the ILSVRC12 classiﬁcation data for pre-training a model that is later reﬁned on the detection data.
Some teams also mention the use of the localization data. Since a good portion of the localization
task bounding boxes are not included in the detection dataset, one can pre-train a general bounding
box regressor with this data the same way classiﬁcation is used for pre-training. The GoogLeNet
entry did not use the localization data for pretraining.

In Table 5, we compare results using a single model only. The top performing model is by Deep
Insight and surprisingly only improves by 0.3 points with an ensemble of 3 models while the
GoogLeNet obtains signiﬁcantly stronger results with the ensemble.

9 Conclusions

Our results seem to yield a solid evidence that approximating the expected optimal sparse structure
by readily available dense building blocks is a viable method for improving neural networks for
computer vision. The main advantage of this method is a signiﬁcant quality gain at a modest in-
crease of computational requirements compared to shallower and less wide networks. Also note that
our detection work was competitive despite of neither utilizing context nor performing bounding box

10

regression and this fact provides further evidence of the strength of the Inception architecture. Al-
though it is expected that similar quality of result can be achieved by much more expensive networks
of similar depth and width, our approach yields solid evidence that moving to sparser architectures
is feasible and useful idea in general. This suggest promising future work towards creating sparser
and more reﬁned structures in automated ways on the basis of [2].

10 Acknowledgements

We would like to thank Sanjeev Arora and Aditya Bhaskara for fruitful discussions on [2]. Also
we are indebted to the DistBelief [4] team for their support especially to Rajat Monga, Jon Shlens,
Alex Krizhevsky, Jeff Dean, Ilya Sutskever and Andrea Frome. We would also like to thank to Tom
Duerig and Ning Ye for their help on photometric distortions. Also our work would not have been
possible without the support of Chuck Rosenberg and Hartwig Adam.

References

[1] Know your meme: We need to go deeper. http://knowyourmeme.com/memes/

we-need-to-go-deeper. Accessed: 2014-09-15.

[2] Sanjeev Arora, Aditya Bhaskara, Rong Ge, and Tengyu Ma. Provable bounds for learning

[3]

some deep representations. CoRR, abs/1310.6343, 2013.
¨Umit V. C¸ ataly¨urek, Cevdet Aykanat, and Bora Uc¸ar. On two-dimensional sparse matrix par-
titioning: Models, methods, and a recipe. SIAM J. Sci. Comput., 32(2):656–683, February
2010.

[4] Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Mark Mao,
Marc’aurelio Ranzato, Andrew Senior, Paul Tucker, Ke Yang, Quoc V. Le, and Andrew Y.
Ng. Large scale distributed deep networks. In P. Bartlett, F.c.n. Pereira, C.j.c. Burges, L. Bot-
tou, and K.q. Weinberger, editors, Advances in Neural Information Processing Systems 25,
pages 1232–1240. 2012.

[5] Dumitru Erhan, Christian Szegedy, Alexander Toshev, and Dragomir Anguelov. Scalable ob-
ject detection using deep neural networks. In Computer Vision and Pattern Recognition, 2014.
CVPR 2014. IEEE Conference on, 2014.

[6] Ross B. Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies
In Computer Vision and Pattern

for accurate object detection and semantic segmentation.
Recognition, 2014. CVPR 2014. IEEE Conference on, 2014.

[7] Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhut-
dinov. Improving neural networks by preventing co-adaptation of feature detectors. CoRR,
abs/1207.0580, 2012.

[8] Andrew G. Howard. Some improvements on deep convolutional neural network based image

classiﬁcation. CoRR, abs/1312.5402, 2013.

[9] Alex Krizhevsky, Ilya Sutskever, and Geoff Hinton. Imagenet classiﬁcation with deep con-
volutional neural networks. In Advances in Neural Information Processing Systems 25, pages
1106–1114, 2012.

[10] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel.
Backpropagation applied to handwritten zip code recognition. Neural Comput., 1(4):541–551,
December 1989.

[11] Yann LeCun, L´eon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning

applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

[12] Min Lin, Qiang Chen, and Shuicheng Yan. Network in network. CoRR, abs/1312.4400, 2013.
[13] B. T. Polyak and A. B. Juditsky. Acceleration of stochastic approximation by averaging. SIAM

J. Control Optim., 30(4):838–855, July 1992.

[14] Pierre Sermanet, David Eigen, Xiang Zhang, Micha¨el Mathieu, Rob Fergus, and Yann Le-
Cun. Overfeat: Integrated recognition, localization and detection using convolutional net-
works. CoRR, abs/1312.6229, 2013.

11

[15] Thomas Serre, Lior Wolf, Stanley M. Bileschi, Maximilian Riesenhuber, and Tomaso Poggio.
IEEE Trans. Pattern Anal. Mach.

Robust object recognition with cortex-like mechanisms.
Intell., 29(3):411–426, 2007.

[16] Fengguang Song and Jack Dongarra. Scaling up matrix computations on shared-memory
manycore systems with 1000 cpu cores. In Proceedings of the 28th ACM International Con-
ference on Supercomputing, ICS ’14, pages 333–342, New York, NY, USA, 2014. ACM.
[17] Ilya Sutskever, James Martens, George E. Dahl, and Geoffrey E. Hinton. On the importance
In Proceedings of the 30th International
of initialization and momentum in deep learning.
Conference on Machine Learning, ICML 2013, Atlanta, GA, USA, 16-21 June 2013, volume 28
of JMLR Proceedings, pages 1139–1147. JMLR.org, 2013.

[18] Christian Szegedy, Alexander Toshev, and Dumitru Erhan. Deep neural networks for object
In Christopher J. C. Burges, L´eon Bottou, Zoubin Ghahramani, and Kilian Q.
detection.
Weinberger, editors, Advances in Neural Information Processing Systems 26: 27th Annual
Conference on Neural Information Processing Systems 2013. Proceedings of a meeting held
December 5-8, 2013, Lake Tahoe, Nevada, United States., pages 2553–2561, 2013.

[19] Alexander Toshev and Christian Szegedy. Deeppose: Human pose estimation via deep neural

networks. CoRR, abs/1312.4659, 2013.

[20] Koen E. A. van de Sande, Jasper R. R. Uijlings, Theo Gevers, and Arnold W. M. Smeulders.
Segmentation as selective search for object recognition. In Proceedings of the 2011 Interna-
tional Conference on Computer Vision, ICCV ’11, pages 1879–1886, Washington, DC, USA,
2011. IEEE Computer Society.

[21] Matthew D. Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In
David J. Fleet, Tom´as Pajdla, Bernt Schiele, and Tinne Tuytelaars, editors, Computer Vision
- ECCV 2014 - 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Pro-
ceedings, Part I, volume 8689 of Lecture Notes in Computer Science, pages 818–833. Springer,
2014.

12


