8
1
0
2

n
a
J

8
2

]

V
C
.
s
c
[

5
v
3
9
9
6
0
.
8
0
6
1
:
v
i
X
r
a

Densely Connected Convolutional Networks

Gao Huang∗
Cornell University
gh349@cornell.edu

Zhuang Liu∗
Tsinghua University
liuzhuang13@mails.tsinghua.edu.cn

Laurens van der Maaten
Facebook AI Research
lvdmaaten@fb.com

Kilian Q. Weinberger
Cornell University
kqw4@cornell.edu

Abstract

Recent work has shown that convolutional networks can
be substantially deeper, more accurate, and efﬁcient to train
if they contain shorter connections between layers close to
the input and those close to the output. In this paper, we
embrace this observation and introduce the Dense Convo-
lutional Network (DenseNet), which connects each layer
to every other layer in a feed-forward fashion. Whereas
traditional convolutional networks with L layers have L
connections—one between each layer and its subsequent
layer—our network has L(L+1)
direct connections. For
2
each layer, the feature-maps of all preceding layers are
used as inputs, and its own feature-maps are used as inputs
into all subsequent layers. DenseNets have several com-
pelling advantages: they alleviate the vanishing-gradient
problem, strengthen feature propagation, encourage fea-
ture reuse, and substantially reduce the number of parame-
ters. We evaluate our proposed architecture on four highly
competitive object recognition benchmark tasks (CIFAR-10,
CIFAR-100, SVHN, and ImageNet). DenseNets obtain sig-
niﬁcant improvements over the state-of-the-art on most of
them, whilst requiring less computation to achieve high per-
formance. Code and pre-trained models are available at
https://github.com/liuzhuang13/DenseNet.

1. Introduction

Convolutional neural networks (CNNs) have become
the dominant machine learning approach for visual object
recognition. Although they were originally introduced over
20 years ago [18], improvements in computer hardware and
network structure have enabled the training of truly deep
CNNs only recently. The original LeNet5 [19] consisted of
5 layers, VGG featured 19 [29], and only last year Highway

∗Authors contributed equally

Figure 1: A 5-layer dense block with a growth rate of k = 4.
Each layer takes all preceding feature-maps as input.

Networks [34] and Residual Networks (ResNets) [11] have
surpassed the 100-layer barrier.

As CNNs become increasingly deep, a new research
problem emerges: as information about the input or gra-
dient passes through many layers, it can vanish and “wash
out” by the time it reaches the end (or beginning) of the
network. Many recent publications address this or related
problems. ResNets [11] and Highway Networks [34] by-
pass signal from one layer to the next via identity connec-
tions. Stochastic depth [13] shortens ResNets by randomly
dropping layers during training to allow better information
and gradient ﬂow. FractalNets [17] repeatedly combine sev-
eral parallel layer sequences with different number of con-
volutional blocks to obtain a large nominal depth, while
maintaining many short paths in the network. Although
these different approaches vary in network topology and
training procedure, they all share a key characteristic: they
create short paths from early layers to later layers.

1

x0x1H1x2H2H3H4x3x4

In this paper, we propose an architecture that distills this
insight into a simple connectivity pattern: to ensure maxi-
mum information ﬂow between layers in the network, we
connect all layers (with matching feature-map sizes) di-
rectly with each other. To preserve the feed-forward nature,
each layer obtains additional inputs from all preceding lay-
ers and passes on its own feature-maps to all subsequent
layers. Figure 1 illustrates this layout schematically. Cru-
cially, in contrast to ResNets, we never combine features
through summation before they are passed into a layer; in-
stead, we combine features by concatenating them. Hence,
the (cid:96)th layer has (cid:96) inputs, consisting of the feature-maps
of all preceding convolutional blocks. Its own feature-maps
are passed on to all L−(cid:96) subsequent layers. This introduces
L(L+1)
connections in an L-layer network, instead of just
2
L, as in traditional architectures. Because of its dense con-
nectivity pattern, we refer to our approach as Dense Convo-
lutional Network (DenseNet).

A possibly counter-intuitive effect of this dense connec-
tivity pattern is that it requires fewer parameters than tra-
ditional convolutional networks, as there is no need to re-
learn redundant feature-maps. Traditional feed-forward ar-
chitectures can be viewed as algorithms with a state, which
is passed on from layer to layer. Each layer reads the state
from its preceding layer and writes to the subsequent layer.
It changes the state but also passes on information that needs
to be preserved. ResNets [11] make this information preser-
vation explicit through additive identity transformations.
Recent variations of ResNets [13] show that many layers
contribute very little and can in fact be randomly dropped
during training. This makes the state of ResNets similar
to (unrolled) recurrent neural networks [21], but the num-
ber of parameters of ResNets is substantially larger because
each layer has its own weights. Our proposed DenseNet ar-
chitecture explicitly differentiates between information that
is added to the network and information that is preserved.
DenseNet layers are very narrow (e.g., 12 ﬁlters per layer),
adding only a small set of feature-maps to the “collective
knowledge” of the network and keep the remaining feature-
maps unchanged—and the ﬁnal classiﬁer makes a decision
based on all feature-maps in the network.

Besides better parameter efﬁciency, one big advantage of
DenseNets is their improved ﬂow of information and gra-
dients throughout the network, which makes them easy to
train. Each layer has direct access to the gradients from the
loss function and the original input signal, leading to an im-
plicit deep supervision [20]. This helps training of deeper
network architectures. Further, we also observe that dense
connections have a regularizing effect, which reduces over-
ﬁtting on tasks with smaller training set sizes.

We evaluate DenseNets on four highly competitive
benchmark datasets (CIFAR-10, CIFAR-100, SVHN, and
ImageNet). Our models tend to require much fewer param-

eters than existing algorithms with comparable accuracy.
Further, we signiﬁcantly outperform the current state-of-
the-art results on most of the benchmark tasks.

2. Related Work

The exploration of network architectures has been a part
of neural network research since their initial discovery. The
recent resurgence in popularity of neural networks has also
revived this research domain. The increasing number of lay-
ers in modern networks ampliﬁes the differences between
architectures and motivates the exploration of different con-
nectivity patterns and the revisiting of old research ideas.

A cascade structure similar to our proposed dense net-
work layout has already been studied in the neural networks
literature in the 1980s [3]. Their pioneering work focuses on
fully connected multi-layer perceptrons trained in a layer-
by-layer fashion. More recently, fully connected cascade
networks to be trained with batch gradient descent were
proposed [40]. Although effective on small datasets, this
approach only scales to networks with a few hundred pa-
rameters. In [9, 23, 31, 41], utilizing multi-level features
in CNNs through skip-connnections has been found to be
effective for various vision tasks. Parallel to our work, [1]
derived a purely theoretical framework for networks with
cross-layer connections similar to ours.

Highway Networks [34] were amongst the ﬁrst architec-
tures that provided a means to effectively train end-to-end
networks with more than 100 layers. Using bypassing paths
along with gating units, Highway Networks with hundreds
of layers can be optimized without difﬁculty. The bypass-
ing paths are presumed to be the key factor that eases the
training of these very deep networks. This point is further
supported by ResNets [11], in which pure identity mappings
are used as bypassing paths. ResNets have achieved im-
pressive, record-breaking performance on many challeng-
ing image recognition, localization, and detection tasks,
such as ImageNet and COCO object detection [11]. Re-
cently, stochastic depth was proposed as a way to success-
fully train a 1202-layer ResNet [13]. Stochastic depth im-
proves the training of deep residual networks by dropping
layers randomly during training. This shows that not all
layers may be needed and highlights that there is a great
amount of redundancy in deep (residual) networks. Our pa-
per was partly inspired by that observation. ResNets with
pre-activation also facilitate the training of state-of-the-art
networks with > 1000 layers [12].

An orthogonal approach to making networks deeper
(e.g., with the help of skip connections) is to increase the
network width. The GoogLeNet [36, 37] uses an “Incep-
tion module” which concatenates feature-maps produced
by ﬁlters of different sizes. In [38], a variant of ResNets
with wide generalized residual blocks was proposed.
In
fact, simply increasing the number of ﬁlters in each layer of

Figure 2: A deep DenseNet with three dense blocks. The layers between two adjacent blocks are referred to as transition layers and change
feature-map sizes via convolution and pooling.

ResNets can improve its performance provided the depth is
sufﬁcient [42]. FractalNets also achieve competitive results
on several datasets using a wide network structure [17].

Instead of drawing representational power from ex-
tremely deep or wide architectures, DenseNets exploit the
potential of the network through feature reuse, yielding con-
densed models that are easy to train and highly parameter-
efﬁcient. Concatenating feature-maps learned by different
layers increases variation in the input of subsequent layers
and improves efﬁciency. This constitutes a major difference
between DenseNets and ResNets. Compared to Inception
networks [36, 37], which also concatenate features from dif-
ferent layers, DenseNets are simpler and more efﬁcient.

There are other notable network architecture innovations
which have yielded competitive results. The Network in
Network (NIN) [22] structure includes micro multi-layer
perceptrons into the ﬁlters of convolutional layers to ex-
tract more complicated features. In Deeply Supervised Net-
work (DSN) [20], internal layers are directly supervised
by auxiliary classiﬁers, which can strengthen the gradients
received by earlier layers. Ladder Networks [27, 25] in-
troduce lateral connections into autoencoders, producing
impressive accuracies on semi-supervised learning tasks.
In [39], Deeply-Fused Nets (DFNs) were proposed to im-
prove information ﬂow by combining intermediate layers
of different base networks. The augmentation of networks
with pathways that minimize reconstruction losses was also
shown to improve image classiﬁcation models [43].

3. DenseNets

Consider a single image x0 that is passed through a con-
volutional network. The network comprises L layers, each
of which implements a non-linear transformation H(cid:96)(·),
where (cid:96) indexes the layer. H(cid:96)(·) can be a composite func-
tion of operations such as Batch Normalization (BN) [14],
rectiﬁed linear units (ReLU) [6], Pooling [19], or Convolu-
tion (Conv). We denote the output of the (cid:96)th layer as x(cid:96).

ResNets. Traditional convolutional
feed-forward net-
works connect the output of the (cid:96)th layer as input to the
((cid:96) + 1)th layer [16], which gives rise to the following
layer transition: x(cid:96) = H(cid:96)(x(cid:96)−1). ResNets [11] add a
skip-connection that bypasses the non-linear transforma-
tions with an identity function:

x(cid:96) = H(cid:96)(x(cid:96)−1) + x(cid:96)−1.

(1)

An advantage of ResNets is that the gradient can ﬂow di-
rectly through the identity function from later layers to the
earlier layers. However, the identity function and the output
of H(cid:96) are combined by summation, which may impede the
information ﬂow in the network.

Dense connectivity. To further improve the information
ﬂow between layers we propose a different connectivity
pattern: we introduce direct connections from any layer
to all subsequent layers. Figure 1 illustrates the layout of
the resulting DenseNet schematically. Consequently, the
(cid:96)th layer receives the feature-maps of all preceding layers,
x0, . . . , x(cid:96)−1, as input:

x(cid:96) = H(cid:96)([x0, x1, . . . , x(cid:96)−1]),

(2)

where [x0, x1, . . . , x(cid:96)−1] refers to the concatenation of the
feature-maps produced in layers 0, . . . , (cid:96) − 1. Because of its
dense connectivity we refer to this network architecture as
Dense Convolutional Network (DenseNet). For ease of im-
plementation, we concatenate the multiple inputs of H(cid:96)(·)
in eq. (2) into a single tensor.

Composite function. Motivated by [12], we deﬁne H(cid:96)(·)
as a composite function of three consecutive operations:
batch normalization (BN) [14], followed by a rectiﬁed lin-
ear unit (ReLU) [6] and a 3 × 3 convolution (Conv).

Pooling layers. The concatenation operation used in
Eq. (2) is not viable when the size of feature-maps changes.
However, an essential part of convolutional networks is
down-sampling layers that change the size of feature-maps.
To facilitate down-sampling in our architecture we divide
the network into multiple densely connected dense blocks;
see Figure 2. We refer to layers between blocks as transition
layers, which do convolution and pooling. The transition
layers used in our experiments consist of a batch normal-
ization layer and an 1×1 convolutional layer followed by a
2×2 average pooling layer.

If each function H(cid:96) produces k feature-
Growth rate.
maps, it follows that the (cid:96)th layer has k0 + k × ((cid:96) − 1) input
feature-maps, where k0 is the number of channels in the in-
put layer. An important difference between DenseNet and
existing network architectures is that DenseNet can have
very narrow layers, e.g., k = 12. We refer to the hyper-
parameter k as the growth rate of the network. We show in
Section 4 that a relatively small growth rate is sufﬁcient to

ConvolutionPoolingDense Block 1ConvolutionPoolingPoolingLinearConvolutionInputPrediction“horse”Dense Block 2Dense Block 3Layers
Convolution
Pooling
Dense Block
(1)
Transition Layer
(1)
Dense Block
(2)
Transition Layer
(2)
Dense Block
(3)
Transition Layer
(3)
Dense Block
(4)
Classiﬁcation
Layer

Output Size
112 × 112
56 × 56

56 × 56

56 × 56
28 × 28

28 × 28

28 × 28
14 × 14

14 × 14

14 × 14
7 × 7

7 × 7

1 × 1

DenseNet-121

DenseNet-169

DenseNet-201

DenseNet-264

(cid:20) 1 × 1 conv
3 × 3 conv

(cid:21)

× 6

(cid:20) 1 × 1 conv
3 × 3 conv

(cid:21)

× 12

(cid:20) 1 × 1 conv
3 × 3 conv

(cid:21)

× 24

(cid:20) 1 × 1 conv
3 × 3 conv

(cid:21)

× 16

7 × 7 conv, stride 2
3 × 3 max pool, stride 2

(cid:20) 1 × 1 conv
3 × 3 conv

(cid:21)

× 6

(cid:20) 1 × 1 conv
3 × 3 conv

1 × 1 conv
2 × 2 average pool, stride 2
(cid:20) 1 × 1 conv
3 × 3 conv

× 12

(cid:21)

(cid:20) 1 × 1 conv
3 × 3 conv

1 × 1 conv
2 × 2 average pool, stride 2
(cid:20) 1 × 1 conv
3 × 3 conv

× 32

(cid:21)

(cid:20) 1 × 1 conv
3 × 3 conv

1 × 1 conv
2 × 2 average pool, stride 2
(cid:20) 1 × 1 conv
3 × 3 conv

× 32

(cid:21)

(cid:20) 1 × 1 conv
3 × 3 conv

7 × 7 global average pool
1000D fully-connected, softmax

(cid:21)

(cid:21)

(cid:21)

(cid:21)

× 6

(cid:20) 1 × 1 conv
3 × 3 conv

(cid:21)

× 6

× 12

(cid:20) 1 × 1 conv
3 × 3 conv

(cid:21)

× 12

× 48

(cid:20) 1 × 1 conv
3 × 3 conv

(cid:21)

× 64

× 32

(cid:20) 1 × 1 conv
3 × 3 conv

(cid:21)

× 48

Table 1: DenseNet architectures for ImageNet. The growth rate for all the networks is k = 32. Note that each “conv” layer shown in the
table corresponds the sequence BN-ReLU-Conv.

obtain state-of-the-art results on the datasets that we tested
on. One explanation for this is that each layer has access
to all the preceding feature-maps in its block and, therefore,
to the network’s “collective knowledge”. One can view the
feature-maps as the global state of the network. Each layer
adds k feature-maps of its own to this state. The growth
rate regulates how much new information each layer con-
tributes to the global state. The global state, once written,
can be accessed from everywhere within the network and,
unlike in traditional network architectures, there is no need
to replicate it from layer to layer.

Bottleneck layers. Although each layer only produces k
output feature-maps, it typically has many more inputs. It
has been noted in [37, 11] that a 1×1 convolution can be in-
troduced as bottleneck layer before each 3×3 convolution
to reduce the number of input feature-maps, and thus to
improve computational efﬁciency. We ﬁnd this design es-
pecially effective for DenseNet and we refer to our network
with such a bottleneck layer, i.e., to the BN-ReLU-Conv(1×
1)-BN-ReLU-Conv(3×3) version of H(cid:96), as DenseNet-B. In
our experiments, we let each 1×1 convolution produce 4k
feature-maps.

Compression. To further improve model compactness,
we can reduce the number of feature-maps at transition
layers. If a dense block contains m feature-maps, we let
the following transition layer generate (cid:98)θm(cid:99) output feature-
maps, where 0 < θ ≤ 1 is referred to as the compression fac-
tor. When θ = 1, the number of feature-maps across transi-
tion layers remains unchanged. We refer the DenseNet with
θ < 1 as DenseNet-C, and we set θ = 0.5 in our experiment.
When both the bottleneck and transition layers with θ < 1
are used, we refer to our model as DenseNet-BC.

Implementation Details. On all datasets except Ima-
geNet, the DenseNet used in our experiments has three
dense blocks that each has an equal number of layers. Be-
fore entering the ﬁrst dense block, a convolution with 16 (or
twice the growth rate for DenseNet-BC) output channels is
performed on the input images. For convolutional layers
with kernel size 3×3, each side of the inputs is zero-padded
by one pixel to keep the feature-map size ﬁxed. We use 1×1
convolution followed by 2×2 average pooling as transition
layers between two contiguous dense blocks. At the end of
the last dense block, a global average pooling is performed
and then a softmax classiﬁer is attached. The feature-map
sizes in the three dense blocks are 32× 32, 16×16, and
8×8, respectively. We experiment with the basic DenseNet
structure with conﬁgurations {L = 40, k = 12}, {L =
100, k = 12} and {L = 100, k = 24}. For DenseNet-
BC, the networks with conﬁgurations {L = 100, k = 12},
{L = 250, k = 24} and {L = 190, k = 40} are evaluated.

In our experiments on ImageNet, we use a DenseNet-BC
structure with 4 dense blocks on 224×224 input images.
The initial convolution layer comprises 2k convolutions of
size 7×7 with stride 2; the number of feature-maps in all
other layers also follow from setting k. The exact network
conﬁgurations we used on ImageNet are shown in Table 1.

4. Experiments

We empirically demonstrate DenseNet’s effectiveness on
several benchmark datasets and compare with state-of-the-
art architectures, especially with ResNet and its variants.

Method

Network in Network [22]
All-CNN [32]
Deeply Supervised Net [20]
Highway Network [34]
FractalNet [17]
with Dropout/Drop-path
ResNet [11]
ResNet (reported by [13])
ResNet with Stochastic Depth [13]

Wide ResNet [42]

with Dropout
ResNet (pre-activation) [12]

DenseNet (k = 12)
DenseNet (k = 12)
DenseNet (k = 24)
DenseNet-BC (k = 12)
DenseNet-BC (k = 24)
DenseNet-BC (k = 40)

Depth
-
-
-
-
21
21
110
110
110
1202
16
28
16
164
1001
40
100
100
100
250
190

Params
-
-
-
-
38.6M
38.6M
1.7M
1.7M
1.7M
10.2M
11.0M
36.5M
2.7M
1.7M
10.2M
1.0M
7.0M
27.2M
0.8M
15.3M
25.6M

C10
10.41
9.08
9.69
-
10.18
7.33
-
13.63
11.66
-
-
-
-
11.26∗
10.56∗
7.00
5.77
5.83
5.92
5.19
-

C10+
8.81
7.25
7.97
7.72
5.22
4.60
6.61
6.41
5.23
4.91
4.81
4.17
-
5.46
4.62
5.24
4.10
3.74
4.51
3.62
3.46

C100
35.68
-
-
-
35.34
28.20
-
44.74
37.80
-
-
-
-
35.58∗
33.47∗
27.55
23.79
23.42
24.15
19.64
-

C100+
-
33.71
34.57
32.39
23.30
23.73
-
27.22
24.58
-
22.07
20.50
-
24.33
22.71
24.42
20.20
19.25
22.27
17.60
17.18

SVHN
2.35
-
1.92
-
2.01
1.87
-
2.01
1.75
-
-
-
1.64
-
-
1.79
1.67
1.59
1.76
1.74
-

Table 2: Error rates (%) on CIFAR and SVHN datasets. k denotes network’s growth rate. Results that surpass all competing methods are
bold and the overall best results are blue. “+” indicates standard data augmentation (translation and/or mirroring). ∗ indicates results run
by ourselves. All the results of DenseNets without data augmentation (C10, C100, SVHN) are obtained using Dropout. DenseNets achieve
lower error rates while using fewer parameters than ResNet. Without data augmentation, DenseNet performs better by a large margin.

4.1. Datasets

CIFAR. The two CIFAR datasets [15] consist of colored
natural images with 32×32 pixels. CIFAR-10 (C10) con-
sists of images drawn from 10 and CIFAR-100 (C100) from
100 classes. The training and test sets contain 50,000 and
10,000 images respectively, and we hold out 5,000 training
images as a validation set. We adopt a standard data aug-
mentation scheme (mirroring/shifting) that is widely used
for these two datasets [11, 13, 17, 22, 28, 20, 32, 34]. We
denote this data augmentation scheme by a “+” mark at the
end of the dataset name (e.g., C10+). For preprocessing,
we normalize the data using the channel means and stan-
dard deviations. For the ﬁnal run we use all 50,000 training
images and report the ﬁnal test error at the end of training.

SVHN. The Street View House Numbers (SVHN) dataset
[24] contains 32×32 colored digit images. There are 73,257
images in the training set, 26,032 images in the test set, and
531,131 images for additional training. Following common
practice [7, 13, 20, 22, 30] we use all the training data with-
out any data augmentation, and a validation set with 6,000
images is split from the training set. We select the model
with the lowest validation error during training and report
the test error. We follow [42] and divide the pixel values by
255 so they are in the [0, 1] range.

ImageNet. The ILSVRC 2012 classiﬁcation dataset [2]
consists 1.2 million images for training, and 50,000 for val-
idation, from 1, 000 classes. We adopt the same data aug-
mentation scheme for training images as in [8, 11, 12], and
apply a single-crop or 10-crop with size 224×224 at test
time. Following [11, 12, 13], we report classiﬁcation errors
on the validation set.

4.2. Training

All the networks are trained using stochastic gradient de-
scent (SGD). On CIFAR and SVHN we train using batch
size 64 for 300 and 40 epochs, respectively. The initial
learning rate is set to 0.1, and is divided by 10 at 50% and
75% of the total number of training epochs. On ImageNet,
we train models for 90 epochs with a batch size of 256.
The learning rate is set to 0.1 initially, and is lowered by
10 times at epoch 30 and 60. Note that a naive implemen-
tation of DenseNet may contain memory inefﬁciencies. To
reduce the memory consumption on GPUs, please refer to
our technical report on the memory-efﬁcient implementa-
tion of DenseNets [26].

Following [8], we use a weight decay of 10−4 and a
Nesterov momentum [35] of 0.9 without dampening. We
adopt the weight initialization introduced by [10]. For the
three datasets without data augmentation, i.e., C10, C100

Model

top-1

top-5

DenseNet-121 25.02 / 23.61 7.71 / 6.66

DenseNet-169 23.80 / 22.08 6.85 / 5.92

DenseNet-201 22.58 / 21.46 6.34 / 5.54

DenseNet-264 22.15 / 20.80 6.12 / 5.29

Table 3: The top-1 and top-5 error rates on the
ImageNet validation set, with single-crop / 10-
crop testing.

Figure 3: Comparison of the DenseNets and ResNets top-1 error rates (single-crop
testing) on the ImageNet validation dataset as a function of learned parameters (left)
and FLOPs during test-time (right).

and SVHN, we add a dropout layer [33] after each convolu-
tional layer (except the ﬁrst one) and set the dropout rate to
0.2. The test errors were only evaluated once for each task
and model setting.

4.3. Classiﬁcation Results on CIFAR and SVHN

We train DenseNets with different depths, L, and growth
rates, k. The main results on CIFAR and SVHN are shown
in Table 2. To highlight general trends, we mark all results
that outperform the existing state-of-the-art in boldface and
the overall best result in blue.

Accuracy. Possibly the most noticeable trend may orig-
inate from the bottom row of Table 2, which shows that
DenseNet-BC with L = 190 and k = 40 outperforms
the existing state-of-the-art consistently on all the CIFAR
datasets. Its error rates of 3.46% on C10+ and 17.18% on
C100+ are signiﬁcantly lower than the error rates achieved
by wide ResNet architecture [42]. Our best results on
C10 and C100 (without data augmentation) are even more
encouraging: both are close to 30% lower than Fractal-
Net with drop-path regularization [17]. On SVHN, with
dropout, the DenseNet with L = 100 and k = 24 also
surpasses the current best result achieved by wide ResNet.
However, the 250-layer DenseNet-BC doesn’t further im-
prove the performance over its shorter counterpart. This
may be explained by that SVHN is a relatively easy task,
and extremely deep models may overﬁt to the training set.

Capacity. Without compression or bottleneck layers,
there is a general trend that DenseNets perform better as
L and k increase. We attribute this primarily to the corre-
sponding growth in model capacity. This is best demon-
strated by the column of C10+ and C100+. On C10+, the
error drops from 5.24% to 4.10% and ﬁnally to 3.74% as
the number of parameters increases from 1.0M, over 7.0M
to 27.2M. On C100+, we observe a similar trend. This sug-
gests that DenseNets can utilize the increased representa-
tional power of bigger and deeper models. It also indicates
that they do not suffer from overﬁtting or the optimization
difﬁculties of residual networks [11].

Parameter Efﬁciency. The results in Table 2 indicate that
DenseNets utilize parameters more efﬁciently than alterna-
tive architectures (in particular, ResNets). The DenseNet-
BC with bottleneck structure and dimension reduction at
transition layers is particularly parameter-efﬁcient. For ex-
ample, our 250-layer model only has 15.3M parameters, but
it consistently outperforms other models such as FractalNet
and Wide ResNets that have more than 30M parameters. We
also highlight that DenseNet-BC with L = 100 and k = 12
achieves comparable performance (e.g., 4.51% vs 4.62% er-
ror on C10+, 22.27% vs 22.71% error on C100+) as the
1001-layer pre-activation ResNet using 90% fewer parame-
ters. Figure 4 (right panel) shows the training loss and test
errors of these two networks on C10+. The 1001-layer deep
ResNet converges to a lower training loss value but a similar
test error. We analyze this effect in more detail below.

Overﬁtting. One positive side-effect of the more efﬁcient
use of parameters is a tendency of DenseNets to be less
prone to overﬁtting. We observe that on the datasets without
data augmentation, the improvements of DenseNet architec-
tures over prior work are particularly pronounced. On C10,
the improvement denotes a 29% relative reduction in error
from 7.33% to 5.19%. On C100, the reduction is about 30%
from 28.20% to 19.64%. In our experiments, we observed
potential overﬁtting in a single setting: on C10, a 4× growth
of parameters produced by increasing k = 12 to k = 24 lead
to a modest increase in error from 5.77% to 5.83%. The
DenseNet-BC bottleneck and compression layers appear to
be an effective way to counter this trend.

4.4. Classiﬁcation Results on ImageNet

We evaluate DenseNet-BC with different depths and
growth rates on the ImageNet classiﬁcation task, and com-
pare it with state-of-the-art ResNet architectures. To en-
sure a fair comparison between the two architectures, we
eliminate all other factors such as differences in data pre-
processing and optimization settings by adopting the pub-
licly available Torch implementation for ResNet by [8]1.

1https://github.com/facebook/fb.resnet.torch

012345678x 10721.522.523.524.525.526.527.5#parametersvalidation error (%)ResNet−34ResNet−101ResNet−152DenseNet−121 DenseNet−169DenseNet−201DenseNet−264ResNetsDenseNets−BC0.50.7511.251.51.7522.252.5x 101021.522.523.524.525.526.527.5#flopsvalidation error (%)ResNet−34ResNet−101ResNet−152DenseNet−121 DenseNet−169DenseNet−201DenseNet−264ResNetsDenseNets−BCResNet−50ResNet−50Figure 4: Left: Comparison of the parameter efﬁciency on C10+ between DenseNet variations. Middle: Comparison of the parameter
efﬁciency between DenseNet-BC and (pre-activation) ResNets. DenseNet-BC requires about 1/3 of the parameters as ResNet to achieve
comparable accuracy. Right: Training and testing curves of the 1001-layer pre-activation ResNet [12] with more than 10M parameters and
a 100-layer DenseNet with only 0.8M parameters.

We simply replace the ResNet model with the DenseNet-
BC network, and keep all the experiment settings exactly
the same as those used for ResNet.

We report the single-crop and 10-crop validation errors
of DenseNets on ImageNet in Table 3. Figure 3 shows
the single-crop top-1 validation errors of DenseNets and
ResNets as a function of the number of parameters (left) and
FLOPs (right). The results presented in the ﬁgure reveal that
DenseNets perform on par with the state-of-the-art ResNets,
whilst requiring signiﬁcantly fewer parameters and compu-
tation to achieve comparable performance. For example, a
DenseNet-201 with 20M parameters model yields similar
validation error as a 101-layer ResNet with more than 40M
parameters. Similar trends can be observed from the right
panel, which plots the validation error as a function of the
number of FLOPs: a DenseNet that requires as much com-
putation as a ResNet-50 performs on par with a ResNet-101,
which requires twice as much computation.

It is worth noting that our experimental setup implies
that we use hyperparameter settings that are optimized for
ResNets but not for DenseNets. It is conceivable that more
extensive hyper-parameter searches may further improve
the performance of DenseNet on ImageNet.

5. Discussion

Superﬁcially, DenseNets are quite similar to ResNets:
Eq. (2) differs from Eq. (1) only in that the inputs to H(cid:96)(·)
are concatenated instead of summed. However, the implica-
tions of this seemingly small modiﬁcation lead to substan-
tially different behaviors of the two network architectures.

Model compactness. As a direct consequence of the in-
put concatenation, the feature-maps learned by any of the
DenseNet layers can be accessed by all subsequent layers.
This encourages feature reuse throughout the network, and
leads to more compact models.

The left two plots in Figure 4 show the result of an
experiment that aims to compare the parameter efﬁciency
of all variants of DenseNets (left) and also a comparable

ResNet architecture (middle). We train multiple small net-
works with varying depths on C10+ and plot their test ac-
curacies as a function of network parameters.
In com-
parison with other popular network architectures, such as
AlexNet [16] or VGG-net [29], ResNets with pre-activation
use fewer parameters while typically achieving better re-
sults [12]. Hence, we compare DenseNet (k = 12) against
this architecture. The training setting for DenseNet is kept
the same as in the previous section.

The graph shows that DenseNet-BC is consistently the
most parameter efﬁcient variant of DenseNet. Further, to
achieve the same level of accuracy, DenseNet-BC only re-
quires around 1/3 of the parameters of ResNets (middle
plot). This result is in line with the results on ImageNet
we presented in Figure 3. The right plot in Figure 4 shows
that a DenseNet-BC with only 0.8M trainable parameters
is able to achieve comparable accuracy as the 1001-layer
(pre-activation) ResNet [12] with 10.2M parameters.

Implicit Deep Supervision. One explanation for the im-
proved accuracy of dense convolutional networks may be
that individual layers receive additional supervision from
the loss function through the shorter connections. One can
interpret DenseNets to perform a kind of “deep supervi-
sion”. The beneﬁts of deep supervision have previously
been shown in deeply-supervised nets (DSN; [20]), which
have classiﬁers attached to every hidden layer, enforcing the
intermediate layers to learn discriminative features.

DenseNets perform a similar deep supervision in an im-
plicit fashion: a single classiﬁer on top of the network pro-
vides direct supervision to all layers through at most two or
three transition layers. However, the loss function and gra-
dient of DenseNets are substantially less complicated, as the
same loss function is shared between all layers.

Stochastic vs. deterministic connection. There is an
interesting connection between dense convolutional net-
works and stochastic depth regularization of residual net-
works [13]. In stochastic depth, layers in residual networks
are randomly dropped, which creates direct connections be-

012345678#parameters×10546810121416testerror(%)DenseNet DenseNet-CDenseNet-B DenseNet-BC012345678#parameters⇥10546810121416testerror(%)ResNetDenseNet-BC3x fewer parameters050100150200250300epoch46810121416testerror(%)Testerror:ResNet-1001(10.2M)Testerror:DenseNet-BC-100(0.8M)Trainingloss:ResNet-1001(10.2M)Trainingloss:DenseNet-BC-100(0.8M)10−310−210−1100traininglosstween the surrounding layers. As the pooling layers are
never dropped, the network results in a similar connectiv-
ity pattern as DenseNet:
there is a small probability for
any two layers, between the same pooling layers, to be di-
rectly connected—if all intermediate layers are randomly
dropped. Although the methods are ultimately quite dif-
ferent, the DenseNet interpretation of stochastic depth may
provide insights into the success of this regularizer.

Feature Reuse. By design, DenseNets allow layers ac-
cess to feature-maps from all of its preceding layers (al-
though sometimes through transition layers). We conduct
an experiment to investigate if a trained network takes ad-
vantage of this opportunity. We ﬁrst train a DenseNet on
C10+ with L = 40 and k = 12. For each convolutional
layer (cid:96) within a block, we compute the average (absolute)
weight assigned to connections with layer s. Figure 5 shows
a heat-map for all three dense blocks. The average absolute
weight serves as a surrogate for the dependency of a convo-
lutional layer on its preceding layers. A red dot in position
((cid:96), s) indicates that the layer (cid:96) makes, on average, strong use
of feature-maps produced s-layers before. Several observa-
tions can be made from the plot:

1. All layers spread their weights over many inputs within
the same block. This indicates that features extracted
by very early layers are, indeed, directly used by deep
layers throughout the same dense block.

2. The weights of the transition layers also spread their
weight across all layers within the preceding dense
block, indicating information ﬂow from the ﬁrst to the
last layers of the DenseNet through few indirections.
3. The layers within the second and third dense block
consistently assign the least weight to the outputs of
the transition layer (the top row of the triangles), in-
dicating that the transition layer outputs many redun-
dant features (with low weight on average). This is in
keeping with the strong results of DenseNet-BC where
exactly these outputs are compressed.

4. Although the ﬁnal classiﬁcation layer, shown on the
very right, also uses weights across the entire dense
block, there seems to be a concentration towards ﬁnal
feature-maps, suggesting that there may be some more
high-level features produced late in the network.

6. Conclusion

We proposed a new convolutional network architec-
ture, which we refer to as Dense Convolutional Network
(DenseNet). It introduces direct connections between any
two layers with the same feature-map size. We showed that
DenseNets scale naturally to hundreds of layers, while ex-
In our experiments,
hibiting no optimization difﬁculties.

Figure 5: The average absolute ﬁlter weights of convolutional lay-
ers in a trained DenseNet. The color of pixel (s, (cid:96)) encodes the av-
erage L1 norm (normalized by number of input feature-maps) of
the weights connecting convolutional layer s to (cid:96) within a dense
block. Three columns highlighted by black rectangles correspond
to two transition layers and the classiﬁcation layer. The ﬁrst row
encodes weights connected to the input layer of the dense block.

DenseNets tend to yield consistent improvement in accu-
racy with growing number of parameters, without any signs
of performance degradation or overﬁtting. Under multi-
ple settings, it achieved state-of-the-art results across sev-
eral highly competitive datasets. Moreover, DenseNets
require substantially fewer parameters and less computa-
tion to achieve state-of-the-art performances. Because we
adopted hyperparameter settings optimized for residual net-
works in our study, we believe that further gains in accuracy
of DenseNets may be obtained by more detailed tuning of
hyperparameters and learning rate schedules.

Whilst following a simple connectivity rule, DenseNets
naturally integrate the properties of identity mappings, deep
supervision, and diversiﬁed depth. They allow feature reuse
throughout the networks and can consequently learn more
compact and, according to our experiments, more accurate
models. Because of their compact internal representations
and reduced feature redundancy, DenseNets may be good
feature extractors for various computer vision tasks that
build on convolutional features, e.g.,
[4, 5]. We plan to
study such feature transfer with DenseNets in future work.

Acknowledgements. The authors are supported in part by
the NSF III-1618134, III-1526012, IIS-1149882, the Of-
ﬁce of Naval Research Grant N00014-17-1-2175 and the
Bill and Melinda Gates foundation. GH is supported by
the International Postdoctoral Exchange Fellowship Pro-
gram of China Postdoctoral Council (No.20150015). ZL
is supported by the National Basic Research Program of
China Grants 2011CBA00300, 2011CBA00301, the NSFC
61361136003. We also thank Daniel Sedra, Geoff Pleiss
and Yu Sun for many insightful discussions.

References

[1] C. Cortes, X. Gonzalvo, V. Kuznetsov, M. Mohri, and
S. Yang. Adanet: Adaptive structural learning of artiﬁcial
neural networks. arXiv preprint arXiv:1607.01097, 2016. 2

Dense Block 1Source layer (s)Dense Block 291Dense Block 3Target layer ()  00.10.20.30.40.50.60.70.80.9  1Transition layer 1 Transition layer 2 Classification layer135724681012Target layer ()Target layer ()135791113579112468101224681012[2] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-
Fei. Imagenet: A large-scale hierarchical image database. In
CVPR, 2009. 5

[3] S. E. Fahlman and C. Lebiere. The cascade-correlation learn-

ing architecture. In NIPS, 1989. 2

[4] J. R. Gardner, M. J. Kusner, Y. Li, P. Upchurch, K. Q.
Weinberger, and J. E. Hopcroft. Deep manifold traversal:
Changing labels with convolutional features. arXiv preprint
arXiv:1511.06421, 2015. 8

[5] L. Gatys, A. Ecker, and M. Bethge. A neural algorithm of

artistic style. Nature Communications, 2015. 8

[6] X. Glorot, A. Bordes, and Y. Bengio. Deep sparse rectiﬁer

neural networks. In AISTATS, 2011. 3

[7] I. Goodfellow, D. Warde-Farley, M. Mirza, A. Courville, and

Y. Bengio. Maxout networks. In ICML, 2013. 5

[8] S. Gross and M. Wilber. Training and investigating residual

nets, 2016. 5, 7

[9] B. Hariharan, P. Arbeláez, R. Girshick, and J. Malik. Hyper-
columns for object segmentation and ﬁne-grained localiza-
tion. In CVPR, 2015. 2

[10] K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
rectiﬁers: Surpassing human-level performance on imagenet
classiﬁcation. In ICCV, 2015. 5

[11] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning
for image recognition. In CVPR, 2016. 1, 2, 3, 4, 5, 6
[12] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in

deep residual networks. In ECCV, 2016. 2, 3, 5, 7

[13] G. Huang, Y. Sun, Z. Liu, D. Sedra, and K. Q. Weinberger.
Deep networks with stochastic depth. In ECCV, 2016. 1, 2,
5, 8

[14] S. Ioffe and C. Szegedy. Batch normalization: Accelerating
deep network training by reducing internal covariate shift. In
ICML, 2015. 3

[15] A. Krizhevsky and G. Hinton. Learning multiple layers of

features from tiny images. Tech Report, 2009. 5
[16] A. Krizhevsky, I. Sutskever, and G. E. Hinton.

classiﬁcation with deep convolutional neural networks.
NIPS, 2012. 3, 7

Imagenet
In

[17] G. Larsson, M. Maire, and G. Shakhnarovich. Fractalnet:
Ultra-deep neural networks without residuals. arXiv preprint
arXiv:1605.07648, 2016. 1, 3, 5, 6

[18] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E.
Howard, W. Hubbard, and L. D. Jackel. Backpropagation
applied to handwritten zip code recognition. Neural compu-
tation, 1(4):541–551, 1989. 1

[19] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-
based learning applied to document recognition. Proceed-
ings of the IEEE, 86(11):2278–2324, 1998. 1, 3

[20] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu. Deeply-

supervised nets. In AISTATS, 2015. 2, 3, 5, 7

[21] Q. Liao and T. Poggio. Bridging the gaps between residual
learning, recurrent neural networks and visual cortex. arXiv
preprint arXiv:1604.03640, 2016. 2

[22] M. Lin, Q. Chen, and S. Yan. Network in network. In ICLR,

2014. 3, 5

[23] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional
networks for semantic segmentation. In CVPR, 2015. 2

[24] Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, and A. Y.
Ng. Reading digits in natural images with unsupervised fea-
ture learning, 2011. In NIPS Workshop, 2011. 5

[25] M. Pezeshki, L. Fan, P. Brakel, A. Courville, and Y. Bengio.
In ICML,

Deconstructing the ladder network architecture.
2016. 3

[26] G. Pleiss, D. Chen, G. Huang, T. Li, L. van der Maaten,
and K. Q. Weinberger. Memory-efﬁcient implementation of
densenets. arXiv preprint arXiv:1707.06990, 2017. 5
[27] A. Rasmus, M. Berglund, M. Honkala, H. Valpola, and
T. Raiko. Semi-supervised learning with ladder networks.
In NIPS, 2015. 3

[28] A. Romero, N. Ballas, S. E. Kahou, A. Chassang, C. Gatta,
and Y. Bengio. Fitnets: Hints for thin deep nets. In ICLR,
2015. 5

[29] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh,
S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein,
et al.
Imagenet large scale visual recognition challenge.
IJCV. 1, 7

[30] P. Sermanet, S. Chintala, and Y. LeCun. Convolutional neu-
ral networks applied to house numbers digit classiﬁcation. In
ICPR, pages 3288–3291. IEEE, 2012. 5

[31] P. Sermanet, K. Kavukcuoglu, S. Chintala, and Y. LeCun.
Pedestrian detection with unsupervised multi-stage feature
learning. In CVPR, 2013. 2

[32] J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Ried-
miller. Striving for simplicity: The all convolutional net.
arXiv preprint arXiv:1412.6806, 2014. 5

[33] N. Srivastava, G. E. Hinton, A. Krizhevsky, I. Sutskever, and
R. Salakhutdinov. Dropout: a simple way to prevent neural
networks from overﬁtting. JMLR, 2014. 6

[34] R. K. Srivastava, K. Greff, and J. Schmidhuber. Training

very deep networks. In NIPS, 2015. 1, 2, 5

[35] I. Sutskever, J. Martens, G. Dahl, and G. Hinton. On the
importance of initialization and momentum in deep learning.
In ICML, 2013. 5

[36] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed,
D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich.
Going deeper with convolutions. In CVPR, 2015. 2, 3
[37] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna.
Rethinking the inception architecture for computer vision. In
CVPR, 2016. 2, 3, 4
[38] S. Targ, D. Almeida,

in
resnet: Generalizing residual architectures. arXiv preprint
arXiv:1603.08029, 2016. 2

and K. Lyman.

Resnet

[39] J. Wang, Z. Wei, T. Zhang, and W. Zeng. Deeply-fused nets.

arXiv preprint arXiv:1605.07716, 2016. 3

[40] B. M. Wilamowski and H. Yu. Neural network learning
without backpropagation. IEEE Transactions on Neural Net-
works, 21(11):1793–1803, 2010. 2

[41] S. Yang and D. Ramanan. Multi-scale recognition with dag-

cnns. In ICCV, 2015. 2

[42] S. Zagoruyko and N. Komodakis. Wide residual networks.

arXiv preprint arXiv:1605.07146, 2016. 3, 5, 6

[43] Y. Zhang, K. Lee, and H. Lee. Augmenting supervised neural
networks with unsupervised objectives for large-scale image
classiﬁcation. In ICML, 2016. 3


