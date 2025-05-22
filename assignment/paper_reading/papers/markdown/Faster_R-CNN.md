6
1
0
2

n
a
J

6

]

V
C
.
s
c
[

3
v
7
9
4
1
0
.
6
0
5
1
:
v
i
X
r
a

1

Faster R-CNN: Towards Real-Time Object
Detection with Region Proposal Networks

Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun

Abstract—State-of-the-art object detection networks depend on region proposal algorithms to hypothesize object locations.
Advances like SPPnet [1] and Fast R-CNN [2] have reduced the running time of these detection networks, exposing region
proposal computation as a bottleneck. In this work, we introduce a Region Proposal Network (RPN) that shares full-image
convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional
network that simultaneously predicts object bounds and objectness scores at each position. The RPN is trained end-to-end to
generate high-quality region proposals, which are used by Fast R-CNN for detection. We further merge RPN and Fast R-CNN
into a single network by sharing their convolutional features—using the recently popular terminology of neural networks with
“attention” mechanisms, the RPN component tells the uniﬁed network where to look. For the very deep VGG-16 model [3],
our detection system has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection
accuracy on PASCAL VOC 2007, 2012, and MS COCO datasets with only 300 proposals per image. In ILSVRC and COCO
2015 competitions, Faster R-CNN and RPN are the foundations of the 1st-place winning entries in several tracks. Code has been
made publicly available.

Index Terms—Object Detection, Region Proposal, Convolutional Neural Network.

(cid:70)

1 INTRODUCTION

Recent advances in object detection are driven by
the success of region proposal methods (e.g., [4])
and region-based convolutional neural networks (R-
CNNs) [5]. Although region-based CNNs were com-
putationally expensive as originally developed in [5],
their cost has been drastically reduced thanks to shar-
ing convolutions across proposals [1], [2]. The latest
incarnation, Fast R-CNN [2], achieves near real-time
rates using very deep networks [3], when ignoring the
time spent on region proposals. Now, proposals are the
test-time computational bottleneck in state-of-the-art
detection systems.

Region proposal methods typically rely on inex-
pensive features and economical inference schemes.
Selective Search [4], one of the most popular meth-
ods, greedily merges superpixels based on engineered
low-level features. Yet when compared to efﬁcient
detection networks [2], Selective Search is an order of
magnitude slower, at 2 seconds per image in a CPU
implementation. EdgeBoxes [6] currently provides the
best tradeoff between proposal quality and speed,
at 0.2 seconds per image. Nevertheless, the region
proposal step still consumes as much running time
as the detection network.

• S. Ren is with University of Science and Technology of China, Hefei,
China. This work was done when S. Ren was an intern at Microsoft
Research. Email: sqren@mail.ustc.edu.cn

• K. He and J. Sun are with Visual Computing Group, Microsoft

Research. E-mail: {kahe,jiansun}@microsoft.com

• R. Girshick is with Facebook AI Research. The majority of this work
was done when R. Girshick was with Microsoft Research. E-mail:
rbg@fb.com

One may note that fast region-based CNNs take
advantage of GPUs, while the region proposal meth-
ods used in research are implemented on the CPU,
making such runtime comparisons inequitable. An ob-
vious way to accelerate proposal computation is to re-
implement it for the GPU. This may be an effective en-
gineering solution, but re-implementation ignores the
down-stream detection network and therefore misses
important opportunities for sharing computation.

In this paper, we show that an algorithmic change—
computing proposals with a deep convolutional neu-
ral network—leads to an elegant and effective solution
where proposal computation is nearly cost-free given
the detection network’s computation. To this end, we
introduce novel Region Proposal Networks (RPNs) that
share convolutional layers with state-of-the-art object
detection networks [1], [2]. By sharing convolutions at
test-time, the marginal cost for computing proposals
is small (e.g., 10ms per image).

Our observation is that the convolutional feature
maps used by region-based detectors, like Fast R-
CNN, can also be used for generating region pro-
posals. On top of these convolutional features, we
construct an RPN by adding a few additional con-
volutional layers that simultaneously regress region
bounds and objectness scores at each location on a
regular grid. The RPN is thus a kind of fully convo-
lutional network (FCN) [7] and can be trained end-to-
end speciﬁcally for the task for generating detection
proposals.

RPNs are designed to efﬁciently predict region pro-
posals with a wide range of scales and aspect ratios. In
contrast to prevalent methods [8], [9], [1], [2] that use

2

Figure 1: Different schemes for addressing multiple scales and sizes. (a) Pyramids of images and feature maps
are built, and the classiﬁer is run at all scales. (b) Pyramids of ﬁlters with multiple scales/sizes are run on
the feature map. (c) We use pyramids of reference boxes in the regression functions.

pyramids of images (Figure 1, a) or pyramids of ﬁlters
(Figure 1, b), we introduce novel “anchor” boxes
that serve as references at multiple scales and aspect
ratios. Our scheme can be thought of as a pyramid
of regression references (Figure 1, c), which avoids
enumerating images or ﬁlters of multiple scales or
aspect ratios. This model performs well when trained
and tested using single-scale images and thus beneﬁts
running speed.

To unify RPNs with Fast R-CNN [2] object detec-
tion networks, we propose a training scheme that
alternates between ﬁne-tuning for the region proposal
task and then ﬁne-tuning for object detection, while
keeping the proposals ﬁxed. This scheme converges
quickly and produces a uniﬁed network with convo-
lutional features that are shared between both tasks.1
We comprehensively evaluate our method on the
PASCAL VOC detection benchmarks [11] where RPNs
with Fast R-CNNs produce detection accuracy bet-
ter than the strong baseline of Selective Search with
Fast R-CNNs. Meanwhile, our method waives nearly
all computational burdens of Selective Search at
test-time—the effective running time for proposals
is just 10 milliseconds. Using the expensive very
deep models of [3], our detection method still has
a frame rate of 5fps (including all steps) on a GPU,
and thus is a practical object detection system in
terms of both speed and accuracy. We also report
results on the MS COCO dataset [12] and investi-
gate the improvements on PASCAL VOC using the
COCO data. Code has been made publicly available
at https://github.com/shaoqingren/faster_
rcnn (in MATLAB) and https://github.com/
rbgirshick/py-faster-rcnn (in Python).

A preliminary version of this manuscript was pub-
lished previously [10]. Since then, the frameworks of
RPN and Faster R-CNN have been adopted and gen-
eralized to other methods, such as 3D object detection
[13], part-based detection [14], instance segmentation
[15], and image captioning [16]. Our fast and effective
object detection system has also been built in com-

1. Since the publication of the conference version of this paper
[10], we have also found that RPNs can be trained jointly with Fast
R-CNN networks leading to less training time.

mercial systems such as at Pinterests [17], with user
engagement improvements reported.

In ILSVRC and COCO 2015 competitions, Faster
R-CNN and RPN are the basis of several 1st-place
entries [18] in the tracks of ImageNet detection, Ima-
geNet localization, COCO detection, and COCO seg-
mentation. RPNs completely learn to propose regions
from data, and thus can easily beneﬁt from deeper
and more expressive features (such as the 101-layer
residual nets adopted in [18]). Faster R-CNN and RPN
are also used by several other leading entries in these
competitions2. These results suggest that our method
is not only a cost-efﬁcient solution for practical usage,
but also an effective way of improving object detec-
tion accuracy.

2 RELATED WORK

Object Proposals. There is a large literature on object
proposal methods. Comprehensive surveys and com-
parisons of object proposal methods can be found in
[19], [20], [21]. Widely used object proposal methods
include those based on grouping super-pixels (e.g.,
Selective Search [4], CPMC [22], MCG [23]) and those
based on sliding windows (e.g., objectness in windows
[24], EdgeBoxes [6]). Object proposal methods were
adopted as external modules independent of the de-
tectors (e.g., Selective Search [4] object detectors, R-
CNN [5], and Fast R-CNN [2]).

Deep Networks for Object Detection. The R-CNN
method [5] trains CNNs end-to-end to classify the
proposal regions into object categories or background.
R-CNN mainly plays as a classiﬁer, and it does not
predict object bounds (except for reﬁning by bounding
box regression). Its accuracy depends on the perfor-
mance of the region proposal module (see compar-
isons in [20]). Several papers have proposed ways of
using deep networks for predicting object bounding
boxes [25], [9], [26], [27]. In the OverFeat method [9],
a fully-connected layer is trained to predict the box
coordinates for the localization task that assumes a
single object. The fully-connected layer is then turned

2. http://image-net.org/challenges/LSVRC/2015/results

multiple scaled imagesmultiple filter sizesmultiple references(a)(b)(c)imagefeature mapimagefeature mapimagefeature map3

single, uniﬁed network for object detection (Figure 2).
Using the recently popular terminology of neural
networks with ‘attention’ [31] mechanisms, the RPN
module tells the Fast R-CNN module where to look.
In Section 3.1 we introduce the designs and properties
of the network for region proposal. In Section 3.2 we
develop algorithms for training both modules with
features shared.

3.1 Region Proposal Networks

A Region Proposal Network (RPN) takes an image
(of any size) as input and outputs a set of rectangular
object proposals, each with an objectness score.3 We
model this process with a fully convolutional network
[7], which we describe in this section. Because our ulti-
mate goal is to share computation with a Fast R-CNN
object detection network [2], we assume that both nets
share a common set of convolutional layers. In our ex-
periments, we investigate the Zeiler and Fergus model
[32] (ZF), which has 5 shareable convolutional layers
and the Simonyan and Zisserman model [3] (VGG-16),
which has 13 shareable convolutional layers.

To generate region proposals, we slide a small
network over the convolutional feature map output
by the last shared convolutional
layer. This small
network takes as input an n × n spatial window of
the input convolutional feature map. Each sliding
window is mapped to a lower-dimensional feature
(256-d for ZF and 512-d for VGG, with ReLU [33]
following). This feature is fed into two sibling fully-
connected layers—a box-regression layer (reg) and a
box-classiﬁcation layer (cls). We use n = 3 in this
paper, noting that the effective receptive ﬁeld on the
input image is large (171 and 228 pixels for ZF and
VGG, respectively). This mini-network is illustrated
at a single position in Figure 3 (left). Note that be-
cause the mini-network operates in a sliding-window
fashion, the fully-connected layers are shared across
all spatial locations. This architecture is naturally im-
plemented with an n × n convolutional layer followed
by two sibling 1 × 1 convolutional layers (for reg and
cls, respectively).

3.1.1 Anchors
At each sliding-window location, we simultaneously
predict multiple region proposals, where the number
of maximum possible proposals for each location is
denoted as k. So the reg layer has 4k outputs encoding
the coordinates of k boxes, and the cls layer outputs
2k scores that estimate probability of object or not
object for each proposal4. The k proposals are param-
eterized relative to k reference boxes, which we call

3. “Region” is a generic term and in this paper we only consider
rectangular regions, as is common for many methods (e.g., [27], [4],
[6]). “Objectness” measures membership to a set of object classes
vs. background.

4. For simplicity we implement the cls layer as a two-class
softmax layer. Alternatively, one may use logistic regression to
produce k scores.

Figure 2: Faster R-CNN is a single, uniﬁed network
for object detection. The RPN module serves as the
‘attention’ of this uniﬁed network.

into a convolutional layer for detecting multiple class-
speciﬁc objects. The MultiBox methods [26], [27] gen-
erate region proposals from a network whose last
fully-connected layer simultaneously predicts mul-
tiple class-agnostic boxes, generalizing the “single-
box” fashion of OverFeat. These class-agnostic boxes
are used as proposals for R-CNN [5]. The MultiBox
proposal network is applied on a single image crop or
multiple large image crops (e.g., 224×224), in contrast
to our fully convolutional scheme. MultiBox does not
share features between the proposal and detection
networks. We discuss OverFeat and MultiBox in more
depth later in context with our method. Concurrent
with our work, the DeepMask method [28] is devel-
oped for learning segmentation proposals.

Shared computation of convolutions [9], [1], [29],
[7], [2] has been attracting increasing attention for ef-
ﬁcient, yet accurate, visual recognition. The OverFeat
paper [9] computes convolutional features from an
image pyramid for classiﬁcation, localization, and de-
tection. Adaptively-sized pooling (SPP) [1] on shared
convolutional feature maps is developed for efﬁcient
region-based object detection [1], [30] and semantic
segmentation [29]. Fast R-CNN [2] enables end-to-end
detector training on shared convolutional features and
shows compelling accuracy and speed.

3 FASTER R-CNN
Our object detection system, called Faster R-CNN, is
composed of two modules. The ﬁrst module is a deep
fully convolutional network that proposes regions,
and the second module is the Fast R-CNN detector [2]
that uses the proposed regions. The entire system is a

imageconv layersfeature mapsRegion Proposal NetworkproposalsclassifierRoIpooling4

Figure 3: Left: Region Proposal Network (RPN). Right: Example detections using RPN proposals on PASCAL
VOC 2007 test. Our method detects objects in a wide range of scales and aspect ratios.

anchors. An anchor is centered at the sliding window
in question, and is associated with a scale and aspect
ratio (Figure 3, left). By default we use 3 scales and
3 aspect ratios, yielding k = 9 anchors at each sliding
position. For a convolutional feature map of a size
W × H (typically ∼2,400), there are W Hk anchors in
total.

Translation-Invariant Anchors

An important property of our approach is that it
is translation invariant, both in terms of the anchors
and the functions that compute proposals relative to
the anchors. If one translates an object in an image,
the proposal should translate and the same function
should be able to predict the proposal in either lo-
cation. This translation-invariant property is guaran-
teed by our method5. As a comparison, the MultiBox
method [27] uses k-means to generate 800 anchors,
which are not translation invariant. So MultiBox does
not guarantee that the same proposal is generated if
an object is translated.

The translation-invariant property also reduces the
model size. MultiBox has a (4 + 1) × 800-dimensional
fully-connected output layer, whereas our method has
a (4 + 2) × 9-dimensional convolutional output layer
in the case of k = 9 anchors. As a result, our output
layer has 2.8 × 104 parameters (512 × (4 + 2) × 9
for VGG-16), two orders of magnitude fewer than
MultiBox’s output layer that has 6.1 × 106 parameters
(1536 × (4 + 1) × 800 for GoogleNet [34] in MultiBox
[27]). If considering the feature projection layers, our
proposal layers still have an order of magnitude fewer
parameters than MultiBox6. We expect our method
to have less risk of overﬁtting on small datasets, like
PASCAL VOC.

5. As is the case of FCNs [7], our network is translation invariant

up to the network’s total stride.

6. Considering the feature projection layers, our proposal layers’
parameter count is 3 × 3 × 512 × 512 + 512 × 6 × 9 = 2.4 × 106;
MultiBox’s proposal layers’ parameter count is 7 × 7 × (64 + 96 +
64 + 64) × 1536 + 1536 × 5 × 800 = 27 × 106.

Multi-Scale Anchors as Regression References

Our design of anchors presents a novel scheme
for addressing multiple scales (and aspect ratios). As
shown in Figure 1, there have been two popular ways
for multi-scale predictions. The ﬁrst way is based on
image/feature pyramids, e.g., in DPM [8] and CNN-
based methods [9], [1], [2]. The images are resized at
multiple scales, and feature maps (HOG [8] or deep
convolutional features [9], [1], [2]) are computed for
each scale (Figure 1(a)). This way is often useful but
is time-consuming. The second way is to use sliding
windows of multiple scales (and/or aspect ratios) on
the feature maps. For example, in DPM [8], models
of different aspect ratios are trained separately using
different ﬁlter sizes (such as 5×7 and 7×5). If this way
is used to address multiple scales, it can be thought
of as a “pyramid of ﬁlters” (Figure 1(b)). The second
way is usually adopted jointly with the ﬁrst way [8].
As a comparison, our anchor-based method is built
on a pyramid of anchors, which is more cost-efﬁcient.
Our method classiﬁes and regresses bounding boxes
with reference to anchor boxes of multiple scales and
aspect ratios. It only relies on images and feature
maps of a single scale, and uses ﬁlters (sliding win-
dows on the feature map) of a single size. We show by
experiments the effects of this scheme for addressing
multiple scales and sizes (Table 8).

Because of this multi-scale design based on anchors,
we can simply use the convolutional features com-
puted on a single-scale image, as is also done by
the Fast R-CNN detector [2]. The design of multi-
scale anchors is a key component for sharing features
without extra cost for addressing scales.

3.1.2 Loss Function
For training RPNs, we assign a binary class label
(of being an object or not) to each anchor. We as-
sign a positive label to two kinds of anchors: (i) the
anchor/anchors with the highest Intersection-over-
Union (IoU) overlap with a ground-truth box, or (ii) an
anchor that has an IoU overlap higher than 0.7 with

car:1.000dog:0.997person:0.992person:0.979horse:0.993convfeaturemapintermediatelayer256-d2kscores4kcoordinatesslidingwindowreglayerclslayerkanchorboxesbus:0.996person:0.736boat:0.970person:0.989person:0.983person:0.983person:0.925cat:0.982dog:0.994any ground-truth box. Note that a single ground-truth
box may assign positive labels to multiple anchors.
Usually the second condition is sufﬁcient to determine
the positive samples; but we still adopt the ﬁrst
condition for the reason that in some rare cases the
second condition may ﬁnd no positive sample. We
assign a negative label to a non-positive anchor if its
IoU ratio is lower than 0.3 for all ground-truth boxes.
Anchors that are neither positive nor negative do not
contribute to the training objective.

With these deﬁnitions, we minimize an objective
function following the multi-task loss in Fast R-CNN
[2]. Our loss function for an image is deﬁned as:

L({pi}, {ti}) =

1
Ncls
1
Nreg

i

(cid:88)

Lcls (pi, p∗
i )

(cid:88)

i

i Lreg (ti, t∗
p∗

i ).

+λ

(1)

Here, i is the index of an anchor in a mini-batch and
pi is the predicted probability of anchor i being an
object. The ground-truth label p∗
is 1 if the anchor
i
is positive, and is 0 if the anchor is negative. ti is a
vector representing the 4 parameterized coordinates
of the predicted bounding box, and t∗
is that of the
i
ground-truth box associated with a positive anchor.
The classiﬁcation loss Lcls is log loss over two classes
(object vs. not object). For the regression loss, we use
Lreg (ti, t∗
i ) where R is the robust loss
function (smooth L1) deﬁned in [2]. The term p∗
i Lreg
means the regression loss is activated only for positive
anchors (p∗
i = 0).
The outputs of the cls and reg layers consist of {pi}
and {ti} respectively.

i = 1) and is disabled otherwise (p∗

i ) = R(ti − t∗

The two terms are normalized by Ncls and Nreg
and weighted by a balancing parameter λ. In our
current implementation (as in the released code), the
cls term in Eqn.(1) is normalized by the mini-batch
size (i.e., Ncls = 256) and the reg term is normalized
by the number of anchor locations (i.e., Nreg ∼ 2, 400).
By default we set λ = 10, and thus both cls and
reg terms are roughly equally weighted. We show
by experiments that the results are insensitive to the
values of λ in a wide range (Table 9). We also note
that the normalization as above is not required and
could be simpliﬁed.

For bounding box regression, we adopt the param-

eterizations of the 4 coordinates following [5]:

tx = (x − xa)/wa,
tw = log(w/wa),
t∗
x = (x∗ − xa)/wa,
w = log(w∗/wa),
t∗

ty = (y − ya)/ha,

th = log(h/ha),

t∗
y = (y∗ − ya)/ha,
h = log(h∗/ha),
t∗

(2)

where x, y, w, and h denote the box’s center coordi-
nates and its width and height. Variables x, xa, and
x∗ are for the predicted box, anchor box, and ground-
truth box respectively (likewise for y, w, h). This can

5

be thought of as bounding-box regression from an
anchor box to a nearby ground-truth box.

Nevertheless, our method achieves bounding-box
regression by a different manner from previous RoI-
based (Region of Interest) methods [1], [2]. In [1],
[2], bounding-box regression is performed on features
pooled from arbitrarily sized RoIs, and the regression
weights are shared by all region sizes. In our formula-
tion, the features used for regression are of the same
spatial size (3 × 3) on the feature maps. To account
for varying sizes, a set of k bounding-box regressors
are learned. Each regressor is responsible for one scale
and one aspect ratio, and the k regressors do not share
weights. As such, it is still possible to predict boxes of
various sizes even though the features are of a ﬁxed
size/scale, thanks to the design of anchors.

3.1.3 Training RPNs

The RPN can be trained end-to-end by back-
propagation and stochastic gradient descent (SGD)
[35]. We follow the “image-centric” sampling strategy
from [2] to train this network. Each mini-batch arises
from a single image that contains many positive and
negative example anchors. It is possible to optimize
for the loss functions of all anchors, but this will
bias towards negative samples as they are dominate.
Instead, we randomly sample 256 anchors in an image
to compute the loss function of a mini-batch, where
the sampled positive and negative anchors have a
ratio of up to 1:1. If there are fewer than 128 positive
samples in an image, we pad the mini-batch with
negative ones.

We randomly initialize all new layers by drawing
weights from a zero-mean Gaussian distribution with
standard deviation 0.01. All other layers (i.e., the
shared convolutional layers) are initialized by pre-
training a model for ImageNet classiﬁcation [36], as
is standard practice [5]. We tune all layers of the
ZF net, and conv3 1 and up for the VGG net to
conserve memory [2]. We use a learning rate of 0.001
for 60k mini-batches, and 0.0001 for the next 20k
mini-batches on the PASCAL VOC dataset. We use a
momentum of 0.9 and a weight decay of 0.0005 [37].
Our implementation uses Caffe [38].

3.2 Sharing Features for RPN and Fast R-CNN

Thus far we have described how to train a network
for region proposal generation, without considering
the region-based object detection CNN that will utilize
these proposals. For the detection network, we adopt
Fast R-CNN [2]. Next we describe algorithms that
learn a uniﬁed network composed of RPN and Fast
R-CNN with shared convolutional layers (Figure 2).
Both RPN and Fast R-CNN, trained independently,
will modify their convolutional
layers in different
ways. We therefore need to develop a technique that
allows for sharing convolutional layers between the

Table 1: the learned average proposal size for each anchor using the ZF net (numbers for s = 600).

anchor
proposal

1282, 2:1 1282, 1:1 1282, 1:2 2562, 2:1 2562, 1:1 2562, 1:2 5122, 2:1 5122, 1:1 5122, 1:2
768×437 499×501 355×715
416×229 261×284 174×332
188×111 113×114

70×92

6

two networks, rather than learning two separate net-
works. We discuss three ways for training networks
with features shared:

(i) Alternating training. In this solution, we ﬁrst train
RPN, and use the proposals to train Fast R-CNN.
The network tuned by Fast R-CNN is then used to
initialize RPN, and this process is iterated. This is the
solution that is used in all experiments in this paper.
(ii) Approximate joint training. In this solution, the
RPN and Fast R-CNN networks are merged into one
network during training as in Figure 2. In each SGD
iteration, the forward pass generates region propos-
als which are treated just like ﬁxed, pre-computed
proposals when training a Fast R-CNN detector. The
backward propagation takes place as usual, where for
the shared layers the backward propagated signals
from both the RPN loss and the Fast R-CNN loss
are combined. This solution is easy to implement. But
this solution ignores the derivative w.r.t. the proposal
boxes’ coordinates that are also network responses,
so is approximate. In our experiments, we have em-
pirically found this solver produces close results, yet
reduces the training time by about 25-50% comparing
with alternating training. This solver is included in
our released Python code.

(iii) Non-approximate joint

training. As discussed
above, the bounding boxes predicted by RPN are
also functions of the input. The RoI pooling layer
[2] in Fast R-CNN accepts the convolutional features
and also the predicted bounding boxes as input, so
a theoretically valid backpropagation solver should
also involve gradients w.r.t. the box coordinates. These
gradients are ignored in the above approximate joint
training. In a non-approximate joint training solution,
we need an RoI pooling layer that is differentiable
w.r.t. the box coordinates. This is a nontrivial problem
and a solution can be given by an “RoI warping” layer
as developed in [15], which is beyond the scope of this
paper.

4-Step Alternating Training. In this paper, we adopt
a pragmatic 4-step training algorithm to learn shared
features via alternating optimization. In the ﬁrst step,
we train the RPN as described in Section 3.1.3. This
network is initialized with an ImageNet-pre-trained
model and ﬁne-tuned end-to-end for the region pro-
posal task. In the second step, we train a separate
detection network by Fast R-CNN using the proposals
generated by the step-1 RPN. This detection net-
work is also initialized by the ImageNet-pre-trained
model. At this point the two networks do not share
convolutional layers. In the third step, we use the
detector network to initialize RPN training, but we

ﬁx the shared convolutional layers and only ﬁne-tune
the layers unique to RPN. Now the two networks
share convolutional layers. Finally, keeping the shared
convolutional layers ﬁxed, we ﬁne-tune the unique
layers of Fast R-CNN. As such, both networks share
the same convolutional layers and form a uniﬁed
network. A similar alternating training can be run
for more iterations, but we have observed negligible
improvements.

3.3 Implementation Details

We train and test both region proposal and object
detection networks on images of a single scale [1], [2].
We re-scale the images such that their shorter side
is s = 600 pixels [2]. Multi-scale feature extraction
(using an image pyramid) may improve accuracy but
does not exhibit a good speed-accuracy trade-off [2].
On the re-scaled images, the total stride for both ZF
and VGG nets on the last convolutional layer is 16
pixels, and thus is ∼10 pixels on a typical PASCAL
image before resizing (∼500×375). Even such a large
stride provides good results, though accuracy may be
further improved with a smaller stride.

For anchors, we use 3 scales with box areas of 1282,
2562, and 5122 pixels, and 3 aspect ratios of 1:1, 1:2,
and 2:1. These hyper-parameters are not carefully cho-
sen for a particular dataset, and we provide ablation
experiments on their effects in the next section. As dis-
cussed, our solution does not need an image pyramid
or ﬁlter pyramid to predict regions of multiple scales,
saving considerable running time. Figure 3 (right)
shows the capability of our method for a wide range
of scales and aspect ratios. Table 1 shows the learned
average proposal size for each anchor using the ZF
net. We note that our algorithm allows predictions
that are larger than the underlying receptive ﬁeld.
Such predictions are not impossible—one may still
roughly infer the extent of an object if only the middle
of the object is visible.

The anchor boxes that cross image boundaries need
to be handled with care. During training, we ignore
all cross-boundary anchors so they do not contribute
to the loss. For a typical 1000 × 600 image, there
will be roughly 20000 (≈ 60 × 40 × 9) anchors in
total. With the cross-boundary anchors ignored, there
are about 6000 anchors per image for training. If the
boundary-crossing outliers are not ignored in training,
they introduce large, difﬁcult to correct error terms in
the objective, and training does not converge. During
testing, however, we still apply the fully convolutional
RPN to the entire image. This may generate cross-
boundary proposal boxes, which we clip to the image
boundary.

Table 2: Detection results on PASCAL VOC 2007 test set (trained on VOC 2007 trainval). The detectors are
Fast R-CNN with ZF, but using various proposal methods for training and testing.

train-time region proposals

test-time region proposals

method

# boxes

method

# proposals

mAP (%)

7

SS
EB
RPN+ZF, shared

2000
2000
2000

SS
EB
RPN+ZF, shared

ablation experiments follow below

RPN+ZF, unshared
SS
SS
SS
SS
SS
SS
SS
SS
SS
SS

2000
2000
2000
2000
2000
2000
2000
2000
2000
2000
2000

RPN+ZF, unshared
RPN+ZF
RPN+ZF
RPN+ZF
RPN+ZF (no NMS)
RPN+ZF (no cls)
RPN+ZF (no cls)
RPN+ZF (no cls)
RPN+ZF (no reg)
RPN+ZF (no reg)
RPN+VGG

2000
2000
300

300
100
300
1000
6000
100
300
1000
300
1000
300

58.7
58.6
59.9

58.7
55.1
56.8
56.3
55.2
44.6
51.4
55.8
52.1
51.3
59.2

Some RPN proposals highly overlap with each
other. To reduce redundancy, we adopt non-maximum
suppression (NMS) on the proposal regions based on
their cls scores. We ﬁx the IoU threshold for NMS
at 0.7, which leaves us about 2000 proposal regions
per image. As we will show, NMS does not harm the
ultimate detection accuracy, but substantially reduces
the number of proposals. After NMS, we use the
top-N ranked proposal regions for detection. In the
following, we train Fast R-CNN using 2000 RPN pro-
posals, but evaluate different numbers of proposals at
test-time.

4 EXPERIMENTS
4.1 Experiments on PASCAL VOC

We comprehensively evaluate our method on the
PASCAL VOC 2007 detection benchmark [11]. This
dataset consists of about 5k trainval images and 5k
test images over 20 object categories. We also provide
results on the PASCAL VOC 2012 benchmark for a
few models. For the ImageNet pre-trained network,
we use the “fast” version of ZF net [32] that has
5 convolutional layers and 3 fully-connected layers,
and the public VGG-16 model7 [3] that has 13 con-
volutional layers and 3 fully-connected layers. We
primarily evaluate detection mean Average Precision
(mAP), because this is the actual metric for object
detection (rather than focusing on object proposal
proxy metrics).

Table 2 (top) shows Fast R-CNN results when
trained and tested using various region proposal
methods. These results use the ZF net. For Selective
Search (SS) [4], we generate about 2000 proposals by
the “fast” mode. For EdgeBoxes (EB) [6], we generate
the proposals by the default EB setting tuned for 0.7

7. www.robots.ox.ac.uk/∼vgg/research/very deep/

IoU. SS has an mAP of 58.7% and EB has an mAP
of 58.6% under the Fast R-CNN framework. RPN
with Fast R-CNN achieves competitive results, with
an mAP of 59.9% while using up to 300 proposals8.
Using RPN yields a much faster detection system than
using either SS or EB because of shared convolutional
computations; the fewer proposals also reduce the
region-wise fully-connected layers’ cost (Table 5).

Ablation Experiments on RPN. To investigate the be-
havior of RPNs as a proposal method, we conducted
several ablation studies. First, we show the effect of
sharing convolutional layers between the RPN and
Fast R-CNN detection network. To do this, we stop
after the second step in the 4-step training process.
Using separate networks reduces the result slightly to
58.7% (RPN+ZF, unshared, Table 2). We observe that
this is because in the third step when the detector-
tuned features are used to ﬁne-tune the RPN, the
proposal quality is improved.

Next, we disentangle the RPN’s inﬂuence on train-
ing the Fast R-CNN detection network. For this pur-
pose, we train a Fast R-CNN model by using the
2000 SS proposals and ZF net. We ﬁx this detector
and evaluate the detection mAP by changing the
proposal regions used at test-time. In these ablation
experiments, the RPN does not share features with
the detector.

Replacing SS with 300 RPN proposals at test-time
leads to an mAP of 56.8%. The loss in mAP is because
of the inconsistency between the training/testing pro-
posals. This result serves as the baseline for the fol-
lowing comparisons.

Somewhat surprisingly, the RPN still leads to a
competitive result (55.1%) when using the top-ranked

8. For RPN, the number of proposals (e.g., 300) is the maximum
number for an image. RPN may produce fewer proposals after
NMS, and thus the average number of proposals is smaller.

Table 3: Detection results on PASCAL VOC 2007 test set. The detector is Fast R-CNN and VGG-16. Training
data: “07”: VOC 2007 trainval, “07+12”: union set of VOC 2007 trainval and VOC 2012 trainval. For RPN,
the train-time proposals for Fast R-CNN are 2000. †: this number was reported in [2]; using the repository
provided by this paper, this result is higher (68.1).

8

method

# proposals

data

SS
SS
RPN+VGG, unshared
RPN+VGG, shared
RPN+VGG, shared
RPN+VGG, shared

2000
2000
300
300
300
300

07
07+12
07
07
07+12
COCO+07+12

mAP (%)
66.9†
70.0
68.5
69.9
73.2
78.8

Table 4: Detection results on PASCAL VOC 2012 test set. The detector is Fast R-CNN and VGG-16. Training
data: “07”: VOC 2007 trainval, “07++12”: union set of VOC 2007 trainval+test and VOC 2012 trainval. For
RPN, the train-time proposals for Fast R-CNN are 2000. †: http://host.robots.ox.ac.uk:8080/anonymous/HZJTQA.html. ‡:
http://host.robots.ox.ac.uk:8080/anonymous/YNPLXB.html. §: http://host.robots.ox.ac.uk:8080/anonymous/XEDH10.html.
# proposals

mAP (%)

method

data

SS
SS
RPN+VGG, shared†
RPN+VGG, shared‡
RPN+VGG, shared§

2000
2000
300
300
300

12
07++12
12
07++12
COCO+07++12

65.7
68.4
67.0
70.4
75.9

Table 5: Timing (ms) on a K40 GPU, except SS proposal is evaluated in a CPU. “Region-wise” includes NMS,
pooling, fully-connected, and softmax layers. See our released code for the proﬁling of running time.

model

VGG
VGG
ZF

system

conv

proposal

region-wise

SS + Fast R-CNN
RPN + Fast R-CNN
RPN + Fast R-CNN

146
141
31

1510
10
3

174
47
25

total

1830
198
59

rate

0.5 fps
5 fps
17 fps

100 proposals at test-time, indicating that the top-
ranked RPN proposals are accurate. On the other
extreme, using the top-ranked 6000 RPN proposals
(without NMS) has a comparable mAP (55.2%), sug-
gesting NMS does not harm the detection mAP and
may reduce false alarms.

Next, we separately investigate the roles of RPN’s
cls and reg outputs by turning off either of them
at test-time. When the cls layer is removed at test-
time (thus no NMS/ranking is used), we randomly
sample N proposals from the unscored regions. The
mAP is nearly unchanged with N = 1000 (55.8%), but
degrades considerably to 44.6% when N = 100. This
shows that the cls scores account for the accuracy of
the highest ranked proposals.

On the other hand, when the reg layer is removed
at test-time (so the proposals become anchor boxes),
the mAP drops to 52.1%. This suggests that the high-
quality proposals are mainly due to the regressed box
bounds. The anchor boxes, though having multiple
scales and aspect ratios, are not sufﬁcient for accurate
detection.

We also evaluate the effects of more powerful net-
works on the proposal quality of RPN alone. We use
VGG-16 to train the RPN, and still use the above
detector of SS+ZF. The mAP improves from 56.8%

(using RPN+ZF) to 59.2% (using RPN+VGG). This is a
promising result, because it suggests that the proposal
quality of RPN+VGG is better than that of RPN+ZF.
Because proposals of RPN+ZF are competitive with
SS (both are 58.7% when consistently used for training
and testing), we may expect RPN+VGG to be better
than SS. The following experiments justify this hy-
pothesis.

Performance of VGG-16. Table 3 shows the results
of VGG-16 for both proposal and detection. Using
RPN+VGG, the result is 68.5% for unshared features,
slightly higher than the SS baseline. As shown above,
this is because the proposals generated by RPN+VGG
are more accurate than SS. Unlike SS that is pre-
deﬁned, the RPN is actively trained and beneﬁts from
better networks. For the feature-shared variant, the
result is 69.9%—better than the strong SS baseline, yet
with nearly cost-free proposals. We further train the
RPN and detection network on the union set of PAS-
CAL VOC 2007 trainval and 2012 trainval. The mAP
is 73.2%. Figure 5 shows some results on the PASCAL
VOC 2007 test set. On the PASCAL VOC 2012 test set
(Table 4), our method has an mAP of 70.4% trained
on the union set of VOC 2007 trainval+test and VOC
2012 trainval. Table 6 and Table 7 show the detailed
numbers.

Table 6: Results on PASCAL VOC 2007 test set with Fast R-CNN detectors and VGG-16. For RPN, the train-time
proposals for Fast R-CNN are 2000. RPN∗ denotes the unsharing feature version.

method

# box

SS

SS
RPN∗
RPN

RPN

RPN

2000

2000

300

300

300

300

data

07

07+12

07

07

07+12

COCO+07+12

mAP areo

bike

bird

boat

bottle

bus

car

cat

chair

cow table

dog

horse mbike person plant

sheep sofa

train

tv

66.9

70.0

68.5

69.9

73.2

78.8

74.5 78.3 69.2 53.2 36.6 77.3 78.2 82.0 40.7 72.7 67.9 79.6 79.2 73.0 69.0 30.1 65.4 70.2 75.8 65.8

77.0 78.1 69.3 59.4 38.3 81.6 78.6 86.7 42.8 78.8 68.9 84.7 82.0 76.6 69.9 31.8 70.1 74.8 80.4 70.4

74.1 77.2 67.7 53.9 51.0 75.1 79.2 78.9 50.7 78.0 61.1 79.1 81.9 72.2 75.9 37.2 71.4 62.5 77.4 66.4

70.0 80.6 70.1 57.3 49.9 78.2 80.4 82.0 52.2 75.3 67.2 80.3 79.8 75.0 76.3 39.1 68.3 67.3 81.1 67.6

76.5 79.0 70.9 65.5 52.1 83.1 84.7 86.4 52.0 81.9 65.7 84.8 84.6 77.5 76.7 38.8 73.6 73.9 83.0 72.6

84.3 82.0 77.7 68.9 65.7 88.1 88.4 88.9 63.6 86.3 70.8 85.9 87.6 80.1 82.3 53.6 80.4 75.8 86.6 78.9

9

Table 7: Results on PASCAL VOC 2012 test set with Fast R-CNN detectors and VGG-16. For RPN, the train-time
proposals for Fast R-CNN are 2000.
data

horse mbike person plant

mAP areo

sheep sofa

cow table

# box

method

bottle

chair

train

boat

bike

bird

dog

bus

car

cat

tv

SS

SS

RPN

RPN

RPN

2000

2000

300

300

300

12

07++12

12

07++12

COCO+07++12

65.7

68.4

67.0

70.4

75.9

80.3 74.7 66.9 46.9 37.7 73.9 68.6 87.7 41.7 71.1 51.1 86.0 77.8 79.8 69.8 32.1 65.5 63.8 76.4 61.7

82.3 78.4 70.8 52.3 38.7 77.8 71.6 89.3 44.2 73.0 55.0 87.5 80.5 80.8 72.0 35.1 68.3 65.7 80.4 64.2

82.3 76.4 71.0 48.4 45.2 72.1 72.3 87.3 42.2 73.7 50.0 86.8 78.7 78.4 77.4 34.5 70.1 57.1 77.1 58.9

84.9 79.8 74.3 53.9 49.8 77.5 75.9 88.5 45.6 77.1 55.3 86.9 81.7 80.9 79.6 40.1 72.6 60.9 81.2 61.5

87.4 83.6 76.8 62.9 59.6 81.9 82.0 91.3 54.9 82.6 59.0 89.0 85.5 84.7 84.1 52.2 78.9 65.5 85.4 70.2

Table 8: Detection results of Faster R-CNN on PAS-
CAL VOC 2007 test set using different settings of
anchors. The network is VGG-16. The training data
is VOC 2007 trainval. The default setting of using 3
scales and 3 aspect ratios (69.9%) is the same as that
in Table 3.
settings

aspect ratios mAP (%)

1 scale, 1 ratio

1 scale, 3 ratios

anchor scales
1282
2562
1282
2562

1:1
1:1
{2:1, 1:1, 1:2}
{2:1, 1:1, 1:2}
1:1

65.8
66.7
68.8
67.9
69.8
69.9

3 scales, 1 ratio {1282, 2562, 5122}
3 scales, 3 ratios {1282, 2562, 5122} {2:1, 1:1, 1:2}

Table 9: Detection results of Faster R-CNN on PAS-
CAL VOC 2007 test set using different values of λ
in Equation (1). The network is VGG-16. The training
data is VOC 2007 trainval. The default setting of using
λ = 10 (69.9%) is the same as that in Table 3.
100
69.1

λ
mAP (%)

0.1
67.2

1
68.9

10
69.9

In Table 5 we summarize the running time of the
entire object detection system. SS takes 1-2 seconds
depending on content (on average about 1.5s), and
Fast R-CNN with VGG-16 takes 320ms on 2000 SS
proposals (or 223ms if using SVD on fully-connected
layers [2]). Our system with VGG-16 takes in total
198ms for both proposal and detection. With the con-
volutional features shared, the RPN alone only takes
10ms computing the additional layers. Our region-
wise computation is also lower, thanks to fewer pro-
posals (300 per image). Our system has a frame-rate
of 17 fps with the ZF net.

Sensitivities to Hyper-parameters. In Table 8 we
investigate the settings of anchors. By default we use

3 scales and 3 aspect ratios (69.9% mAP in Table 8).
If using just one anchor at each position, the mAP
drops by a considerable margin of 3-4%. The mAP
is higher if using 3 scales (with 1 aspect ratio) or 3
aspect ratios (with 1 scale), demonstrating that using
anchors of multiple sizes as the regression references
is an effective solution. Using just 3 scales with 1
aspect ratio (69.8%) is as good as using 3 scales with
3 aspect ratios on this dataset, suggesting that scales
and aspect ratios are not disentangled dimensions for
the detection accuracy. But we still adopt these two
dimensions in our designs to keep our system ﬂexible.
In Table 9 we compare different values of λ in Equa-
tion (1). By default we use λ = 10 which makes the
two terms in Equation (1) roughly equally weighted
after normalization. Table 9 shows that our result is
impacted just marginally (by ∼ 1%) when λ is within
a scale of about two orders of magnitude (1 to 100).
This demonstrates that the result is insensitive to λ in
a wide range.

Analysis of Recall-to-IoU. Next we compute the
recall of proposals at different IoU ratios with ground-
truth boxes. It is noteworthy that the Recall-to-IoU
metric is just loosely [19], [20], [21] related to the
ultimate detection accuracy. It is more appropriate to
use this metric to diagnose the proposal method than
to evaluate it.

In Figure 4, we show the results of using 300, 1000,
and 2000 proposals. We compare with SS and EB, and
the N proposals are the top-N ranked ones based on
the conﬁdence generated by these methods. The plots
show that the RPN method behaves gracefully when
the number of proposals drops from 2000 to 300. This
explains why the RPN has a good ultimate detection
mAP when using as few as 300 proposals. As we
analyzed before, this property is mainly attributed to
the cls term of the RPN. The recall of SS and EB drops
more quickly than RPN when the proposals are fewer.

10

Figure 4: Recall vs. IoU overlap ratio on the PASCAL VOC 2007 test set.

Table 10: One-Stage Detection vs. Two-Stage Proposal + Detection. Detection results are on the PASCAL
VOC 2007 test set using the ZF model and Fast R-CNN. RPN uses unshared features.

proposals

detector

mAP (%)

Two-Stage
One-Stage
One-Stage

RPN + ZF, unshared
dense, 3 scales, 3 aspect ratios
dense, 3 scales, 3 aspect ratios

300
20000
20000

Fast R-CNN + ZF, 1 scale
Fast R-CNN + ZF, 1 scale
Fast R-CNN + ZF, 5 scales

58.7
53.8
53.9

One-Stage Detection vs. Two-Stage Proposal + De-
tection. The OverFeat paper [9] proposes a detection
method that uses regressors and classiﬁers on sliding
windows over convolutional feature maps. OverFeat
is a one-stage, class-speciﬁc detection pipeline, and ours
is a two-stage cascade consisting of class-agnostic pro-
posals and class-speciﬁc detections. In OverFeat, the
region-wise features come from a sliding window of
one aspect ratio over a scale pyramid. These features
are used to simultaneously determine the location and
category of objects. In RPN, the features are from
square (3×3) sliding windows and predict proposals
relative to anchors with different scales and aspect
ratios. Though both methods use sliding windows, the
region proposal task is only the ﬁrst stage of Faster R-
CNN—the downstream Fast R-CNN detector attends
to the proposals to reﬁne them. In the second stage of
our cascade, the region-wise features are adaptively
pooled [1], [2] from proposal boxes that more faith-
fully cover the features of the regions. We believe
these features lead to more accurate detections.

To compare the one-stage and two-stage systems,
we emulate the OverFeat system (and thus also circum-
vent other differences of implementation details) by
one-stage Fast R-CNN. In this system, the “proposals”
are dense sliding windows of 3 scales (128, 256, 512)
and 3 aspect ratios (1:1, 1:2, 2:1). Fast R-CNN is
trained to predict class-speciﬁc scores and regress box
locations from these sliding windows. Because the
OverFeat system adopts an image pyramid, we also
evaluate using convolutional features extracted from
5 scales. We use those 5 scales as in [1], [2].

Table 10 compares the two-stage system and two
variants of the one-stage system. Using the ZF model,
the one-stage system has an mAP of 53.9%. This is
lower than the two-stage system (58.7%) by 4.8%.
This experiment justiﬁes the effectiveness of cascaded
region proposals and object detection. Similar obser-
vations are reported in [2], [39], where replacing SS

region proposals with sliding windows leads to ∼6%
degradation in both papers. We also note that the one-
stage system is slower as it has considerably more
proposals to process.

4.2 Experiments on MS COCO

We present more results on the Microsoft COCO
object detection dataset [12]. This dataset involves 80
object categories. We experiment with the 80k images
on the training set, 40k images on the validation set,
and 20k images on the test-dev set. We evaluate the
mAP averaged for IoU ∈ [0.5 : 0.05 : 0.95] (COCO’s
standard metric, simply denoted as mAP@[.5,
.95])
and mAP@0.5 (PASCAL VOC’s metric).

There are a few minor changes of our system made
for this dataset. We train our models on an 8-GPU
implementation, and the effective mini-batch size be-
comes 8 for RPN (1 per GPU) and 16 for Fast R-CNN
(2 per GPU). The RPN step and Fast R-CNN step are
both trained for 240k iterations with a learning rate
of 0.003 and then for 80k iterations with 0.0003. We
modify the learning rates (starting with 0.003 instead
of 0.001) because the mini-batch size is changed. For
the anchors, we use 3 aspect ratios and 4 scales
(adding 642), mainly motivated by handling small
objects on this dataset. In addition, in our Fast R-CNN
step, the negative samples are deﬁned as those with
a maximum IoU with ground truth in the interval of
[0, 0.5), instead of [0.1, 0.5) used in [1], [2]. We note
that in the SPPnet system [1], the negative samples
in [0.1, 0.5) are used for network ﬁne-tuning, but the
negative samples in [0, 0.5) are still visited in the SVM
step with hard-negative mining. But the Fast R-CNN
system [2] abandons the SVM step, so the negative
samples in [0, 0.1) are never visited. Including these
[0, 0.1) samples improves mAP@0.5 on the COCO
dataset for both Fast R-CNN and Faster R-CNN sys-
tems (but the impact is negligible on PASCAL VOC).

(cid:2)(cid:3)(cid:4)(cid:2)(cid:3)(cid:5)(cid:2)(cid:3)(cid:6)(cid:2)(cid:3)(cid:7)(cid:2)(cid:3)(cid:8)(cid:9)(cid:2)(cid:2)(cid:3)(cid:10)(cid:2)(cid:3)(cid:11)(cid:2)(cid:3)(cid:5)(cid:2)(cid:3)(cid:7)(cid:9)(cid:2)(cid:3)(cid:4)(cid:2)(cid:3)(cid:4)(cid:5)(cid:6)(cid:6)(cid:2)(cid:3)(cid:3)(cid:4)(cid:5)(cid:6)(cid:7)(cid:5)(cid:7)(cid:8)(cid:9)(cid:10)(cid:8)  (cid:2)(cid:2)(cid:3)(cid:4)(cid:5)(cid:6)(cid:7)(cid:8)(cid:9)(cid:10)(cid:5)(cid:6)(cid:7)(cid:8)(cid:11)(cid:12)(cid:12)(cid:2)(cid:3)(cid:4)(cid:2)(cid:3)(cid:5)(cid:2)(cid:3)(cid:6)(cid:2)(cid:3)(cid:7)(cid:2)(cid:3)(cid:8)(cid:9)(cid:2)(cid:2)(cid:3)(cid:10)(cid:2)(cid:3)(cid:11)(cid:2)(cid:3)(cid:5)(cid:2)(cid:3)(cid:7)(cid:9)(cid:2)(cid:3)(cid:4)(cid:11)(cid:3)(cid:3)(cid:3)(cid:4)(cid:5)(cid:6)(cid:7)(cid:5)(cid:7)(cid:8)(cid:9)(cid:10)(cid:8)  (cid:2)(cid:2)(cid:3)(cid:4)(cid:5)(cid:6)(cid:7)(cid:8)(cid:9)(cid:10)(cid:5)(cid:6)(cid:7)(cid:8)(cid:11)(cid:12)(cid:12)(cid:2)(cid:3)(cid:4)(cid:2)(cid:3)(cid:5)(cid:2)(cid:3)(cid:6)(cid:2)(cid:3)(cid:7)(cid:2)(cid:3)(cid:8)(cid:9)(cid:2)(cid:2)(cid:3)(cid:10)(cid:2)(cid:3)(cid:11)(cid:2)(cid:3)(cid:5)(cid:2)(cid:3)(cid:7)(cid:9)(cid:2)(cid:3)(cid:4)(cid:12)(cid:3)(cid:3)(cid:3)(cid:4)(cid:5)(cid:6)(cid:7)(cid:5)(cid:7)(cid:8)(cid:9)(cid:10)(cid:8)  (cid:2)(cid:2)(cid:3)(cid:4)(cid:5)(cid:6)(cid:7)(cid:8)(cid:9)(cid:10)(cid:5)(cid:6)(cid:7)(cid:8)(cid:11)(cid:12)(cid:12)Table 11: Object detection results (%) on the MS COCO dataset. The model is VGG-16.

COCO val

COCO test-dev

method

proposals

training data

mAP@.5 mAP@[.5, .95] mAP@.5 mAP@[.5, .95]

Fast R-CNN [2]
SS, 2000
Fast R-CNN [impl. in this paper] SS, 2000
Faster R-CNN
Faster R-CNN

COCO train
COCO train
RPN, 300 COCO train
RPN, 300 COCO trainval

-
38.6
41.5
-

-
18.9
21.2
-

35.9
39.3
42.1
42.7

19.7
19.3
21.5
21.9

11

The rest of the implementation details are the same
as on PASCAL VOC. In particular, we keep using
300 proposals and single-scale (s = 600) testing. The
testing time is still about 200ms per image on the
COCO dataset.

In Table 11 we ﬁrst report the results of the Fast
R-CNN system [2] using the implementation in this
paper. Our Fast R-CNN baseline has 39.3% mAP@0.5
on the test-dev set, higher than that reported in [2].
We conjecture that the reason for this gap is mainly
due to the deﬁnition of the negative samples and also
the changes of the mini-batch sizes. We also note that
the mAP@[.5, .95] is just comparable.

Next we evaluate our Faster R-CNN system. Using
the COCO training set to train, Faster R-CNN has
.95] on the
42.1% mAP@0.5 and 21.5% mAP@[.5,
COCO test-dev set. This is 2.8% higher for mAP@0.5
and 2.2% higher for mAP@[.5, .95] than the Fast R-
CNN counterpart under the same protocol (Table 11).
This indicates that RPN performs excellent for im-
proving the localization accuracy at higher IoU thresh-
olds. Using the COCO trainval set to train, Faster R-
CNN has 42.7% mAP@0.5 and 21.9% mAP@[.5, .95] on
the COCO test-dev set. Figure 6 shows some results
on the MS COCO test-dev set.

Faster R-CNN in ILSVRC & COCO 2015 compe-
titions We have demonstrated that Faster R-CNN
beneﬁts more from better features, thanks to the fact
that the RPN completely learns to propose regions by
neural networks. This observation is still valid even
when one increases the depth substantially to over
100 layers [18]. Only by replacing VGG-16 with a 101-
layer residual net (ResNet-101) [18], the Faster R-CNN
system increases the mAP from 41.5%/21.2% (VGG-
16) to 48.4%/27.2% (ResNet-101) on the COCO val
set. With other improvements orthogonal to Faster R-
CNN, He et al. [18] obtained a single-model result of
55.7%/34.9% and an ensemble result of 59.0%/37.4%
on the COCO test-dev set, which won the 1st place
in the COCO 2015 object detection competition. The
same system [18] also won the 1st place in the ILSVRC
2015 object detection competition, surpassing the sec-
ond place by absolute 8.5%. RPN is also a building
block of the 1st-place winning entries in ILSVRC 2015
localization and COCO 2015 segmentation competi-
tions, for which the details are available in [18] and
[15] respectively.

Table 12: Detection mAP (%) of Faster R-CNN on
PASCAL VOC 2007 test set and 2012 test set us-
ing different training data. The model is VGG-16.
“COCO” denotes that the COCO trainval set is used
for training. See also Table 6 and Table 7.

training data

2007 test

2012 test

VOC07
VOC07+12
VOC07++12
COCO (no VOC)
COCO+VOC07+12
COCO+VOC07++12

69.9
73.2
-
76.1
78.8
-

67.0
-
70.4
73.0
-
75.9

4.3 From MS COCO to PASCAL VOC

Large-scale data is of crucial importance for improv-
ing deep neural networks. Next, we investigate how
the MS COCO dataset can help with the detection
performance on PASCAL VOC.

As a simple baseline, we directly evaluate the
COCO detection model on the PASCAL VOC dataset,
without ﬁne-tuning on any PASCAL VOC data. This
evaluation is possible because the categories on
COCO are a superset of those on PASCAL VOC. The
categories that are exclusive on COCO are ignored in
this experiment, and the softmax layer is performed
only on the 20 categories plus background. The mAP
under this setting is 76.1% on the PASCAL VOC 2007
test set (Table 12). This result is better than that trained
on VOC07+12 (73.2%) by a good margin, even though
the PASCAL VOC data are not exploited.

Then we ﬁne-tune the COCO detection model on
the VOC dataset. In this experiment, the COCO model
is in place of the ImageNet-pre-trained model (that
is used to initialize the network weights), and the
Faster R-CNN system is ﬁne-tuned as described in
Section 3.2. Doing so leads to 78.8% mAP on the
PASCAL VOC 2007 test set. The extra data from
the COCO set increases the mAP by 5.6%. Table 6
shows that the model trained on COCO+VOC has
the best AP for every individual category on PASCAL
VOC 2007. Similar improvements are observed on the
PASCAL VOC 2012 test set (Table 12 and Table 7). We
note that the test-time speed of obtaining these strong
results is still about 200ms per image.

5 CONCLUSION
We have presented RPNs for efﬁcient and accurate
region proposal generation. By sharing convolutional

12

Figure 5: Selected examples of object detection results on the PASCAL VOC 2007 test set using the Faster
R-CNN system. The model is VGG-16 and the training data is 07+12 trainval (73.2% mAP on the 2007 test
set). Our method detects objects of a wide range of scales and aspect ratios. Each output box is associated
with a category label and a softmax score in [0, 1]. A score threshold of 0.6 is used to display these images.
The running time for obtaining these results is 198ms per image, including all steps.

features with the down-stream detection network, the
region proposal step is nearly cost-free. Our method
enables a uniﬁed, deep-learning-based object detec-
tion system to run at near real-time frame rates. The
learned RPN also improves region proposal quality
and thus the overall object detection accuracy.

REFERENCES

[1] K. He, X. Zhang, S. Ren, and J. Sun, “Spatial pyramid pooling
in deep convolutional networks for visual recognition,” in
European Conference on Computer Vision (ECCV), 2014.

[2] R. Girshick, “Fast R-CNN,” in IEEE International Conference on

Computer Vision (ICCV), 2015.

[3] K. Simonyan and A. Zisserman, “Very deep convolutional

bottle : 0.726person : 0.992dog : 0.981bicycle : 0.987bicycle : 0.977777bicycle : 0.972person : 0.995person : 0.994e : 0.987e:0987bicycbicycbicycb444person : 0.981person : 0.975person : 0.972person : 0.948person : 0.919horse : 0.984person : 0.670bird : 0.997bird : 0.727car : 1.000car : 0.982car car car : 0.981car : 0.880bottle : 0.826chair : 0.630diningtable : 0.862pottedplant : 0.728boat : 0.995boat : 0.948boat : 0.808:0808:boat : 0.692boat : 0.992boat : 0.846boat : 0.693bottle : 0.9620962bottle : 0.851diningtable : 0.791person : 0.962person : 0.930pottedplant : 0.951dog : 0.987person : 0.940940940person : 0.893cat : 0.998car : 1.000person : 0.917boat : 0.895boat : 0.877boat : 0.749boat : 0.671person : 0.988car : 0.9555555car : 0.745.745horse : 0.991person : 0.988person : 0.797bird : 0.978bird : 0.972bird : 0.941bird : 0.902person : 0.918cow : 0.998cow : 0.995aeroplane : 0.992aeroplane : 0.986sheep : 0.970bird : 0.998bird : 0.980bird : 0.806pottedplant : 0.993pottedplant : 0.940pottedplant : 0.869pottedplant : 0.820pottedplant : 0.715aeroplane : 0.998car : 0.907907907person : 0.993person : 0.987chair : 0.984chair : 0.978chair : 0.976chair : 0.962984984diningtable : 0.997bottle : 0.789chair : 0.723diningtable : 0.903e:0.789person : 0.968tvmonitor : 0.993tvmonitor : 0.945aeroplane : 0.978person : 0.988bottle : 0.903bottle : 0.884bottle : 0.858bbbottle :0botbottle : 0.616chair : 0.982chair : 0.852person : 0.983person : 0.959:0903person : 0.897person : 0.870tvmonitor : 0.993dog : 0.697person : 0.961person : 0.960personpersonperson : 0.958person : 0.757bus : 0.999person : 0.996perperperperson : 0.995person : 0.994person : 0.985cow : 0.985cow : 0.979cow : 0.979cow : 0.974cow : 0.892person : 0.998car : 0.999person : 0.929person : 0.994person : 0.991person : 0.988perspperson : 0.976person : 0.964car : 0.997car : 0.980person : 0.993personpersonpersonpersonperson : 0.9860993:n86n:n:nperson : 0.95913

Figure 6: Selected examples of object detection results on the MS COCO test-dev set using the Faster R-CNN
system. The model is VGG-16 and the training data is COCO trainval (42.7% mAP@0.5 on the test-dev set).
Each output box is associated with a category label and a softmax score in [0, 1]. A score threshold of 0.6 is
used to display these images. For each image, one color represents one object category in that image.

[4]

networks for large-scale image recognition,” in International
Conference on Learning Representations (ICLR), 2015.
J. R. Uijlings, K. E. van de Sande, T. Gevers, and A. W. Smeul-
ders, “Selective search for object recognition,” International
Journal of Computer Vision (IJCV), 2013.

[5] R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature
hierarchies for accurate object detection and semantic seg-
mentation,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2014.

[6] C. L. Zitnick and P. Doll´ar, “Edge boxes: Locating object
proposals from edges,” in European Conference on Computer
Vision (ECCV), 2014.

[7]

J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional
networks for semantic segmentation,” in IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), 2015.

[8] P. F. Felzenszwalb, R. B. Girshick, D. McAllester, and D. Ra-
manan, “Object detection with discriminatively trained part-
based models,” IEEE Transactions on Pattern Analysis and Ma-
chine Intelligence (TPAMI), 2010.

[9] P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus,
and Y. LeCun, “Overfeat: Integrated recognition, localization
and detection using convolutional networks,” in International
Conference on Learning Representations (ICLR), 2014.

[10] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards

cup : 0.807bowl : 0.847bowl : 0.816bowl : 0.744bowl : 0.710chair : 0.772dining table : 0.618oven : 0.969refrigerator : 0.631cup : 0.990pizza : 0.919dining table : 0.888person : 0.984personpersoncar : 0.816pizza : 0.965clock : 0.988person : 0.998kite : 0.934toothbrush : 0.668teddy bear : 0.999teddy bear : 0.890teddy bear : 0.802teddy bear : 0.738bowl : 0.602potted plant : 0.769toilet : 0.921sink : 0.969sink : 0.994sink : 0.992sink : 0.97666sink : 0.938person : 0.970: 0.970ersonperson : 0.869bus : 0.999bottle : 0.768cup : 0.720chair : 0.644tv : 0.964tv : 0.959laptop : 0.986mouse : 0.871mouse : 0.677mkeyboard : 0.956book : 0.611person : 0.986boat : 0.758boat : 0.746boat : 0.613bench : 0.971train : 0.965traffic light : 0.869traffic light : 0.713chair : 0.631couch : 0.991couch : 0.719couch : 0.627dining table : 0.637dog : 0.966frisbee : 0.998bird : 0.987bird : 0.968bird : 0.894person : 0.723cup : 0.986cup : 0.931bowl : 0.958sandwich : 0.629dining table : 0.941zebra : 0.996zebra : 0.993zebra : 0.970970zebra : 0.848person : 0.917person : 0.792: 0.7920792tv : 0.711laptop : 0.973mouse : 0.981keyboard : 0.638keyboard : 0.615person : 0.999person : 0.999person : 0.999persopersotennis racket : 0.960bird : 0.956bird : 0.906bird : 0.746horse : 0.990person : 0.993bottle : 0.982oven : 0.655refrigerator : 0.699clock : 0.982bed : 0.999person : 0.808bottle : 0.627pizza : 0.995pizza : 0.985pizza : 0.982pizza : 0.938dining table : 0.956person : 0.998skis : 0.919bowl : 0.759broccoli : 0.953person : 0.999person : 0.934surfboard : 0.979person : 0.940person : 0.927person : 0.8640.940person : 0.854person : 0.82555person : 0.813person : 0.716person : 0.692ppperson : 0.691927927person : 0.665person : 0.618boat : 0.992umbrella : 0.885giraffe : 0.993giraffe : 0.989giraffe : 0.988person : 0.867airplane : 0.997person : 0.970person : 0.950person : 0.931pperson : 0.916person : 0.897person : 0.842person : 0.841person:0.84person : 0.772bicycle : 0.891bicycle : 0.639car : 0.957motorcycle : 0.827motorcycle : 0.713traffic light : 0.802umbrella : 0.824person : 0.800clock : 0.986clock : 0.981person : 0.996person : 0.976person : 0.975rson : 0.975rsonson : 0onperson : 0.958person : 0.950person : 0.9410.9760976person : 0.939pepeperson : 0.9289589580975n : 0.n:0.0.9750.9750.person : 0.823on : 0.950050person : 0.805person : 0.766person : 0.759.9414person : 0.673dog : 0.996dog : 0.6910939pbackpack : 0.756handbag : 0.84814

[36] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma,
Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg,
and L. Fei-Fei, “ImageNet Large Scale Visual Recognition
Challenge,” in International Journal of Computer Vision (IJCV),
2015.

[37] A. Krizhevsky, I. Sutskever, and G. Hinton, “Imagenet classi-
ﬁcation with deep convolutional neural networks,” in Neural
Information Processing Systems (NIPS), 2012.

[38] Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Gir-
shick, S. Guadarrama, and T. Darrell, “Caffe: Convolutional
architecture for fast feature embedding,” arXiv:1408.5093, 2014.
[39] K. Lenc and A. Vedaldi, “R-CNN minus R,” in British Machine

Vision Conference (BMVC), 2015.

real-time object detection with region proposal networks,” in
Neural Information Processing Systems (NIPS), 2015.

[11] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and
A. Zisserman, “The PASCAL Visual Object Classes Challenge
2007 (VOC2007) Results,” 2007.

[12] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ra-
manan, P. Doll´ar, and C. L. Zitnick, “Microsoft COCO: Com-
mon Objects in Context,” in European Conference on Computer
Vision (ECCV), 2014.

[13] S. Song and J. Xiao, “Deep sliding shapes for amodal 3d object

detection in rgb-d images,” arXiv:1511.02300, 2015.

[14] J. Zhu, X. Chen, and A. L. Yuille, “DeePM: A deep part-based
model for object detection and semantic part localization,”
arXiv:1511.07131, 2015.

[15] J. Dai, K. He, and J. Sun, “Instance-aware semantic segmenta-
tion via multi-task network cascades,” arXiv:1512.04412, 2015.
[16] J. Johnson, A. Karpathy, and L. Fei-Fei, “Densecap: Fully
localization networks for dense captioning,”

convolutional
arXiv:1511.07571, 2015.

[17] D. Kislyuk, Y. Liu, D. Liu, E. Tzeng, and Y. Jing, “Human cu-
ration and convnets: Powering item-to-item recommendations
on pinterest,” arXiv:1511.04003, 2015.

[18] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning

for image recognition,” arXiv:1512.03385, 2015.

[19] J. Hosang, R. Benenson, and B. Schiele, “How good are de-
tection proposals, really?” in British Machine Vision Conference
(BMVC), 2014.

[20] J. Hosang, R. Benenson, P. Doll´ar, and B. Schiele, “What makes
for effective detection proposals?” IEEE Transactions on Pattern
Analysis and Machine Intelligence (TPAMI), 2015.

[21] N. Chavali, H. Agrawal, A. Mahendru, and D. Batra,
“Object-Proposal Evaluation Protocol is ’Gameable’,” arXiv:
1505.05836, 2015.

[22] J. Carreira and C. Sminchisescu, “CPMC: Automatic ob-
ject segmentation using constrained parametric min-cuts,”
IEEE Transactions on Pattern Analysis and Machine Intelligence
(TPAMI), 2012.

[23] P. Arbel´aez, J. Pont-Tuset, J. T. Barron, F. Marques, and J. Malik,
“Multiscale combinatorial grouping,” in IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), 2014.

[24] B. Alexe, T. Deselaers, and V. Ferrari, “Measuring the object-
ness of image windows,” IEEE Transactions on Pattern Analysis
and Machine Intelligence (TPAMI), 2012.

[25] C. Szegedy, A. Toshev, and D. Erhan, “Deep neural networks
for object detection,” in Neural Information Processing Systems
(NIPS), 2013.

[26] D. Erhan, C. Szegedy, A. Toshev, and D. Anguelov, “Scalable
object detection using deep neural networks,” in IEEE Confer-
ence on Computer Vision and Pattern Recognition (CVPR), 2014.
[27] C. Szegedy, S. Reed, D. Erhan, and D. Anguelov, “Scalable,
high-quality object detection,” arXiv:1412.1441 (v1), 2015.
[28] P. O. Pinheiro, R. Collobert, and P. Dollar, “Learning to
segment object candidates,” in Neural Information Processing
Systems (NIPS), 2015.

[29] J. Dai, K. He, and J. Sun, “Convolutional feature masking
for joint object and stuff segmentation,” in IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), 2015.

[30] S. Ren, K. He, R. Girshick, X. Zhang, and J. Sun, “Ob-
feature maps,”

ject detection networks on convolutional
arXiv:1504.06066, 2015.

[31] J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and
Y. Bengio, “Attention-based models for speech recognition,”
in Neural Information Processing Systems (NIPS), 2015.

[32] M. D. Zeiler and R. Fergus, “Visualizing and understanding
convolutional neural networks,” in European Conference on
Computer Vision (ECCV), 2014.

[33] V. Nair and G. E. Hinton, “Rectiﬁed linear units improve
restricted boltzmann machines,” in International Conference on
Machine Learning (ICML), 2010.

[34] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov,
D. Erhan, and A. Rabinovich, “Going deeper with convo-
lutions,” in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2015.

[35] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard,
W. Hubbard, and L. D. Jackel, “Backpropagation applied to
handwritten zip code recognition,” Neural computation, 1989.


