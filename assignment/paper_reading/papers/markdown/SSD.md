6
1
0
2
c
e
D
9
2

SSD: Single Shot MultiBox Detector

Wei Liu1, Dragomir Anguelov2, Dumitru Erhan3, Christian Szegedy3,
Scott Reed4, Cheng-Yang Fu1, Alexander C. Berg1

1UNC Chapel Hill 2Zoox Inc. 3Google Inc. 4University of Michigan, Ann-Arbor
1wliu@cs.unc.edu, 2drago@zoox.com, 3{dumitru,szegedy}@google.com,
4reedscot@umich.edu, 1{cyfu,aberg}@cs.unc.edu

]

V
C
.
s
c
[

5
v
5
2
3
2
0
.
2
1
5
1
:
v
i
X
r
a

Abstract. We present a method for detecting objects in images using a single
deep neural network. Our approach, named SSD, discretizes the output space of
bounding boxes into a set of default boxes over different aspect ratios and scales
per feature map location. At prediction time, the network generates scores for the
presence of each object category in each default box and produces adjustments to
the box to better match the object shape. Additionally, the network combines pre-
dictions from multiple feature maps with different resolutions to naturally handle
objects of various sizes. SSD is simple relative to methods that require object
proposals because it completely eliminates proposal generation and subsequent
pixel or feature resampling stages and encapsulates all computation in a single
network. This makes SSD easy to train and straightforward to integrate into sys-
tems that require a detection component. Experimental results on the PASCAL
VOC, COCO, and ILSVRC datasets conﬁrm that SSD has competitive accuracy
to methods that utilize an additional object proposal step and is much faster, while
providing a uniﬁed framework for both training and inference. For 300 × 300 in-
put, SSD achieves 74.3% mAP1 on VOC2007 test at 59 FPS on a Nvidia Titan
X and for 512 × 512 input, SSD achieves 76.9% mAP, outperforming a compa-
rable state-of-the-art Faster R-CNN model. Compared to other single stage meth-
ods, SSD has much better accuracy even with a smaller input image size. Code is
available at: https://github.com/weiliu89/caffe/tree/ssd .

Keywords: Real-time Object Detection; Convolutional Neural Network

1 Introduction

Current state-of-the-art object detection systems are variants of the following approach:
hypothesize bounding boxes, resample pixels or features for each box, and apply a high-
quality classiﬁer. This pipeline has prevailed on detection benchmarks since the Selec-
tive Search work [1] through the current leading results on PASCAL VOC, COCO, and
ILSVRC detection all based on Faster R-CNN[2] albeit with deeper features such as
[3]. While accurate, these approaches have been too computationally intensive for em-
bedded systems and, even with high-end hardware, too slow for real-time applications.

1 We achieved even better results using an improved data augmentation scheme in follow-on
experiments: 77.2% mAP for 300×300 input and 79.8% mAP for 512×512 input on VOC2007.
Please see Sec. 3.6 for details.

2

Liu et al.

Often detection speed for these approaches is measured in seconds per frame (SPF),
and even the fastest high-accuracy detector, Faster R-CNN, operates at only 7 frames
per second (FPS). There have been many attempts to build faster detectors by attacking
each stage of the detection pipeline (see related work in Sec. 4), but so far, signiﬁcantly
increased speed comes only at the cost of signiﬁcantly decreased detection accuracy.

This paper presents the ﬁrst deep network based object detector that does not re-
sample pixels or features for bounding box hypotheses and and is as accurate as ap-
proaches that do. This results in a signiﬁcant improvement in speed for high-accuracy
detection (59 FPS with mAP 74.3% on VOC2007 test, vs. Faster R-CNN 7 FPS with
mAP 73.2% or YOLO 45 FPS with mAP 63.4%). The fundamental improvement in
speed comes from eliminating bounding box proposals and the subsequent pixel or fea-
ture resampling stage. We are not the ﬁrst to do this (cf [4,5]), but by adding a series
of improvements, we manage to increase the accuracy signiﬁcantly over previous at-
tempts. Our improvements include using a small convolutional ﬁlter to predict object
categories and offsets in bounding box locations, using separate predictors (ﬁlters) for
different aspect ratio detections, and applying these ﬁlters to multiple feature maps from
the later stages of a network in order to perform detection at multiple scales. With these
modiﬁcations—especially using multiple layers for prediction at different scales—we
can achieve high-accuracy using relatively low resolution input, further increasing de-
tection speed. While these contributions may seem small independently, we note that
the resulting system improves accuracy on real-time detection for PASCAL VOC from
63.4% mAP for YOLO to 74.3% mAP for our SSD. This is a larger relative improve-
ment in detection accuracy than that from the recent, very high-proﬁle work on residual
networks [3]. Furthermore, signiﬁcantly improving the speed of high-quality detection
can broaden the range of settings where computer vision is useful.

We summarize our contributions as follows:

– We introduce SSD, a single-shot detector for multiple categories that is faster than
the previous state-of-the-art for single shot detectors (YOLO), and signiﬁcantly
more accurate, in fact as accurate as slower techniques that perform explicit region
proposals and pooling (including Faster R-CNN).

– The core of SSD is predicting category scores and box offsets for a ﬁxed set of
default bounding boxes using small convolutional ﬁlters applied to feature maps.
– To achieve high detection accuracy we produce predictions of different scales from
feature maps of different scales, and explicitly separate predictions by aspect ratio.
– These design features lead to simple end-to-end training and high accuracy, even
on low resolution input images, further improving the speed vs accuracy trade-off.
– Experiments include timing and accuracy analysis on models with varying input
size evaluated on PASCAL VOC, COCO, and ILSVRC and are compared to a
range of recent state-of-the-art approaches.

2 The Single Shot Detector (SSD)

This section describes our proposed SSD framework for detection (Sec. 2.1) and the
associated training methodology (Sec. 2.2). Afterwards, Sec. 3 presents dataset-speciﬁc
model details and experimental results.

SSD: Single Shot MultiBox Detector

3

Fig. 1: SSD framework. (a) SSD only needs an input image and ground truth boxes for
each object during training. In a convolutional fashion, we evaluate a small set (e.g. 4)
of default boxes of different aspect ratios at each location in several feature maps with
4 in (b) and (c)). For each default box, we predict
different scales (e.g. 8
both the shape offsets and the conﬁdences for all object categories ((c1, c2,
, cp)).
At training time, we ﬁrst match these default boxes to the ground truth boxes. For
example, we have matched two default boxes with the cat and one with the dog, which
are treated as positives and the rest as negatives. The model loss is a weighted sum
between localization loss (e.g. Smooth L1 [6]) and conﬁdence loss (e.g. Softmax).

8 and 4

· · ·

×

×

2.1 Model

The SSD approach is based on a feed-forward convolutional network that produces
a ﬁxed-size collection of bounding boxes and scores for the presence of object class
instances in those boxes, followed by a non-maximum suppression step to produce the
ﬁnal detections. The early network layers are based on a standard architecture used for
high quality image classiﬁcation (truncated before any classiﬁcation layers), which we
will call the base network2. We then add auxiliary structure to the network to produce
detections with the following key features:

Multi-scale feature maps for detection We add convolutional feature layers to the end
of the truncated base network. These layers decrease in size progressively and allow
predictions of detections at multiple scales. The convolutional model for predicting
detections is different for each feature layer (cf Overfeat[4] and YOLO[5] that operate
on a single scale feature map).

Convolutional predictors for detection Each added feature layer (or optionally an ex-
isting feature layer from the base network) can produce a ﬁxed set of detection predic-
tions using a set of convolutional ﬁlters. These are indicated on top of the SSD network
n with p channels, the basic el-
architecture in Fig. 2. For a feature layer of size m
p small kernel
ement for predicting parameters of a potential detection is a 3
that produces either a score for a category, or a shape offset relative to the default box
coordinates. At each of the m
n locations where the kernel is applied, it produces an
output value. The bounding box offset output values are measured relative to a default

×

×

×

×

3

2 We use the VGG-16 network as a base, but other networks should also produce good results.

(a)ImagewithGTboxes(b)8×8featuremap(c)4×4featuremaploc:∆(cx,cy,w,h)conf:(c1,c2,···,cp)4

Liu et al.

Fig. 2: A comparison between two single shot detection models: SSD and YOLO [5].
Our SSD model adds several feature layers to the end of a base network, which predict
the offsets to default boxes of different scales and aspect ratios and their associated
conﬁdences. SSD with a 300
448
YOLO counterpart in accuracy on VOC2007 test while also improving the speed.

300 input size signiﬁcantly outperforms its 448

×

×

box position relative to each feature map location (cf the architecture of YOLO[5] that
uses an intermediate fully connected layer instead of a convolutional ﬁlter for this step).
Default boxes and aspect ratios We associate a set of default bounding boxes with
each feature map cell, for multiple feature maps at the top of the network. The default
boxes tile the feature map in a convolutional manner, so that the position of each box
relative to its corresponding cell is ﬁxed. At each feature map cell, we predict the offsets
relative to the default box shapes in the cell, as well as the per-class scores that indicate
the presence of a class instance in each of those boxes. Speciﬁcally, for each box out of
k at a given location, we compute c class scores and the 4 offsets relative to the original
default box shape. This results in a total of (c + 4)k ﬁlters that are applied around each
location in the feature map, yielding (c + 4)kmn outputs for a m
n feature map. For
an illustration of default boxes, please refer to Fig. 1. Our default boxes are similar to
the anchor boxes used in Faster R-CNN [2], however we apply them to several feature
maps of different resolutions. Allowing different default box shapes in several feature
maps let us efﬁciently discretize the space of possible output box shapes.

×

2.2 Training

The key difference between training SSD and training a typical detector that uses region
proposals, is that ground truth information needs to be assigned to speciﬁc outputs in
the ﬁxed set of detector outputs. Some version of this is also required for training in
YOLO[5] and for the region proposal stage of Faster R-CNN[2] and MultiBox[7]. Once
this assignment is determined, the loss function and back propagation are applied end-
to-end. Training also involves choosing the set of default boxes and scales for detection
as well as the hard negative mining and data augmentation strategies.

3003003VGG-16 through Conv5_3 layer1919Conv7(FC7)10241010Conv8_251255Conv9_22563Conv10_22562563838Conv4_331ImageConv: 1x1x1024Conv: 1x1x256Conv:  3x3x512-s2Conv: 1x1x128Conv: 3x3x256-s2Conv: 1x1x128Conv: 3x3x256-s1Detections:8732  per ClassClassifier : Conv: 3x3x(4x(Classes+4))5124484483Image7710247730Fully ConnectedYOLO Customized ArchitectureNon-Maximum Suppression Fully ConnectedNon-Maximum Suppression  Detections: 98 per classConv11_274.3mAP   59FPS63.4mAP   45FPSClassifier : Conv: 3x3x(6x(Classes+4))1919Conv6(FC6)1024Conv: 3x3x1024SSDYOLOExtra Feature LayersConv: 1x1x128Conv: 3x3x256-s1Conv: 3x3x(4x(Classes+4))SSD: Single Shot MultiBox Detector

5

Matching strategy During training we need to determine which default boxes corre-
spond to a ground truth detection and train the network accordingly. For each ground
truth box we are selecting from default boxes that vary over location, aspect ratio, and
scale. We begin by matching each ground truth box to the default box with the best
jaccard overlap (as in MultiBox [7]). Unlike MultiBox, we then match default boxes to
any ground truth with jaccard overlap higher than a threshold (0.5). This simpliﬁes the
learning problem, allowing the network to predict high scores for multiple overlapping
default boxes rather than requiring it to pick only the one with maximum overlap.

Training objective The SSD training objective is derived from the MultiBox objec-
tive [7,8] but is extended to handle multiple object categories. Let xp
be an
indicator for matching the i-th default box to the j-th ground truth box of category p.
In the matching strategy above, we can have (cid:80)
1. The overall objective loss
function is a weighted sum of the localization loss (loc) and the conﬁdence loss (conf):

1, 0
}
{

ij ≥

ij =

i xp

L(x, c, l, g) =

1
N

(Lconf (x, c) + αLloc(x, l, g))

(1)

where N is the number of matched default boxes. If N = 0, wet set the loss to 0. The
localization loss is a Smooth L1 loss [6] between the predicted box (l) and the ground
truth box (g) parameters. Similar to Faster R-CNN [2], we regress to offsets for the
center (cx, cy) of the default bounding box (d) and for its width (w) and height (h).

Lloc(x, l, g) =

N
(cid:88)

i∈P os

ˆgcx
j = (gcx

j −

ˆgw
j = log

(cid:88)

m∈{cx,cy,w,h}
i )/dw
dcx
i
(cid:16) gw
(cid:17)
j
dw
i

ijsmoothL1(lm
xk

i −

ˆgm
j )

j = (gcy
ˆgcy

ˆgh
j = log

j −
(cid:16) gh
j
dh
i

dcy
i )/dh
i
(cid:17)

(2)

The conﬁdence loss is the softmax loss over multiple classes conﬁdences (c).

Lconf (x, c) =

N
(cid:88)

−

i∈P os

ijlog(ˆcp
xp
i )

(cid:88)

−

i∈N eg

log(ˆc0

i ) where

ˆcp
i =

exp(cp
i )
p exp(cp
i )

(cid:80)

(3)

and the weight term α is set to 1 by cross validation.

Choosing scales and aspect ratios for default boxes To handle different object scales,
some methods [4,9] suggest processing the image at different sizes and combining the
results afterwards. However, by utilizing feature maps from several different layers in a
single network for prediction we can mimic the same effect, while also sharing parame-
ters across all object scales. Previous works [10,11] have shown that using feature maps
from the lower layers can improve semantic segmentation quality because the lower
layers capture more ﬁne details of the input objects. Similarly, [12] showed that adding
global context pooled from a feature map can help smooth the segmentation results.

6

Liu et al.

Motivated by these methods, we use both the lower and upper feature maps for detec-
4) which are used in the
tion. Figure 1 shows two exemplar feature maps (8
framework. In practice, we can use many more with small computational overhead.

8 and 4

×

×

Feature maps from different levels within a network are known to have different
(empirical) receptive ﬁeld sizes [13]. Fortunately, within the SSD framework, the de-
fault boxes do not necessary need to correspond to the actual receptive ﬁelds of each
layer. We design the tiling of default boxes so that speciﬁc feature maps learn to be
responsive to particular scales of the objects. Suppose we want to use m feature maps
for prediction. The scale of the default boxes for each feature map is computed as:

sk = smin +

smax
m

smin
1

(k

−

1),

k

[1, m]

∈

(4)

−
−

2 , 1
3 }

. We can compute the width (wa

k = sk√ar) and height (ha

where smin is 0.2 and smax is 0.9, meaning the lowest layer has a scale of 0.2 and
the highest layer has a scale of 0.9, and all layers in between are regularly spaced.
We impose different aspect ratios for the default boxes, and denote them as ar ∈
1, 2, 3, 1
k = sk/√ar)
{
for each default box. For the aspect ratio of 1, we also add a default box whose scale is
s(cid:48)
k = √sksk+1, resulting in 6 default boxes per feature map location. We set the center
of each default box to ( i+0.5
is the size of the k-th square feature
map, i, j
). In practice, one can also design a distribution of default boxes to
[0,
best ﬁt a speciﬁc dataset. How to design the optimal tiling is an open question as well.
By combining predictions for all default boxes with different scales and aspect ratios
from all locations of many feature maps, we have a diverse set of predictions, covering
various input object sizes and shapes. For example, in Fig. 1, the dog is matched to a
default box in the 4
8 feature
map. This is because those boxes have different scales and do not match the dog box,
and therefore are considered as negatives during training.

4 feature map, but not to any default boxes in the 8

|fk| ), where

|fk| , j+0.5

fk|
|

fk|

×

×

∈

|

Hard negative mining After the matching step, most of the default boxes are nega-
tives, especially when the number of possible default boxes is large. This introduces a
signiﬁcant imbalance between the positive and negative training examples. Instead of
using all the negative examples, we sort them using the highest conﬁdence loss for each
default box and pick the top ones so that the ratio between the negatives and positives is
at most 3:1. We found that this leads to faster optimization and a more stable training.

Data augmentation To make the model more robust to various input object sizes and
shapes, each training image is randomly sampled by one of the following options:

– Use the entire original input image.
– Sample a patch so that the minimum jaccard overlap with the objects is 0.1, 0.3,

0.5, 0.7, or 0.9.

– Randomly sample a patch.

The size of each sampled patch is [0.1, 1] of the original image size, and the aspect ratio
is between 1
2 and 2. We keep the overlapped part of the ground truth box if the center of
it is in the sampled patch. After the aforementioned sampling step, each sampled patch
is resized to ﬁxed size and is horizontally ﬂipped with probability of 0.5, in addition to
applying some photo-metric distortions similar to those described in [14].

3 Experimental Results

SSD: Single Shot MultiBox Detector

7

2

Base network Our experiments are all based on VGG16 [15], which is pre-trained on
the ILSVRC CLS-LOC dataset [16]. Similar to DeepLab-LargeFOV [17], we convert
fc6 and fc7 to convolutional layers, subsample parameters from fc6 and fc7, change
s1, and use the `a trous algorithm [18] to ﬁll the
pool5 from 2
”holes”. We remove all the dropout layers and the fc8 layer. We ﬁne-tune the resulting
model using SGD with initial learning rate 10−3, 0.9 momentum, 0.0005 weight decay,
and batch size 32. The learning rate decay policy is slightly different for each dataset,
and we will describe details later. The full training and testing code is built on Caffe [19]
and is open source at: https://github.com/weiliu89/caffe/tree/ssd .

s2 to 3

×

−

×

−

3

3.1 PASCAL VOC2007

On this dataset, we compare against Fast R-CNN [6] and Faster R-CNN [2] on VOC2007
test (4952 images). All methods ﬁne-tune on the same pre-trained VGG16 network.
Figure 2 shows the architecture details of the SSD300 model. We use conv4 3,
conv7 (fc7), conv8 2, conv9 2, conv10 2, and conv11 2 to predict both location and
conﬁdences. We set default box with scale 0.1 on conv4 33. We initialize the parameters
for all the newly added convolutional layers with the ”xavier” method [20]. For conv4 3,
conv10 2 and conv11 2, we only associate 4 default boxes at each feature map location
– omitting aspect ratios of 1
3 and 3. For all other layers, we put 6 default boxes as
described in Sec. 2.2. Since, as pointed out in [12], conv4 3 has a different feature
scale compared to the other layers, we use the L2 normalization technique introduced
in [12] to scale the feature norm at each location in the feature map to 20 and learn the
scale during back propagation. We use the 10−3 learning rate for 40k iterations, then
continue training for 10k iterations with 10−4 and 10−5. When training on VOC2007
trainval, Table 1 shows that our low resolution SSD300 model is already more
accurate than Fast R-CNN. When we train SSD on a larger 512
512 input image, it is
even more accurate, surpassing Faster R-CNN by 1.7% mAP. If we train SSD with more
(i.e. 07+12) data, we see that SSD300 is already better than Faster R-CNN by 1.1%
and that SSD512 is 3.6% better. If we take models trained on COCO trainval35k
as described in Sec. 3.4 and ﬁne-tuning them on the 07+12 dataset with SSD512, we
achieve the best results: 81.6% mAP.

×

To understand the performance of our two SSD models in more details, we used the
detection analysis tool from [21]. Figure 3 shows that SSD can detect various object
categories with high quality (large white area). The majority of its conﬁdent detections
are correct. The recall is around 85-90%, and is much higher with “weak” (0.1 jaccard
overlap) criteria. Compared to R-CNN [22], SSD has less localization error, indicating
that SSD can localize objects better because it directly learns to regress the object shape
and classify object categories instead of using two decoupled steps. However, SSD has
more confusions with similar object categories (especially for animals), partly because
we share locations for multiple categories. Figure 4 shows that SSD is very sensitive
to the bounding box size. In other words, it has much worse performance on smaller

3 For SSD512 model, we add extra conv12 2 for prediction, set smin to 0.15, and 0.07 on conv4 3.

8

Liu et al.

tv

cat

car

bird

bike

train

chair

boat bottle bus

07
07+12

07
07+12

data
07
07+12
07
07+12

cow table dog horse mbike person plant sheep sofa

mAP aero
Method
66.9 74.5 78.3 69.2 53.2 36.6 77.3 78.2 82.0 40.7 72.7 67.9 79.6 79.2 73.0 69.0 30.1 65.4 70.2 75.8 65.8
Fast [6]
70.0 77.0 78.1 69.3 59.4 38.3 81.6 78.6 86.7 42.8 78.8 68.9 84.7 82.0 76.6 69.9 31.8 70.1 74.8 80.4 70.4
Fast [6]
69.9 70.0 80.6 70.1 57.3 49.9 78.2 80.4 82.0 52.2 75.3 67.2 80.3 79.8 75.0 76.3 39.1 68.3 67.3 81.1 67.6
Faster [2]
73.2 76.5 79.0 70.9 65.5 52.1 83.1 84.7 86.4 52.0 81.9 65.7 84.8 84.6 77.5 76.7 38.8 73.6 73.9 83.0 72.6
Faster [2]
Faster [2] 07+12+COCO 78.8 84.3 82.0 77.7 68.9 65.7 88.1 88.4 88.9 63.6 86.3 70.8 85.9 87.6 80.1 82.3 53.6 80.4 75.8 86.6 78.9
68.0 73.4 77.5 64.1 59.0 38.9 75.2 80.8 78.5 46.0 67.8 69.2 76.6 82.1 77.0 72.5 41.2 64.2 69.1 78.0 68.5
SSD300
74.3 75.5 80.2 72.3 66.3 47.6 83.0 84.2 86.1 54.7 78.3 73.9 84.5 85.3 82.6 76.2 48.6 73.9 76.0 83.4 74.0
SSD300
07+12+COCO 79.6 80.9 86.3 79.0 76.2 57.6 87.3 88.2 88.6 60.5 85.4 76.7 87.5 89.2 84.5 81.4 55.0 81.9 81.5 85.9 78.9
SSD300
71.6 75.1 81.4 69.8 60.8 46.3 82.6 84.7 84.1 48.5 75.0 67.4 82.3 83.9 79.4 76.6 44.9 69.9 69.1 78.1 71.8
SSD512
76.8 82.4 84.7 78.4 73.8 53.2 86.2 87.5 86.0 57.8 83.1 70.2 84.9 85.2 83.9 79.7 50.3 77.9 73.9 82.5 75.3
SSD512
07+12+COCO 81.6 86.6 88.3 82.4 76.0 66.3 88.6 88.9 89.1 65.1 88.4 73.6 86.5 88.9 85.3 84.6 59.1 85.0 80.4 87.4 81.2
SSD512
Table 1: PASCAL VOC2007 test detection results. Both Fast and Faster R-CNN
use input images whose minimum dimension is 600. The two SSD models have exactly
512). It
the same settings except that they have different input sizes (300
is obvious that larger input size leads to better results, and more data always helps. Data:
”07”: VOC2007 trainval, ”07+12”: union of VOC2007 and VOC2012 trainval.
”07+12+COCO”: ﬁrst train on COCO trainval35k then ﬁne-tune on 07+12.

300 vs. 512

×

×

300 to 512

objects than bigger objects. This is not surprising because those small objects may not
even have any information at the very top layers. Increasing the input size (e.g. from
512) can help improve detecting small objects, but there is still a lot
300
of room to improve. On the positive side, we can clearly see that SSD performs really
well on large objects. And it is very robust to different object aspect ratios because we
use default boxes of various aspect ratios per feature map location.

×

×

3.2 Model analysis

To understand SSD better, we carried out controlled experiments to examine how each
component affects performance. For all the experiments, we use the same settings and
input size (300

300), except for speciﬁed changes to the settings or component(s).

×

more data augmentation?

(cid:52)
(cid:52)
(cid:52)
(cid:52)
VOC2007 test mAP 65.5 71.6 73.7 74.2 74.3
Table 2: Effects of various design choices and components on SSD performance.

2 , 2} box? (cid:52)
3 , 3} box? (cid:52)
use atrous? (cid:52)

include { 1
include { 1

(cid:52)
(cid:52)
(cid:52)

(cid:52)

(cid:52)

(cid:52)

SSD300
(cid:52)
(cid:52)

Data augmentation is crucial. Fast and Faster R-CNN use the original image and the
horizontal ﬂip to train. We use a more extensive sampling strategy, similar to YOLO [5].
Table 2 shows that we can improve 8.8% mAP with this sampling strategy. We do not
know how much our sampling strategy will beneﬁt Fast and Faster R-CNN, but they are
likely to beneﬁt less because they use a feature pooling step during classiﬁcation that is
relatively robust to object translation by design.

SSD: Single Shot MultiBox Detector

9

Fig. 3: Visualization of performance for SSD512 on animals, vehicles, and furni-
ture from VOC2007 test. The top row shows the cumulative fraction of detections
that are correct (Cor) or false positive due to poor localization (Loc), confusion with
similar categories (Sim), with others (Oth), or with background (BG). The solid red
line reﬂects the change of recall with strong criteria (0.5 jaccard overlap) as the num-
ber of detections increases. The dashed red line is using the weak criteria (0.1 jaccard
overlap). The bottom row shows the distribution of top-ranked false positive types.

Fig. 4: Sensitivity and impact of different object characteristics on VOC2007 test
set using [21]. The plot on the left shows the effects of BBox Area per category, and
the right plot shows the effect of Aspect Ratio. Key: BBox Area: XS=extra-small;
S=small; M=medium; L=large; XL =extra-large. Aspect Ratio: XT=extra-tall/narrow;
T=tall; M=medium; W=wide; XW =extra-wide.

animalstotal detections (x 357)0.1250.250.51248percentage of each type020406080100CorLocSimOthBGvehiclestotal detections (x 415)0.1250.250.51248percentage of each type020406080100CorLocSimOthBGfurnituretotal detections (x 400)0.1250.250.51248percentage of each type020406080100CorLocSimOthBGanimalstotal false positives255010020040080016003200percentage of each type020406080100LocSimOthBGvehiclestotal false positives255010020040080016003200percentage of each type020406080100LocSimOthBGfurnituretotal false positives255010020040080016003200percentage of each type020406080100LocSimOthBGXSSMLXLXSSMLXLXSSMLXLXSSMLXLXSSMLXLXSSMLXLXSSMLXL00.20.40.60.810.280.780.840.980.930.490.880.880.910.980.170.670.820.920.960.370.640.770.910.940.470.880.950.990.990.090.360.670.700.540.220.700.940.990.99airplanebicyclebirdboatcatchairtableSSD300: BBox AreaXTTMWXWXTTMWXWXTTMWXWXTTMWXWXTTMWXWXTTMWXWXTTMWXW00.20.40.60.810.770.730.860.810.900.680.870.920.890.760.650.790.780.840.720.710.830.750.730.760.870.880.940.910.870.460.570.560.580.460.670.870.870.920.76airplanebicyclebirdboatcatchairtableSSD300: Aspect RatioXSSMLXLXSSMLXLXSSMLXLXSSMLXLXSSMLXLXSSMLXLXSSMLXL00.20.40.60.810.690.870.870.980.960.680.960.910.910.970.470.830.820.920.990.600.790.810.920.970.530.930.950.990.980.290.470.690.700.380.230.680.910.970.93airplanebicyclebirdboatcatchairtableSSD512: BBox AreaXTTMWXWXTTMWXWXTTMWXWXTTMWXWXTTMWXWXTTMWXWXTTMWXW00.20.40.60.810.910.830.900.890.900.830.930.960.880.770.750.860.850.850.770.950.840.820.820.720.900.900.950.930.850.540.620.630.570.420.590.880.840.860.76airplanebicyclebirdboatcatchairtableSSD512: Aspect Ratio10

Liu et al.

More default box shapes is better. As described in Sec. 2.2, by default we use 6
default boxes per location. If we remove the boxes with 1
3 and 3 aspect ratios, the
performance drops by 0.6%. By further removing the boxes with 1
2 and 2 aspect ratios,
the performance drops another 2.1%. Using a variety of default box shapes seems to
make the task of predicting boxes easier for the network.

Atrous is faster. As described in Sec. 3, we used the atrous version of a subsampled
VGG16, following DeepLab-LargeFOV [17]. If we use the full VGG16, keeping pool5
s2 and not subsampling parameters from fc6 and fc7, and add conv5 3 for
with 2
prediction, the result is about the same while the speed is about 20% slower.

×

−

2

Prediction source layers from:

conv4 3 conv7 conv8 2 conv9 2 conv10 2 conv11 2

(cid:52)
(cid:52)
(cid:52)
(cid:52)

(cid:52)
(cid:52)
(cid:52)

(cid:52)
(cid:52)

(cid:52)

(cid:52)
(cid:52)
(cid:52)
(cid:52)
(cid:52)

(cid:52)
(cid:52)
(cid:52)
(cid:52)
(cid:52)
(cid:52)

mAP

use boundary boxes? # Boxes
No
63.4
63.1
68.4
69.2
64.4
64.0

Yes
74.3
74.6
73.8
70.7
64.2
62.4

8732
8764
8942
9864
9025
8664

Table 3: Effects of using multiple output layers.

Multiple output layers at different resolutions is better. A major contribution of
SSD is using default boxes of different scales on different output layers. To measure
the advantage gained, we progressively remove layers and compare results. For a fair
comparison, every time we remove a layer, we adjust the default box tiling to keep the
total number of boxes similar to the original (8732). This is done by stacking more
scales of boxes on remaining layers and adjusting scales of boxes if needed. We do not
exhaustively optimize the tiling for each setting. Table 3 shows a decrease in accuracy
with fewer layers, dropping monotonically from 74.3 to 62.4. When we stack boxes of
multiple scales on a layer, many are on the image boundary and need to be handled
carefully. We tried the strategy used in Faster R-CNN [2], ignoring boxes which are
on the boundary. We observe some interesting trends. For example, it hurts the perfor-
mance by a large margin if we use very coarse feature maps (e.g. conv11 2 (1
1)
or conv10 2 (3
3)). The reason might be that we do not have enough large boxes to
cover large objects after the pruning. When we use primarily ﬁner resolution maps, the
performance starts increasing again because even after pruning a sufﬁcient number of
large boxes remains. If we only use conv7 for prediction, the performance is the worst,
reinforcing the message that it is critical to spread boxes of different scales over dif-
ferent layers. Besides, since our predictions do not rely on ROI pooling as in [6], we
do not have the collapsing bins problem in low-resolution feature maps [23]. The SSD
architecture combines predictions from feature maps of various resolutions to achieve
comparable accuracy to Faster R-CNN, while using lower resolution input images.

×

×

SSD: Single Shot MultiBox Detector

11

3.3 PASCAL VOC2012

We use the same settings as those used for our basic VOC2007 experiments above,
except that we use VOC2012 trainval and VOC2007 trainval and test (21503
images) for training, and test on VOC2012 test (10991 images). We train the models
with 10−3 learning rate for 60k iterations, then 10−4 for 20k iterations. Table 4 shows
the results of our SSD300 and SSD5124 model. We see the same performance trend
as we observed on VOC2007 test. Our SSD300 improves accuracy over Fast/Faster R-
CNN. By increasing the training and testing image size to 512
512, we are 4.5% more
accurate than Faster R-CNN. Compared to YOLO, SSD is signiﬁcantly more accurate,
likely due to the use of convolutional default boxes from multiple feature maps and our
matching strategy during training. When ﬁne-tuned from models trained on COCO, our
SSD512 achieves 80.0% mAP, which is 4.1% higher than Faster R-CNN.

×

cat

car

bird

data
07++12
07++12

boat bottle bus

07++12
07++12

chair cow table dog horse mbike person plant sheep sofa

mAP aero bike
Method
68.4 82.3 78.4 70.8 52.3 38.7 77.8 71.6 89.3 44.2 73.0 55.0 87.5 80.5 80.8 72.0 35.1 68.3 65.7 80.4 64.2
Fast[6]
70.4 84.9 79.8 74.3 53.9 49.8 77.5 75.9 88.5 45.6 77.1 55.3 86.9 81.7 80.9 79.6 40.1 72.6 60.9 81.2 61.5
Faster[2]
Faster[2] 07++12+COCO 75.9 87.4 83.6 76.8 62.9 59.6 81.9 82.0 91.3 54.9 82.6 59.0 89.0 85.5 84.7 84.1 52.2 78.9 65.5 85.4 70.2
57.9 77.0 67.2 57.7 38.3 22.7 68.3 55.9 81.4 36.2 60.8 48.5 77.2 72.3 71.3 63.5 28.9 52.2 54.8 73.9 50.8
YOLO[5]
72.4 85.6 80.1 70.5 57.6 46.2 79.4 76.1 89.2 53.0 77.0 60.8 87.0 83.1 82.3 79.4 45.9 75.9 69.5 81.9 67.5
SSD300
07++12+COCO 77.5 90.2 83.3 76.3 63.0 53.6 83.8 82.8 92.0 59.7 82.7 63.5 89.3 87.6 85.9 84.3 52.6 82.5 74.1 88.4 74.2
SSD300
74.9 87.4 82.3 75.8 59.0 52.6 81.7 81.5 90.0 55.4 79.0 59.8 88.4 84.3 84.7 83.3 50.2 78.0 66.3 86.3 72.0
SSD512
07++12+COCO 80.0 90.7 86.8 80.5 67.8 60.8 86.3 85.5 93.5 63.2 85.7 64.4 90.9 89.0 88.9 86.8 57.2 85.1 72.8 88.4 75.9
SSD512
Table 4: PASCAL VOC2012 test detection results. Fast and Faster R-CNN use
images with minimum dimension 600, while the image size for YOLO is 448
448.
data: ”07++12”: union of VOC2007 trainval and test and VOC2012 trainval.
”07++12+COCO”: ﬁrst train on COCO trainval35k then ﬁne-tune on 07++12.

07++12

×

train

tv

3.4 COCO

To further validate the SSD framework, we trained our SSD300 and SSD512 architec-
tures on the COCO dataset. Since objects in COCO tend to be smaller than PASCAL
VOC, we use smaller default boxes for all layers. We follow the strategy mentioned in
Sec. 2.2, but now our smallest default box has a scale of 0.15 instead of 0.2, and the
scale of the default box on conv4 3 is 0.07 (e.g. 21 pixels for a 300

300 image)5.

We use the trainval35k [24] for training. We ﬁrst train the model with 10−3
learning rate for 160k iterations, and then continue training for 40k iterations with
10−4 and 40k iterations with 10−5. Table 5 shows the results on test-dev2015.
Similar to what we observed on the PASCAL VOC dataset, SSD300 is better than Fast
R-CNN in both mAP@0.5 and mAP@[0.5:0.95]. SSD300 has a similar mAP@0.75 as
ION [24] and Faster R-CNN [25], but is worse in mAP@0.5. By increasing the im-
age size to 512
512, our SSD512 is better than Faster R-CNN [25] in both criteria.
Interestingly, we observe that SSD512 is 5.3% better in mAP@0.75, but is only 1.2%
better in mAP@0.5. We also observe that it has much better AP (4.8%) and AR (4.6%)
for large objects, but has relatively less improvement in AP (1.3%) and AR (2.0%) for

×

×

4
http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?cls=mean&challengeid=11&compid=4
5 For SSD512 model, we add extra conv12 2 for prediction, set smin to 0.1, and 0.04 on conv4 3.

12

Liu et al.

Method

data

Fast [6]
Fast [24]
Faster [2]
ION [24]
Faster [25]
SSD300
SSD512

train
train
trainval
train
trainval
trainval35k
trainval35k

19.7
20.5
21.9
23.6
24.2
23.2
26.8

Avg. Precision, IoU:
0.5:0.95 0.5 0.75

-

100
-

S M L
-
-
-

Avg. Precision, Area: Avg. Recall, #Dets: Avg. Recall, Area:
S M L
10
-
-
-
-

1
35.9
-
39.9 19.4 4.1 20.0 35.8 21.3 29.5 30.1 7.3 32.1 52.0
42.7
-
43.2 23.6 6.4 24.1 38.3 23.2 32.7 33.5 10.1 37.7 53.6
45.3 23.5 7.7 26.4 37.1 23.8 34.0 34.6 12.0 38.5 54.4
41.2 23.4 5.3 23.2 39.6 22.5 33.2 35.3 9.6 37.6 56.5
46.5 27.8 9.0 28.9 41.9 24.8 37.5 39.8 14.0 43.5 59.0

-

-

-

-

-

-

-

-

-

Table 5: COCO test-dev2015 detection results.

small objects. Compared to ION, the improvement in AR for large and small objects is
more similar (5.4% vs. 3.9%). We conjecture that Faster R-CNN is more competitive
on smaller objects with SSD because it performs two box reﬁnement steps, in both the
RPN part and in the Fast R-CNN part. In Fig. 5, we show some detection examples on
COCO test-dev with the SSD512 model.

3.5 Preliminary ILSVRC results

We applied the same network architecture we used for COCO to the ILSVRC DET
dataset [16]. We train a SSD300 model using the ILSVRC2014 DET train and val1
as used in [22]. We ﬁrst train the model with 10−3 learning rate for 320k iterations, and
then continue training for 80k iterations with 10−4 and 40k iterations with 10−5. We
can achieve 43.4 mAP on the val2 set [22]. Again, it validates that SSD is a general
framework for high quality real-time detection.

3.6 Data Augmentation for Small Object Accuracy

Without a follow-up feature resampling step as in Faster R-CNN, the classiﬁcation task
for small objects is relatively hard for SSD, as demonstrated in our analysis (see Fig. 4).
The data augmentation strategy described in Sec. 2.2 helps to improve the performance
dramatically, especially on small datasets such as PASCAL VOC. The random crops
generated by the strategy can be thought of as a ”zoom in” operation and can generate
many larger training examples. To implement a ”zoom out” operation that creates more
of the
small training examples, we ﬁrst randomly place an image on a canvas of 16
original image size ﬁlled with mean values before we do any random crop operation.
Because we have more training images by introducing this new ”expansion” data aug-
mentation trick, we have to double the training iterations. We have seen a consistent
increase of 2%-3% mAP across multiple datasets, as shown in Table 6. In speciﬁc, Fig-
ure 6 shows that the new augmentation trick signiﬁcantly improves the performance on
small objects. This result underscores the importance of the data augmentation strategy
for the ﬁnal model accuracy.

×

An alternative way of improving SSD is to design a better tiling of default boxes so
that its position and scale are better aligned with the receptive ﬁeld of each position on
a feature map. We leave this for future work.

SSD: Single Shot MultiBox Detector

13

Fig. 5: Detection examples on COCO test-dev with SSD512 model. We show
detections with scores higher than 0.6. Each color corresponds to an object category.

VOC2007 test

VOC2012 test

Method

07+12 07+12+COCO 07++12 07++12+COCO

SSD300
SSD512
SSD300*
SSD512*

0.5
74.3
76.8
77.2
79.8

0.5
79.6
81.6
81.2
83.2

0.5
72.4
74.9
75.8
78.5

0.5
77.5
80.0
79.3
82.2

COCO test-dev2015
trainval35k

0.5:0.95
23.2
26.8
25.1
28.8

0.5
41.2
46.5
43.1
48.5

0.75
23.4
27.8
25.8
30.3

Table 6: Results on multiple datasets when we add the image expansion data aug-
mentation trick. SSD300* and SSD512* are the models that are trained with the new
data augmentation.

cup: 0.91bowl: 0.87pizza: 0.96person: 0.98cow: 1.00cow: 0.94cow: 0.90cup: 0.70chair: 0.92chair: 0.87chair: 0.80chair: 0.75couch: 0.87dining table: 0.85tv: 0.94tv: 0.77person: 1.00person: 0.99person: 0.87person: 0.82frisbee: 0.90person: 1.00bicycle: 0.94traffic light: 0.71person: 0.92chair: 0.74mouse: 0.63keyboard: 0.82cup: 0.98bowl: 0.98bowl: 0.97bowl: 0.81sandwich: 0.99dining table: 0.86person: 1.00motorcycle: 0.99backpack: 0.82person: 1.00person: 0.90baseball glove: 0.62person: 0.93person: 0.88car: 0.99car: 0.96car: 0.83umbrella: 0.86cat: 0.99cup: 0.92tv: 0.89laptop: 0.99keyboard: 0.99book: 0.90bicycle: 0.84bus: 0.98bus: 0.94bus: 0.74person: 1.00person: 0.98person: 0.98skateboard: 0.97cup: 0.99cup: 0.81cake: 0.86cake: 0.83dining table: 0.95person: 0.86person: 0.82person: 0.81boat: 0.97person: 1.00person: 0.98sports ball: 0.67baseball bat: 0.99baseball glove: 0.92person: 0.98car: 0.98car: 0.86car: 0.83car: 0.80car: 0.63fire hydrant: 0.98person: 1.00person: 1.00person: 0.98person: 0.84person: 0.83bench: 0.84umbrella: 0.99umbrella: 0.95umbrella: 0.92person: 1.00car: 0.99stop sign: 0.72person: 0.88dog: 0.99frisbee: 0.93person: 0.98person: 0.97person: 0.95person: 0.94person: 0.90person: 0.74bowl: 0.88chair: 0.84dining table: 0.92person: 0.89car: 1.00car: 1.00car: 0.98motorcycle: 0.88person: 0.99person: 0.99tennis racket: 0.97chair: 0.80chair: 0.77chair: 0.72chair: 0.72chair: 0.66person: 0.86bench: 0.73horse: 0.98person: 0.94bottle: 0.97bottle: 0.97cup: 0.99cup: 0.60fork: 0.71sandwich: 0.89dining table: 0.86person: 0.99person: 0.99person: 0.94person: 0.84person: 0.67elephant: 0.98elephant: 0.89elephant: 0.69car: 0.85car: 0.83car: 0.71car: 0.69tv: 0.88laptop: 0.99laptop: 0.99keyboard: 0.63keyboard: 0.63person: 0.82dog: 0.67cup: 0.96couch: 0.70person: 1.00person: 0.96baseball bat: 0.80baseball bat: 0.68bowl: 0.71bowl: 0.65chair: 0.92dining table: 0.80vase: 0.79person: 1.00person: 1.00person: 0.85bicycle: 0.98bicycle: 0.98bicycle: 0.86backpack: 0.72person: 0.99person: 0.93person: 0.86person: 0.83truck: 0.60dog: 1.00chair: 0.65dining table: 0.87vase: 1.00person: 1.00person: 0.95person: 0.94skateboard: 0.87person: 0.99car: 0.99car: 0.98car: 0.97car: 0.92car: 0.91car: 0.80car: 0.78car: 0.74car: 0.72truck: 0.9614

Liu et al.

Fig. 6: Sensitivity and impact of object size with new data augmentation on
VOC2007 test set using [21]. The top row shows the effects of BBox Area per cat-
egory for the original SSD300 and SSD512 model, and the bottom row corresponds to
the SSD300* and SSD512* model trained with the new data augmentation trick. It is
obvious that the new data augmentation trick helps detecting small objects signiﬁcantly.

3.7

Inference time

Considering the large number of boxes generated from our method, it is essential to
perform non-maximum suppression (nms) efﬁciently during inference. By using a con-
ﬁdence threshold of 0.01, we can ﬁlter out most boxes. We then apply nms with jaccard
overlap of 0.45 per class and keep the top 200 detections per image. This step costs
about 1.7 msec per image for SSD300 and 20 VOC classes, which is close to the total
time (2.4 msec) spent on all newly added layers. We measure the speed with batch size
8 using Titan X and cuDNN v4 with Intel Xeon E5-2667v3@3.20GHz.

Table 7 shows the comparison between SSD, Faster R-CNN[2], and YOLO[5]. Both
our SSD300 and SSD512 method outperforms Faster R-CNN in both speed and accu-
racy. Although Fast YOLO[5] can run at 155 FPS, it has lower accuracy by almost 22%
mAP. To the best of our knowledge, SSD300 is the ﬁrst real-time method to achieve
above 70% mAP. Note that about 80% of the forward time is spent on the base network
(VGG16 in our case). Therefore, using a faster base network could even further improve
the speed, which can possibly make the SSD512 model real-time as well.

4 Related Work

There are two established classes of methods for object detection in images, one based
on sliding windows and the other based on region proposal classiﬁcation. Before the
advent of convolutional neural networks, the state of the art for those two approaches
– Deformable Part Model (DPM) [26] and Selective Search [1] – had comparable
performance. However, after the dramatic improvement brought on by R-CNN [22],
which combines selective search region proposals and convolutional network based
post-classiﬁcation, region proposal object detection methods became prevalent.

The original R-CNN approach has been improved in a variety of ways. The ﬁrst
set of approaches improve the quality and speed of post-classiﬁcation, since it requires

XSSMLXLXSSMLXLXSSMLXLXSSMLXLXSSMLXLXSSMLXLXSSMLXL00.20.40.60.810.280.780.840.980.930.490.880.880.910.980.170.670.820.920.960.370.640.770.910.940.470.880.950.990.990.090.360.670.700.540.220.700.940.990.99airplanebicyclebirdboatcatchairtableSSD300: BBox AreaXSSMLXLXSSMLXLXSSMLXLXSSMLXLXSSMLXLXSSMLXLXSSMLXL00.20.40.60.810.690.870.870.980.960.680.960.910.910.970.470.830.820.920.990.600.790.810.920.970.530.930.950.990.980.290.470.690.700.380.230.680.910.970.93airplanebicyclebirdboatcatchairtableSSD512: BBox AreaXSSMLXLXSSMLXLXSSMLXLXSSMLXLXSSMLXLXSSMLXLXSSMLXL00.20.40.60.810.400.860.870.980.940.670.940.890.930.970.380.750.860.900.960.590.710.790.880.990.690.900.961.000.970.250.460.720.730.510.430.750.920.990.99airplanebicyclebirdboatcatchairtableSSD300*: BBox AreaXSSMLXLXSSMLXLXSSMLXLXSSMLXLXSSMLXLXSSMLXLXSSMLXL00.20.40.60.810.850.900.870.990.920.750.950.900.930.970.620.870.860.920.990.770.760.780.920.970.730.890.980.990.980.420.580.720.740.450.320.700.940.970.98airplanebicyclebirdboatcatchairtableSSD512*: BBox AreaSSD: Single Shot MultiBox Detector

15

Method
Faster R-CNN (VGG16)
Fast YOLO
YOLO (VGG16)
SSD300
SSD512
SSD300
SSD512

mAP
73.2
52.7
66.4
74.3
76.8
74.3
76.8

FPS
7
155
21
46
19
59
22

batch size
1
1
1
1
1
8
8

# Boxes
Input resolution
∼ 6000 ∼ 1000 × 600

98
98
8732
24564
8732
24564

448 × 448
448 × 448
300 × 300
512 × 512
300 × 300
512 × 512

Table 7: Results on Pascal VOC2007 test. SSD300 is the only real-time detection
method that can achieve above 70% mAP. By using a larger input image, SSD512 out-
performs all methods on accuracy while maintaining a close to real-time speed.

the classiﬁcation of thousands of image crops, which is expensive and time-consuming.
SPPnet [9] speeds up the original R-CNN approach signiﬁcantly. It introduces a spatial
pyramid pooling layer that is more robust to region size and scale and allows the classi-
ﬁcation layers to reuse features computed over feature maps generated at several image
resolutions. Fast R-CNN [6] extends SPPnet so that it can ﬁne-tune all layers end-to-
end by minimizing a loss for both conﬁdences and bounding box regression, which was
ﬁrst introduced in MultiBox [7] for learning objectness.

The second set of approaches improve the quality of proposal generation using deep
neural networks. In the most recent works like MultiBox [7,8], the Selective Search
region proposals, which are based on low-level image features, are replaced by pro-
posals generated directly from a separate deep neural network. This further improves
the detection accuracy but results in a somewhat complex setup, requiring the training
of two neural networks with a dependency between them. Faster R-CNN [2] replaces
selective search proposals by ones learned from a region proposal network (RPN), and
introduces a method to integrate the RPN with Fast R-CNN by alternating between ﬁne-
tuning shared convolutional layers and prediction layers for these two networks. This
way region proposals are used to pool mid-level features and the ﬁnal classiﬁcation
step is less expensive. Our SSD is very similar to the region proposal network (RPN) in
Faster R-CNN in that we also use a ﬁxed set of (default) boxes for prediction, similar
to the anchor boxes in the RPN. But instead of using these to pool features and evaluate
another classiﬁer, we simultaneously produce a score for each object category in each
box. Thus, our approach avoids the complication of merging RPN with Fast R-CNN
and is easier to train, faster, and straightforward to integrate in other tasks.

Another set of methods, which are directly related to our approach, skip the proposal
step altogether and predict bounding boxes and conﬁdences for multiple categories di-
rectly. OverFeat [4], a deep version of the sliding window method, predicts a bounding
box directly from each location of the topmost feature map after knowing the conﬁ-
dences of the underlying object categories. YOLO [5] uses the whole topmost feature
map to predict both conﬁdences for multiple categories and bounding boxes (which
are shared for these categories). Our SSD method falls in this category because we do
not have the proposal step but use the default boxes. However, our approach is more
ﬂexible than the existing methods because we can use default boxes of different aspect

16

Liu et al.

ratios on each feature location from multiple feature maps at different scales. If we only
use one default box per location from the topmost feature map, our SSD would have
similar architecture to OverFeat [4]; if we use the whole topmost feature map and add a
fully connected layer for predictions instead of our convolutional predictors, and do not
explicitly consider multiple aspect ratios, we can approximately reproduce YOLO [5].

5 Conclusions

This paper introduces SSD, a fast single-shot object detector for multiple categories. A
key feature of our model is the use of multi-scale convolutional bounding box outputs
attached to multiple feature maps at the top of the network. This representation allows
us to efﬁciently model the space of possible box shapes. We experimentally validate
that given appropriate training strategies, a larger number of carefully chosen default
bounding boxes results in improved performance. We build SSD models with at least an
order of magnitude more box predictions sampling location, scale, and aspect ratio, than
existing methods [5,7]. We demonstrate that given the same VGG-16 base architecture,
SSD compares favorably to its state-of-the-art object detector counterparts in terms of
both accuracy and speed. Our SSD512 model signiﬁcantly outperforms the state-of-the-
art Faster R-CNN [2] in terms of accuracy on PASCAL VOC and COCO, while being
faster. Our real time SSD300 model runs at 59 FPS, which is faster than the current
3
×
real time YOLO [5] alternative, while producing markedly superior detection accuracy.
Apart from its standalone utility, we believe that our monolithic and relatively sim-
ple SSD model provides a useful building block for larger systems that employ an object
detection component. A promising future direction is to explore its use as part of a sys-
tem using recurrent neural networks to detect and track objects in video simultaneously.

6 Acknowledgment

This work was started as an internship project at Google and continued at UNC. We
would like to thank Alex Toshev for helpful discussions and are indebted to the Im-
age Understanding and DistBelief teams at Google. We also thank Philip Ammirato
and Patrick Poirson for helpful comments. We thank NVIDIA for providing GPUs and
acknowledge support from NSF 1452851, 1446631, 1526367, 1533771.

References

1. Uijlings, J.R., van de Sande, K.E., Gevers, T., Smeulders, A.W.: Selective search for object

recognition. IJCV (2013)

2. Ren, S., He, K., Girshick, R., Sun, J.: Faster R-CNN: Towards real-time object detection

with region proposal networks. In: NIPS. (2015)

3. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR.

(2016)

4. Sermanet, P., Eigen, D., Zhang, X., Mathieu, M., Fergus, R., LeCun, Y.: Overfeat: Integrated
recognition, localization and detection using convolutional networks. In: ICLR. (2014)

SSD: Single Shot MultiBox Detector

17

5. Redmon, J., Divvala, S., Girshick, R., Farhadi, A.: You only look once: Uniﬁed, real-time

object detection. In: CVPR. (2016)

6. Girshick, R.: Fast R-CNN. In: ICCV. (2015)
7. Erhan, D., Szegedy, C., Toshev, A., Anguelov, D.: Scalable object detection using deep

neural networks. In: CVPR. (2014)

8. Szegedy, C., Reed, S., Erhan, D., Anguelov, D.: Scalable, high-quality object detection.

arXiv preprint arXiv:1412.1441 v3 (2015)

9. He, K., Zhang, X., Ren, S., Sun, J.: Spatial pyramid pooling in deep convolutional networks

for visual recognition. In: ECCV. (2014)

10. Long, J., Shelhamer, E., Darrell, T.: Fully convolutional networks for semantic segmentation.

In: CVPR. (2015)

11. Hariharan, B., Arbel´aez, P., Girshick, R., Malik, J.: Hypercolumns for object segmentation

and ﬁne-grained localization. In: CVPR. (2015)

12. Liu, W., Rabinovich, A., Berg, A.C.: ParseNet: Looking wider to see better. In: ILCR. (2016)
13. Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., Torralba, A.: Object detectors emerge in deep

scene cnns. In: ICLR. (2015)

14. Howard, A.G.: Some improvements on deep convolutional neural network based image

classiﬁcation. arXiv preprint arXiv:1312.5402 (2013)

15. Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recog-

nition. In: NIPS. (2015)

16. Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A.,
Khosla, A., Bernstein, M., Berg, A.C., Fei-Fei, L.: Imagenet large scale visual recognition
challenge. IJCV (2015)

17. Chen, L.C., Papandreou, G., Kokkinos, I., Murphy, K., Yuille, A.L.: Semantic image seg-

mentation with deep convolutional nets and fully connected crfs. In: ICLR. (2015)

18. Holschneider, M., Kronland-Martinet, R., Morlet, J., Tchamitchian, P.: A real-time algorithm
for signal analysis with the help of the wavelet transform. In: Wavelets. Springer (1990)
286–297

19. Jia, Y., Shelhamer, E., Donahue, J., Karayev, S., Long, J., Girshick, R., Guadarrama, S.,
Darrell, T.: Caffe: Convolutional architecture for fast feature embedding. In: MM. (2014)
20. Glorot, X., Bengio, Y.: Understanding the difﬁculty of training deep feedforward neural

networks. In: AISTATS. (2010)

21. Hoiem, D., Chodpathumwan, Y., Dai, Q.: Diagnosing error in object detectors. In: ECCV

2012. (2012)

22. Girshick, R., Donahue, J., Darrell, T., Malik, J.: Rich feature hierarchies for accurate object

detection and semantic segmentation. In: CVPR. (2014)

23. Zhang, L., Lin, L., Liang, X., He, K.: Is faster r-cnn doing well for pedestrian detection. In:

ECCV. (2016)

24. Bell, S., Zitnick, C.L., Bala, K., Girshick, R.: Inside-outside net: Detecting objects in context

with skip pooling and recurrent neural networks. In: CVPR. (2016)

25. COCO:

Common Objects

in Context.

http://mscoco.org/dataset/

#detections-leaderboard (2016) [Online; accessed 25-July-2016].

26. Felzenszwalb, P., McAllester, D., Ramanan, D.: A discriminatively trained, multiscale, de-

formable part model. In: CVPR. (2008)


