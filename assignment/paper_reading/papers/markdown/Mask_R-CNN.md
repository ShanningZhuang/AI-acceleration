Mask R-CNN

Kaiming He Georgia Gkioxari

Piotr Doll´ar Ross Girshick

Facebook AI Research (FAIR)

8
1
0
2

n
a
J

4
2

]

V
C
.
s
c
[

3
v
0
7
8
6
0
.
3
0
7
1
:
v
i
X
r
a

Abstract

We present a conceptually simple, ﬂexible, and general
framework for object instance segmentation. Our approach
efﬁciently detects objects in an image while simultaneously
generating a high-quality segmentation mask for each in-
stance. The method, called Mask R-CNN, extends Faster
R-CNN by adding a branch for predicting an object mask in
parallel with the existing branch for bounding box recogni-
tion. Mask R-CNN is simple to train and adds only a small
overhead to Faster R-CNN, running at 5 fps. Moreover,
Mask R-CNN is easy to generalize to other tasks, e.g., al-
lowing us to estimate human poses in the same framework.
We show top results in all three tracks of the COCO suite
of challenges, including instance segmentation, bounding-
box object detection, and person keypoint detection. With-
out bells and whistles, Mask R-CNN outperforms all ex-
isting, single-model entries on every task, including the
COCO 2016 challenge winners. We hope our simple and
effective approach will serve as a solid baseline and help
ease future research in instance-level recognition. Code
has been made available at: https://github.com/
facebookresearch/Detectron.

1. Introduction

The vision community has rapidly improved object de-
tection and semantic segmentation results over a short pe-
riod of time. In large part, these advances have been driven
by powerful baseline systems, such as the Fast/Faster R-
CNN [12, 36] and Fully Convolutional Network (FCN) [30]
frameworks for object detection and semantic segmenta-
tion, respectively. These methods are conceptually intuitive
and offer ﬂexibility and robustness, together with fast train-
ing and inference time. Our goal in this work is to develop a
comparably enabling framework for instance segmentation.
Instance segmentation is challenging because it requires
the correct detection of all objects in an image while also
precisely segmenting each instance. It therefore combines
elements from the classical computer vision tasks of ob-
ject detection, where the goal is to classify individual ob-
jects and localize each using a bounding box, and semantic

Figure 1. The Mask R-CNN framework for instance segmentation.

segmentation, where the goal is to classify each pixel into
a ﬁxed set of categories without differentiating object in-
stances.1 Given this, one might expect a complex method
is required to achieve good results. However, we show that
a surprisingly simple, ﬂexible, and fast system can surpass
prior state-of-the-art instance segmentation results.

Our method, called Mask R-CNN, extends Faster R-CNN
[36] by adding a branch for predicting segmentation masks
on each Region of Interest (RoI), in parallel with the ex-
isting branch for classiﬁcation and bounding box regres-
sion (Figure 1). The mask branch is a small FCN applied
to each RoI, predicting a segmentation mask in a pixel-to-
pixel manner. Mask R-CNN is simple to implement and
train given the Faster R-CNN framework, which facilitates
a wide range of ﬂexible architecture designs. Additionally,
the mask branch only adds a small computational overhead,
enabling a fast system and rapid experimentation.

In principle Mask R-CNN is an intuitive extension of
Faster R-CNN, yet constructing the mask branch properly
is critical for good results. Most importantly, Faster R-
CNN was not designed for pixel-to-pixel alignment be-
tween network inputs and outputs. This is most evident in
how RoIPool [18, 12], the de facto core operation for at-
tending to instances, performs coarse spatial quantization
for feature extraction. To ﬁx the misalignment, we pro-
pose a simple, quantization-free layer, called RoIAlign, that
faithfully preserves exact spatial locations. Despite being

1Following common terminology, we use object detection to denote
detection via bounding boxes, not masks, and semantic segmentation to
denote per-pixel classiﬁcation without differentiating instances. Yet we
note that instance segmentation is both semantic and a form of detection.

1

RoIAlignRoIAlignclassboxconvconvconvconv

Figure 2. Mask R-CNN results on the COCO test set. These results are based on ResNet-101 [19], achieving a mask AP of 35.7 and
running at 5 fps. Masks are shown in color, and bounding box, category, and conﬁdences are also shown.

a seemingly minor change, RoIAlign has a large impact: it
improves mask accuracy by relative 10% to 50%, showing
bigger gains under stricter localization metrics. Second, we
found it essential to decouple mask and class prediction: we
predict a binary mask for each class independently, without
competition among classes, and rely on the network’s RoI
classiﬁcation branch to predict the category.
In contrast,
FCNs usually perform per-pixel multi-class categorization,
which couples segmentation and classiﬁcation, and based
on our experiments works poorly for instance segmentation.
Without bells and whistles, Mask R-CNN surpasses all
previous state-of-the-art single-model results on the COCO
instance segmentation task [28],
including the heavily-
engineered entries from the 2016 competition winner. As
a by-product, our method also excels on the COCO object
detection task. In ablation experiments, we evaluate multi-
ple basic instantiations, which allows us to demonstrate its
robustness and analyze the effects of core factors.

Our models can run at about 200ms per frame on a GPU,
and training on COCO takes one to two days on a single
8-GPU machine. We believe the fast train and test speeds,
together with the framework’s ﬂexibility and accuracy, will
beneﬁt and ease future research on instance segmentation.

Finally, we showcase the generality of our framework
via the task of human pose estimation on the COCO key-
point dataset [28]. By viewing each keypoint as a one-hot
binary mask, with minimal modiﬁcation Mask R-CNN can
be applied to detect instance-speciﬁc poses. Mask R-CNN
surpasses the winner of the 2016 COCO keypoint compe-
tition, and at the same time runs at 5 fps. Mask R-CNN,
therefore, can be seen more broadly as a ﬂexible framework
for instance-level recognition and can be readily extended
to more complex tasks.

We have released code to facilitate future research.

2. Related Work

R-CNN: The Region-based CNN (R-CNN) approach [13]
to bounding-box object detection is to attend to a manage-
able number of candidate object regions [42, 20] and evalu-
ate convolutional networks [25, 24] independently on each
RoI. R-CNN was extended [18, 12] to allow attending to
RoIs on feature maps using RoIPool, leading to fast speed
and better accuracy. Faster R-CNN [36] advanced this
stream by learning the attention mechanism with a Region
Proposal Network (RPN). Faster R-CNN is ﬂexible and ro-
bust to many follow-up improvements (e.g., [38, 27, 21]),
and is the current leading framework in several benchmarks.

Instance Segmentation: Driven by the effectiveness of R-
CNN, many approaches to instance segmentation are based
on segment proposals. Earlier methods [13, 15, 16, 9] re-
sorted to bottom-up segments [42, 2]. DeepMask [33] and
following works [34, 8] learn to propose segment candi-
dates, which are then classiﬁed by Fast R-CNN. In these
methods, segmentation precedes recognition, which is slow
and less accurate. Likewise, Dai et al. [10] proposed a com-
plex multiple-stage cascade that predicts segment proposals
from bounding-box proposals, followed by classiﬁcation.
Instead, our method is based on parallel prediction of masks
and class labels, which is simpler and more ﬂexible.

Most recently, Li et al. [26] combined the segment pro-
posal system in [8] and object detection system in [11] for
“fully convolutional instance segmentation” (FCIS). The
common idea in [8, 11, 26] is to predict a set of position-
sensitive output channels fully convolutionally.
These
channels simultaneously address object classes, boxes, and
masks, making the system fast. But FCIS exhibits system-
atic errors on overlapping instances and creates spurious
edges (Figure 6), showing that it is challenged by the fun-
damental difﬁculties of segmenting instances.

2

dining table.96person1.00person1.00person1.00person1.00person1.00person1.00person1.00person.94bottle.99bottle.99bottle.99motorcycle1.00motorcycle1.00person1.00person1.00person.96person1.00person.83person.96person.98person.90person.92person.99person.91bus.99person1.00person1.00person1.00backpack.93person1.00person.99person1.00backpack.99person.99person.98person.89person.95person1.00person1.00car1.00traffic light.96person.96truck1.00person.99car.99person.85motorcycle.95car.99car.92person.99person1.00traffic light.92traffic light.84traffic light.95car.93person.87person1.00person1.00umbrella.98umbrella.98backpack1.00handbag.96elephant1.00person1.00person1.00person.99sheep1.00person1.00sheep.99sheep.91sheep1.00sheep.99sheep.99sheep.95person.99sheep1.00sheep.96sheep.99sheep.99sheep.96sheep.96sheep.96sheep.86sheep.82sheep.93dining table.99chair.99chair.90chair.99chair.98chair.96chair.86chair.99bowl.81chair.96tv.99bottle.99wine glass.99wine glass1.00bowl.85knife.83wine glass1.00wine glass.93wine glass.97fork.95Another family of solutions [23, 4, 3, 29] to instance seg-
mentation are driven by the success of semantic segmen-
tation. Starting from per-pixel classiﬁcation results (e.g.,
FCN outputs), these methods attempt to cut the pixels of
the same category into different instances. In contrast to the
segmentation-ﬁrst strategy of these methods, Mask R-CNN
is based on an instance-ﬁrst strategy. We expect a deeper in-
corporation of both strategies will be studied in the future.

3. Mask R-CNN

Mask R-CNN is conceptually simple: Faster R-CNN has
two outputs for each candidate object, a class label and a
bounding-box offset; to this we add a third branch that out-
puts the object mask. Mask R-CNN is thus a natural and in-
tuitive idea. But the additional mask output is distinct from
the class and box outputs, requiring extraction of much ﬁner
spatial layout of an object. Next, we introduce the key ele-
ments of Mask R-CNN, including pixel-to-pixel alignment,
which is the main missing piece of Fast/Faster R-CNN.

Faster R-CNN: We begin by brieﬂy reviewing the Faster
R-CNN detector [36]. Faster R-CNN consists of two stages.
The ﬁrst stage, called a Region Proposal Network (RPN),
proposes candidate object bounding boxes. The second
stage, which is in essence Fast R-CNN [12], extracts fea-
tures using RoIPool from each candidate box and performs
classiﬁcation and bounding-box regression. The features
used by both stages can be shared for faster inference. We
refer readers to [21] for latest, comprehensive comparisons
between Faster R-CNN and other frameworks.

Mask R-CNN: Mask R-CNN adopts the same two-stage
procedure, with an identical ﬁrst stage (which is RPN). In
the second stage, in parallel to predicting the class and box
offset, Mask R-CNN also outputs a binary mask for each
RoI. This is in contrast to most recent systems, where clas-
siﬁcation depends on mask predictions (e.g. [33, 10, 26]).
Our approach follows the spirit of Fast R-CNN [12] that
applies bounding-box classiﬁcation and regression in par-
allel (which turned out to largely simplify the multi-stage
pipeline of original R-CNN [13]).

Formally, during training, we deﬁne a multi-task loss on
each sampled RoI as L = Lcls + Lbox + Lmask. The clas-
siﬁcation loss Lcls and bounding-box loss Lbox are identi-
cal as those deﬁned in [12]. The mask branch has a Km2-
dimensional output for each RoI, which encodes K binary
masks of resolution m × m, one for each of the K classes.
To this we apply a per-pixel sigmoid, and deﬁne Lmask as
the average binary cross-entropy loss. For an RoI associated
with ground-truth class k, Lmask is only deﬁned on the k-th
mask (other mask outputs do not contribute to the loss).

Our deﬁnition of Lmask allows the network to generate
masks for every class without competition among classes;
we rely on the dedicated classiﬁcation branch to predict the

Figure 3. RoIAlign: The dashed grid rep-
resents a feature map, the solid lines an RoI
(with 2×2 bins in this example), and the dots
the 4 sampling points in each bin. RoIAlign
computes the value of each sampling point
by bilinear interpolation from the nearby grid
points on the feature map. No quantization is
performed on any coordinates involved in the
RoI, its bins, or the sampling points.

class label used to select the output mask. This decouples
mask and class prediction. This is different from common
practice when applying FCNs [30] to semantic segmenta-
tion, which typically uses a per-pixel softmax and a multino-
mial cross-entropy loss. In that case, masks across classes
compete; in our case, with a per-pixel sigmoid and a binary
loss, they do not. We show by experiments that this formu-
lation is key for good instance segmentation results.

Mask Representation: A mask encodes an input object’s
spatial layout. Thus, unlike class labels or box offsets
that are inevitably collapsed into short output vectors by
fully-connected (fc) layers, extracting the spatial structure
of masks can be addressed naturally by the pixel-to-pixel
correspondence provided by convolutions.

Speciﬁcally, we predict an m × m mask from each RoI
using an FCN [30]. This allows each layer in the mask
branch to maintain the explicit m × m object spatial lay-
out without collapsing it into a vector representation that
lacks spatial dimensions. Unlike previous methods that re-
sort to fc layers for mask prediction [33, 34, 10], our fully
convolutional representation requires fewer parameters, and
is more accurate as demonstrated by experiments.

This pixel-to-pixel behavior requires our RoI features,
which themselves are small feature maps, to be well aligned
to faithfully preserve the explicit per-pixel spatial corre-
spondence. This motivated us to develop the following
RoIAlign layer that plays a key role in mask prediction.

RoIAlign: RoIPool [12] is a standard operation for extract-
ing a small feature map (e.g., 7×7) from each RoI. RoIPool
ﬁrst quantizes a ﬂoating-number RoI to the discrete granu-
larity of the feature map, this quantized RoI is then subdi-
vided into spatial bins which are themselves quantized, and
ﬁnally feature values covered by each bin are aggregated
(usually by max pooling). Quantization is performed, e.g.,
on a continuous coordinate x by computing [x/16], where
16 is a feature map stride and [·] is rounding; likewise, quan-
tization is performed when dividing into bins (e.g., 7×7).
These quantizations introduce misalignments between the
RoI and the extracted features. While this may not impact
classiﬁcation, which is robust to small translations, it has a
large negative effect on predicting pixel-accurate masks.

To address this, we propose an RoIAlign layer that re-
moves the harsh quantization of RoIPool, properly aligning
the extracted features with the input. Our proposed change
is simple: we avoid any quantization of the RoI boundaries

3

or bins (i.e., we use x/16 instead of [x/16]). We use bi-
linear interpolation [22] to compute the exact values of the
input features at four regularly sampled locations in each
RoI bin, and aggregate the result (using max or average),
see Figure 3 for details. We note that the results are not sen-
sitive to the exact sampling locations, or how many points
are sampled, as long as no quantization is performed.

RoIAlign leads to large improvements as we show in
§4.2. We also compare to the RoIWarp operation proposed
in [10]. Unlike RoIAlign, RoIWarp overlooked the align-
ment issue and was implemented in [10] as quantizing RoI
just like RoIPool. So even though RoIWarp also adopts
bilinear resampling motivated by [22], it performs on par
with RoIPool as shown by experiments (more details in Ta-
ble 2c), demonstrating the crucial role of alignment.

Network Architecture: To demonstrate the generality of
our approach, we instantiate Mask R-CNN with multiple
architectures. For clarity, we differentiate between: (i) the
convolutional backbone architecture used for feature ex-
traction over an entire image, and (ii) the network head
for bounding-box recognition (classiﬁcation and regression)
and mask prediction that is applied separately to each RoI.
We denote the backbone architecture using the nomen-
clature network-depth-features. We evaluate ResNet [19]
and ResNeXt [45] networks of depth 50 or 101 layers. The
original implementation of Faster R-CNN with ResNets
[19] extracted features from the ﬁnal convolutional layer
of the 4-th stage, which we call C4. This backbone with
ResNet-50, for example, is denoted by ResNet-50-C4. This
is a common choice used in [19, 10, 21, 39].

We also explore another more effective backbone re-
cently proposed by Lin et al. [27], called a Feature Pyra-
mid Network (FPN). FPN uses a top-down architecture with
lateral connections to build an in-network feature pyramid
from a single-scale input. Faster R-CNN with an FPN back-
bone extracts RoI features from different levels of the fea-
ture pyramid according to their scale, but otherwise the
rest of the approach is similar to vanilla ResNet. Using a
ResNet-FPN backbone for feature extraction with Mask R-
CNN gives excellent gains in both accuracy and speed. For
further details on FPN, we refer readers to [27].

For the network head we closely follow architectures
presented in previous work to which we add a fully con-
volutional mask prediction branch. Speciﬁcally, we ex-
tend the Faster R-CNN box heads from the ResNet [19]
and FPN [27] papers. Details are shown in Figure 4. The
head on the ResNet-C4 backbone includes the 5-th stage of
ResNet (namely, the 9-layer ‘res5’ [19]), which is compute-
intensive. For FPN, the backbone already includes res5 and
thus allows for a more efﬁcient head that uses fewer ﬁlters.
We note that our mask branches have a straightforward
structure. More complex designs have the potential to im-
prove performance but are not the focus of this work.

Faster R-CNN
w/ ResNet [19]

Faster R-CNN
w/ FPN [27]

Figure 4. Head Architecture: We extend two existing Faster R-
CNN heads [19, 27]. Left/Right panels show the heads for the
ResNet C4 and FPN backbones, from [19] and [27], respectively,
to which a mask branch is added. Numbers denote spatial resolu-
tion and channels. Arrows denote either conv, deconv, or fc layers
as can be inferred from context (conv preserves spatial dimension
while deconv increases it). All convs are 3×3, except the output
conv which is 1×1, deconvs are 2×2 with stride 2, and we use
ReLU [31] in hidden layers. Left: ‘res5’ denotes ResNet’s ﬁfth
stage, which for simplicity we altered so that the ﬁrst conv oper-
ates on a 7×7 RoI with stride 1 (instead of 14×14 / stride 2 as in
[19]). Right: ‘×4’ denotes a stack of four consecutive convs.

3.1. Implementation Details

We set hyper-parameters following existing Fast/Faster
R-CNN work [12, 36, 27]. Although these decisions were
made for object detection in original papers [12, 36, 27], we
found our instance segmentation system is robust to them.

Training: As in Fast R-CNN, an RoI is considered positive
if it has IoU with a ground-truth box of at least 0.5 and
negative otherwise. The mask loss Lmask is deﬁned only on
positive RoIs. The mask target is the intersection between
an RoI and its associated ground-truth mask.

We adopt image-centric training [12]. Images are resized
such that their scale (shorter edge) is 800 pixels [27]. Each
mini-batch has 2 images per GPU and each image has N
sampled RoIs, with a ratio of 1:3 of positive to negatives
[12]. N is 64 for the C4 backbone (as in [12, 36]) and 512
for FPN (as in [27]). We train on 8 GPUs (so effective mini-
batch size is 16) for 160k iterations, with a learning rate of
0.02 which is decreased by 10 at the 120k iteration. We
use a weight decay of 0.0001 and momentum of 0.9. With
ResNeXt [45], we train with 1 image per GPU and the same
number of iterations, with a starting learning rate of 0.01.

The RPN anchors span 5 scales and 3 aspect ratios, fol-
lowing [27]. For convenient ablation, RPN is trained sep-
arately and does not share features with Mask R-CNN, un-
less speciﬁed. For every entry in this paper, RPN and Mask
R-CNN have the same backbones and so they are shareable.

Inference: At test time, the proposal number is 300 for the
C4 backbone (as in [36]) and 1000 for FPN (as in [27]). We
run the box prediction branch on these proposals, followed
by non-maximum suppression [14]. The mask branch is
then applied to the highest scoring 100 detection boxes. Al-
though this differs from the parallel computation used in
training, it speeds up inference and improves accuracy (due
to the use of fewer, more accurate RoIs). The mask branch

4

aveRoIRoI14×14×2567×7×25614×14×256102428×28×2561024mask14×14×256classbox2048RoIres57×7×10247×7×2048×4classbox14×14×80mask28×28×80Figure 5. More results of Mask R-CNN on COCO test images, using ResNet-101-FPN and running at 5 fps, with 35.7 mask AP (Table 1).

backbone
ResNet-101-C4
MNC [10]
ResNet-101-C5-dilated
FCIS [26] +OHEM
FCIS+++ [26] +OHEM ResNet-101-C5-dilated
Mask R-CNN
Mask R-CNN
Mask R-CNN

ResNet-101-C4
ResNet-101-FPN
ResNeXt-101-FPN

AP
24.6
29.2
33.6
33.1
35.7
37.1

AP50 AP75
24.8
44.3
-
49.5
-
54.5
34.8
54.9
37.8
58.0
39.4
60.0

APS
4.7
7.1
-
12.1
15.5
16.9

APM APL
43.6
25.9
50.0
31.3
-
-
51.1
35.6
52.4
38.1
53.5
39.9

Table 1. Instance segmentation mask AP on COCO test-dev. MNC [10] and FCIS [26] are the winners of the COCO 2015 and 2016
segmentation challenges, respectively. Without bells and whistles, Mask R-CNN outperforms the more complex FCIS+++, which includes
multi-scale train/test, horizontal ﬂip test, and OHEM [38]. All entries are single-model results.

can predict K masks per RoI, but we only use the k-th mask,
where k is the predicted class by the classiﬁcation branch.
The m×m ﬂoating-number mask output is then resized to
the RoI size, and binarized at a threshold of 0.5.

Note that since we only compute masks on the top 100
detection boxes, Mask R-CNN adds a small overhead to its
Faster R-CNN counterpart (e.g., ∼20% on typical models).

4. Experiments: Instance Segmentation

We perform a thorough comparison of Mask R-CNN to
the state of the art along with comprehensive ablations on
the COCO dataset [28]. We report the standard COCO met-
rics including AP (averaged over IoU thresholds), AP50,
AP75, and APS, APM , APL (AP at different scales). Un-
less noted, AP is evaluating using mask IoU. As in previous
work [5, 27], we train using the union of 80k train images
and a 35k subset of val images (trainval35k), and re-
port ablations on the remaining 5k val images (minival).
We also report results on test-dev [28].

4.1. Main Results

We compare Mask R-CNN to the state-of-the-art meth-
ods in instance segmentation in Table 1. All instantia-
tions of our model outperform baseline variants of pre-
vious state-of-the-art models. This includes MNC [10]
and FCIS [26], the winners of the COCO 2015 and 2016
segmentation challenges, respectively. Without bells and
whistles, Mask R-CNN with ResNet-101-FPN backbone
outperforms FCIS+++ [26], which includes multi-scale
train/test, horizontal ﬂip test, and online hard example min-
ing (OHEM) [38]. While outside the scope of this work, we
expect many such improvements to be applicable to ours.

Mask R-CNN outputs are visualized in Figures 2 and 5.
Mask R-CNN achieves good results even under challeng-
ing conditions. In Figure 6 we compare our Mask R-CNN
baseline and FCIS+++ [26]. FCIS+++ exhibits systematic
artifacts on overlapping instances, suggesting that it is chal-
lenged by the fundamental difﬁculty of instance segmenta-
tion. Mask R-CNN shows no such artifacts.

5

horse1.00horse1.00horse1.00bus1.00bus1.00car.98truck.88car.93car.78car.98car.91car.96car.99car.94car.99car.98truck.86car.99car.95car1.00car.93car.98car.95car.97car.87car.99car.82car.78car.93car.95car.97person.99traffic light.73person1.00person.99person.95person.93person.93person1.00person.98skateboard.82suitcase1.00suitcase.99suitcase.96suitcase1.00suitcase.93suitcase.98suitcase.88suitcase.72stop sign.88person1.00person1.00person1.00person1.00person.99person.99bench.76skateboard.91skateboard.83handbag.81surfboard1.00person1.00person1.00surfboard1.00person1.00person.98surfboard1.00person1.00surfboard.98surfboard1.00person.91person.74person1.00person1.00person1.00person1.00person1.00person1.00person.98person.99person1.00person.99umbrella1.00person.95umbrella.99umbrella.97umbrella.97umbrella.96umbrella1.00backpack.96umbrella.98backpack.95person.80backpack.98bicycle.93umbrella.89person.89handbag.97handbag.85person1.00person1.00person1.00person1.00person1.00person1.00motorcycle.72kite.89person.99kite.95person.99person1.00person.81person.72kite.93person.89kite1.00person.98person1.00kite.84kite.97person.80handbag.80person.99kite.82person.98person.96kite.98person.99person.82kite.81person.95person.84kite.98kite.72kite.99kite.84kite.99person.94person.72person.98kite.95person.98person.77kite.73person.78person.71person.87kite.88kite.88person.94kite.86kite.89zebra.99zebra1.00zebra1.00zebra.99zebra1.00zebra.96zebra.74zebra.96zebra.99zebra.90zebra.88zebra.76dining table.91dining table.78chair.97person.99person.86chair.94chair.98person.95chair.95person.97chair.92chair.99person.97person.99person.94person.99person.87person.99chair.83person.94person.99person.98chair.87chair.95person.97person.96chair.99person.86person.89chair.89wine glass.93person.98person.88person.97person.88person.88person.91chair.96person.95person.77person.92wine glass.94cup.83wine glass.94wine glass.83cup.91chair.85dining table.96wine glass.91person.96cup.98person.83dining table.75cup.96person.72wine glass.80chair.98person.81person.82dining table.81chair.85chair.78cup.75person.77cup.71wine glass.80cup.79cup.93cup.71person.99person.99person1.00person1.00frisbee1.00person.80person.82elephant1.00elephant1.00elephant1.00elephant.97elephant.99person1.00person1.00dining table.95person1.00person.88wine glass1.00bottle.97wine glass1.00wine glass.99tv.98tv.84person1.00bench.97person.98person1.00person1.00handbag.73person.86potted plant.92bird.93person.76person.98person.78person.78backpack.88handbag.91cell phone.77clock.73person.99person1.00person.98person1.00person1.00person1.00person.99person.99person.99person1.00person1.00person.98person.99handbag.88person1.00person.98person.92handbag.99person.97person.95handbag.88traffic light.99person.95person.87person.95traffic light.87traffic light.71person.80person.95person.95person.73person.74tie.85car.99car.86car.97car1.00car.95car.97traffic light1.00traffic light.99car.99person.99car.95car.97car.98car.98car.91car1.00car.96car.96bicycle.86car.97car.97car.97car.94car.95car.94car.81person.87parking meter.98car.89donut1.00donut.90donut.88donut.81donut.95donut.96donut1.00donut.98donut.99donut.94donut.97donut.99donut.98donut1.00donut.95donut1.00donut.98donut.98donut.99donut.96donut.89donut.96donut.95donut.98donut.89donut.93donut.95donut.90donut.89donut.89donut.89donut.86donut.86person1.00person1.00person1.00person1.00person1.00person1.00person1.00dog1.00baseball bat.99baseball bat.85baseball bat.98truck.92truck.99truck.96truck.99truck.97bus.99truck.93bus.90person1.00person1.00horse.77horse.99cow.93person.96person1.00person.99horse.97person.98person.97person.98person.96person1.00tennis racket1.00chair.73person.90person.77person.97person.81person.87person.71person.96person.99person.98person.94chair.97chair.80chair.71chair.94chair.92chair.99chair.93chair.99chair.91chair.81chair.98chair.83chair.81chair.81chair.93sports ball.99person1.00couch.82person1.00person.99person1.00person1.00person1.00person.99skateboard.99person.90person.98person.99person.91person.99person1.00person.80skateboard.98Figure 6. FCIS+++ [26] (top) vs. Mask R-CNN (bottom, ResNet-101-FPN). FCIS exhibits systematic artifacts on overlapping objects.

net-depth-features

AP
30.3
ResNet-50-C4
32.7
ResNet-101-C4
33.6
ResNet-50-FPN
35.4
ResNet-101-FPN
ResNeXt-101-FPN 36.7

AP50 AP75
31.5
51.2
34.3
54.2
35.3
55.2
37.5
57.3
38.9
59.5

softmax
sigmoid

AP
24.8
30.3
+5.5

AP50
44.1
51.2
+7.1

AP75
25.1
31.5
+6.4

RoIPool [12]

RoIWarp [10]

RoIAlign

align? bilinear? agg.
max
max
ave
max
ave

(cid:88)
(cid:88)
(cid:88)
(cid:88)

(cid:88)
(cid:88)

AP
26.9
27.2
27.1
30.2
30.3

AP50 AP75
26.4
48.8
27.1
49.2
27.1
48.9
31.8
51.0
31.5
51.2

(a) Backbone Architecture: Better back-
bones bring expected gains: deeper networks
do better, FPN outperforms C4 features, and
ResNeXt improves on ResNet.

(b) Multinomial vs. Independent Masks
(ResNet-50-C4): Decoupling via per-
class binary masks (sigmoid) gives large
gains over multinomial masks (softmax).

(c) RoIAlign (ResNet-50-C4): Mask results with various RoI
layers. Our RoIAlign layer improves AP by ∼3 points and
AP75 by ∼5 points. Using proper alignment is the only fac-
tor that contributes to the large gap between RoI layers.

RoIPool
RoIAlign

AP
23.6
30.9
+7.3

AP50
46.5
51.8
+ 5.3

AP75
21.6
32.1
+10.5

APbb
28.2
34.0
+5.8

APbb
50
52.7
55.3
+2.6

APbb
75
26.9
36.4
+9.5

mask branch
fc: 1024→1024→80·282
fc: 1024→1024→1024→80·282

MLP
MLP
FCN conv: 256→256→256→256→256→80

AP
31.5
31.5
33.6

AP50
53.7
54.0
55.2

AP75
32.8
32.6
35.3

(d) RoIAlign (ResNet-50-C5, stride 32): Mask-level and box-level
AP using large-stride features. Misalignments are more severe than
with stride-16 features (Table 2c), resulting in big accuracy gaps.

(e) Mask Branch (ResNet-50-FPN): Fully convolutional networks (FCN) vs.
multi-layer perceptrons (MLP, fully-connected) for mask prediction. FCNs im-
prove results as they take advantage of explicitly encoding spatial layout.

Table 2. Ablations. We train on trainval35k, test on minival, and report mask AP unless otherwise noted.

4.2. Ablation Experiments

We run a number of ablations to analyze Mask R-CNN.

Results are shown in Table 2 and discussed in detail next.

Architecture: Table 2a shows Mask R-CNN with various
backbones. It beneﬁts from deeper networks (50 vs. 101)
and advanced designs including FPN and ResNeXt. We
note that not all frameworks automatically beneﬁt from
deeper or advanced networks (see benchmarking in [21]).

Multinomial vs. Independent Masks: Mask R-CNN de-
couples mask and class prediction: as the existing box
branch predicts the class label, we generate a mask for each
class without competition among classes (by a per-pixel sig-
moid and a binary loss). In Table 2b, we compare this to
using a per-pixel softmax and a multinomial loss (as com-
monly used in FCN [30]). This alternative couples the tasks
of mask and class prediction, and results in a severe loss
in mask AP (5.5 points). This suggests that once the in-
stance has been classiﬁed as a whole (by the box branch),
it is sufﬁcient to predict a binary mask without concern for
the categories, which makes the model easier to train.

Class-Speciﬁc vs. Class-Agnostic Masks: Our default in-
stantiation predicts class-speciﬁc masks, i.e., one m×m

Interestingly, Mask R-CNN with class-
mask per class.
agnostic masks (i.e., predicting a single m×m output re-
gardless of class) is nearly as effective: it has 29.7 mask AP
vs. 30.3 for the class-speciﬁc counterpart on ResNet-50-C4.
This further highlights the division of labor in our approach
which largely decouples classiﬁcation and segmentation.

RoIAlign: An evaluation of our proposed RoIAlign layer is
shown in Table 2c. For this experiment we use the ResNet-
50-C4 backbone, which has stride 16. RoIAlign improves
AP by about 3 points over RoIPool, with much of the gain
coming at high IoU (AP75). RoIAlign is insensitive to
max/average pool; we use average in the rest of the paper.

Additionally, we compare with RoIWarp proposed in
MNC [10] that also adopt bilinear sampling. As discussed
in §3, RoIWarp still quantizes the RoI, losing alignment
with the input. As can be seen in Table 2c, RoIWarp per-
forms on par with RoIPool and much worse than RoIAlign.
This highlights that proper alignment is key.

We also evaluate RoIAlign with a ResNet-50-C5 back-
bone, which has an even larger stride of 32 pixels. We use
the same head as in Figure 4 (right), as the res5 head is not
applicable. Table 2d shows that RoIAlign improves mask
AP by a massive 7.3 points, and mask AP75 by 10.5 points

6

person1.00person1.00person1.00person1.00umbrella1.00umbrella.99car.99car.93giraffe1.00giraffe1.00person1.00person1.00person1.00person1.00person.95sports ball1.00sports ball.98person1.00person1.00person1.00tie.95tie1.00FCISMask R-CNNtrain1.00train.99train.80person1.00person1.00person1.00person1.00person1.00person1.00skateboard.98person.99person.99skateboard.99handbag.93Faster R-CNN+++ [19]
Faster R-CNN w FPN [27]
Faster R-CNN by G-RMI [21]
Faster R-CNN w TDM [39]
Faster R-CNN, RoIAlign
Mask R-CNN
Mask R-CNN

backbone

APbb
34.9
ResNet-101-C4
36.2
ResNet-101-FPN
Inception-ResNet-v2 [41]
34.7
Inception-ResNet-v2-TDM 36.8
37.3
ResNet-101-FPN
38.2
ResNet-101-FPN
39.8
ResNeXt-101-FPN

APbb
55.7
59.1
55.5
57.7
59.6
60.3
62.3

50 APbb
75
37.4
39.0
36.7
39.2
40.3
41.7
43.4

APbb
S
15.6
18.2
13.5
16.2
19.8
20.1
22.1

APbb
38.7
39.0
38.1
39.8
40.2
41.1
43.2

M APbb
L
50.9
48.2
52.0
52.1
48.8
50.2
51.2

Table 3. Object detection single-model results (bounding box AP), vs. state-of-the-art on test-dev. Mask R-CNN using ResNet-101-
FPN outperforms the base variants of all previous state-of-the-art models (the mask output is ignored in these experiments). The gains of
Mask R-CNN over [27] come from using RoIAlign (+1.1 APbb), multitask training (+0.9 APbb), and ResNeXt-101 (+1.6 APbb).

(50% relative improvement). Moreover, we note that with
RoIAlign, using stride-32 C5 features (30.9 AP) is more ac-
curate than using stride-16 C4 features (30.3 AP, Table 2c).
RoIAlign largely resolves the long-standing challenge of
using large-stride features for detection and segmentation.

Finally, RoIAlign shows a gain of 1.5 mask AP and 0.5
box AP when used with FPN, which has ﬁner multi-level
strides. For keypoint detection that requires ﬁner alignment,
RoIAlign shows large gains even with FPN (Table 6).

Mask Branch: Segmentation is a pixel-to-pixel task and
we exploit the spatial layout of masks by using an FCN.
In Table 2e, we compare multi-layer perceptrons (MLP)
and FCNs, using a ResNet-50-FPN backbone. Using FCNs
gives a 2.1 mask AP gain over MLPs. We note that we
choose this backbone so that the conv layers of the FCN
head are not pre-trained, for a fair comparison with MLP.

4.3. Bounding Box Detection Results

We compare Mask R-CNN to the state-of-the-art COCO
bounding-box object detection in Table 3. For this result,
even though the full Mask R-CNN model is trained, only
the classiﬁcation and box outputs are used at inference (the
mask output is ignored). Mask R-CNN using ResNet-101-
FPN outperforms the base variants of all previous state-of-
the-art models, including the single-model variant of G-
RMI [21], the winner of the COCO 2016 Detection Chal-
lenge. Using ResNeXt-101-FPN, Mask R-CNN further im-
proves results, with a margin of 3.0 points box AP over
the best previous single model entry from [39] (which used
Inception-ResNet-v2-TDM).

As a further comparison, we trained a version of Mask
R-CNN but without the mask branch, denoted by “Faster
R-CNN, RoIAlign” in Table 3. This model performs better
than the model presented in [27] due to RoIAlign. On the
other hand, it is 0.9 points box AP lower than Mask R-CNN.
This gap of Mask R-CNN on box detection is therefore due
solely to the beneﬁts of multi-task training.

Lastly, we note that Mask R-CNN attains a small gap
between its mask and box AP: e.g., 2.7 points between 37.1
(mask, Table 1) and 39.8 (box, Table 3). This indicates that
our approach largely closes the gap between object detec-
tion and the more challenging instance segmentation task.

4.4. Timing

Inference: We train a ResNet-101-FPN model that shares
features between the RPN and Mask R-CNN stages, follow-
ing the 4-step training of Faster R-CNN [36]. This model
runs at 195ms per image on an Nvidia Tesla M40 GPU (plus
15ms CPU time resizing the outputs to the original resolu-
tion), and achieves statistically the same mask AP as the
unshared one. We also report that the ResNet-101-C4 vari-
ant takes ∼400ms as it has a heavier box head (Figure 4), so
we do not recommend using the C4 variant in practice.

Although Mask R-CNN is fast, we note that our design
is not optimized for speed, and better speed/accuracy trade-
offs could be achieved [21], e.g., by varying image sizes and
proposal numbers, which is beyond the scope of this paper.

Training: Mask R-CNN is also fast to train. Training with
ResNet-50-FPN on COCO trainval35k takes 32 hours
in our synchronized 8-GPU implementation (0.72s per 16-
image mini-batch), and 44 hours with ResNet-101-FPN. In
fact, fast prototyping can be completed in less than one day
when training on the train set. We hope such rapid train-
ing will remove a major hurdle in this area and encourage
more people to perform research on this challenging topic.

5. Mask R-CNN for Human Pose Estimation

Our framework can easily be extended to human pose
estimation. We model a keypoint’s location as a one-hot
mask, and adopt Mask R-CNN to predict K masks, one for
each of K keypoint types (e.g., left shoulder, right elbow).
This task helps demonstrate the ﬂexibility of Mask R-CNN.
We note that minimal domain knowledge for human pose
is exploited by our system, as the experiments are mainly to
demonstrate the generality of the Mask R-CNN framework.
We expect that domain knowledge (e.g., modeling struc-
tures [6]) will be complementary to our simple approach.

Implementation Details: We make minor modiﬁcations to
the segmentation system when adapting it for keypoints.
For each of the K keypoints of an instance, the training
target is a one-hot m × m binary mask where only a single
pixel is labeled as foreground. During training, for each vis-
ible ground-truth keypoint, we minimize the cross-entropy
loss over an m2-way softmax output (which encourages a

7

Figure 7. Keypoint detection results on COCO test using Mask R-CNN (ResNet-50-FPN), with person segmentation masks predicted
from the same model. This model has a keypoint AP of 63.1 and runs at 5 fps.

APkp APkp
CMU-Pose+++ [6]
84.9
61.8
G-RMI [32]†
84.0
62.4
Mask R-CNN, keypoint-only
87.0
62.7
87.3
Mask R-CNN, keypoint & mask 63.1

50 APkp
67.5
68.5
68.4
68.7

75 APkp
57.1
59.1
57.4
57.8

M APkp
L
68.2
68.1
71.1
71.4

Table 4. Keypoint detection AP on COCO test-dev. Ours is a
single model (ResNet-50-FPN) that runs at 5 fps. CMU-Pose+++
[6] is the 2016 competition winner that uses multi-scale testing,
post-processing with CPM [44], and ﬁltering with an object detec-
tor, adding a cumulative ∼5 points (clariﬁed in personal commu-
nication). †: G-RMI was trained on COCO plus MPII [1] (25k im-
ages), using two models (Inception-ResNet-v2 for bounding box
detection and ResNet-101 for keypoints).

single point to be detected). We note that as in instance seg-
mentation, the K keypoints are still treated independently.
We adopt the ResNet-FPN variant, and the keypoint head
architecture is similar to that in Figure 4 (right). The key-
point head consists of a stack of eight 3×3 512-d conv lay-
ers, followed by a deconv layer and 2× bilinear upscaling,
producing an output resolution of 56×56. We found that
a relatively high resolution output (compared to masks) is
required for keypoint-level localization accuracy.

Models are trained on all COCO trainval35k im-
ages that contain annotated keypoints. To reduce overﬁt-
ting, as this training set is smaller, we train using image
scales randomly sampled from [640, 800] pixels; inference
is on a single scale of 800 pixels. We train for 90k iterations,
starting from a learning rate of 0.02 and reducing it by 10 at
60k and 80k iterations. We use bounding-box NMS with a
threshold of 0.5. Other details are identical as in §3.1.

Main Results and Ablations: We evaluate the person key-
point AP (APkp) and experiment with a ResNet-50-FPN
backbone; more backbones will be studied in the appendix.
Table 4 shows that our result (62.7 APkp) is 0.9 points higher
than the COCO 2016 keypoint detection winner [6] that
uses a multi-stage processing pipeline (see caption of Ta-
ble 4). Our method is considerably simpler and faster.

More importantly, we have a uniﬁed model that can si-

Faster R-CNN
Mask R-CNN, mask-only
Mask R-CNN, keypoint-only
Mask R-CNN, keypoint & mask

APbb
52.5
53.6
50.7
52.0

person APmask
person
-
45.8
-
45.1

APkp
-
-
64.2
64.7

Table 5. Multi-task learning of box, mask, and keypoint about the
person category, evaluated on minival. All entries are trained
on the same data for fair comparisons. The backbone is ResNet-
50-FPN. The entries with 64.2 and 64.7 AP on minival have
test-dev AP of 62.7 and 63.1, respectively (see Table 4).

RoIPool
RoIAlign

APkp APkp
86.2
59.8
86.6
64.2

50 APkp
66.7
69.7

75 APkp
55.1
58.7

M APkp
L
67.4
73.0

Table 6. RoIAlign vs. RoIPool
minival. The backbone is ResNet-50-FPN.

for keypoint detection on

multaneously predict boxes, segments, and keypoints while
running at 5 fps. Adding a segment branch (for the per-
son category) improves the APkp to 63.1 (Table 4) on
test-dev. More ablations of multi-task learning on
minival are in Table 5. Adding the mask branch to the
box-only (i.e., Faster R-CNN) or keypoint-only versions
consistently improves these tasks. However, adding the
keypoint branch reduces the box/mask AP slightly, suggest-
ing that while keypoint detection beneﬁts from multitask
training, it does not in turn help the other tasks. Neverthe-
less, learning all three tasks jointly enables a uniﬁed system
to efﬁciently predict all outputs simultaneously (Figure 7).
We also investigate the effect of RoIAlign on keypoint
detection (Table 6). Though this ResNet-50-FPN backbone
has ﬁner strides (e.g., 4 pixels on the ﬁnest level), RoIAlign
still shows signiﬁcant improvement over RoIPool and in-
creases APkp by 4.4 points. This is because keypoint detec-
tions are more sensitive to localization accuracy. This again
indicates that alignment is essential for pixel-level localiza-
tion, including masks and keypoints.

Given the effectiveness of Mask R-CNN for extracting
object bounding boxes, masks, and keypoints, we expect it
be an effective framework for other instance-level tasks.

8

training data

InstanceCut [23] fine + coarse
DWT [4]
SAIS [17]
DIN [3]
SGN [29]
Mask R-CNN
Mask R-CNN
Table 7. Results on Cityscapes val (‘AP [val]’ column) and test (remaining columns) sets. Our method uses ResNet-50-FPN.

fine
fine
fine + coarse
fine + coarse
fine
fine + COCO

train mcycle bicycle
9.3
15.2
7.9
15.0
10.3
19.0
17.1
23.4
17.7
30.8
19.1
18.6
24.1
30.9

AP [val]
15.8
19.8
-
-
29.2
31.5
36.4

person
10.0
15.1
14.6
16.5
21.8
30.5
34.8

AP50
27.9
30.0
36.7
38.8
44.9
49.9
58.1

truck
14.0
17.1
16.0
20.6
24.8
22.8
30.1

rider
8.0
11.7
12.9
16.7
20.1
23.7
27.0

AP
13.0
15.6
17.4
20.0
25.0
26.2
32.0

car
23.7
32.9
35.7
25.7
39.4
46.9
49.1

bus
19.5
20.4
23.2
30.0
33.2
32.2
40.9

4.7
4.9
7.8
10.1
12.4
16.0
18.7

Appendix A: Experiments on Cityscapes

We further report instance segmentation results on the
Cityscapes [7] dataset. This dataset has fine annota-
tions for 2975 train, 500 val, and 1525 test images. It has
20k coarse training images without instance annotations,
which we do not use. All images are 2048×1024 pixels.
The instance segmentation task involves 8 object categories,
whose numbers of instances on the fine training set are:
train mcycle bicycle
0.7k
0.2k

person
17.9k

car
26.9k

truck
0.5k

rider
1.8k

bus
0.4k

3.7k

Instance segmentation performance on this task is measured
by the COCO-style mask AP (averaged over IoU thresh-
olds); AP50 (i.e., mask AP at an IoU of 0.5) is also reported.

Implementation: We apply our Mask R-CNN models with
the ResNet-FPN-50 backbone; we found the 101-layer
counterpart performs similarly due to the small dataset size.
We train with image scale (shorter side) randomly sampled
from [800, 1024], which reduces overﬁtting; inference is on
a single scale of 1024 pixels. We use a mini-batch size of
1 image per GPU (so 8 on 8 GPUs) and train the model
for 24k iterations, starting from a learning rate of 0.01 and
reducing it to 0.001 at 18k iterations. It takes ∼4 hours of
training on a single 8-GPU machine under this setting.

Results: Table 7 compares our results to the state of the
art on the val and test sets. Without using the coarse
training set, our method achieves 26.2 AP on test, which
is over 30% relative improvement over the previous best en-
try (DIN [3]), and is also better than the concurrent work of
SGN’s 25.0 [29]. Both DIN and SGN use fine + coarse
data. Compared to the best entry using fine data only
(17.4 AP), we achieve a ∼50% improvement.

For the person and car categories, the Cityscapes dataset
exhibits a large number of within-category overlapping in-
stances (on average 6 people and 9 cars per image). We
argue that within-category overlap is a core difﬁculty of in-
stance segmentation. Our method shows massive improve-
ment on these two categories over the other best entries (rel-
ative ∼40% improvement on person from 21.8 to 30.5 and
∼20% improvement on car from 39.4 to 46.9), even though
our method does not exploit the coarse data.

A main challenge of the Cityscapes dataset is training
models in a low-data regime, particularly for the categories
of truck, bus, and train, which have about 200-500 train-

Figure 8. Mask R-CNN results on Cityscapes test (32.0 AP).
The bottom-right image shows a failure prediction.

ing samples each. To partially remedy this issue, we further
report a result using COCO pre-training. To do this, we ini-
tialize the corresponding 7 categories in Cityscapes from a
pre-trained COCO Mask R-CNN model (rider being ran-
domly initialized). We ﬁne-tune this model for 4k iterations
in which the learning rate is reduced at 3k iterations, which
takes ∼1 hour for training given the COCO model.

The COCO pre-trained Mask R-CNN model achieves
32.0 AP on test, almost a 6 point improvement over the
fine-only counterpart. This indicates the important role
the amount of training data plays.
It also suggests that
methods on Cityscapes might be inﬂuenced by their low-
shot learning performance. We show that using COCO pre-
training is an effective strategy on this dataset.

Finally, we observed a bias between the val and test
AP, as is also observed from the results of [23, 4, 29]. We
found that this bias is mainly caused by the truck, bus,
and train categories, with the fine-only model having
val/test AP of 28.8/22.8, 53.5/32.2, and 33.0/18.6, re-
spectively. This suggests that there is a domain shift on
these categories, which also have little training data. COCO
pre-training helps to improve results the most on these cat-
egories; however, the domain shift persists with 38.0/30.1,
57.5/40.9, and 41.2/30.9 val/test AP, respectively. Note
that for the person and car categories we do not see any
such bias (val/test AP are within ±1 point).

Example results on Cityscapes are shown in Figure 8.

9

car:1.00car:0.98car:0.98car:0.95car:0.81car:0.52person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:0.99person:0.99person:0.99person:0.99person:0.98person:0.98person:0.98person:0.98person:0.94person:0.94person:0.82person:0.82person:0.79person:0.73person:0.67person:0.66person:0.59truck:0.66bus:1.00bus:0.95rider:0.59bicycle:0.83bicycle:0.56car:1.00car:1.00car:1.00car:1.00car:1.00car:1.00car:1.00car:1.00car:1.00car:0.99car:0.95car:0.95car:0.95car:0.69car:0.68car:0.68car:0.64car:0.57car:0.52person:1.00person:0.99person:0.99person:0.99person:0.99person:0.98person:0.98person:0.98person:0.97person:0.93person:0.92person:0.91person:0.86person:0.84person:0.82person:0.73person:0.72person:0.72person:0.72person:0.63rider:0.68car:1.00car:1.00car:1.00car:1.00car:1.00car:1.00car:1.00car:1.00car:1.00car:1.00car:1.00car:1.00car:1.00car:1.00car:1.00car:1.00car:1.00car:0.98car:0.97car:0.88car:0.76car:0.72car:0.72car:0.65car:0.50person:1.00person:1.00person:0.98person:0.93person:0.85person:0.78person:0.73person:0.58person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:1.00person:0.99person:0.99person:0.98person:0.97person:0.96person:0.92person:0.91person:0.70person:0.59bicycle:0.99bicycle:0.97car:1.00car:1.00car:0.99car:0.89person:1.00person:1.00person:1.00person:1.00person:1.00person:0.96person:0.93person:0.89person:0.88person:0.75rider:0.94car:1.00car:1.00car:1.00car:1.00car:1.00car:1.00car:1.00car:0.99car:0.89car:0.67person:1.00person:1.00person:1.00person:1.00person:0.82bus:0.75AP
description
backbone
36.7
X-101-FPN
original baseline
+ updated baseline X-101-FPN
37.0
+ e2e training
37.6
X-101-FPN
+ ImageNet-5k
38.6
X-101-FPN
+ train-time augm. X-101-FPN
39.2
+ deeper
39.7
X-152-FPN
+ Non-local [43] X-152-FPN-NL 40.3
+ test-time augm. X-152-FPN-NL 41.8
Table 8. Enhanced detection results of Mask R-CNN on COCO
minival. Each row adds an extra component to the above row.
We denote ResNeXt model by ‘X’ for notational brevity.

AP50 AP75 APbb APbb
61.5
59.5
63.0
59.7
64.1
60.4
65.1
61.7
65.9
62.5
66.4
63.2
67.8
64.4
69.3
66.0

50 APbb
75
43.2
43.7
45.2
46.6
47.2
48.4
48.9
51.5

39.6
40.5
41.7
42.7
43.5
44.1
45.0
47.3

38.9
39.0
39.9
40.9
41.6
42.2
42.8
44.8

APkp APkp
backbone
description
86.6
64.2
R-50-FPN
original baseline
+ updated baseline
86.6
R-50-FPN
65.1
+ deeper
87.7
R-101-FPN 66.1
+ ResNeXt
88.0
X-101-FPN 67.3
+ data distillation [35] X-101-FPN 69.1
88.9
+ test-time augm.
89.3
X-101-FPN 70.4
Table 9. Enhanced keypoint results of Mask R-CNN on COCO
minival. Each row adds an extra component to the above row.
Here we use only keypoint annotations but no mask annotations.
We denote ResNet by ‘R’ and ResNeXt by ‘X’ for brevity.

M APkp
L
73.0
73.6
75.0
75.6
77.1
78.1

75 APkp
58.7
59.9
60.5
62.2
64.1
65.8

50 APkp
69.7
70.9
71.7
73.3
75.3
76.8

Appendix B: Enhanced Results on COCO

As a general framework, Mask R-CNN is compat-
ible with complementary techniques developed for de-
including improvements made to
tection/segmentation,
Fast/Faster R-CNN and FCNs.
In this appendix we de-
scribe some techniques that improve over our original re-
sults. Thanks to its generality and ﬂexibility, Mask R-CNN
was used as the framework by the three winning teams in
the COCO 2017 instance segmentation competition, which
all signiﬁcantly outperformed the previous state of the art.

Instance Segmentation and Object Detection

We report some enhanced results of Mask R-CNN in Ta-
ble 8. Overall, the improvements increase mask AP 5.1
points (from 36.7 to 41.8) and box AP 7.7 points (from 39.6
to 47.3). Each model improvement increases both mask AP
and box AP consistently, showing good generalization of
the Mask R-CNN framework. We detail the improvements
next. These results, along with future updates, can be repro-
duced by our released code at https://github.com/
facebookresearch/Detectron, and can serve as
higher baselines for future research.

Updated baseline: We start with an updated baseline
with a different set of hyper-parameters. We lengthen the
training to 180k iterations, in which the learning rate is re-
duced by 10 at 120k and 160k iterations. We also change
the NMS threshold to 0.5 (from a default value of 0.3). The
updated baseline has 37.0 mask AP and 40.5 box AP.

End-to-end training: All previous results used stage-
wise training, i.e., training RPN as the ﬁrst stage and Mask
R-CNN as the second. Following [37], we evaluate end-
to-end (‘e2e’) training that jointly trains RPN and Mask R-
CNN. We adopt the ‘approximate’ version in [37] that only
computes partial gradients in the RoIAlign layer by ignor-
ing the gradient w.r.t. RoI coordinates. Table 8 shows that
e2e training improves mask AP by 0.6 and box AP by 1.2.
ImageNet-5k pre-training: Following [45], we experi-
ment with models pre-trained on a 5k-class subset of Ima-
geNet (in contrast to the standard 1k-class subset). This 5×
increase in pre-training data improves both mask and box 1
AP. As a reference, [40] used ∼250× more images (300M)
and reported a 2-3 box AP improvement on their baselines.

Train-time augmentation: Scale augmentation at train
time further improves results. During training, we randomly
sample a scale from [640, 800] pixels and we increase the
number of iterations to 260k (with the learning rate reduced
by 10 at 200k and 240k iterations). Train-time augmenta-
tion improves mask AP by 0.6 and box AP by 0.8.

Model architecture:

By upgrading the 101-layer
ResNeXt to its 152-layer counterpart [19], we observe an
increase of 0.5 mask AP and 0.6 box AP. This shows a
deeper model can still improve results on COCO.

Using the recently proposed non-local (NL) model [43],
we achieve 40.3 mask AP and 45.0 box AP. This result is
without test-time augmentation, and the method runs at 3fps
on an Nvidia Tesla P100 GPU at test time.

Test-time augmentation: We combine the model results
evaluated using scales of [400, 1200] pixels with a step of
100 and on their horizontal ﬂips. This gives us a single-
model result of 41.8 mask AP and 47.3 box AP.

The above result is the foundation of our submission to
the COCO 2017 competition (which also used an ensemble,
not discussed here). The ﬁrst three winning teams for the
instance segmentation task were all reportedly based on an
extension of the Mask R-CNN framework.

Keypoint Detection

We report enhanced results of keypoint detection in Ta-
ble 9. As an updated baseline, we extend the training sched-
ule to 130k iterations in which the learning rate is reduced
by 10 at 100k and 120k iterations. This improves APkp by
about 1 point. Replacing ResNet-50 with ResNet-101 and
ResNeXt-101 increases APkp to 66.1 and 67.3, respectively.
With a recent method called data distillation [35], we are
able to exploit the additional 120k unlabeled images pro-
vided by COCO. In brief, data distillation is a self-training
strategy that uses a model trained on labeled data to pre-
dict annotations on unlabeled images, and in turn updates
the model with these new annotations. Mask R-CNN pro-
vides an effective framework for such a self-training strat-
egy. With data distillation, Mask R-CNN APkp improve by
1.8 points to 69.1. We observe that Mask R-CNN can ben-
eﬁt from extra data, even if that data is unlabeled.

By using the same test-time augmentation as used for

instance segmentation, we further boost APkp to 70.4.

10

Acknowledgements: We would like to acknowledge Ilija
Radosavovic for contributions to code release and enhanced
results, and the Caffe2 team for engineering support.

[21] J. Huang, V. Rathod, C. Sun, M. Zhu, A. Korattikara,
A. Fathi, I. Fischer, Z. Wojna, Y. Song, S. Guadarrama, et al.
Speed/accuracy trade-offs for modern convolutional object
detectors. In CVPR, 2017. 2, 3, 4, 6, 7

References

[1] M. Andriluka, L. Pishchulin, P. Gehler, and B. Schiele. 2D
human pose estimation: New benchmark and state of the art
analysis. In CVPR, 2014. 8

[2] P. Arbel´aez, J. Pont-Tuset, J. T. Barron, F. Marques, and
In CVPR,

J. Malik. Multiscale combinatorial grouping.
2014. 2

[3] A. Arnab and P. H. Torr. Pixelwise instance segmentation
with a dynamically instantiated network. In CVPR, 2017. 3,
9

[4] M. Bai and R. Urtasun. Deep watershed transform for in-

stance segmentation. In CVPR, 2017. 3, 9

[5] S. Bell, C. L. Zitnick, K. Bala, and R. Girshick.

Inside-
outside net: Detecting objects in context with skip pooling
and recurrent neural networks. In CVPR, 2016. 5

[6] Z. Cao, T. Simon, S.-E. Wei, and Y. Sheikh. Realtime multi-
person 2d pose estimation using part afﬁnity ﬁelds. In CVPR,
2017. 7, 8

[7] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler,
R. Benenson, U. Franke, S. Roth, and B. Schiele. The
Cityscapes dataset for semantic urban scene understanding.
In CVPR, 2016. 9

[8] J. Dai, K. He, Y. Li, S. Ren, and J. Sun. Instance-sensitive

fully convolutional networks. In ECCV, 2016. 2

[9] J. Dai, K. He, and J. Sun. Convolutional feature masking for
joint object and stuff segmentation. In CVPR, 2015. 2
[10] J. Dai, K. He, and J. Sun. Instance-aware semantic segmen-
tation via multi-task network cascades. In CVPR, 2016. 2, 3,
4, 5, 6

[11] J. Dai, Y. Li, K. He, and J. Sun. R-FCN: Object detection via
region-based fully convolutional networks. In NIPS, 2016. 2

[12] R. Girshick. Fast R-CNN. In ICCV, 2015. 1, 2, 3, 4, 6
[13] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich fea-
ture hierarchies for accurate object detection and semantic
segmentation. In CVPR, 2014. 2, 3

[14] R. Girshick, F. Iandola, T. Darrell, and J. Malik. Deformable
In CVPR,

part models are convolutional neural networks.
2015. 4

[15] B. Hariharan, P. Arbel´aez, R. Girshick, and J. Malik. Simul-
taneous detection and segmentation. In ECCV. 2014. 2
[16] B. Hariharan, P. Arbel´aez, R. Girshick, and J. Malik. Hyper-
columns for object segmentation and ﬁne-grained localiza-
tion. In CVPR, 2015. 2

[17] Z. Hayder, X. He, and M. Salzmann. Shape-aware instance

segmentation. In CVPR, 2017. 9

[18] K. He, X. Zhang, S. Ren, and J. Sun. Spatial pyramid pooling
in deep convolutional networks for visual recognition.
In
ECCV. 2014. 1, 2

[19] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning

for image recognition. In CVPR, 2016. 2, 4, 7, 10

[20] J. Hosang, R. Benenson, P. Doll´ar, and B. Schiele. What
makes for effective detection proposals? PAMI, 2015. 2

[22] M.

Jaderberg, K. Simonyan, A. Zisserman,
transformer networks.

Spatial

and
In

K. Kavukcuoglu.
NIPS, 2015. 4

[23] A. Kirillov, E. Levinkov, B. Andres, B. Savchynskyy, and
C. Rother. Instancecut: from edges to instances with multi-
cut. In CVPR, 2017. 3, 9

[24] A. Krizhevsky, I. Sutskever, and G. Hinton. ImageNet clas-
siﬁcation with deep convolutional neural networks. In NIPS,
2012. 2

[25] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E.
Howard, W. Hubbard, and L. D. Jackel. Backpropagation
applied to handwritten zip code recognition. Neural compu-
tation, 1989. 2

[26] Y. Li, H. Qi, J. Dai, X. Ji, and Y. Wei. Fully convolutional
instance-aware semantic segmentation. In CVPR, 2017. 2,
3, 5, 6

[27] T.-Y. Lin, P. Doll´ar, R. Girshick, K. He, B. Hariharan, and
S. Belongie. Feature pyramid networks for object detection.
In CVPR, 2017. 2, 4, 5, 7

[28] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ra-
manan, P. Doll´ar, and C. L. Zitnick. Microsoft COCO: Com-
mon objects in context. In ECCV, 2014. 2, 5

[29] S. Liu, J. Jia, S. Fidler, and R. Urtasun. SGN: Sequen-
tial grouping networks for instance segmentation. In ICCV,
2017. 3, 9

[30] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional
networks for semantic segmentation. In CVPR, 2015. 1, 3, 6
[31] V. Nair and G. E. Hinton. Rectiﬁed linear units improve re-

stricted boltzmann machines. In ICML, 2010. 4

[32] G. Papandreou, T. Zhu, N. Kanazawa, A. Toshev, J. Tomp-
son, C. Bregler, and K. Murphy. Towards accurate multi-
person pose estimation in the wild. In CVPR, 2017. 8
[33] P. O. Pinheiro, R. Collobert, and P. Dollar. Learning to seg-

ment object candidates. In NIPS, 2015. 2, 3

[34] P. O. Pinheiro, T.-Y. Lin, R. Collobert, and P. Doll´ar. Learn-

ing to reﬁne object segments. In ECCV, 2016. 2, 3

[35] I. Radosavovic, P. Doll´ar, R. Girshick, G. Gkioxari, and
K. He. Data distillation: Towards omni-supervised learning.
arXiv:1712.04440, 2017. 10

[36] S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: To-
wards real-time object detection with region proposal net-
works. In NIPS, 2015. 1, 2, 3, 4, 7

[37] S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: To-
wards real-time object detection with region proposal net-
works. In TPAMI, 2017. 10

[38] A. Shrivastava, A. Gupta, and R. Girshick. Training region-
based object detectors with online hard example mining. In
CVPR, 2016. 2, 5

[39] A. Shrivastava, R. Sukthankar, J. Malik, and A. Gupta. Be-
yond skip connections: Top-down modulation for object de-
tection. arXiv:1612.06851, 2016. 4, 7

[40] C. Sun, A. Shrivastava, S. Singh, and A. Gupta. Revisiting
unreasonable effectiveness of data in deep learning era. In
ICCV, 2017. 10

11

[41] C. Szegedy, S. Ioffe, and V. Vanhoucke.

Inception-v4,
inception-resnet and the impact of residual connections on
learning. In ICLR Workshop, 2016. 7

[42] J. R. Uijlings, K. E. van de Sande, T. Gevers, and A. W.
IJCV,
Smeulders. Selective search for object recognition.
2013. 2

[43] X. Wang, R. Girshick, A. Gupta, and K. He. Non-local neural

networks. arXiv:1711.07971, 2017. 10

[44] S.-E. Wei, V. Ramakrishna, T. Kanade, and Y. Sheikh. Con-

volutional pose machines. In CVPR, 2016. 8

[45] S. Xie, R. Girshick, P. Doll´ar, Z. Tu, and K. He. Aggregated
residual transformations for deep neural networks. In CVPR,
2017. 4, 10

12


