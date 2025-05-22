CornerNet: Detecting Objects as Paired Keypoints

Hei Law · Jia Deng

9
1
0
2

r
a

M
8
1

]

V
C
.
s
c
[

2
v
4
4
2
1
0
.
8
0
8
1
:
v
i
X
r
a

Abstract We propose CornerNet, a new approach to
object detection where we detect an object bounding
box as a pair of keypoints, the top-left corner and the
bottom-right corner, using a single convolution neural
network. By detecting objects as paired keypoints, we
eliminate the need for designing a set of anchor boxes
commonly used in prior single-stage detectors. In addi-
tion to our novel formulation, we introduce corner pool-
ing, a new type of pooling layer that helps the network
better localize corners. Experiments show that Corner-
Net achieves a 42.2% AP on MS COCO, outperforming
all existing one-stage detectors.

Keywords Object Detection

1 Introduction

Object detectors based on convolutional neural net-
works (ConvNets) (Krizhevsky et al., 2012; Simonyan
and Zisserman, 2014; He et al., 2016) have achieved
state-of-the-art results on various challenging bench-
marks (Lin et al., 2014; Deng et al., 2009; Everingham
et al., 2015). A common component of state-of-the-art
approaches is anchor boxes (Ren et al., 2015; Liu et al.,
2016), which are boxes of various sizes and aspect ra-
tios that serve as detection candidates. Anchor boxes
are extensively used in one-stage detectors (Liu et al.,
2016; Fu et al., 2017; Redmon and Farhadi, 2016; Lin
et al., 2017), which can achieve results highly competi-
tive with two-stage detectors (Ren et al., 2015; Girshick
et al., 2014; Girshick, 2015; He et al., 2017) while being

H. Law
Princeton University, Princeton, NJ, USA
E-mail: heilaw@cs.princeton.edu

J. Deng
Princeton Universtiy, Princeton, NJ, USA

more eﬃcient. One-stage detectors place anchor boxes
densely over an image and generate ﬁnal box predic-
tions by scoring anchor boxes and reﬁning their coordi-
nates through regression.

But the use of anchor boxes has two drawbacks.
First, we typically need a very large set of anchor boxes,
e.g. more than 40k in DSSD (Fu et al., 2017) and more
than 100k in RetinaNet (Lin et al., 2017). This is be-
cause the detector is trained to classify whether each
anchor box suﬃciently overlaps with a ground truth
box, and a large number of anchor boxes is needed to
ensure suﬃcient overlap with most ground truth boxes.
As a result, only a tiny fraction of anchor boxes will
overlap with ground truth; this creates a huge imbal-
ance between positive and negative anchor boxes and
slows down training (Lin et al., 2017).

Second, the use of anchor boxes introduces many
hyperparameters and design choices. These include how
many boxes, what sizes, and what aspect ratios. Such
choices have largely been made via ad-hoc heuristics,
and can become even more complicated when combined
with multiscale architectures where a single network
makes separate predictions at multiple resolutions, with
each scale using diﬀerent features and its own set of an-
chor boxes (Liu et al., 2016; Fu et al., 2017; Lin et al.,
2017).

In this paper we introduce CornerNet, a new one-
stage approach to object detection that does away with
anchor boxes. We detect an object as a pair of keypoints—
the top-left corner and bottom-right corner of the bound-
ing box. We use a single convolutional network to pre-
dict a heatmap for the top-left corners of all instances
of the same object category, a heatmap for all bottom-
right corners, and an embedding vector for each de-
tected corner. The embeddings serve to group a pair of
corners that belong to the same object—the network is
trained to predict similar embeddings for them. Our ap-

2

Hei Law, Jia Deng

Fig. 1 We detect an object as a pair of bounding box corners grouped together. A convolutional network outputs a heatmap
for all top-left corners, a heatmap for all bottom-right corners, and an embedding vector for each detected corner. The network
is trained to predict similar embeddings for corners that belong to the same object.

proach greatly simpliﬁes the output of the network and
eliminates the need for designing anchor boxes. Our ap-
proach is inspired by the associative embedding method
proposed by Newell et al. (2017), who detect and group
keypoints in the context of multiperson human-pose es-
timation. Fig. 1 illustrates the overall pipeline of our
approach.

Another novel component of CornerNet is corner
pooling, a new type of pooling layer that helps a con-
volutional network better localize corners of bounding
boxes. A corner of a bounding box is often outside the
object—consider the case of a circle as well as the ex-
amples in Fig. 2. In such cases a corner cannot be lo-
calized based on local evidence. Instead, to determine
whether there is a top-left corner at a pixel location,
we need to look horizontally towards the right for the
topmost boundary of the object, and look vertically to-
wards the bottom for the leftmost boundary. This mo-
tivates our corner pooling layer: it takes in two feature
maps; at each pixel location it max-pools all feature
vectors to the right from the ﬁrst feature map, max-
pools all feature vectors directly below from the second
feature map, and then adds the two pooled results to-
gether. An example is shown in Fig. 3.

We hypothesize two reasons why detecting corners
would work better than bounding box centers or pro-
posals. First, the center of a box can be harder to lo-
calize because it depends on all 4 sides of the object,
whereas locating a corner depends on 2 sides and is thus
easier, and even more so with corner pooling, which en-
codes some explicit prior knowledge about the deﬁni-
tion of corners. Second, corners provide a more eﬃcient

way of densely discretizing the space of boxes: we just
need O(wh) corners to represent O(w2h2) possible an-
chor boxes.

We demonstrate the eﬀectiveness of CornerNet on
MS COCO (Lin et al., 2014). CornerNet achieves a
42.2% AP, outperforming all existing one-stage detec-
tors. In addition, through ablation studies we show that
corner pooling is critical to the superior performance of
CornerNet. Code is available at https://github.com/
princeton-vl/CornerNet.

2 Related Works

2.1 Two-stage object detectors

Two-stage approach was ﬁrst introduced and popular-
ized by R-CNN (Girshick et al., 2014). Two-stage de-
tectors generate a sparse set of regions of interest (RoIs)
and classify each of them by a network. R-CNN gener-
ates RoIs using a low level vision algorithm (Uijlings
et al., 2013; Zitnick and Doll´ar, 2014). Each region is
then extracted from the image and processed by a Con-
vNet independently, which creates lots of redundant
computations. Later, SPP (He et al., 2014) and Fast-
RCNN (Girshick, 2015) improve R-CNN by designing
a special pooling layer that pools each region from fea-
ture maps instead. However, both still rely on separate
proposal algorithms and cannot be trained end-to-end.
Faster-RCNN (Ren et al., 2015) does away low level
proposal algorithms by introducing a region proposal
network (RPN), which generates proposals from a set of

ConvNetEmbeddingsHeatmapsTop-Left CornersBottom-Right CornersCornerNet: Detecting Objects as Paired Keypoints

3

Fig. 2 Often there is no local evidence to determine the location of a bounding box corner. We address this issue by proposing
a new type of pooling layer.

Fig. 3 Corner pooling: for each channel, we take the maximum values (red dots) in two directions (red lines), each from a
separate feature map, and add the two maximums together (blue dot).

pre-determined candidate boxes, usually known as an-
chor boxes. This not only makes the detectors more eﬃ-
cient but also allows the detectors to be trained end-to-
end. R-FCN (Dai et al., 2016) further improves the eﬃ-
ciency of Faster-RCNN by replacing the fully connected
sub-detection network with a fully convolutional sub-
detection network. Other works focus on incorporating
sub-category information (Xiang et al., 2016), generat-
ing object proposals at multiple scales with more con-
textual information (Bell et al., 2016; Cai et al., 2016;
Shrivastava et al., 2016; Lin et al., 2016), selecting bet-
ter features (Zhai et al., 2017), improving speed (Li
et al., 2017), cascade procedure (Cai and Vasconcelos,
2017) and better training procedure (Singh and Davis,
2017).

2.2 One-stage object detectors

On the other hand, YOLO (Redmon et al., 2016) and
SSD (Liu et al., 2016) have popularized the one-stage
approach, which removes the RoI pooling step and de-
tects objects in a single network. One-stage detectors
are usually more computationally eﬃcient than two-

stage detectors while maintaining competitive perfor-
mance on diﬀerent challenging benchmarks.

SSD places anchor boxes densely over feature maps
from multiple scales, directly classiﬁes and reﬁnes each
anchor box. YOLO predicts bounding box coordinates
directly from an image, and is later improved in YOLO9000 (Red-
mon and Farhadi, 2016) by switching to anchor boxes.
DSSD (Fu et al., 2017) and RON (Kong et al., 2017)
adopt networks similar to the hourglass network (Newell
et al., 2016), enabling them to combine low-level and
high-level features via skip connections to predict bound-
ing boxes more accurately. However, these one-stage
detectors are still outperformed by the two-stage de-
tectors until the introduction of RetinaNet (Lin et al.,
2017). In (Lin et al., 2017), the authors suggest that
the dense anchor boxes create a huge imbalance be-
tween positive and negative anchor boxes during train-
ing. This imbalance causes the training to be ineﬃcient
and hence the performance to be suboptimal. They pro-
pose a new loss, Focal Loss, to dynamically adjust the
weights of each anchor box and show that their one-
stage detector can outperform the two-stage detectors.
ReﬁneDet (Zhang et al., 2017) proposes to ﬁlter the an-

feature mapsoutputtop-left corner pooling4

Hei Law, Jia Deng

chor boxes to reduce the number of negative boxes, and
to coarsely adjust the anchor boxes.

3 CornerNet

3.1 Overview

DeNet (Tychsen-Smith and Petersson, 2017a) is a
two-stage detector which generates RoIs without using
anchor boxes. It ﬁrst determines how likely each loca-
tion belongs to either the top-left, top-right, bottom-
left or bottom-right corner of a bounding box. It then
generates RoIs by enumerating all possible corner com-
binations, and follows the standard two-stage approach
to classify each RoI. Our approach is very diﬀerent from
DeNet. First, DeNet does not identify if two corners
are from the same objects and relies on a sub-detection
network to reject poor RoIs. In contrast, our approach
is a one-stage approach which detects and groups the
corners using a single ConvNet. Second, DeNet selects
features at manually determined locations relative to
a region for classiﬁcation, while our approach does not
require any feature selection step. Third, we introduce
corner pooling, a novel type of layer to enhance corner
detection.

Point Linking Network (PLN) (Wang et al., 2017)
is an one-stage detector without anchor boxes. It ﬁrst
predicts the locations of the four corners and the center
of a bounding box. Then, at each corner location, it pre-
dicts how likely each pixel location in the image is the
center. Similarly, at the center location, it predicts how
likely each pixel location belongs to either the top-left,
top-right, bottom-left or bottom-right corner. It com-
bines the predictions from each corner and center pair
to generate a bounding box. Finally, it merges the four
bounding boxes to give a bounding box. CornerNet is
very diﬀerent from PLN. First, CornerNet groups the
corners by predicting embedding vectors, while PLN
groups the corner and center by predicting pixel loca-
tions. Second, CornerNet uses corner pooling to better
localize the corners.

Our approach is inspired by Newell et al. (2017) on
Associative Embedding in the context of multi-person
pose estimation. Newell et al. propose an approach that
detects and groups human joints in a single network. In
their approach each detected human joint has an em-
bedding vector. The joints are grouped based on the
distances between their embeddings. To the best of our
knowledge, we are the ﬁrst to formulate the task of
object detection as a task of detecting and grouping
corners with embeddings. Another novelty of ours is
the corner pooling layers that help better localize the
corners. We also signiﬁcantly modify the hourglass ar-
chitecture and add our novel variant of focal loss (Lin
et al., 2017) to help better train the network.

In CornerNet, we detect an object as a pair of keypoints—
the top-left corner and bottom-right corner of the bound-
ing box. A convolutional network predicts two sets of
heatmaps to represent the locations of corners of dif-
ferent object categories, one set for the top-left corners
and the other for the bottom-right corners. The network
also predicts an embedding vector for each detected cor-
ner (Newell et al., 2017) such that the distance between
the embeddings of two corners from the same object
is small. To produce tighter bounding boxes, the net-
work also predicts oﬀsets to slightly adjust the locations
of the corners. With the predicted heatmaps, embed-
dings and oﬀsets, we apply a simple post-processing
algorithm to obtain the ﬁnal bounding boxes.

Fig. 4 provides an overview of CornerNet. We use
the hourglass network (Newell et al., 2016) as the back-
bone network of CornerNet. The hourglass network is
followed by two prediction modules. One module is for
the top-left corners, while the other one is for the bottom-
right corners. Each module has its own corner pooling
module to pool features from the hourglass network be-
fore predicting the heatmaps, embeddings and oﬀsets.
Unlike many other object detectors, we do not use fea-
tures from diﬀerent scales to detect objects of diﬀerent
sizes. We only apply both modules to the output of the
hourglass network.

3.2 Detecting Corners

We predict two sets of heatmaps, one for top-left corners
and one for bottom-right corners. Each set of heatmaps
has C channels, where C is the number of categories,
and is of size H × W . There is no background channel.
Each channel is a binary mask indicating the locations
of the corners for a class.

For each corner, there is one ground-truth positive
location, and all other locations are negative. During
training, instead of equally penalizing negative loca-
tions, we reduce the penalty given to negative locations
within a radius of the positive location. This is because
a pair of false corner detections, if they are close to
their respective ground truth locations, can still pro-
duce a box that suﬃciently overlaps the ground-truth
box (Fig. 5). We determine the radius by the size of an
object by ensuring that a pair of points within the ra-
dius would generate a bounding box with at least t IoU
with the ground-truth annotation (we set t to 0.3 in all
experiments). Given the radius, the amount of penalty
reduction is given by an unnormalized 2D Gaussian,

CornerNet: Detecting Objects as Paired Keypoints

5

Fig. 4 Overview of CornerNet. The backbone network is followed by two prediction modules, one for the top-left corners and
the other for the bottom-right corners. Using the predictions from both modules, we locate and group the corners.

n (cid:99), (cid:98) y

output is usually smaller than the image. Hence, a lo-
cation (x, y) in the image is mapped to the location
n (cid:99)(cid:1) in the heatmaps, where n is the downsam-
(cid:0)(cid:98) x
pling factor. When we remap the locations from the
heatmaps to the input image, some precision may be
lost, which can greatly aﬀect the IoU of small bounding
boxes with their ground truths. To address this issue we
predict location oﬀsets to slightly adjust the corner lo-
cations before remapping them to the input resolution.

Fig. 5 “Ground-truth” heatmaps for training. Boxes (green
dotted rectangles) whose corners are within the radii of the
positive locations (orange circles) still have large overlaps
with the ground-truth annotations (red solid rectangles).

2σ2 , whose center is at the positive location and

e− x2+y2
whose σ is 1/3 of the radius.

Let pcij be the score at location (i, j) for class c
in the predicted heatmaps, and let ycij be the “ground-
truth” heatmap augmented with the unnormalized Gaus-
sians. We design a variant of focal loss (Lin et al., 2017):

Ldet = −1
N

C
(cid:80)
c=1

H
(cid:80)
i=1

W
(cid:80)
j=1

(cid:26)

(1 − pcij)α log (pcij)

if ycij = 1
(1 − ycij)β (pcij)α log (1 − pcij) otherwise

(1)

where N is the number of objects in an image, and
α and β are the hyper-parameters which control the
contribution of each point (we set α to 2 and β to 4 in
all experiments). With the Gaussian bumps encoded in
ycij, the (1 − ycij) term reduces the penalty around the
ground truth locations.

Many networks (He et al., 2016; Newell et al., 2016)
involve downsampling layers to gather global informa-
tion and to reduce memory usage. When they are ap-
plied to an image fully convolutionally, the size of the

,

(cid:107)

−

−

(cid:107)(cid:17)

(2)

yk
n

ok =

(cid:106) xk
n

(cid:16) xk
n

(cid:106) yk
n
where ok is the oﬀset, xk and yk are the x and y coor-
dinate for corner k. In particular, we predict one set of
oﬀsets shared by the top-left corners of all categories,
and another set shared by the bottom-right corners. For
training, we apply the smooth L1 Loss (Girshick, 2015)
at ground-truth corner locations:

Loﬀ =

1
N

N
(cid:88)

k=1

SmoothL1Loss (ok, ˆok)

(3)

3.3 Grouping Corners

Multiple objects may appear in an image, and thus mul-
tiple top-left and bottom-right corners may be detected.
We need to determine if a pair of the top-left corner and
bottom-right corner is from the same bounding box.
Our approach is inspired by the Associative Embed-
ding method proposed by Newell et al. (2017) for the
task of multi-person pose estimation. Newell et al. de-
tect all human joints and generate an embedding for
each detected joint. They group the joints based on the
distances between the embeddings.

The idea of associative embedding is also applicable
to our task. The network predicts an embedding vector
for each detected corner such that if a top-left corner
and a bottom-right corner belong to the same bound-
ing box, the distance between their embeddings should

Hourglass NetworkOffsetsEmbeddingsHeatmapsCorner PoolingTop-left CornersBottom-right cornersPrediction ModulePrediction ModulePrediction Module6

Hei Law, Jia Deng

be small. We can then group the corners based on the
distances between the embeddings of the top-left and
bottom-right corners. The actual values of the embed-
dings are unimportant. Only the distances between the
embeddings are used to group the corners.

We follow Newell et al. (2017) and use embeddings
of 1 dimension. Let etk be the embedding for the top-left
corner of object k and ebk for the bottom-right corner.
As in Newell and Deng (2017), we use the “pull” loss to
train the network to group the corners and the “push”
loss to separate the corners:

Lpull =

1
N

N
(cid:88)

k=1

(cid:104)

(etk − ek)2 + (ebk − ek)2(cid:105)

,

(4)

Lpush =

1
N (N − 1)

N
(cid:88)

N
(cid:88)

k=1

j=1
j(cid:54)=k

max (0, ∆ − |ek − ej|) ,

(5)

where ek is the average of etk and ebk and we set ∆
to be 1 in all our experiments. Similar to the oﬀset
loss, we only apply the losses at the ground-truth corner
location.

3.4 Corner Pooling

As shown in Fig. 2, there is often no local visual ev-
idence for the presence of corners. To determine if a
pixel is a top-left corner, we need to look horizontally
towards the right for the topmost boundary of an ob-
ject and vertically towards the bottom for the leftmost
boundary. We thus propose corner pooling to better lo-
calize the corners by encoding explicit prior knowledge.
Suppose we want to determine if a pixel at location
(i, j) is a top-left corner. Let ft and fl be the feature
maps that are the inputs to the top-left corner pooling
layer, and let ftij and flij be the vectors at location
(i, j) in ft and fl respectively. With H × W feature
maps, the corner pooling layer ﬁrst max-pools all fea-
ture vectors between (i, j) and (i, H) in ft to a feature
vector tij, and max-pools all feature vectors between
(i, j) and (W, j) in fl to a feature vector lij. Finally,
it adds tij and lij together. This computation can be
expressed by the following equations:

tij =

(cid:26)max (cid:0)ftij , t(i+1)j
ftHj

(cid:1) if i < H
otherwise

lij =

(cid:26)max (cid:0)flij , li(j+1)
fliW

(cid:1) if j < W
otherwise

(6)

(7)

where we apply an elementwise max operation. Both
tij and lij can be computed eﬃciently by dynamic pro-
gramming as shown Fig. 8.

We deﬁne bottom-right corner pooling layer in a
similar way. It max-pools all feature vectors between
(0, j) and (i, j), and all feature vectors between (i, 0)
and (i, j) before adding the pooled results. The corner
pooling layers are used in the prediction modules to
predict heatmaps, embeddings and oﬀsets.

The architecture of the prediction module is shown
in Fig. 7. The ﬁrst part of the module is a modiﬁed
version of the residual block (He et al., 2016). In this
modiﬁed residual block, we replace the ﬁrst 3 × 3 con-
volution module with a corner pooling module, which
ﬁrst processes the features from the backbone network
by two 3 × 3 convolution modules 1 with 128 channels
and then applies a corner pooling layer. Following the
design of a residual block, we then feed the pooled fea-
tures into a 3 × 3 Conv-BN layer with 256 channels and
add back the projection shortcut. The modiﬁed residual
block is followed by a 3×3 convolution module with 256
channels, and 3 Conv-ReLU-Conv layers to produce the
heatmaps, embeddings and oﬀsets.

3.5 Hourglass Network

CornerNet uses the hourglass network (Newell et al.,
2016) as its backbone network. The hourglass network
was ﬁrst introduced for the human pose estimation task.
It is a fully convolutional neural network that consists
of one or more hourglass modules. An hourglass module
ﬁrst downsamples the input features by a series of con-
volution and max pooling layers. It then upsamples the
features back to the original resolution by a series of up-
sampling and convolution layers. Since details are lost
in the max pooling layers, skip layers are added to bring
back the details to the upsampled features. The hour-
glass module captures both global and local features
in a single uniﬁed structure. When multiple hourglass
modules are stacked in the network, the hourglass mod-
ules can reprocess the features to capture higher-level of
information. These properties make the hourglass net-
work an ideal choice for object detection as well. In fact,
many current detectors (Shrivastava et al., 2016; Fu
et al., 2017; Lin et al., 2016; Kong et al., 2017) already
adopted networks similar to the hourglass network.

Our hourglass network consists of two hourglasses,
and we make some modiﬁcations to the architecture
of the hourglass module. Instead of using max pool-

1 Unless otherwise speciﬁed, our convolution module con-
sists of a convolution layer, a BN layer (Ioﬀe and Szegedy,
2015) and a ReLU layer

CornerNet: Detecting Objects as Paired Keypoints

7

Fig. 6 The top-left corner pooling layer can be implemented very eﬃciently. We scan from right to left for the horizontal
max-pooling and from bottom to top for the vertical max-pooling. We then add two max-pooled feature maps.

Fig. 7 The prediction module starts with a modiﬁed residual block, in which we replace the ﬁrst convolution module with
our corner pooling module. The modiﬁed residual block is then followed by a convolution module. We have multiple branches
for predicting the heatmaps, embeddings and oﬀsets.

ing, we simply use stride 2 to reduce feature resolu-
tion. We reduce feature resolutions 5 times and in-
crease the number of feature channels along the way
(256, 384, 384, 384, 512). When we upsample the features,
we apply 2 residual modules followed by a nearest neigh-
bor upsampling. Every skip connection also consists of
2 residual modules. There are 4 residual modules with
512 channels in the middle of an hourglass module. Be-
fore the hourglass modules, we reduce the image res-
olution by 4 times using a 7 × 7 convolution module
with stride 2 and 128 channels followed by a residual
block (He et al., 2016) with stride 2 and 256 channels.

Following (Newell et al., 2016), we also add interme-
diate supervision in training. However, we do not add
back the intermediate predictions to the network as we
ﬁnd that this hurts the performance of the network. We
apply a 1 × 1 Conv-BN module to both the input and
output of the ﬁrst hourglass module. We then merge
them by element-wise addition followed by a ReLU and
a residual block with 256 channels, which is then used as
the input to the second hourglass module. The depth of
the hourglass network is 104. Unlike many other state-
of-the-art detectors, we only use the features from the
last layer of the whole network to make predictions.

4 Experiments

4.1 Training Details

We implement CornerNet in PyTorch (Paszke et al.,
2017). The network is randomly initialized under the
default setting of PyTorch with no pretraining on any
external dataset. As we apply focal loss, we follow (Lin
et al., 2017) to set the biases in the convolution layers
that predict the corner heatmaps. During training, we
set the input resolution of the network to 511 × 511,
which leads to an output resolution of 128 × 128. To
reduce overﬁtting, we adopt standard data augmenta-
tion techniques including random horizontal ﬂipping,
random scaling, random cropping and random color
jittering, which includes adjusting the brightness, sat-
uration and contrast of an image. Finally, we apply
PCA (Krizhevsky et al., 2012) to the input image.

We use Adam (Kingma and Ba, 2014) to optimize

the full training loss:

L = Ldet + αLpull + βLpush + γLoﬀ

(8)

where α, β and γ are the weights for the pull, push and
oﬀset loss respectively. We set both α and β to 0.1 and

203124561132033224442233366666112033422176910Top-left Corner PoolingBackboneHeatmapsEmbeddingsOffsets1x1 Conv3x3 Conv-ReLUReLU1x1 Conv-BN3x3 Conv-BN3x3 Conv-BN-ReLUTop-left Corner Pooling Module8

Hei Law, Jia Deng

Table 1 Ablation on corner pooling on MS COCO validation.

w/o corner pooling
w/ corner pooling

improvement

AP

36.5
38.4

+2.0

AP50
52.0
53.8

+2.1

AP75
38.8
40.9

+2.1

APs
17.5
18.6

+1.1

APm
38.9
40.5

+2.4

APl
49.4
51.8

+3.6

Table 2 Reducing the penalty given to the negative locations near positive locations helps signiﬁcantly improve the perfor-
mance of the network

w/o reducing penalty
ﬁxed radius
object-dependent radius

AP

32.9
35.6
38.4

AP50
49.1
52.5
53.8

AP75
34.8
37.7
40.9

APs
19.0
18.7
18.6

APm
37.0
38.5
40.5

APl
40.7
46.0
51.8

Table 3 Corner pooling consistently improves the network performance on detecting corners in diﬀerent image quadrants,
showing that corner pooling is eﬀective and stable over both small and large areas.

mAP w/o pooling mAP w/ pooling

improvement

Top-Left Corners
Top-Left Quad.
Bottom-Right Quad.

Bottom-Right Corners
Top-Left Quad.
Bottom-Right Quad.

66.1
60.8

53.4
65.0

69.2
63.5

56.2
67.6

+3.1
+2.7

+2.8
+2.6

γ to 1. We ﬁnd that 1 or larger values of α and β lead
to poor performance. We use a batch size of 49 and
train the network on 10 Titan X (PASCAL) GPUs (4
images on the master GPU, 5 images per GPU for the
rest of the GPUs). To conserve GPU resources, in our
ablation experiments, we train the networks for 250k
iterations with a learning rate of 2.5 × 10−4. When we
compare our results with other detectors, we train the
networks for an extra 250k iterations and reduce the
learning rate to 2.5 × 10−5 for the last 50k iterations.

4.2 Testing Details

During testing, we use a simple post-processing algo-
rithm to generate bounding boxes from the heatmaps,
embeddings and oﬀsets. We ﬁrst apply non-maximal
suppression (NMS) by using a 3×3 max pooling layer on
the corner heatmaps. Then we pick the top 100 top-left
and top 100 bottom-right corners from the heatmaps.
The corner locations are adjusted by the correspond-
ing oﬀsets. We calculate the L1 distances between the
embeddings of the top-left and bottom-right corners.
Pairs that have distances greater than 0.5 or contain
corners from diﬀerent categories are rejected. The aver-
age scores of the top-left and bottom-right corners are
used as the detection scores.

Instead of resizing an image to a ﬁxed size, we main-
tain the original resolution of the image and pad it with

zeros before feeding it to CornerNet. Both the original
and ﬂipped images are used for testing. We combine the
detections from the original and ﬂipped images, and ap-
ply soft-nms (Bodla et al., 2017) to suppress redundant
detections. Only the top 100 detections are reported.
The average inference time is 244ms per image on a
Titan X (PASCAL) GPU.

4.3 MS COCO

We evaluate CornerNet on the very challenging MS
COCO dataset (Lin et al., 2014). MS COCO contains
80k images for training, 40k for validation and 20k for
testing. All images in the training set and 35k images in
the validation set are used for training. The remaining
5k images in validation set are used for hyper-parameter
searching and ablation study. All results on the test set
are submitted to an external server for evaluation. To
provide fair comparisons with other detectors, we re-
port our main results on the test-dev set. MS COCO
uses average precisions (APs) at diﬀerent IoUs and APs
for diﬀerent object sizes as the main evaluation metrics.

CornerNet: Detecting Objects as Paired Keypoints

9

Fig. 8 Qualitative examples showing corner pooling helps better localize the corners.

Table 4 The hourglass network is crucial to the performance of CornerNet.

FPN (w/ ResNet-101) + Corners
Hourglass + Anchors
Hourglass + Corners

AP

30.2
32.9
38.4

AP50 AP75
32.0
44.1
35.6
53.1
40.9
53.8

APs
13.3
16.5
18.6

APm
33.3
38.5
40.5

APl
42.7
45.0
51.8

4.4 Ablation Study

4.4.1 Corner Pooling

Corner pooling is a key component of CornerNet. To
understand its contribution to performance, we train
another network without corner pooling but with the
same number of parameters.

Tab. 1 shows that adding corner pooling gives sig-
niﬁcant improvement: 2.0% on AP, 2.1% on AP50 and
2.1% on AP75. We also see that corner pooling is es-
pecially helpful for medium and large objects, improv-
ing their APs by 2.4% and 3.6% respectively. This is
expected because the topmost, bottommost, leftmost,
rightmost boundaries of medium and large objects are
likely to be further away from the corner locations.
Fig. 8 shows four qualitative examples with and with-
out corner pooling.

4.4.2 Stability of Corner Pooling over Larger Area

Corner pooling pools over diﬀerent sizes of area in dif-
ferent quadrants of an image. For example, the top-left
corner pooling pools over larger areas both horizontally
and vertically in the upper-left quadrant of an image,
compared to the lower-right quadrant. Therefore, the
location of a corner may aﬀect the stability of the cor-
ner pooling.

We evaluate the performance of our network on de-
tecting both the top-left and bottom-right corners in

diﬀerent quadrants of an image. Detecting corners can
be seen as a binary classiﬁcation task i.e. the ground-
truth location of a corner is positive, and any location
outside of a small radius of the corner is negative. We
measure the performance using mAPs over all cate-
gories on the MS COCO validation set.

Tab. 3 shows that without corner pooling, the top-
left corner mAPs of upper-left and lower-right quad-
rant are 66.1% and 60.8% respectively. Top-left cor-
ner pooling improves the mAPs by 3.1% (to 69.2%)
and 2.7% (to 63.5%) respectively. Similarly, bottom-
right corner pooling improves the bottom-right corner
mAPs of upper-left quadrant by 2.8% (from 53.4% to
56.2%), and lower-right quadrant by 2.6% (from 65.0%
to 67.6%). Corner pooling gives similar improvement to
corners at diﬀerent quadrants, show that corner pooling
is eﬀective and stable over both small and large areas.

4.4.3 Reducing Penalty to Negative Locations

We reduce the penalty given to negative locations around
a positive location, within a radius determined by the
size of the object (Sec. 3.2). To understand how this
helps train CornerNet, we train one network with no
penalty reduction and another network with a ﬁxed ra-
dius of 2.5. We compare them with CornerNet on the
validation set.

Tab. 2 shows that a ﬁxed radius improves AP over
the baseline by 2.7%, APm by 1.5% and APl by 5.3%.
Object-dependent radius further improves the AP by

w/o corner poolingw/ corner pooling10

Hei Law, Jia Deng

Table 5 CornerNet performs much better at high IoUs than other state-of-the-art detectors.

RetinaNet (Lin et al., 2017)
Cascade R-CNN (Cai and Vasconcelos, 2017)
Cascade R-CNN + IoU Net (Jiang et al., 2018)
CornerNet

AP

39.8
38.9
41.4
40.6

AP50 AP60 AP70 AP80 AP90
15.1
48.2
59.5
15.8
46.9
57.8
19.5
49.6
59.3
23.4
46.8
56.1

36.4
35.8
39.4
38.8

55.6
53.4
55.3
52.0

Table 6 Error analysis. We replace the predicted heatmaps and oﬀsets with the ground-truth values. Using the ground-truth
heatmaps alone improves the AP from 38.4% to 73.1%, suggesting that the main bottleneck of CornerNet is detecting corners.

w/ gt heatmaps
w/ gt heatmaps + oﬀsets

AP

38.4
73.1
86.1

AP50
53.8
87.7
88.9

AP75
40.9
78.4
85.5

APs
18.6
60.9
84.8

APm
40.5
81.2
87.2

APl
51.8
81.8
82.0

Fig. 9 Qualitative example showing errors in predicting corners and embeddings. The ﬁrst row shows images where CornerNet
mistakenly combines boundary evidence from diﬀerent objects. The second row shows images where CornerNet predicts similar
embeddings for corners from diﬀerent objects.

2.8%, APm by 2.0% and APl by 5.8%. In addition,
we see that the penalty reduction especially beneﬁts
medium and large objects.

4.4.4 Hourglass Network

CornerNet uses the hourglass network (Newell et al.,
2016) as its backbone network. Since the hourglass net-
work is not commonly used in other state-of-the-art de-
tectors, we perform an experiment to study the contri-
bution of the hourglass network in CornerNet. We train
a CornerNet in which we replace the hourglass network
with FPN (w/ ResNet-101) (Lin et al., 2017), which is
more commonly used in state-of-the-art object detec-
tors. We only use the ﬁnal output of FPN for predic-
tions. Meanwhile, we train an anchor box based detec-

tor which uses the hourglass network as its backbone.
Each hourglass module predicts anchor boxes at multi-
ple resolutions by using features at multiple scales dur-
ing upsampling stage. We follow the anchor box design
in RetinaNet (Lin et al., 2017) and add intermediate
supervisions during training. In both experiments, we
initialize the networks from scratch and follow the same
training procedure as we train CornerNet (Sec. 4.1).

Tab. 4 shows that CornerNet with hourglass net-
work outperforms CornerNet with FPN by 8.2% AP,
and the anchor box based detector with hourglass net-
work by 5.5% AP. The results suggest that the choice of
the backbone network is important and the hourglass
network is crucial to the performance of CornerNet.

CornerNet: Detecting Objects as Paired Keypoints

11

Table 7 CornerNet versus others on MS COCO test-dev. CornerNet outperforms all one-stage detectors and achieves results
competitive to two-stage detectors

Method

Backbone

AP

AP50 AP75 APs APm APl AR1 AR10 AR100 ARs ARm ARl

Two-stage detectors
ResNet-101
DeNet (Tychsen-Smith and Petersson, 2017a)
ResNet-101
CoupleNet (Zhu et al., 2017)
Inception-ResNet-v2 (Szegedy et al., 2017)
Faster R-CNN by G-RMI (Huang et al., 2017)
ResNet-101
Faster R-CNN+++ (He et al., 2016)
ResNet-101
Faster R-CNN w/ FPN (Lin et al., 2016)
Inception-ResNet-v2
Faster R-CNN w/ TDM (Shrivastava et al., 2016)
Aligned-Inception-ResNet
D-FCN (Dai et al., 2017)
ResNet-101
Regionlets (Xu et al., 2017)
ResNeXt-101
Mask R-CNN (He et al., 2017)
Aligned-Inception-ResNet
Soft-NMS (Bodla et al., 2017)
LH R-CNN (Li et al., 2017)
ResNet-101
Fitness-NMS (Tychsen-Smith and Petersson, 2017b) ResNet-101
ResNet-101
Cascade R-CNN (Cai and Vasconcelos, 2017)
DPN-98 (Chen et al., 2017)
D-RFCN + SNIP (Singh and Davis, 2017)

One-stage detectors
YOLOv2 (Redmon and Farhadi, 2016)
DSOD300 (Shen et al., 2017a)
GRP-DSOD320 (Shen et al., 2017b)
SSD513 (Liu et al., 2016)
DSSD513 (Fu et al., 2017)
ReﬁneDet512 (single scale) (Zhang et al., 2017)
RetinaNet800 (Lin et al., 2017)
ReﬁneDet512 (multi scale) (Zhang et al., 2017)
CornerNet511 (single scale)
CornerNet511 (multi scale)

DarkNet-19
DS/64-192-48-1
DS/64-192-48-1
ResNet-101
ResNet-101
ResNet-101
ResNet-101
ResNet-101
Hourglass-104
Hourglass-104

33.8
34.4
34.7
34.9
36.2
36.8
37.5
39.3
39.8
40.9
41.5
41.8
42.8
45.7

21.6
29.3
30.0
31.2
33.2
36.4
39.1
41.8
40.6
42.2

53.4
54.8
55.5
55.7
59.1
57.7
58.0
59.8
62.3
62.8
-
60.9
62.1
67.3

44.0
47.3
47.9
50.4
53.3
57.5
59.1
62.9
56.4
57.8

36.1
37.2
36.7
37.4
39.0
39.2
-
-
43.4
-
-
44.9
46.3
51.1

19.2
30.6
31.8
33.3
35.2
39.5
42.3
45.7
43.2
45.2

12.3
13.4
13.5
15.6
18.2
16.2
19.4
21.7
22.1
23.3
25.2
21.5
23.7
29.3

5.0
9.4
10.9
10.2
13.0
16.6
21.8
25.6
19.1
20.7

36.1
38.1
38.1
38.7
39.0
39.8
40.1
43.7
43.2
43.6
45.3
45.0
45.5
48.8

22.4
31.5
33.6
34.5
35.4
39.9
42.7
45.1
42.8
44.8

50.8
50.8
52.0
50.9
48.2
52.1
52.5
50.9
51.2
53.3
53.1
57.5
55.2
57.1

35.5
47.0
46.3
49.8
51.1
51.4
50.2
54.1
54.3
56.6

29.6
30.0
-
-
-
31.6
-
-
-
-
-
-
-
-

20.7
27.3
28.0
28.3
28.9
-
-
-
35.3
36.6

42.6
45.0
-
-
-
49.3
-
-
-
-
-
-
-
-

31.6
40.7
42.1
42.1
43.5
-
-
-
54.7
55.9

43.5
46.4
-
-
-
51.9
-
-
-
-
-
-
-
-

33.3
43.0
44.5
44.4
46.2
-
-
-
59.4
60.3

19.2
20.7
-
-
-
28.1
-
-
-
-
-
-
-
-

9.8
16.7
18.8
17.6
21.8
-
-
-
37.4
39.5

46.9
53.1
-
-
-
56.6
-
-
-
-
-
-
-
-

36.5
47.1
49.1
49.2
49.1
-
-
-
62.4
63.2

64.3
68.5
-
-
-
71.1
-
-
-
-
-
-
-
-

54.4
65.0
65.0
65.8
66.4
-
-
-
77.2
77.3

Fig. 10 Example bounding box predictions overlaid on predicted heatmaps of corners.

4.4.5 Quality of the Bounding Boxes

nerNet is able to generate bounding boxes of higher
quality compared to other state-of-the-art detectors.

A good detector should predict high quality bound-
ing boxes that cover objects tightly. To understand the
quality of the bounding boxes predicted by CornerNet,
we evaluate the performance of CornerNet at multi-
ple IoU thresholds, and compare the results with other
state-of-the-art detectors, including RetinaNet (Lin et al.,
2017), Cascade R-CNN (Cai and Vasconcelos, 2017)
and IoU-Net (Jiang et al., 2018).

Tab. 5 shows that CornerNet achieves a much higher
AP at 0.9 IoU than other detectors, outperforming Cas-
cade R-CNN + IoU-Net by 3.9%, Cascade R-CNN by
7.6% and RetinaNet 2 by 7.3%. This suggests that Cor-

2 We

use

the

best model

publicly

available

on

https://github.com/facebookresearch/Detectron/blob/
master/MODEL_ZOO.md

4.4.6 Error Analysis

CornerNet simultaneously outputs heatmaps, oﬀsets,
and embeddings, all of which aﬀect detection perfor-
mance. An object will be missed if either corner is
missed; precise oﬀsets are needed to generate tight bound-
ing boxes; incorrect embeddings will result in many
false bounding boxes. To understand how each part con-
tributes to the ﬁnal error, we perform an error analysis
by replacing the predicted heatmaps and oﬀsets with
the ground-truth values and evaluting performance on
the validation set.

Tab. 6 shows that using the ground-truth corner
heatmaps alone improves the AP from 38.4% to 73.1%.

12

Hei Law, Jia Deng

Fig. 11 Qualitative examples on MS COCO.

APs, APm and APl also increase by 42.3%, 40.7% and
30.0% respectively. If we replace the predicted oﬀsets
with the ground-truth oﬀsets, the AP further increases
by 13.0% to 86.1%. This suggests that although there
is still ample room for improvement in both detecting
and grouping corners, the main bottleneck is detecting
corners. Fig. 9 shows some qualitative examples where
the corner locations or embeddings are incorrect.

4.5 Comparisons with state-of-the-art detectors

We compare CornerNet with other state-of-the-art de-
tectors on MS COCO test-dev (Tab. 7). With multi-

scale evaluation, CornerNet achieves an AP of 42.2%,
the state of the art among existing one-stage methods
and competitive with two-stage methods.

5 Conclusion

We have presented CornerNet, a new approach to ob-
ject detection that detects bounding boxes as pairs of
corners. We evaluate CornerNet on MS COCO and
demonstrate competitive results.

CornerNet: Detecting Objects as Paired Keypoints

13

Acknowledgements This work is partially supported by
a grant from Toyota Research Institute and a DARPA grant
FA8750-18-2-0019. This article solely reﬂects the opinions and
conclusions of its authors.

References

Bell, S., Lawrence Zitnick, C., Bala, K., and Girshick,
R. (2016). Inside-outside net: Detecting objects in
context with skip pooling and recurrent neural net-
works. In Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition, pages
2874–2883.

Bodla, N., Singh, B., Chellappa, R., and Davis, L. S.
(2017). Soft-nmsimproving object detection with
one line of code. In 2017 IEEE International Con-
ference on Computer Vision (ICCV), pages 5562–
5570. IEEE.

Cai, Z., Fan, Q., Feris, R. S., and Vasconcelos, N.
(2016). A uniﬁed multi-scale deep convolutional
neural network for fast object detection. In Euro-
pean Conference on Computer Vision, pages 354–
370. Springer.

Cai, Z. and Vasconcelos, N. (2017). Cascade r-cnn:
Delving into high quality object detection. arXiv
preprint arXiv:1712.00726.

Chen, Y., Li, J., Xiao, H., Jin, X., Yan, S., and Feng, J.
(2017). Dual path networks. In Advances in Neural
Information Processing Systems, pages 4470–4478.
Dai, J., Li, Y., He, K., and Sun, J. (2016). R-fcn: Ob-
ject detection via region-based fully convolutional
networks. arXiv preprint arXiv:1605.06409.
Dai, J., Qi, H., Xiong, Y., Li, Y., Zhang, G., Hu, H.,
and Wei, Y. (2017). Deformable convolutional net-
works. CoRR, abs/1703.06211, 1(2):3.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and
Imagenet: A large-scale hi-
Fei-Fei, L. (2009).
erarchical image database.
In Computer Vision
and Pattern Recognition, 2009. CVPR 2009. IEEE
Conference on, pages 248–255. IEEE.

Everingham, M., Eslami, S. A., Van Gool, L., Williams,
C. K., Winn, J., and Zisserman, A. (2015). The
pascal visual object classes challenge: A retrospec-
tive.
International journal of computer vision,
111(1):98–136.

Fu, C.-Y., Liu, W., Ranga, A., Tyagi, A., and Berg,
A. C. (2017). Dssd: Deconvolutional single shot
detector. arXiv preprint arXiv:1701.06659.

Girshick, R. (2015).

Fast r-cnn.

arXiv preprint

arXiv:1504.08083.

Girshick, R., Donahue, J., Darrell, T., and Malik, J.
(2014). Rich feature hierarchies for accurate object
detection and semantic segmentation. In Proceed-

ings of the IEEE conference on computer vision
and pattern recognition, pages 580–587.

He, K., Gkioxari, G., Doll´ar, P., and Girshick,
R. (2017). Mask r-cnn. arxiv preprint arxiv:
170306870.

He, K., Zhang, X., Ren, S., and Sun, J. (2014). Spatial
pyramid pooling in deep convolutional networks
for visual recognition. In European Conference on
Computer Vision, pages 346–361. Springer.
He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep
residual learning for image recognition. In Proceed-
ings of the IEEE conference on computer vision
and pattern recognition, pages 770–778.

Huang, J., Rathod, V., Sun, C., Zhu, M., Korattikara,
A., Fathi, A., Fischer, I., Wojna, Z., Song, Y.,
Guadarrama, S., et al. (2017). Speed/accuracy
trade-oﬀs for modern convolutional object detec-
tors. In IEEE CVPR.

Ioﬀe, S. and Szegedy, C. (2015). Batch normalization:
Accelerating deep network training by reducing in-
ternal covariate shift. In International conference
on machine learning, pages 448–456.

Jiang, B., Luo, R., Mao, J., Xiao, T., and Jiang, Y.
(2018). Acquisition of localization conﬁdence for
accurate object detection.
In Computer Vision–
ECCV 2018, pages 816–832. Springer.

Kingma, D. P. and Ba, J.

Adam: A
method for stochastic optimization. arXiv preprint
arXiv:1412.6980.

(2014).

Kong, T., Sun, F., Yao, A., Liu, H., Lu, M., and Chen,
Y. (2017). Ron: Reverse connection with object-
ness prior networks for object detection. arXiv
preprint arXiv:1707.01691.

Krizhevsky, A., Sutskever, I., and Hinton, G. E. (2012).
Imagenet classiﬁcation with deep convolutional
neural networks. In Advances in neural informa-
tion processing systems, pages 1097–1105.

Li, Z., Peng, C., Yu, G., Zhang, X., Deng, Y., and Sun,
J. (2017). Light-head r-cnn: In defense of two-stage
object detector. arXiv preprint arXiv:1711.07264.
Lin, T.-Y., Doll´ar, P., Girshick, R., He, K., Hariha-
ran, B., and Belongie, S. (2016). Feature pyra-
mid networks for object detection. arXiv preprint
arXiv:1612.03144.

Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Doll´ar,
P. (2017). Focal loss for dense object detection.
arXiv preprint arXiv:1708.02002.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona,
P., Ramanan, D., Doll´ar, P., and Zitnick, C. L.
(2014). Microsoft coco: Common objects in con-
text. In European conference on computer vision,
pages 740–755. Springer.

14

Hei Law, Jia Deng

volume 4, page 12.

Tychsen-Smith, L. and Petersson, L. (2017a). Denet:
Scalable real-time object detection with directed
sparse sampling. arXiv preprint arXiv:1703.10295.
(2017b).
ﬁtness
arXiv preprint

Tychsen-Smith, L.
Improving
object
nms and bounded iou loss.
arXiv:1711.00164.

and Petersson, L.

localization with

Uijlings, J. R., van de Sande, K. E., Gevers, T., and
Smeulders, A. W. (2013). Selective search for ob-
ject recognition. International journal of computer
vision, 104(2):154–171.

Wang, X., Chen, K., Huang, Z., Yao, C., and Liu, W.
(2017). Point linking network for object detection.
arXiv preprint arXiv:1706.03646.

Xiang, Y., Choi, W., Lin, Y., and Savarese, S. (2016).
Subcategory-aware convolutional neural networks
for object proposals and detection. arXiv preprint
arXiv:1604.04693.

Xu, H., Lv, X., Wang, X., Ren, Z., and Chellappa, R.
(2017). Deep regionlets for object detection. arXiv
preprint arXiv:1712.02408.

Zhai, Y., Fu, J., Lu, Y., and Li, H. (2017). Feature selec-
tive networks for object detection. arXiv preprint
arXiv:1711.08879.

Zhang, S., Wen, L., Bian, X., Lei, Z., and Li, S. Z.
(2017). Single-shot reﬁnement neural network for
object detection. arXiv preprint arXiv:1711.06897.
Zhu, Y., Zhao, C., Wang, J., Zhao, X., Wu, Y., and Lu,
H. (2017). Couplenet: Coupling global structure
with local parts for object detection. In Proc. of
Intl Conf. on Computer Vision (ICCV).

Zitnick, C. L. and Doll´ar, P. (2014). Edge boxes: Lo-
cating object proposals from edges. In European
Conference on Computer Vision, pages 391–405.
Springer.

Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed,
S., Fu, C.-Y., and Berg, A. C. (2016). Ssd: Single
shot multibox detector. In European conference on
computer vision, pages 21–37. Springer.

Newell, A. and Deng, J. (2017). Pixels to graphs by
associative embedding. In Advances in Neural In-
formation Processing Systems, pages 2168–2177.

Newell, A., Huang, Z., and Deng, J. (2017). Associative
embedding: End-to-end learning for joint detection
and grouping. In Advances in Neural Information
Processing Systems, pages 2274–2284.

Newell, A., Yang, K., and Deng, J. (2016). Stacked
hourglass networks for human pose estimation. In
European Conference on Computer Vision, pages
483–499. Springer.

Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang,
E., DeVito, Z., Lin, Z., Desmaison, A., Antiga, L.,
and Lerer, A. (2017). Automatic diﬀerentiation in
pytorch.

Redmon, J., Divvala, S., Girshick, R., and Farhadi, A.
(2016). You only look once: Uniﬁed, real-time ob-
ject detection. In Proceedings of the IEEE confer-
ence on computer vision and pattern recognition,
pages 779–788.

Redmon, J. and Farhadi, A. (2016). Yolo9000: better,

faster, stronger. arXiv preprint, 1612.

Ren, S., He, K., Girshick, R., and Sun, J. (2015). Faster
r-cnn: Towards real-time object detection with re-
gion proposal networks. In Advances in neural in-
formation processing systems, pages 91–99.

Shen, Z., Liu, Z., Li, J., Jiang, Y.-G., Chen, Y., and
Xue, X. (2017a). Dsod: Learning deeply supervised
object detectors from scratch. In The IEEE Inter-
national Conference on Computer Vision (ICCV),
volume 3, page 7.

Shen, Z., Shi, H., Feris, R., Cao, L., Yan, S., Liu,
D., Wang, X., Xue, X., and Huang, T. S.
(2017b). Learning object detectors from scratch
with gated recurrent feature pyramids.
arXiv
preprint arXiv:1712.00886.

Shrivastava, A., Sukthankar, R., Malik, J., and Gupta,
A. (2016). Beyond skip connections: Top-down
modulation for object detection. arXiv preprint
arXiv:1612.06851.

Simonyan, K. and Zisserman, A. (2014). Very deep con-
volutional networks for large-scale image recogni-
tion. arXiv preprint arXiv:1409.1556.

Singh, B. and Davis, L. S. (2017). An analysis of scale
invariance in object detection-snip. arXiv preprint
arXiv:1711.08189.

Szegedy, C., Ioﬀe, S., Vanhoucke, V., and Alemi, A. A.
(2017). Inception-v4, inception-resnet and the im-
pact of residual connections on learning. In AAAI,


