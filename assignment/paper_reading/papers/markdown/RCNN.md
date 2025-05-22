Rich feature hierarchies for accurate object detection and semantic segmentation
Tech report (v5)

Ross Girshick Jeff Donahue Trevor Darrell Jitendra Malik
UC Berkeley
{rbg,jdonahue,trevor,malik}@eecs.berkeley.edu

4
1
0
2

t
c
O
2
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
4
2
5
2
.
1
1
3
1
:
v
i
X
r
a

Abstract

Object detection performance, as measured on the
canonical PASCAL VOC dataset, has plateaued in the last
few years. The best-performing methods are complex en-
semble systems that typically combine multiple low-level
image features with high-level context.
In this paper, we
propose a simple and scalable detection algorithm that im-
proves mean average precision (mAP) by more than 30%
relative to the previous best result on VOC 2012—achieving
a mAP of 53.3%. Our approach combines two key insights:
(1) one can apply high-capacity convolutional neural net-
works (CNNs) to bottom-up region proposals in order to
localize and segment objects and (2) when labeled training
data is scarce, supervised pre-training for an auxiliary task,
followed by domain-speciﬁc ﬁne-tuning, yields a signiﬁcant
performance boost. Since we combine region proposals
with CNNs, we call our method R-CNN: Regions with CNN
features. We also compare R-CNN to OverFeat, a recently
proposed sliding-window detector based on a similar CNN
architecture. We ﬁnd that R-CNN outperforms OverFeat
by a large margin on the 200-class ILSVRC2013 detection
dataset. Source code for the complete system is available at
http://www.cs.berkeley.edu/˜rbg/rcnn.

1. Introduction

Features matter. The last decade of progress on various
visual recognition tasks has been based considerably on the
use of SIFT [29] and HOG [7]. But if we look at perfor-
mance on the canonical visual recognition task, PASCAL
VOC object detection [15], it is generally acknowledged
that progress has been slow during 2010-2012, with small
gains obtained by building ensemble systems and employ-
ing minor variants of successful methods.

SIFT and HOG are blockwise orientation histograms,
a representation we could associate roughly with complex
cells in V1, the ﬁrst cortical area in the primate visual path-
way. But we also know that recognition occurs several
stages downstream, which suggests that there might be hier-

Figure 1: Object detection system overview. Our system (1)
takes an input image, (2) extracts around 2000 bottom-up region
proposals, (3) computes features for each proposal using a large
convolutional neural network (CNN), and then (4) classiﬁes each
region using class-speciﬁc linear SVMs. R-CNN achieves a mean
average precision (mAP) of 53.7% on PASCAL VOC 2010. For
comparison, [39] reports 35.1% mAP using the same region pro-
posals, but with a spatial pyramid and bag-of-visual-words ap-
proach. The popular deformable part models perform at 33.4%.
On the 200-class ILSVRC2013 detection dataset, R-CNN’s
mAP is 31.4%, a large improvement over OverFeat [34], which
had the previous best result at 24.3%.

[19],

Fukushima’s

archical, multi-stage processes for computing features that
are even more informative for visual recognition.
“neocognitron”

a biologically-
inspired hierarchical and shift-invariant model for pattern
recognition, was an early attempt at just such a process.
The neocognitron, however, lacked a supervised training
algorithm. Building on Rumelhart et al.
[33], LeCun et
al. [26] showed that stochastic gradient descent via back-
propagation was effective for training convolutional neural
networks (CNNs), a class of models that extend the neocog-
nitron.

CNNs saw heavy use in the 1990s (e.g., [27]), but then
fell out of fashion with the rise of support vector machines.
In 2012, Krizhevsky et al. [25] rekindled interest in CNNs
by showing substantially higher image classiﬁcation accu-
racy on the ImageNet Large Scale Visual Recognition Chal-
lenge (ILSVRC) [9, 10]. Their success resulted from train-
ing a large CNN on 1.2 million labeled images, together
with a few twists on LeCun’s CNN (e.g., max(x, 0) rectify-
ing non-linearities and “dropout” regularization).

The signiﬁcance of the ImageNet result was vigorously

1

1. Input image2. Extract region proposals (~2k)3. Compute CNN featuresaeroplane? no....person? yes.tvmonitor? no.4. Classify regionswarped region...CNNR-CNN: Regions with CNN features

debated during the ILSVRC 2012 workshop. The central
issue can be distilled to the following: To what extent do
the CNN classiﬁcation results on ImageNet generalize to
object detection results on the PASCAL VOC Challenge?

We answer this question by bridging the gap between
image classiﬁcation and object detection. This paper is the
ﬁrst to show that a CNN can lead to dramatically higher ob-
ject detection performance on PASCAL VOC as compared
to systems based on simpler HOG-like features. To achieve
this result, we focused on two problems: localizing objects
with a deep network and training a high-capacity model
with only a small quantity of annotated detection data.

Unlike image classiﬁcation, detection requires localiz-
ing (likely many) objects within an image. One approach
frames localization as a regression problem. However, work
from Szegedy et al. [38], concurrent with our own, indi-
cates that this strategy may not fare well in practice (they
report a mAP of 30.5% on VOC 2007 compared to the
58.5% achieved by our method). An alternative is to build a
sliding-window detector. CNNs have been used in this way
for at least two decades, typically on constrained object cat-
egories, such as faces [32, 40] and pedestrians [35]. In order
to maintain high spatial resolution, these CNNs typically
only have two convolutional and pooling layers. We also
considered adopting a sliding-window approach. However,
units high up in our network, which has ﬁve convolutional
layers, have very large receptive ﬁelds (195 × 195 pixels)
and strides (32×32 pixels) in the input image, which makes
precise localization within the sliding-window paradigm an
open technical challenge.

Instead, we solve the CNN localization problem by oper-
ating within the “recognition using regions” paradigm [21],
which has been successful for both object detection [39] and
semantic segmentation [5]. At test time, our method gener-
ates around 2000 category-independent region proposals for
the input image, extracts a ﬁxed-length feature vector from
each proposal using a CNN, and then classiﬁes each region
with category-speciﬁc linear SVMs. We use a simple tech-
nique (afﬁne image warping) to compute a ﬁxed-size CNN
input from each region proposal, regardless of the region’s
shape. Figure 1 presents an overview of our method and
highlights some of our results. Since our system combines
region proposals with CNNs, we dub the method R-CNN:
Regions with CNN features.

In this updated version of this paper, we provide a head-
to-head comparison of R-CNN and the recently proposed
OverFeat [34] detection system by running R-CNN on the
200-class ILSVRC2013 detection dataset. OverFeat uses a
sliding-window CNN for detection and until now was the
best performing method on ILSVRC2013 detection. We
show that R-CNN signiﬁcantly outperforms OverFeat, with
a mAP of 31.4% versus 24.3%.

A second challenge faced in detection is that labeled data

is scarce and the amount currently available is insufﬁcient
for training a large CNN. The conventional solution to this
problem is to use unsupervised pre-training, followed by su-
pervised ﬁne-tuning (e.g., [35]). The second principle con-
tribution of this paper is to show that supervised pre-training
on a large auxiliary dataset (ILSVRC), followed by domain-
speciﬁc ﬁne-tuning on a small dataset (PASCAL), is an
effective paradigm for learning high-capacity CNNs when
data is scarce. In our experiments, ﬁne-tuning for detection
improves mAP performance by 8 percentage points. After
ﬁne-tuning, our system achieves a mAP of 54% on VOC
2010 compared to 33% for the highly-tuned, HOG-based
deformable part model (DPM) [17, 20]. We also point read-
ers to contemporaneous work by Donahue et al. [12], who
show that Krizhevsky’s CNN can be used (without ﬁne-
tuning) as a blackbox feature extractor, yielding excellent
performance on several recognition tasks including scene
classiﬁcation, ﬁne-grained sub-categorization, and domain
adaptation.

Our system is also quite efﬁcient. The only class-speciﬁc
computations are a reasonably small matrix-vector product
and greedy non-maximum suppression. This computational
property follows from features that are shared across all cat-
egories and that are also two orders of magnitude lower-
dimensional than previously used region features (cf. [39]).
Understanding the failure modes of our approach is also
critical for improving it, and so we report results from the
detection analysis tool of Hoiem et al.
[23]. As an im-
mediate consequence of this analysis, we demonstrate that
a simple bounding-box regression method signiﬁcantly re-
duces mislocalizations, which are the dominant error mode.
Before developing technical details, we note that because
R-CNN operates on regions it is natural to extend it to the
task of semantic segmentation. With minor modiﬁcations,
we also achieve competitive results on the PASCAL VOC
segmentation task, with an average segmentation accuracy
of 47.9% on the VOC 2011 test set.

2. Object detection with R-CNN

Our object detection system consists of three modules.
The ﬁrst generates category-independent region proposals.
These proposals deﬁne the set of candidate detections avail-
able to our detector. The second module is a large convo-
lutional neural network that extracts a ﬁxed-length feature
vector from each region. The third module is a set of class-
speciﬁc linear SVMs. In this section, we present our design
decisions for each module, describe their test-time usage,
detail how their parameters are learned, and show detection
results on PASCAL VOC 2010-12 and on ILSVRC2013.

2.1. Module design

Region proposals. A variety of recent papers offer meth-
ods for generating category-independent region proposals.

2

are low-dimensional when compared to other common ap-
proaches, such as spatial pyramids with bag-of-visual-word
encodings. The features used in the UVA detection system
[39], for example, are two orders of magnitude larger than
ours (360k vs. 4k-dimensional).

The result of such sharing is that the time spent com-
puting region proposals and features (13s/image on a GPU
or 53s/image on a CPU) is amortized over all classes. The
only class-speciﬁc computations are dot products between
features and SVM weights and non-maximum suppression.
In practice, all dot products for an image are batched into
a single matrix-matrix product. The feature matrix is typi-
cally 2000 × 4096 and the SVM weight matrix is 4096 × N ,
where N is the number of classes.

This analysis shows that R-CNN can scale to thousands
of object classes without resorting to approximate tech-
niques, such as hashing. Even if there were 100k classes,
the resulting matrix multiplication takes only 10 seconds on
a modern multi-core CPU. This efﬁciency is not merely the
result of using region proposals and shared features. The
UVA system, due to its high-dimensional features, would
be two orders of magnitude slower while requiring 134GB
of memory just to store 100k linear predictors, compared to
just 1.5GB for our lower-dimensional features.

It is also interesting to contrast R-CNN with the recent
work from Dean et al. on scalable detection using DPMs
and hashing [8]. They report a mAP of around 16% on VOC
2007 at a run-time of 5 minutes per image when introducing
10k distractor classes. With our approach, 10k detectors can
run in about a minute on a CPU, and because no approxi-
mations are made mAP would remain at 59% (Section 3.2).

2.3. Training

Supervised pre-training. We discriminatively pre-trained
the CNN on a large auxiliary dataset (ILSVRC2012 clas-
siﬁcation) using image-level annotations only (bounding-
box labels are not available for this data). Pre-training
was performed using the open source Caffe CNN library
In brief, our CNN nearly matches the performance
[24].
of Krizhevsky et al. [25], obtaining a top-1 error rate 2.2
percentage points higher on the ILSVRC2012 classiﬁcation
validation set. This discrepancy is due to simpliﬁcations in
the training process.

Domain-speciﬁc ﬁne-tuning. To adapt our CNN to the
new task (detection) and the new domain (warped proposal
windows), we continue stochastic gradient descent (SGD)
training of the CNN parameters using only warped region
proposals. Aside from replacing the CNN’s ImageNet-
speciﬁc 1000-way classiﬁcation layer with a randomly ini-
tialized (N + 1)-way classiﬁcation layer (where N is the
number of object classes, plus 1 for background), the CNN
architecture is unchanged. For VOC, N = 20 and for
ILSVRC2013, N = 200. We treat all region proposals with

Figure 2: Warped training samples from VOC 2007 train.

Examples include: objectness [1], selective search [39],
category-independent object proposals [14], constrained
parametric min-cuts (CPMC) [5], multi-scale combinatorial
grouping [3], and Cires¸an et al. [6], who detect mitotic cells
by applying a CNN to regularly-spaced square crops, which
are a special case of region proposals. While R-CNN is ag-
nostic to the particular region proposal method, we use se-
lective search to enable a controlled comparison with prior
detection work (e.g., [39, 41]).

Feature extraction. We extract a 4096-dimensional fea-
ture vector from each region proposal using the Caffe [24]
implementation of the CNN described by Krizhevsky et
al.
[25]. Features are computed by forward propagating
a mean-subtracted 227 × 227 RGB image through ﬁve con-
volutional layers and two fully connected layers. We refer
readers to [24, 25] for more network architecture details.

In order to compute features for a region proposal, we
must ﬁrst convert the image data in that region into a form
that is compatible with the CNN (its architecture requires
inputs of a ﬁxed 227 × 227 pixel size). Of the many possi-
ble transformations of our arbitrary-shaped regions, we opt
for the simplest. Regardless of the size or aspect ratio of the
candidate region, we warp all pixels in a tight bounding box
around it to the required size. Prior to warping, we dilate the
tight bounding box so that at the warped size there are ex-
actly p pixels of warped image context around the original
box (we use p = 16). Figure 2 shows a random sampling
of warped training regions. Alternatives to warping are dis-
cussed in Appendix A.

2.2. Test-time detection

At test time, we run selective search on the test image
to extract around 2000 region proposals (we use selective
search’s “fast mode” in all experiments). We warp each
proposal and forward propagate it through the CNN in or-
der to compute features. Then, for each class, we score
each extracted feature vector using the SVM trained for that
class. Given all scored regions in an image, we apply a
greedy non-maximum suppression (for each class indepen-
dently) that rejects a region if it has an intersection-over-
union (IoU) overlap with a higher scoring selected region
larger than a learned threshold.

Run-time analysis. Two properties make detection efﬁ-
cient. First, all CNN parameters are shared across all cate-
gories. Second, the feature vectors computed by the CNN

3

aeroplanebicyclebirdcar≥ 0.5 IoU overlap with a ground-truth box as positives for
that box’s class and the rest as negatives. We start SGD at
a learning rate of 0.001 (1/10th of the initial pre-training
rate), which allows ﬁne-tuning to make progress while not
clobbering the initialization. In each SGD iteration, we uni-
formly sample 32 positive windows (over all classes) and
96 background windows to construct a mini-batch of size
128. We bias the sampling towards positive windows be-
cause they are extremely rare compared to background.

Object category classiﬁers. Consider training a binary
classiﬁer to detect cars.
It’s clear that an image region
tightly enclosing a car should be a positive example. Simi-
larly, it’s clear that a background region, which has nothing
to do with cars, should be a negative example. Less clear
is how to label a region that partially overlaps a car. We re-
solve this issue with an IoU overlap threshold, below which
regions are deﬁned as negatives. The overlap threshold, 0.3,
was selected by a grid search over {0, 0.1, . . . , 0.5} on a
validation set. We found that selecting this threshold care-
fully is important. Setting it to 0.5, as in [39], decreased
mAP by 5 points. Similarly, setting it to 0 decreased mAP
by 4 points. Positive examples are deﬁned simply to be the
ground-truth bounding boxes for each class.

Once features are extracted and training labels are ap-
plied, we optimize one linear SVM per class. Since the
training data is too large to ﬁt in memory, we adopt the
standard hard negative mining method [17, 37]. Hard neg-
ative mining converges quickly and in practice mAP stops
increasing after only a single pass over all images.

In Appendix B we discuss why the positive and negative
examples are deﬁned differently in ﬁne-tuning versus SVM
training. We also discuss the trade-offs involved in training
detection SVMs rather than simply using the outputs from
the ﬁnal softmax layer of the ﬁne-tuned CNN.

densely sampled SIFT, Extended OpponentSIFT, and RGB-
SIFT descriptors, each vector quantized with 4000-word
codebooks. Classiﬁcation is performed with a histogram
intersection kernel SVM. Compared to their multi-feature,
non-linear kernel SVM approach, we achieve a large im-
provement in mAP, from 35.1% to 53.7% mAP, while also
being much faster (Section 2.2). Our method achieves sim-
ilar performance (53.3% mAP) on VOC 2011/12 test.

2.5. Results on ILSVRC2013 detection

We ran R-CNN on the 200-class ILSVRC2013 detection
dataset using the same system hyperparameters that we used
for PASCAL VOC. We followed the same protocol of sub-
mitting test results to the ILSVRC2013 evaluation server
only twice, once with and once without bounding-box re-
gression.

Figure 3 compares R-CNN to the entries in the ILSVRC
2013 competition and to the post-competition OverFeat re-
sult [34]. R-CNN achieves a mAP of 31.4%, which is sig-
niﬁcantly ahead of the second-best result of 24.3% from
OverFeat. To give a sense of the AP distribution over
classes, box plots are also presented and a table of per-
class APs follows at the end of the paper in Table 8. Most
of the competing submissions (OverFeat, NEC-MU, UvA-
Euvision, Toronto A, and UIUC-IFP) used convolutional
neural networks, indicating that there is signiﬁcant nuance
in how CNNs can be applied to object detection, leading to
greatly varying outcomes.

In Section 4, we give an overview of the ILSVRC2013
detection dataset and provide details about choices that we
made when running R-CNN on it.

3. Visualization, ablation, and modes of error

2.4. Results on PASCAL VOC 2010-12

3.1. Visualizing learned features

Following the PASCAL VOC best practices [15], we
validated all design decisions and hyperparameters on the
VOC 2007 dataset (Section 3.2). For ﬁnal results on the
VOC 2010-12 datasets, we ﬁne-tuned the CNN on VOC
2012 train and optimized our detection SVMs on VOC 2012
trainval. We submitted test results to the evaluation server
only once for each of the two major algorithm variants (with
and without bounding-box regression).

Table 1 shows complete results on VOC 2010. We com-
pare our method against four strong baselines, including
SegDPM [18], which combines DPM detectors with the
output of a semantic segmentation system [4] and uses ad-
ditional inter-detector context and image-classiﬁer rescor-
ing. The most germane comparison is to the UVA system
from Uijlings et al. [39], since our systems use the same re-
gion proposal algorithm. To classify regions, their method
builds a four-level spatial pyramid and populates it with

First-layer ﬁlters can be visualized directly and are easy
to understand [25]. They capture oriented edges and oppo-
nent colors. Understanding the subsequent layers is more
challenging. Zeiler and Fergus present a visually attrac-
tive deconvolutional approach in [42]. We propose a simple
(and complementary) non-parametric method that directly
shows what the network learned.

The idea is to single out a particular unit (feature) in the
network and use it as if it were an object detector in its own
right. That is, we compute the unit’s activations on a large
set of held-out region proposals (about 10 million), sort the
proposals from highest to lowest activation, perform non-
maximum suppression, and then display the top-scoring re-
gions. Our method lets the selected unit “speak for itself”
by showing exactly which inputs it ﬁres on. We avoid aver-
aging in order to see different visual modes and gain insight
into the invariances computed by the unit.

4

cat

VOC 2010 test aero bike bird boat bottle bus
car
DPM v5 [20]†
49.2 53.8 13.1 15.3 35.5 53.4 49.7 27.0 17.2 28.8 14.7 17.8 46.4
UVA [39]
56.2 42.4 15.3 12.6 21.8 49.3 36.8 46.1 12.9 32.1 30.0 36.5 43.5
Regionlets [41] 65.0 48.9 25.9 24.6 24.5 56.1 54.5 51.2 17.0 28.9 30.2 35.8 40.2
SegDPM [18]† 61.4 53.4 25.6 25.2 35.5 51.7 50.6 50.8 19.3 33.8 26.8 40.4 48.3
67.1 64.1 46.7 32.0 30.5 56.4 57.2 65.9 27.0 47.3 40.9 66.6 57.8
R-CNN
71.8 65.8 53.0 36.8 35.9 59.7 60.0 69.9 27.9 50.6 41.4 70.0 62.0
R-CNN BB

chair cow table dog horse mbike person plant sheep sofa train
51.2
52.9
55.7
54.4
65.9
69.0

tv mAP
34.2 20.7 43.8 38.3 33.4
41.1 31.8 47.0 44.8 35.1
43.9 32.6 54.0 45.9 39.7
38.7 35.0 52.8 43.1 40.4
56.5 38.1 52.8 50.2 50.2
59.4 39.3 61.2 52.4 53.7

47.7
32.9
43.5
47.1
53.6
58.1

10.8
15.3
14.3
14.8
26.7
29.5

Table 1: Detection average precision (%) on VOC 2010 test. R-CNN is most directly comparable to UVA and Regionlets since all
methods use selective search region proposals. Bounding-box regression (BB) is described in Section C. At publication time, SegDPM
was the top-performer on the PASCAL VOC leaderboard. †DPM and SegDPM use context rescoring not used by the other methods.

Figure 3: (Left) Mean average precision on the ILSVRC2013 detection test set. Methods preceeded by * use outside training data
(images and labels from the ILSVRC classiﬁcation dataset in all cases). (Right) Box plots for the 200 average precision values per
method. A box plot for the post-competition OverFeat result is not shown because per-class APs are not yet available (per-class APs for
R-CNN are in Table 8 and also included in the tech report source uploaded to arXiv.org; see R-CNN-ILSVRC2013-APs.txt). The red
line marks the median AP, the box bottom and top are the 25th and 75th percentiles. The whiskers extend to the min and max AP of each
method. Each AP is plotted as a green dot over the whiskers (best viewed digitally with zoom).

Figure 4: Top regions for six pool5 units. Receptive ﬁelds and activation values are drawn in white. Some units are aligned to concepts,
such as people (row 1) or text (4). Other units capture texture and material properties, such as dot arrays (2) and specular reﬂections (6).

5

020406080100UIUC−IFP Delta GPU_UCLA SYSU_Vision Toronto A *OverFeat (1) *NEC−MU UvA−Euvision *OverFeat (2) *R−CNN BB mean average precision (mAP) in %ILSVRC2013 detection test set mAP  1.0%6.1%9.8%10.5%11.5%19.4%20.9%22.6%24.3%31.4%competition resultpost competition result0102030405060708090100*R−CNN BBUvA−Euvision*NEC−MU*OverFeat (1)Toronto ASYSU_VisionGPU_UCLADeltaUIUC−IFPaverage precision (AP) in %ILSVRC2013 detection test set class AP box plots1.01.00.90.90.90.90.90.90.90.90.90.90.90.90.90.91.00.90.90.80.80.80.70.70.70.70.70.70.70.70.60.61.00.80.70.70.70.70.70.70.70.70.70.70.70.70.60.61.00.90.80.80.80.70.70.70.70.70.70.70.70.70.70.71.01.00.90.90.90.80.80.80.80.80.80.80.80.80.80.81.00.90.80.80.80.70.70.70.70.70.70.70.70.70.70.7cat

VOC 2007 test
car
aero bike bird boat bottle bus
51.8 60.2 36.4 27.8 23.2 52.8 60.6 49.2 18.3 47.8 44.3 40.8 56.6
R-CNN pool5
59.3 61.8 43.1 34.0 25.1 53.1 60.6 52.8 21.7 47.8 42.7 47.8 52.5
R-CNN fc6
57.6 57.9 38.5 31.8 23.7 51.2 58.9 51.4 20.0 50.5 40.9 46.0 51.6
R-CNN fc7
58.2 63.3 37.9 27.6 26.1 54.1 66.9 51.4 26.7 55.5 43.4 43.1 57.7
R-CNN FT pool5
63.5 66.0 47.9 37.7 29.9 62.5 70.2 60.2 32.0 57.9 47.0 53.5 60.1
R-CNN FT fc6
R-CNN FT fc7
64.2 69.7 50.0 41.9 32.0 62.6 71.0 60.7 32.7 58.5 46.5 56.1 60.6
R-CNN FT fc7 BB 68.1 72.8 56.8 43.0 36.8 66.3 74.2 67.6 34.4 63.5 54.5 61.2 69.1

chair cow table dog horse mbike person plant sheep sofa train
58.7
58.5
55.9
59.0
64.2
66.8
68.6

tv mAP
46.1 36.7 51.3 55.7 44.2
48.3 34.0 53.1 58.0 46.2
48.1 35.3 51.0 57.4 44.7
50.8 40.6 53.1 56.4 47.3
55.0 50.0 57.7 63.0 53.1
52.8 48.9 57.9 64.7 54.2
62.9 51.1 62.5 64.8 58.5

23.4
25.6
23.3
28.1
31.3
31.5
33.4

42.4
44.6
43.3
45.8
52.2
54.2
58.7

DPM v5 [20]
DPM ST [28]
DPM HSC [31]

33.2 60.3 10.2 16.1 27.3 54.3 58.2 23.0 20.0 24.1 26.7 12.7 58.1
8.0 55.9
23.8 58.2 10.5
32.2 58.3 11.5 16.3 30.6 49.9 54.8 23.5 21.5 27.7 34.0 13.7 58.1

8.5 27.1 50.4 52.0

7.3 19.2 22.8 18.1

48.2
44.8
51.6

43.2
32.4
39.9

12.0
13.3
12.4

21.1 36.1 46.0 43.5 33.7
15.9 22.8 46.2 44.9 29.1
23.5 34.4 47.4 45.2 34.3

Table 2: Detection average precision (%) on VOC 2007 test. Rows 1-3 show R-CNN performance without ﬁne-tuning. Rows 4-6 show
results for the CNN pre-trained on ILSVRC 2012 and then ﬁne-tuned (FT) on VOC 2007 trainval. Row 7 includes a simple bounding-box
regression (BB) stage that reduces localization errors (Section C). Rows 8-10 present DPM methods as a strong baseline. The ﬁrst uses
only HOG, while the next two use different feature learning approaches to augment or replace HOG.

VOC 2007 test
aero bike bird boat bottle bus
car
R-CNN T-Net
64.2 69.7 50.0 41.9 32.0 62.6 71.0 60.7 32.7 58.5 46.5 56.1 60.6
R-CNN T-Net BB 68.1 72.8 56.8 43.0 36.8 66.3 74.2 67.6 34.4 63.5 54.5 61.2 69.1
R-CNN O-Net
71.6 73.5 58.1 42.2 39.4 70.7 76.0 74.5 38.7 71.0 56.9 74.5 67.9
R-CNN O-Net BB 73.4 77.0 63.4 45.4 44.6 75.1 78.1 79.8 40.5 73.7 62.2 79.4 78.1

chair cow table dog horse mbike person plant sheep sofa train
66.8
68.6
69.6
73.1

tv mAP
52.8 48.9 57.9 64.7 54.2
62.9 51.1 62.5 64.8 58.5
62.1 64.0 66.5 71.2 62.2
66.8 67.2 70.4 71.1 66.0

31.5
33.4
35.7
35.6

54.2
58.7
59.3
64.2

cat

Table 3: Detection average precision (%) on VOC 2007 test for two different CNN architectures. The ﬁrst two rows are results from
Table 2 using Krizhevsky et al.’s architecture (T-Net). Rows three and four use the recently proposed 16-layer architecture from Simonyan
and Zisserman (O-Net) [43].

We visualize units from layer pool5, which is the max-
pooled output of the network’s ﬁfth and ﬁnal convolutional
layer. The pool5 feature map is 6 × 6 × 256 = 9216-
dimensional. Ignoring boundary effects, each pool5 unit has
a receptive ﬁeld of 195×195 pixels in the original 227×227
pixel input. A central pool5 unit has a nearly global view,
while one near the edge has a smaller, clipped support.

Each row in Figure 4 displays the top 16 activations for
a pool5 unit from a CNN that we ﬁne-tuned on VOC 2007
trainval. Six of the 256 functionally unique units are visu-
alized (Appendix D includes more). These units were se-
lected to show a representative sample of what the network
learns. In the second row, we see a unit that ﬁres on dog
faces and dot arrays. The unit corresponding to the third row
is a red blob detector. There are also detectors for human
faces and more abstract patterns such as text and triangular
structures with windows. The network appears to learn a
representation that combines a small number of class-tuned
features together with a distributed representation of shape,
texture, color, and material properties. The subsequent fully
connected layer fc6 has the ability to model a large set of
compositions of these rich features.

3.2. Ablation studies

Performance layer-by-layer, without ﬁne-tuning. To un-
derstand which layers are critical for detection performance,
we analyzed results on the VOC 2007 dataset for each of the
CNN’s last three layers. Layer pool5 was brieﬂy described
in Section 3.1. The ﬁnal two layers are summarized below.

Layer fc6 is fully connected to pool5. To compute fea-
tures, it multiplies a 4096×9216 weight matrix by the pool5
feature map (reshaped as a 9216-dimensional vector) and
then adds a vector of biases. This intermediate vector is
component-wise half-wave rectiﬁed (x ← max(0, x)).

Layer fc7 is the ﬁnal layer of the network. It is imple-
mented by multiplying the features computed by fc6 by a
4096 × 4096 weight matrix, and similarly adding a vector
of biases and applying half-wave rectiﬁcation.

We start by looking at results from the CNN without
ﬁne-tuning on PASCAL, i.e.
all CNN parameters were
pre-trained on ILSVRC 2012 only. Analyzing performance
layer-by-layer (Table 2 rows 1-3) reveals that features from
fc7 generalize worse than features from fc6. This means
that 29%, or about 16.8 million, of the CNN’s parameters
can be removed without degrading mAP. More surprising is
that removing both fc7 and fc6 produces quite good results
even though pool5 features are computed using only 6% of
the CNN’s parameters. Much of the CNN’s representational
power comes from its convolutional layers, rather than from
the much larger densely connected layers. This ﬁnding sug-
gests potential utility in computing a dense feature map, in
the sense of HOG, of an arbitrary-sized image by using only
the convolutional layers of the CNN. This representation
would enable experimentation with sliding-window detec-
tors, including DPM, on top of pool5 features.

Performance layer-by-layer, with ﬁne-tuning. We now
look at results from our CNN after having ﬁne-tuned its pa-

6

rameters on VOC 2007 trainval. The improvement is strik-
ing (Table 2 rows 4-6): ﬁne-tuning increases mAP by 8.0
percentage points to 54.2%. The boost from ﬁne-tuning is
much larger for fc6 and fc7 than for pool5, which suggests
that the pool5 features learned from ImageNet are general
and that most of the improvement is gained from learning
domain-speciﬁc non-linear classiﬁers on top of them.

Comparison to recent feature learning methods. Rela-
tively few feature learning methods have been tried on PAS-
CAL VOC detection. We look at two recent approaches that
build on deformable part models. For reference, we also in-
clude results for the standard HOG-based DPM [20].

The ﬁrst DPM feature learning method, DPM ST [28],
augments HOG features with histograms of “sketch token”
probabilities.
Intuitively, a sketch token is a tight distri-
bution of contours passing through the center of an image
patch. Sketch token probabilities are computed at each pixel
by a random forest that was trained to classify 35 × 35 pixel
patches into one of 150 sketch tokens or background.

The second method, DPM HSC [31], replaces HOG with
histograms of sparse codes (HSC). To compute an HSC,
sparse code activations are solved for at each pixel using
a learned dictionary of 100 7 × 7 pixel (grayscale) atoms.
The resulting activations are rectiﬁed in three ways (full and
both half-waves), spatially pooled, unit (cid:96)2 normalized, and
then power transformed (x ← sign(x)|x|α).

All R-CNN variants strongly outperform the three DPM
baselines (Table 2 rows 8-10), including the two that use
feature learning. Compared to the latest version of DPM,
which uses only HOG features, our mAP is more than 20
percentage points higher: 54.2% vs. 33.7%—a 61% rela-
tive improvement. The combination of HOG and sketch to-
kens yields 2.5 mAP points over HOG alone, while HSC
improves over HOG by 4 mAP points (when compared
internally to their private DPM baselines—both use non-
public implementations of DPM that underperform the open
source version [20]). These methods achieve mAPs of
29.1% and 34.3%, respectively.

3.3. Network architectures

Most results in this paper use the network architecture
from Krizhevsky et al. [25]. However, we have found that
the choice of architecture has a large effect on R-CNN de-
tection performance. In Table 3 we show results on VOC
2007 test using the 16-layer deep network recently proposed
by Simonyan and Zisserman [43]. This network was one of
the top performers in the recent ILSVRC 2014 classiﬁca-
tion challenge. The network has a homogeneous structure
consisting of 13 layers of 3 × 3 convolution kernels, with
ﬁve max pooling layers interspersed, and topped with three
fully-connected layers. We refer to this network as “O-Net”
for OxfordNet and the baseline as “T-Net” for TorontoNet.

for

To use O-Net in R-CNN, we downloaded the pub-
licly available pre-trained network weights
the
VGG ILSVRC 16 layers model from the Caffe Model
Zoo.1 We then ﬁne-tuned the network using the same pro-
tocol as we used for T-Net. The only difference was to use
smaller minibatches (24 examples) as required in order to
ﬁt within GPU memory. The results in Table 3 show that R-
CNN with O-Net substantially outperforms R-CNN with T-
Net, increasing mAP from 58.5% to 66.0%. However there
is a considerable drawback in terms of compute time, with
the forward pass of O-Net taking roughly 7 times longer
than T-Net.

3.4. Detection error analysis

We applied the excellent detection analysis tool from
Hoiem et al.
[23] in order to reveal our method’s error
modes, understand how ﬁne-tuning changes them, and to
see how our error types compare with DPM. A full sum-
mary of the analysis tool is beyond the scope of this pa-
per and we encourage readers to consult [23] to understand
some ﬁner details (such as “normalized AP”). Since the
analysis is best absorbed in the context of the associated
plots, we present the discussion within the captions of Fig-
ure 5 and Figure 6.

3.5. Bounding-box regression

Based on the error analysis, we implemented a sim-
ple method to reduce localization errors.
Inspired by the
bounding-box regression employed in DPM [17], we train a
linear regression model to predict a new detection window
given the pool5 features for a selective search region pro-
posal. Full details are given in Appendix C. Results in Ta-
ble 1, Table 2, and Figure 5 show that this simple approach
ﬁxes a large number of mislocalized detections, boosting
mAP by 3 to 4 points.

3.6. Qualitative results

Qualitative detection results on ILSVRC2013 are pre-
sented in Figure 8 and Figure 9 at the end of the paper. Each
image was sampled randomly from the val2 set and all de-
tections from all detectors with a precision greater than 0.5
are shown. Note that these are not curated and give a re-
alistic impression of the detectors in action. More qualita-
tive results are presented in Figure 10 and Figure 11, but
these have been curated. We selected each image because it
contained interesting, surprising, or amusing results. Here,
also, all detections at precision greater than 0.5 are shown.

4. The ILSVRC2013 detection dataset

In Section 2 we presented results on the ILSVRC2013
detection dataset. This dataset is less homogeneous than

1https://github.com/BVLC/caffe/wiki/Model-Zoo

7

Figure 6: Sensitivity to object characteristics. Each plot shows the mean (over classes) normalized AP (see [23]) for the highest and
lowest performing subsets within six different object characteristics (occlusion, truncation, bounding-box area, aspect ratio, viewpoint, part
visibility). We show plots for our method (R-CNN) with and without ﬁne-tuning (FT) and bounding-box regression (BB) as well as for
DPM voc-release5. Overall, ﬁne-tuning does not reduce sensitivity (the difference between max and min), but does substantially improve
both the highest and lowest performing subsets for nearly all characteristics. This indicates that ﬁne-tuning does more than simply improve
the lowest performing subsets for aspect ratio and bounding-box area, as one might conjecture based on how we warp network inputs.
Instead, ﬁne-tuning improves robustness for all characteristics including occlusion, truncation, viewpoint, and part visibility.

val and test splits are drawn from the same image distribu-
tion. These images are scene-like and similar in complexity
(number of objects, amount of clutter, pose variability, etc.)
to PASCAL VOC images. The val and test splits are exhaus-
tively annotated, meaning that in each image all instances
from all 200 classes are labeled with bounding boxes. The
train set, in contrast, is drawn from the ILSVRC2013 clas-
siﬁcation image distribution. These images have more vari-
able complexity with a skew towards images of a single cen-
tered object. Unlike val and test, the train images (due to
their large number) are not exhaustively annotated. In any
given train image, instances from the 200 classes may or
may not be labeled. In addition to these image sets, each
class has an extra set of negative images. Negative images
are manually checked to validate that they do not contain
any instances of their associated class. The negative im-
age sets were not used in this work. More information on
how ILSVRC was collected and annotated can be found in
[11, 36].

The nature of these splits presents a number of choices
for training R-CNN. The train images cannot be used for
hard negative mining, because annotations are not exhaus-
tive. Where should negative examples come from? Also,
the train images have different statistics than val and test.
Should the train images be used at all, and if so, to what
extent? While we have not thoroughly evaluated a large
number of choices, we present what seemed like the most
obvious path based on previous experience.

Our general strategy is to rely heavily on the val set and
use some of the train images as an auxiliary source of pos-
itive examples. To use val for both training and valida-
tion, we split it into roughly equally sized “val1” and “val2”
sets. Since some classes have very few examples in val (the
smallest has only 31 and half have fewer than 110), it is
important to produce an approximately class-balanced par-
tition. To do this, a large number of candidate splits were
generated and the one with the smallest maximum relative

Figure 5: Distribution of top-ranked false positive (FP) types.
Each plot shows the evolving distribution of FP types as more FPs
are considered in order of decreasing score. Each FP is catego-
rized into 1 of 4 types: Loc—poor localization (a detection with
an IoU overlap with the correct class between 0.1 and 0.5, or a du-
plicate); Sim—confusion with a similar category; Oth—confusion
with a dissimilar object category; BG—a FP that ﬁred on back-
ground. Compared with DPM (see [23]), signiﬁcantly more of
our errors result from poor localization, rather than confusion with
background or other object classes, indicating that the CNN fea-
tures are much more discriminative than HOG. Loose localiza-
tion likely results from our use of bottom-up region proposals and
the positional invariance learned from pre-training the CNN for
whole-image classiﬁcation. Column three shows how our simple
bounding-box regression method ﬁxes many localization errors.

PASCAL VOC, requiring choices about how to use it. Since
these decisions are non-trivial, we cover them in this sec-
tion.

4.1. Dataset overview

The ILSVRC2013 detection dataset is split into three
sets: train (395,918), val (20,121), and test (40,152), where
the number of images in each set is in parentheses. The

8

occtrnsizeaspviewpart00.20.40.60.80.2120.6120.4200.5570.2010.7200.3440.6060.3510.6770.2440.6090.516normalized APR−CNN fc6: sensitivity and impactocctrnsizeaspviewpart00.20.40.60.80.1790.7010.4980.6340.3350.7660.4420.6720.4290.7230.3250.6850.593normalized APR−CNN FT fc7: sensitivity and impactocctrnsizeaspviewpart00.20.40.60.80.2110.7310.5420.6760.3850.7860.4840.7090.4530.7790.3680.7200.633normalized APR−CNN FT fc7 BB: sensitivity and impactocctrnsizeaspviewpart00.20.40.60.80.1320.3390.2160.3470.0560.4870.1260.4530.1370.3910.0940.3880.297normalized APDPM voc−release5: sensitivity and impacttotal false positivespercentage of each typeR−CNN fc6: animals  2510040016006400020406080100LocSimOthBGtotal false positivespercentage of each typeR−CNN FT fc7: animals  2510040016006400020406080100LocSimOthBGtotal false positivespercentage of each typeR−CNN FT fc7 BB: animals  2510040016006400020406080100LocSimOthBGtotal false positivespercentage of each typeR−CNN fc6: furniture  2510040016006400020406080100LocSimOthBGtotal false positivespercentage of each typeR−CNN FT fc7: furniture  2510040016006400020406080100LocSimOthBGtotal false positivespercentage of each typeR−CNN FT fc7 BB: furniture  2510040016006400020406080100LocSimOthBGclass imbalance was selected.2 Each candidate split was
generated by clustering val images using their class counts
as features, followed by a randomized local search that may
improve the split balance. The particular split used here has
a maximum relative imbalance of about 11% and a median
relative imbalance of 4%. The val1/val2 split and code used
to produce them will be publicly available to allow other re-
searchers to compare their methods on the val splits used in
this report.

4.2. Region proposals

We followed the same region proposal approach that was
used for detection on PASCAL. Selective search [39] was
run in “fast mode” on each image in val1, val2, and test (but
not on images in train). One minor modiﬁcation was re-
quired to deal with the fact that selective search is not scale
invariant and so the number of regions produced depends
on the image resolution. ILSVRC image sizes range from
very small to a few that are several mega-pixels, and so we
resized each image to a ﬁxed width (500 pixels) before run-
ning selective search. On val, selective search resulted in an
average of 2403 region proposals per image with a 91.6%
recall of all ground-truth bounding boxes (at 0.5 IoU thresh-
old). This recall is notably lower than in PASCAL, where
it is approximately 98%, indicating signiﬁcant room for im-
provement in the region proposal stage.

4.3. Training data

For training data, we formed a set of images and boxes
that includes all selective search and ground-truth boxes
from val1 together with up to N ground-truth boxes per
class from train (if a class has fewer than N ground-truth
boxes in train, then we take all of them). We’ll call this
dataset of images and boxes val1+trainN .
In an ablation
study, we show mAP on val2 for N ∈ {0, 500, 1000} (Sec-
tion 4.5).

Training data is required for three procedures in R-CNN:
(1) CNN ﬁne-tuning, (2) detector SVM training, and (3)
bounding-box regressor training. CNN ﬁne-tuning was run
for 50k SGD iteration on val1+trainN using the exact same
settings as were used for PASCAL. Fine-tuning on a sin-
gle NVIDIA Tesla K20 took 13 hours using Caffe. For
SVM training, all ground-truth boxes from val1+trainN
were used as positive examples for their respective classes.
Hard negative mining was performed on a randomly se-
lected subset of 5000 images from val1. An initial experi-
ment indicated that mining negatives from all of val1, versus
a 5000 image subset (roughly half of it), resulted in only a
0.5 percentage point drop in mAP, while cutting SVM train-
ing time in half. No negative examples were taken from

2Relative imbalance is measured as |a − b|/(a + b) where a and b are

class counts in each half of the split.

train because the annotations are not exhaustive. The ex-
tra sets of veriﬁed negative images were not used. The
bounding-box regressors were trained on val1.

4.4. Validation and evaluation

Before submitting results to the evaluation server, we
validated data usage choices and the effect of ﬁne-tuning
and bounding-box regression on the val2 set using the train-
ing data described above. All system hyperparameters (e.g.,
SVM C hyperparameters, padding used in region warp-
ing, NMS thresholds, bounding-box regression hyperpa-
rameters) were ﬁxed at the same values used for PAS-
CAL. Undoubtedly some of these hyperparameter choices
are slightly suboptimal for ILSVRC, however the goal of
this work was to produce a preliminary R-CNN result on
ILSVRC without extensive dataset tuning. After selecting
the best choices on val2, we submitted exactly two result
ﬁles to the ILSVRC2013 evaluation server. The ﬁrst sub-
mission was without bounding-box regression and the sec-
ond submission was with bounding-box regression. For
these submissions, we expanded the SVM and bounding-
box regressor training sets to use val+train1k and val, re-
spectively. We used the CNN that was ﬁne-tuned on
val1+train1k to avoid re-running ﬁne-tuning and feature
computation.

4.5. Ablation study

Table 4 shows an ablation study of the effects of differ-
ent amounts of training data, ﬁne-tuning, and bounding-
box regression. A ﬁrst observation is that mAP on val2
matches mAP on test very closely. This gives us conﬁ-
dence that mAP on val2 is a good indicator of test set per-
formance. The ﬁrst result, 20.9%, is what R-CNN achieves
using a CNN pre-trained on the ILSVRC2012 classiﬁca-
tion dataset (no ﬁne-tuning) and given access to the small
amount of training data in val1 (recall that half of the classes
in val1 have between 15 and 55 examples). Expanding
the training set to val1+trainN improves performance to
24.1%, with essentially no difference between N = 500
and N = 1000. Fine-tuning the CNN using examples from
just val1 gives a modest improvement to 26.5%, however
there is likely signiﬁcant overﬁtting due to the small number
of positive training examples. Expanding the ﬁne-tuning
set to val1+train1k, which adds up to 1000 positive exam-
ples per class from the train set, helps signiﬁcantly, boosting
mAP to 29.7%. Bounding-box regression improves results
to 31.0%, which is a smaller relative gain that what was ob-
served in PASCAL.

4.6. Relationship to OverFeat

There is an interesting relationship between R-CNN and
OverFeat: OverFeat can be seen (roughly) as a special case
of R-CNN. If one were to replace selective search region

9

test set
SVM training set

val2

val2
val1 val1+train.5k val1+train1k val1+train1k val1+train1k val1+train1k val+train1k

test
val+train1k
val1+train1k val1+train1k val1+train1k val1+train1k

val2

val2

val2

val2

test

CNN ﬁne-tuning set n/a
n/a
fc6
20.9
17.7

bbox reg set
CNN feature layer
mAP
median AP

n/a
n/a
fc6
24.1
21.0

n/a
n/a
fc6
24.1
21.4

val1
n/a
fc7
26.5
24.8

n/a
fc7
29.7
29.2

val1
fc7
31.0
29.6

n/a
fc7
30.2
29.0

val
fc7
31.4
30.3

Table 4: ILSVRC2013 ablation study of data usage choices, ﬁne-tuning, and bounding-box regression.

proposals with a multi-scale pyramid of regular square re-
gions and change the per-class bounding-box regressors to
a single bounding-box regressor, then the systems would
be very similar (modulo some potentially signiﬁcant differ-
ences in how they are trained: CNN detection ﬁne-tuning,
using SVMs, etc.).
It is worth noting that OverFeat has
a signiﬁcant speed advantage over R-CNN: it is about 9x
faster, based on a ﬁgure of 2 seconds per image quoted from
[34]. This speed comes from the fact that OverFeat’s slid-
ing windows (i.e., region proposals) are not warped at the
image level and therefore computation can be easily shared
between overlapping windows. Sharing is implemented by
running the entire network in a convolutional fashion over
arbitrary-sized inputs. Speeding up R-CNN should be pos-
sible in a variety of ways and remains as future work.

5. Semantic segmentation

Region classiﬁcation is a standard technique for seman-
tic segmentation, allowing us to easily apply R-CNN to the
PASCAL VOC segmentation challenge. To facilitate a di-
rect comparison with the current leading semantic segmen-
tation system (called O2P for “second-order pooling”) [4],
we work within their open source framework. O2P uses
CPMC to generate 150 region proposals per image and then
predicts the quality of each region, for each class, using
support vector regression (SVR). The high performance of
their approach is due to the quality of the CPMC regions
and the powerful second-order pooling of multiple feature
types (enriched variants of SIFT and LBP). We also note
that Farabet et al. [16] recently demonstrated good results
on several dense scene labeling datasets (not including PAS-
CAL) using a CNN as a multi-scale per-pixel classiﬁer.

We follow [2, 4] and extend the PASCAL segmentation
training set to include the extra annotations made available
by Hariharan et al. [22]. Design decisions and hyperparam-
eters were cross-validated on the VOC 2011 validation set.
Final test results were evaluated only once.

CNN features for segmentation. We evaluate three strate-
gies for computing features on CPMC regions, all of which
begin by warping the rectangular window around the re-
gion to 227 × 227. The ﬁrst strategy (full) ignores the re-

gion’s shape and computes CNN features directly on the
warped window, exactly as we did for detection. However,
these features ignore the non-rectangular shape of the re-
gion. Two regions might have very similar bounding boxes
while having very little overlap. Therefore, the second strat-
egy (fg) computes CNN features only on a region’s fore-
ground mask. We replace the background with the mean
input so that background regions are zero after mean sub-
traction. The third strategy (full+fg) simply concatenates
the full and fg features; our experiments validate their com-
plementarity.

O2P [4]
46.4

full R-CNN
fc7
fc6
42.5
43.0

fg R-CNN
fc7
fc6
42.1
43.7

full+fg R-CNN
fc6
47.9

fc7
45.8

Table 5: Segmentation mean accuracy (%) on VOC 2011 vali-
dation. Column 1 presents O2P; 2-7 use our CNN pre-trained on
ILSVRC 2012.

Results on VOC 2011. Table 5 shows a summary of our
results on the VOC 2011 validation set compared with O2P.
(See Appendix E for complete per-category results.) Within
each feature computation strategy, layer fc6 always outper-
forms fc7 and the following discussion refers to the fc6 fea-
tures. The fg strategy slightly outperforms full, indicating
that the masked region shape provides a stronger signal,
matching our intuition. However, full+fg achieves an aver-
age accuracy of 47.9%, our best result by a margin of 4.2%
(also modestly outperforming O2P), indicating that the con-
text provided by the full features is highly informative even
given the fg features. Notably, training the 20 SVRs on our
full+fg features takes an hour on a single core, compared to
10+ hours for training on O2P features.

In Table 6 we present results on the VOC 2011 test
set, comparing our best-performing method, fc6 (full+fg),
against two strong baselines. Our method achieves the high-
est segmentation accuracy for 11 out of 21 categories, and
the highest overall segmentation accuracy of 47.9%, aver-
aged across categories (but likely ties with the O2P result
under any reasonable margin of error). Still better perfor-
mance could likely be achieved by ﬁne-tuning.

10

VOC 2011 test
R&P [2]
O2P [4]
ours (full+fg R-CNN fc6) 84.2 66.9 23.7 58.3 37.4 55.4 73.3 58.7 56.5

bg
cat
8.1 39.4 36.1 36.3 49.5
83.4 46.8 18.9 36.6 31.2 42.7 57.3 47.4 44.1
85.4 69.7 22.3 45.2 44.4 46.9 66.7 57.8 56.2 13.5 46.1 32.3 41.2 59.1
9.7 45.5 29.5 49.3 40.1

chair cow table dog horse mbike person plant sheep sofa train
48.3
55.3
57.8

tv mean
47.2 22.1 42.0 43.2 40.8
50.4 27.8 46.9 44.6 47.6
60.7 22.7 47.1 41.3 47.9

aero bike bird boat bottle bus

26.3
36.2
33.8

50.7
51.0
53.9

car

Table 6: Segmentation accuracy (%) on VOC 2011 test. We compare against two strong baselines: the “Regions and Parts” (R&P)
method of [2] and the second-order pooling (O2P) method of [4]. Without any ﬁne-tuning, our CNN achieves top segmentation perfor-
mance, outperforming R&P and roughly matching O2P.

6. Conclusion

In recent years, object detection performance had stag-
nated. The best performing systems were complex en-
sembles combining multiple low-level image features with
high-level context from object detectors and scene classi-
ﬁers. This paper presents a simple and scalable object de-
tection algorithm that gives a 30% relative improvement
over the best previous results on PASCAL VOC 2012.

We achieved this performance through two insights. The
ﬁrst is to apply high-capacity convolutional neural net-
works to bottom-up region proposals in order to localize
and segment objects. The second is a paradigm for train-
ing large CNNs when labeled training data is scarce. We
show that it is highly effective to pre-train the network—
with supervision—for a auxiliary task with abundant data
(image classiﬁcation) and then to ﬁne-tune the network for
the target task where data is scarce (detection). We conjec-
ture that the “supervised pre-training/domain-speciﬁc ﬁne-
tuning” paradigm will be highly effective for a variety of
data-scarce vision problems.

We conclude by noting that it is signiﬁcant that we
achieved these results by using a combination of classi-
cal tools from computer vision and deep learning (bottom-
up region proposals and convolutional neural networks).
Rather than opposing lines of scientiﬁc inquiry, the two are
natural and inevitable partners.

Acknowledgments. This research was supported in part
by DARPA Mind’s Eye and MSEE programs, by NSF
awards
and IIS-1212798,
MURI N000014-10-1-0933, and by support from Toyota.
The GPUs used in this research were generously donated
by the NVIDIA Corporation.

IIS-1134072,

IIS-0905647,

Appendix

A. Object proposal transformations

The convolutional neural network used in this work re-
quires a ﬁxed-size input of 227 × 227 pixels. For detec-
tion, we consider object proposals that are arbitrary image
rectangles. We evaluated two approaches for transforming
object proposals into valid CNN inputs.

The ﬁrst method (“tightest square with context”) en-
closes each object proposal inside the tightest square and

11

Figure 7: Different object proposal transformations. (A) the
original object proposal at its actual scale relative to the trans-
formed CNN inputs; (B) tightest square with context; (C) tight-
est square without context; (D) warp. Within each column and
example proposal, the top row corresponds to p = 0 pixels of con-
text padding while the bottom row has p = 16 pixels of context
padding.

then scales (isotropically) the image contained in that
square to the CNN input size. Figure 7 column (B) shows
this transformation. A variant on this method (“tightest
square without context”) excludes the image content that
surrounds the original object proposal. Figure 7 column
(C) shows this transformation. The second method (“warp”)
anisotropically scales each object proposal to the CNN in-
put size. Figure 7 column (D) shows the warp transforma-
tion.

For each of these transformations, we also consider in-
cluding additional image context around the original object
proposal. The amount of context padding (p) is deﬁned as a
border size around the original object proposal in the trans-
formed input coordinate frame. Figure 7 shows p = 0 pix-
els in the top row of each example and p = 16 pixels in
the bottom row. In all methods, if the source rectangle ex-
tends beyond the image, the missing data is replaced with
the image mean (which is then subtracted before inputing
the image into the CNN). A pilot set of experiments showed
that warping with context padding (p = 16 pixels) outper-
formed the alternatives by a large margin (3-5 mAP points).
Obviously more alternatives are possible, including using
replication instead of mean padding. Exhaustive evaluation
of these alternatives is left as future work.

(A)(B)(C)(D)(A)(B)(C)(D)B. Positive vs. negative examples and softmax

Two design choices warrant further discussion. The ﬁrst
is: Why are positive and negative examples deﬁned differ-
ently for ﬁne-tuning the CNN versus training the object de-
tection SVMs? To review the deﬁnitions brieﬂy, for ﬁne-
tuning we map each object proposal to the ground-truth in-
stance with which it has maximum IoU overlap (if any) and
label it as a positive for the matched ground-truth class if the
IoU is at least 0.5. All other proposals are labeled “back-
ground” (i.e., negative examples for all classes). For train-
ing SVMs, in contrast, we take only the ground-truth boxes
as positive examples for their respective classes and label
proposals with less than 0.3 IoU overlap with all instances
of a class as a negative for that class. Proposals that fall
into the grey zone (more than 0.3 IoU overlap, but are not
ground truth) are ignored.

Historically speaking, we arrived at these deﬁnitions be-
cause we started by training SVMs on features computed
by the ImageNet pre-trained CNN, and so ﬁne-tuning was
not a consideration at that point in time. In that setup, we
found that our particular label deﬁnition for training SVMs
was optimal within the set of options we evaluated (which
included the setting we now use for ﬁne-tuning). When we
started using ﬁne-tuning, we initially used the same positive
and negative example deﬁnition as we were using for SVM
training. However, we found that results were much worse
than those obtained using our current deﬁnition of positives
and negatives.

Our hypothesis is that this difference in how positives
and negatives are deﬁned is not fundamentally important
and arises from the fact that ﬁne-tuning data is limited.
Our current scheme introduces many “jittered” examples
(those proposals with overlap between 0.5 and 1, but not
ground truth), which expands the number of positive exam-
ples by approximately 30x. We conjecture that this large
set is needed when ﬁne-tuning the entire network to avoid
overﬁtting. However, we also note that using these jittered
examples is likely suboptimal because the network is not
being ﬁne-tuned for precise localization.

This leads to the second issue: Why, after ﬁne-tuning,
train SVMs at all? It would be cleaner to simply apply the
last layer of the ﬁne-tuned network, which is a 21-way soft-
max regression classiﬁer, as the object detector. We tried
this and found that performance on VOC 2007 dropped
from 54.2% to 50.9% mAP. This performance drop likely
arises from a combination of several factors including that
the deﬁnition of positive examples used in ﬁne-tuning does
not emphasize precise localization and the softmax classi-
ﬁer was trained on randomly sampled negative examples
rather than on the subset of “hard negatives” used for SVM
training.

This result shows that it’s possible to obtain close to
the same level of performance without training SVMs af-

ter ﬁne-tuning. We conjecture that with some additional
tweaks to ﬁne-tuning the remaining performance gap may
be closed. If true, this would simplify and speed up R-CNN
training with no loss in detection performance.

C. Bounding-box regression

We use a simple bounding-box regression stage to im-
prove localization performance. After scoring each selec-
tive search proposal with a class-speciﬁc detection SVM,
we predict a new bounding box for the detection using a
class-speciﬁc bounding-box regressor. This is similar in
spirit to the bounding-box regression used in deformable
part models [17]. The primary difference between the two
approaches is that here we regress from features computed
by the CNN, rather than from geometric features computed
on the inferred DPM part locations.

y, P i

x, P i

The input to our training algorithm is a set of N train-
ing pairs {(P i, Gi)}i=1,...,N , where P i = (P i
w, P i
h)
speciﬁes the pixel coordinates of the center of proposal P i’s
bounding box together with P i’s width and height in pixels.
Hence forth, we drop the superscript i unless it is needed.
Each ground-truth bounding box G is speciﬁed in the same
way: G = (Gx, Gy, Gw, Gh). Our goal is to learn a trans-
formation that maps a proposed box P to a ground-truth box
G.

We parameterize the transformation in terms of four
functions dx(P ), dy(P ), dw(P ), and dh(P ). The ﬁrst
two specify a scale-invariant translation of the center of
P ’s bounding box, while the second two specify log-space
translations of the width and height of P ’s bounding box.
After learning these functions, we can transform an input
proposal P into a predicted ground-truth box ˆG by apply-
ing the transformation

ˆGx = Pwdx(P ) + Px
ˆGy = Phdy(P ) + Py
ˆGw = Pw exp(dw(P ))
ˆGh = Ph exp(dh(P )).

(1)

(2)

(3)

(4)

Each function d(cid:63)(P ) (where (cid:63) is one of x, y, h, w) is
modeled as a linear function of the pool5 features of pro-
posal P , denoted by φ5(P ).
(The dependence of φ5(P )
on the image data is implicitly assumed.) Thus we have
d(cid:63)(P ) = wT
(cid:63)φ5(P ), where w(cid:63) is a vector of learnable
model parameters. We learn w(cid:63) by optimizing the regu-
larized least squares objective (ridge regression):

N
(cid:88)

(ti

(cid:63) − ˆwT

(cid:63)φ5(P i))2 + λ (cid:107) ˆw(cid:63)(cid:107)2 .

(5)

i

w(cid:63) = argmin

ˆw(cid:63)

12

The regression targets t(cid:63) for the training pair (P, G) are de-
ﬁned as

tx = (Gx − Px)/Pw
ty = (Gy − Py)/Ph
tw = log(Gw/Pw)
th = log(Gh/Ph).

(6)

(7)

(8)

(9)

As a standard regularized least squares problem, this can be
solved efﬁciently in closed form.

We found two subtle issues while implementing
bounding-box regression. The ﬁrst is that regularization
is important: we set λ = 1000 based on a validation set.
The second issue is that care must be taken when selecting
which training pairs (P, G) to use. Intuitively, if P is far
from all ground-truth boxes, then the task of transforming
P to a ground-truth box G does not make sense. Using ex-
amples like P would lead to a hopeless learning problem.
Therefore, we only learn from a proposal P if it is nearby
at least one ground-truth box. We implement “nearness” by
assigning P to the ground-truth box G with which it has
maximum IoU overlap (in case it overlaps more than one) if
and only if the overlap is greater than a threshold (which we
set to 0.6 using a validation set). All unassigned proposals
are discarded. We do this once for each object class in order
to learn a set of class-speciﬁc bounding-box regressors.

At test time, we score each proposal and predict its new
detection window only once. In principle, we could iterate
this procedure (i.e., re-score the newly predicted bounding
box, and then predict a new bounding box from it, and so
on). However, we found that iterating does not improve
results.

D. Additional feature visualizations

Figure 12 shows additional visualizations for 20 pool5
units. For each unit, we show the 24 region proposals that
maximally activate that unit out of the full set of approxi-
mately 10 million regions in all of VOC 2007 test.

We label each unit by its (y, x, channel) position in the
6 × 6 × 256 dimensional pool5 feature map. Within each
channel, the CNN computes exactly the same function of
the input region, with the (y, x) position changing only the
receptive ﬁeld.

E. Per-category segmentation results

In Table 7 we show the per-category segmentation ac-
curacy on VOC 2011 val for each of our six segmentation
methods in addition to the O2P method [4]. These results
show which methods are strongest across each of the 20
PASCAL classes, plus the background class.

F. Analysis of cross-dataset redundancy

One concern when training on an auxiliary dataset is that
there might be redundancy between it and the test set. Even
though the tasks of object detection and whole-image clas-
siﬁcation are substantially different, making such cross-set
redundancy much less worrisome, we still conducted a thor-
ough investigation that quantiﬁes the extent to which PAS-
CAL test images are contained within the ILSVRC 2012
training and validation sets. Our ﬁndings may be useful to
researchers who are interested in using ILSVRC 2012 as
training data for the PASCAL image classiﬁcation task.

We performed two checks for duplicate (and near-
duplicate) images. The ﬁrst test is based on exact matches
of ﬂickr image IDs, which are included in the VOC 2007
test annotations (these IDs are intentionally kept secret for
subsequent PASCAL test sets). All PASCAL images, and
about half of ILSVRC, were collected from ﬂickr.com. This
check turned up 31 matches out of 4952 (0.63%).

The second check uses GIST [30] descriptor matching,
which was shown in [13] to have excellent performance at
near-duplicate image detection in large (> 1 million) image
collections. Following [13], we computed GIST descrip-
tors on warped 32 × 32 pixel versions of all ILSVRC 2012
trainval and PASCAL 2007 test images.

Euclidean distance nearest-neighbor matching of GIST
descriptors revealed 38 near-duplicate images (including all
31 found by ﬂickr ID matching). The matches tend to vary
slightly in JPEG compression level and resolution, and to a
lesser extent cropping. These ﬁndings show that the overlap
is small, less than 1%. For VOC 2012, because ﬂickr IDs
are not available, we used the GIST matching method only.
Based on GIST matches, 1.5% of VOC 2012 test images
are in ILSVRC 2012 trainval. The slightly higher rate for
VOC 2012 is likely due to the fact that the two datasets
were collected closer together in time than VOC 2007 and
ILSVRC 2012 were.

G. Document changelog

This document tracks the progress of R-CNN. To help
readers understand how it has changed over time, here’s a
brief changelog describing the revisions.

v1 Initial version.

v2 CVPR 2014 camera-ready revision. Includes substan-
tial improvements in detection performance brought about
by (1) starting ﬁne-tuning from a higher learning rate (0.001
instead of 0.0001), (2) using context padding when prepar-
ing CNN inputs, and (3) bounding-box regression to ﬁx lo-
calization errors.

v3 Results on the ILSVRC2013 detection dataset and com-
parison with OverFeat were integrated into several sections
(primarily Section 2 and Section 4).

13

cat

car

aero bike bird boat bottle bus

VOC 2011 val
bg
84.0 69.0 21.7 47.7 42.2 42.4 64.7 65.8 57.4 12.9 37.4 20.5 43.7 35.7
O2P [4]
full R-CNN fc6
81.3 56.2 23.9 42.9 40.7 38.8 59.2 56.5 53.2 11.4 34.6 16.7 48.1 37.0
full R-CNN fc7
81.0 52.8 25.1 43.8 40.5 42.7 55.4 57.7 51.3
8.7 32.5 11.5 48.1 37.0
fg R-CNN fc6
9.1 36.5 23.6 46.4 38.1
81.4 54.1 21.1 40.6 38.7 53.6 59.9 57.2 52.5
fg R-CNN fc7
7.3 32.1 14.3 48.8 42.9
80.9 50.1 20.0 40.2 34.1 40.9 59.7 59.8 52.7
full+fg R-CNN fc6 83.1 60.4 23.2 48.4 47.3 52.6 61.6 60.6 59.1 10.8 45.8 20.9 57.7 43.3
full+fg R-CNN fc7 82.3 56.7 20.6 49.9 44.2 43.6 59.3 61.3 57.8
7.7 38.4 15.1 53.4 43.7

chair cow table dog horse mbike person plant sheep sofa train
52.7
51.4
50.5
53.2
54.0
57.4
50.8

tv mean
51.0 28.4 59.8 49.7 46.4
44.0 24.3 53.7 51.1 43.0
42.1 21.2 57.7 56.0 42.5
38.7 29.0 53.0 47.5 43.7
42.6 24.9 52.2 48.8 42.1
48.7 28.1 60.0 48.6 47.9
47.8 24.7 60.1 55.2 45.7

35.8
31.5
30.2
32.2
28.9
34.7
34.1

51.0
46.0
46.4
51.3
48.6
52.9
52.0

Table 7: Per-category segmentation accuracy (%) on the VOC 2011 validation set.

v4 The softmax vs. SVM results in Appendix B contained
an error, which has been ﬁxed. We thank Sergio Guadar-
rama for helping to identify this issue.

v5 Added results using the new 16-layer network architec-
ture from Simonyan and Zisserman [43] to Section 3.3 and
Table 3.

References

[1] B. Alexe, T. Deselaers, and V. Ferrari. Measuring the object-

ness of image windows. TPAMI, 2012. 2

[2] P. Arbel´aez, B. Hariharan, C. Gu, S. Gupta, L. Bourdev, and
J. Malik. Semantic segmentation using regions and parts. In
CVPR, 2012. 10, 11

[3] P. Arbel´aez, J. Pont-Tuset, J. Barron, F. Marques, and J. Ma-
lik. Multiscale combinatorial grouping. In CVPR, 2014. 3
[4] J. Carreira, R. Caseiro, J. Batista, and C. Sminchisescu. Se-
mantic segmentation with second-order pooling. In ECCV,
2012. 4, 10, 11, 13, 14

[5] J. Carreira and C. Sminchisescu. CPMC: Automatic ob-
ject segmentation using constrained parametric min-cuts.
TPAMI, 2012. 2, 3

[6] D. Cires¸an, A. Giusti, L. Gambardella, and J. Schmidhu-
ber. Mitosis detection in breast cancer histology images with
deep neural networks. In MICCAI, 2013. 3

[7] N. Dalal and B. Triggs. Histograms of oriented gradients for

human detection. In CVPR, 2005. 1

[8] T. Dean, M. A. Ruzon, M. Segal, J. Shlens, S. Vijaya-
Fast, accurate detection of
narasimhan, and J. Yagnik.
100,000 object classes on a single machine. In CVPR, 2013.
3

[9] J. Deng, A. Berg, S. Satheesh, H. Su, A. Khosla, and L. Fei-
Fei. ImageNet Large Scale Visual Recognition Competition
2012 (ILSVRC2012). http://www.image-net.org/
challenges/LSVRC/2012/. 1

[10] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-
Fei. ImageNet: A large-scale hierarchical image database.
In CVPR, 2009. 1

[11] J. Deng, O. Russakovsky, J. Krause, M. Bernstein, A. C.
Berg, and L. Fei-Fei. Scalable multi-label annotation.
In
CHI, 2014. 8

[12] J. Donahue, Y. Jia, O. Vinyals, J. Hoffman, N. Zhang,
E. Tzeng, and T. Darrell. DeCAF: A Deep Convolutional
Activation Feature for Generic Visual Recognition. In ICML,
2014. 2

[13] M. Douze, H. J´egou, H. Sandhawalia, L. Amsaleg, and
C. Schmid. Evaluation of gist descriptors for web-scale im-
age search. In Proc. of the ACM International Conference on
Image and Video Retrieval, 2009. 13

[14] I. Endres and D. Hoiem. Category independent object pro-

posals. In ECCV, 2010. 3

[15] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and
A. Zisserman. The PASCAL Visual Object Classes (VOC)
Challenge. IJCV, 2010. 1, 4

[16] C. Farabet, C. Couprie, L. Najman, and Y. LeCun. Learning

hierarchical features for scene labeling. TPAMI, 2013. 10

[17] P. Felzenszwalb, R. Girshick, D. McAllester, and D. Ra-
manan. Object detection with discriminatively trained part
based models. TPAMI, 2010. 2, 4, 7, 12

[18] S. Fidler, R. Mottaghi, A. Yuille, and R. Urtasun. Bottom-up

segmentation for top-down detection. In CVPR, 2013. 4, 5

[19] K. Fukushima. Neocognitron: A self-organizing neu-
ral network model for a mechanism of pattern recogni-
tion unaffected by shift in position. Biological cybernetics,
36(4):193–202, 1980. 1

[20] R. Girshick, P. Felzenszwalb, and D. McAllester. Discrimi-
natively trained deformable part models, release 5. http:
//www.cs.berkeley.edu/˜rbg/latent-v5/. 2,
5, 6, 7

[21] C. Gu, J. J. Lim, P. Arbel´aez, and J. Malik. Recognition

using regions. In CVPR, 2009. 2

[22] B. Hariharan, P. Arbel´aez, L. Bourdev, S. Maji, and J. Malik.
Semantic contours from inverse detectors. In ICCV, 2011.
10

[23] D. Hoiem, Y. Chodpathumwan, and Q. Dai. Diagnosing error

in object detectors. In ECCV. 2012. 2, 7, 8

[24] Y. Jia.

Caffe: An open source convolutional archi-
http://caffe.

tecture for fast feature embedding.
berkeleyvision.org/, 2013. 3

[25] A. Krizhevsky, I. Sutskever, and G. Hinton. ImageNet clas-
siﬁcation with deep convolutional neural networks. In NIPS,
2012. 1, 3, 4, 7

[26] Y. LeCun, B. Boser, J. Denker, D. Henderson, R. Howard,
W. Hubbard, and L. Jackel. Backpropagation applied to
handwritten zip code recognition. Neural Comp., 1989. 1

[27] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-
based learning applied to document recognition. Proc. of the
IEEE, 1998. 1

[28] J. J. Lim, C. L. Zitnick, and P. Doll´ar. Sketch tokens: A
learned mid-level representation for contour and object de-
tection. In CVPR, 2013. 6, 7

14

AP class
30.4 hair spray
14.1 hamburger
19.5 hammer
24.6 hamster
46.2 harmonica
21.5 harp

AP class
50.8 centipede
50.0 chain saw
31.8 chair
53.8 chime
30.9 cocktail shaker
54.0 coffee maker
45.0 computer keyboard 39.6 hat with a wide brim 40.5 ping-pong ball
11.8 computer mouse
42.0 corkscrew
2.8 cream
37.5 croquet ball

AP class
13.8 pencil box
34.2 pencil sharpener
9.9 perfume
46.0 person
12.6 piano
50.4 pineapple

21.9 cucumber
17.4 cup or mug
55.3 diaper
41.8 digital clock
65.3 dishwasher
37.2 dog
11.3 domestic cat
62.7 dragonﬂy
52.9 drum
38.8 dumbbell
12.7 electric fan
41.1 elephant

class
accordion
airplane
ant
antelope
apple
armadillo
artichoke
axe
baby bed
backpack
bagel
balance beam 32.6 crutch
banana
band aid
banjo
baseball
basketball
bathing cap
beaker
bear
bee
bell pepper
bench
bicycle
binder
bird
bookshelf
bow tie
bow
bowl
brassiere
burrito
bus
butterﬂy
camel
can opener
car
cart
cattle
cello

70.9 ﬁg
19.3 ﬁling cabinet
38.8 ﬂower pot
9.0 ﬂute
26.7 fox
31.2 french horn
25.7 frog
57.5 frying pan
88.5 giant panda
37.6 goldﬁsh
28.9 golf ball
44.5 golfcart
48.0 guacamole
32.3 guitar
28.9 hair dryer

6.2 face powder

21.2 head cabbage
24.2 helmet
29.9 hippopotamus
30.0 horizontal bar
23.7 horse
22.8 hotdog
34.0 iPod
10.1 isopod
18.5 jellyﬁsh
19.9 koala bear
76.8 ladle
44.1 ladybug
27.8 lamp
19.9 laptop
14.1 lemon
35.0 lion
56.4 lipstick
22.1 lizard
44.5 lobster
20.6 maillot
20.2 maraca

4.9 microphone
59.3 microwave
24.2 milk can
64.1 miniskirt
21.5 monkey
42.5 motorcycle
28.6 mushroom
51.3 nail
47.9 neck brace
32.3 oboe
33.1 orange
13.0 otter

AP class
11.4 snowplow
9.0 soap dispenser
32.8 soccer ball
41.7 sofa
20.5 spatula
22.6 squirrel
21.0 starﬁsh
19.2 stethoscope
43.7 stove
6.4 strainer
15.2 strawberry
32.0 stretcher
21.2 sunglasses
37.2 swimming trunks
7.9 swine
24.8 syringe
21.3 table
14.1 tape player
29.4 tennis ball
8.0 tick
71.0 tie
16.2 tiger
41.1 toaster
61.1 trafﬁc light
14.0 train
41.6 trombone
2.5 trumpet
34.5 turtle
11.5 tv or monitor

17.4 pitcher
33.4 pizza
38.0 plastic bag
7.0 plate rack
41.7 pomegranate
28.7 popsicle
59.2 porcupine
19.5 power drill
23.7 pretzel
44.3 printer
3.0 puck

58.4 punching bag

9.1 purse
35.4 rabbit
33.3 racket
51.3 ray
23.1 red panda
38.9 refrigerator
32.4 remote control
31.0 rubber eraser
30.1 rugby ball
4.0 ruler
40.1 salt or pepper shaker 24.6 unicycle
40.8 vacuum
33.3 saxophone
57.3 violin
14.9 scorpion
10.6 volleyball
49.6 screwdriver
20.9 wafﬂe iron
42.2 seal
48.9 washer
31.8 sheep
9.0 water bottle
4.5 ski
57.9 watercraft
31.6 skunk
36.2 whale
27.5 snail
33.8 wine bottle
38.8 snake
58.8 zebra
22.2 snowmobile

AP
69.2
16.8
43.7
16.3
6.8
31.3
45.1
18.3
8.1
9.9
26.8
13.2
18.8
9.1
45.3
5.7
21.7
21.4
59.1
42.6
24.6
61.8
29.2
24.7
60.8
13.8
14.4
59.1
41.7
27.2
19.5
13.7
59.7
24.0
39.8
8.1
40.9
48.6
31.2
49.6

Table 8: Per-class average precision (%) on the ILSVRC2013 detection test set.

[29] D. Lowe. Distinctive image features from scale-invariant

keypoints. IJCV, 2004. 1

A holistic representation of the spatial envelope. IJCV, 2001.
13

[30] A. Oliva and A. Torralba. Modeling the shape of the scene:

[31] X. Ren and D. Ramanan. Histograms of sparse codes for

15

Figure 8: Example detections on the val2 set from the conﬁguration that achieved 31.0% mAP on val2. Each image was sampled randomly
(these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the
precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

16

lemon 0.79lemon 0.70lemon 0.56lemon 0.50person 0.88person 0.72cocktail shaker 0.56dog 0.97dog 0.85dog 0.57bird 0.63dog 0.97dog 0.95dog 0.64helmet 0.65helmet 0.52motorcycle 0.65person 0.75person 0.58snowmobile 0.83snowmobile 0.83bow tie 0.86person 0.82bird 0.61dog 0.66dog 0.61domestic cat 0.57bird 0.96dog 0.91dog 0.77sofa 0.71dog 0.95dog 0.55ladybug 1.00person 0.87car 0.96car 0.66car 0.63bird 0.98person 0.65watercraft 1.00watercraft 0.69pretzel 0.78car 0.96person 0.65person 0.58person 0.52person 0.52bird 0.99bird 0.91bird 0.75dog 0.98flower pot 0.62dog 0.97dog 0.56train 1.00train 0.53armadillo 1.00armadillo 0.56bird 0.93dog 0.92swine 0.88bird 1.00butterfly 0.96person 0.90flower pot 0.62snake 0.70turtle 0.54bell pepper 0.81bell pepper 0.62bell pepper 0.54ruler 1.00antelope 0.53mushroom 0.93tv or monitor 0.82tv or monitor 0.76tv or monitor 0.54bird 0.89lipstick 0.80lipstick 0.61person 0.58dog 0.97soccer ball 0.90Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

17

baby bed 0.55helmet 0.51pitcher 0.57dog 0.98hat with a wide brim 0.78person 0.86bird 0.52table 0.60monkey 0.97table 0.68watercraft 0.55person 0.88car 0.61person 0.87person 0.51sunglasses 0.51dog 0.94dog 0.55bird 0.52monkey 0.87monkey 0.81swine 0.50dog 0.97hat with a wide brim 0.96snake 0.74dog 0.93person 0.77dog 0.97guacamole 0.64pretzel 0.69table 0.54dog 0.71person 0.85ladybug 0.90person 0.52zebra 0.83zebra 0.80zebra 0.55zebra 0.52dog 0.98hat with a wide brim 0.60person 0.85person 0.81person 0.73elephant 1.00bird 0.99person 0.58dog 0.98cart 1.00chair 0.79chair 0.64person 0.91person 0.87person 0.57person 0.52computer keyboard 0.52dog 0.97dog 0.92person 0.77bird 0.94butterfly 0.98person 0.73person 0.61bird 1.00bird 0.78person 0.91person 0.75stethoscope 0.83bird 0.83Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing
digitally with zoom is recommended.

18

person 0.81person 0.57person 0.53motorcycle 0.64person 0.73person 0.51bagel 0.57pineapple 1.00bowl 0.63guacamole 1.00tennis ball 0.60lemon 0.88lemon 0.86lemon 0.80lemon 0.78orange 0.78orange 0.73orange 0.71golf ball 1.00golf ball 1.00golf ball 0.89golf ball 0.81golf ball 0.79golf ball 0.76golf ball 0.60golf ball 0.60golf ball 0.51lemon 0.53soccer ball 0.67lamp 0.61table 0.59bee 0.85jellyfish 0.71bowl 0.54hamburger 0.78dumbbell 1.00person 0.52microphone 1.00person 0.85head cabbage 0.83head cabbage 0.75dog 0.74goldfish 0.76person 0.57guitar 1.00guitar 1.00guitar 0.88table 0.63computer keyboard 0.78microwave 0.60table 0.53tick 0.64lemon 0.80tennis ball 0.67rabbit 1.00dog 0.98person 0.81person 0.92sunglasses 0.52watercraft 0.86milk can 1.00milk can 1.00bookshelf 0.50chair 0.86giant panda 0.61person 0.87antelope 0.74cattle 0.81dog 0.87horse 0.78pomegranate 1.00chair 0.86tv or monitor 0.52antelope 0.68bird 0.94snake 0.60dog 0.98dog 0.88person 0.79snake 0.76table 0.62tv or monitor 0.80tv or monitor 0.58tv or monitor 0.54lamp 0.86lamp 0.65table 0.83monkey 1.00monkey 1.00monkey 0.90monkey 0.88monkey 0.52dog 0.88fox 1.00fox 0.81person 0.88watercraft 0.91watercraft 0.56bird 0.95bird 0.78isopod 0.56bird 0.69starfish 0.67dragonfly 0.70dragonfly 0.60hamburger 0.72hamburger 0.60cup or mug 0.72electric fan 1.00electric fan 0.83electric fan 0.78helmet 0.64soccer ball 0.63object detection. In CVPR, 2013. 6, 7

[32] H. A. Rowley, S. Baluja, and T. Kanade. Neural network-

based face detection. TPAMI, 1998. 2

[33] D. E. Rumelhart, G. E. Hinton, and R. J. Williams. Learn-
ing internal representations by error propagation. Parallel
Distributed Processing, 1:318–362, 1986. 1

[34] P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus,
and Y. LeCun. OverFeat: Integrated Recognition, Localiza-
tion and Detection using Convolutional Networks. In ICLR,
2014. 1, 2, 4, 10

[35] P. Sermanet, K. Kavukcuoglu, S. Chintala, and Y. LeCun.
Pedestrian detection with unsupervised multi-stage feature
learning. In CVPR, 2013. 2

[36] H. Su, J. Deng, and L. Fei-Fei. Crowdsourcing annotations
In AAAI Technical Report, 4th

for visual object detection.
Human Computation Workshop, 2012. 8

[37] K. Sung and T. Poggio. Example-based learning for view-
based human face detection. Technical Report A.I. Memo
No. 1521, Massachussets Institute of Technology, 1994. 4

[38] C. Szegedy, A. Toshev, and D. Erhan. Deep neural networks

for object detection. In NIPS, 2013. 2

[39] J. Uijlings, K. van de Sande, T. Gevers, and A. Smeulders.
Selective search for object recognition. IJCV, 2013. 1, 2, 3,
4, 5, 9

[40] R. Vaillant, C. Monrocq, and Y. LeCun. Original approach
for the localisation of objects in images. IEE Proc on Vision,
Image, and Signal Processing, 1994. 2

[41] X. Wang, M. Yang, S. Zhu, and Y. Lin. Regionlets for generic

object detection. In ICCV, 2013. 3, 5

[42] M. Zeiler, G. Taylor, and R. Fergus. Adaptive deconvolu-
tional networks for mid and high level feature learning. In
CVPR, 2011. 4

[43] K. Simonyan and A. Zisserman. Very Deep Convolu-
tional Networks for Large-Scale Image Recognition. arXiv
preprint, arXiv:1409.1556, 2014. 6, 7, 14

19

Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

20

person 0.82snake 0.76frog 0.78bird 0.79goldfish 0.76goldfish 0.76goldfish 0.58person 0.94stethoscope 0.56person 0.95person 0.92person 0.67person 0.60table 0.81jellyfish 0.67lemon 0.52person 0.78person 0.65watercraft 0.55baseball 1.00person 0.94person 0.82person 0.80person 0.61person 0.55person 0.52computer keyboard 0.81dog 0.60person 0.88person 0.79person 0.68person 0.59tv or monitor 0.82lizard 0.58chair 0.50person 0.74table 0.82person 0.94person 0.94person 0.95person 0.81person 0.69rugby ball 0.91person 0.84person 0.59volleyball 0.70pineapple 1.00brassiere 0.71person 0.95person 0.94person 0.94person 0.81person 0.80person 0.80person 0.79person 0.79person 0.69person 0.66person 0.58person 0.56person 0.54swimming trunks 0.56baseball 0.86helmet 0.74person 0.75miniskirt 0.64person 0.92vacuum 1.00dog 0.98dog 0.93person 0.94person 0.75person 0.65person 0.53ski 0.80ski 0.80bird 0.55tiger 1.00tiger 0.67tiger 0.59bird 0.56whale 1.00chair 0.53person 0.92person 0.92person 0.82person 0.78bowl 0.52strawberry 0.79strawberry 0.70burrito 0.54croquet ball 0.91croquet ball 0.91croquet ball 0.91croquet ball 0.91mushroom 0.57watercraft 0.91watercraft 0.87watercraft 0.58plastic bag 0.62plastic bag 0.62whale 0.88car 0.70dog 0.94tv or monitor 0.57cart 0.80person 0.79person 0.53hat with a wide brim 0.89person 0.88person 0.82person 0.79person 0.56person 0.54traffic light 0.79bird 0.59cucumber 0.53cucumber 0.52antelope 1.00antelope 1.00antelope 0.94antelope 0.73antelope 0.63antelope 0.63fox 0.57balance beam 0.50horizontal bar 1.00person 0.80person 0.90snake 0.64dog 0.98dog 0.97helmet 0.69horse 0.92horse 0.69person 0.82person 0.72orange 0.79orange 0.71orange 0.66orange 0.66orange 0.59orange 0.56bird 0.97bird 0.96bird 0.96bird 0.94bird 0.89bird 0.64bird 0.56bird 0.53bird 0.52guitar 1.00person 0.82bicycle 0.92person 0.90person 0.83car 1.00car 0.97dog 0.98dog 0.86dog 0.85dog 0.65dog 0.50person 0.83person 0.80person 0.74person 0.54elephant 0.60Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly
activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 dimensional pool5 feature map.
Each image region is drawn with an overlay of the unit’s receptive ﬁeld in white. The activation value (which we normalize by dividing by
the max activation value over all units in a channel) is shown in the receptive ﬁeld’s upper-left corner. Best viewed digitally with zoom.

21

pool5 feature: (3,3,1) (top 1 − 24)1.00.90.80.80.70.70.70.70.70.70.70.70.70.70.60.60.60.60.60.60.60.60.60.6pool5 feature: (3,3,2) (top 1 − 24)1.00.90.90.90.90.80.80.70.70.70.70.70.70.70.70.70.70.70.70.70.70.70.70.7pool5 feature: (3,3,3) (top 1 − 24)0.90.80.80.80.80.80.80.70.70.70.60.60.60.60.60.60.60.60.60.60.60.60.60.6pool5 feature: (3,3,4) (top 1 − 24)0.90.80.70.70.70.70.70.70.70.70.70.70.70.70.70.70.70.70.70.60.60.60.60.6pool5 feature: (3,3,5) (top 1 − 24)0.90.80.80.80.80.80.80.80.70.70.70.70.70.70.70.70.70.70.70.70.70.70.70.7pool5 feature: (3,3,6) (top 1 − 24)0.90.80.80.80.80.70.70.70.70.70.70.70.70.70.70.70.70.70.70.70.70.70.70.7pool5 feature: (3,3,7) (top 1 − 24)0.90.80.80.80.80.80.70.70.70.70.70.70.70.70.70.70.70.70.70.70.60.60.60.6pool5 feature: (3,3,8) (top 1 − 24)0.90.80.80.80.80.80.80.70.70.70.70.70.70.70.70.70.70.70.70.70.70.70.70.7pool5 feature: (3,3,9) (top 1 − 24)0.80.80.80.70.70.70.70.70.70.70.70.70.70.70.70.60.60.60.60.60.60.60.60.6pool5 feature: (3,3,10) (top 1 − 24)0.90.80.80.70.60.60.60.60.60.60.60.60.60.60.60.60.60.60.60.60.60.60.50.5pool5 feature: (3,3,11) (top 1 − 24)0.70.70.70.70.70.60.60.60.60.60.60.60.60.60.60.60.60.60.60.60.60.60.60.6pool5 feature: (3,3,12) (top 1 − 24)0.90.80.70.70.70.70.70.70.70.70.70.70.70.60.60.60.60.60.60.60.60.60.60.6pool5 feature: (3,3,13) (top 1 − 24)0.90.90.80.80.80.80.80.80.80.80.80.80.80.80.80.80.80.80.80.80.80.80.80.8pool5 feature: (3,3,14) (top 1 − 24)0.90.90.90.80.80.80.80.80.80.80.80.80.80.70.70.70.70.70.70.70.70.70.70.7pool5 feature: (3,3,15) (top 1 − 24)0.80.80.80.80.80.80.80.80.80.80.80.80.70.70.70.70.70.70.70.70.70.70.70.7pool5 feature: (3,3,16) (top 1 − 24)0.90.80.80.70.70.70.70.70.70.70.70.70.60.60.60.60.60.60.60.60.60.60.60.6pool5 feature: (3,3,17) (top 1 − 24)0.90.90.80.80.80.80.70.70.70.70.70.70.70.70.70.70.70.70.70.70.70.70.70.7pool5 feature: (3,3,18) (top 1 − 24)0.80.70.70.70.70.70.70.70.70.70.60.60.60.60.60.60.60.60.60.60.60.60.60.6pool5 feature: (3,3,19) (top 1 − 24)0.90.80.80.70.70.70.70.70.70.60.60.60.60.60.60.60.60.60.60.60.60.60.60.6pool5 feature: (3,3,20) (top 1 − 24)1.00.90.70.70.70.70.70.70.60.60.60.60.60.60.60.60.60.60.60.60.60.60.60.6
