End-to-End Object Detection with Transformers

0
2
0
2

Nicolas Carion(cid:63), Francisco Massa(cid:63), Gabriel Synnaeve, Nicolas Usunier,
Alexander Kirillov, and Sergey Zagoruyko

Facebook AI

Abstract. We present a new method that views object detection as a
direct set prediction problem. Our approach streamlines the detection
pipeline, eﬀectively removing the need for many hand-designed compo-
nents like a non-maximum suppression procedure or anchor generation
that explicitly encode our prior knowledge about the task. The main
ingredients of the new framework, called DEtection TRansformer or
DETR, are a set-based global loss that forces unique predictions via bi-
partite matching, and a transformer encoder-decoder architecture. Given
a ﬁxed small set of learned object queries, DETR reasons about the re-
lations of the objects and the global image context to directly output
the ﬁnal set of predictions in parallel. The new model is conceptually
simple and does not require a specialized library, unlike many other
modern detectors. DETR demonstrates accuracy and run-time perfor-
mance on par with the well-established and highly-optimized Faster R-
CNN baseline on the challenging COCO object detection dataset. More-
over, DETR can be easily generalized to produce panoptic segmentation
in a uniﬁed manner. We show that it signiﬁcantly outperforms com-
petitive baselines. Training code and pretrained models are available at
https://github.com/facebookresearch/detr.

y
a
M
8
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
2
7
8
2
1
.
5
0
0
2
:
v
i
X
r
a

1 Introduction

The goal of object detection is to predict a set of bounding boxes and category
labels for each object of interest. Modern detectors address this set prediction
task in an indirect way, by deﬁning surrogate regression and classiﬁcation prob-
lems on a large set of proposals [37,5], anchors [23], or window centers [53,46].
Their performances are signiﬁcantly inﬂuenced by postprocessing steps to col-
lapse near-duplicate predictions, by the design of the anchor sets and by the
heuristics that assign target boxes to anchors [52]. To simplify these pipelines,
we propose a direct set prediction approach to bypass the surrogate tasks. This
end-to-end philosophy has led to signiﬁcant advances in complex structured pre-
diction tasks such as machine translation or speech recognition, but not yet in
object detection: previous attempts [43,16,4,39] either add other forms of prior
knowledge, or have not proven to be competitive with strong baselines on chal-
lenging benchmarks. This paper aims to bridge this gap.

(cid:63) Equal contribution

2

Carion et al.

Fig. 1: DETR directly predicts (in parallel) the ﬁnal set of detections by combining
a common CNN with a transformer architecture. During training, bipartite matching
uniquely assigns predictions with ground truth boxes. Prediction with no match should
yield a “no object” (∅) class prediction.

We streamline the training pipeline by viewing object detection as a direct set
prediction problem. We adopt an encoder-decoder architecture based on trans-
formers [47], a popular architecture for sequence prediction. The self-attention
mechanisms of transformers, which explicitly model all pairwise interactions be-
tween elements in a sequence, make these architectures particularly suitable for
speciﬁc constraints of set prediction such as removing duplicate predictions.

Our DEtection TRansformer (DETR, see Figure 1) predicts all objects at
once, and is trained end-to-end with a set loss function which performs bipar-
tite matching between predicted and ground-truth objects. DETR simpliﬁes the
detection pipeline by dropping multiple hand-designed components that encode
prior knowledge, like spatial anchors or non-maximal suppression. Unlike most
existing detection methods, DETR doesn’t require any customized layers, and
thus can be reproduced easily in any framework that contains standard CNN
and transformer classes.1.

Compared to most previous work on direct set prediction, the main features of
DETR are the conjunction of the bipartite matching loss and transformers with
(non-autoregressive) parallel decoding [29,12,10,8]. In contrast, previous work
focused on autoregressive decoding with RNNs [43,41,30,36,42]. Our matching
loss function uniquely assigns a prediction to a ground truth object, and is
invariant to a permutation of predicted objects, so we can emit them in parallel.
We evaluate DETR on one of the most popular object detection datasets,
COCO [24], against a very competitive Faster R-CNN baseline [37]. Faster R-
CNN has undergone many design iterations and its performance was greatly
improved since the original publication. Our experiments show that our new
model achieves comparable performances. More precisely, DETR demonstrates
signiﬁcantly better performance on large objects, a result likely enabled by the
non-local computations of the transformer. It obtains, however, lower perfor-
mances on small objects. We expect that future work will improve this aspect
in the same way the development of FPN [22] did for Faster R-CNN.

Training settings for DETR diﬀer from standard object detectors in mul-
tiple ways. The new model requires extra-long training schedule and beneﬁts

1 In our work we use standard implementations of Transformers [47] and ResNet [15]

backbones from standard deep learning libraries.

transformer encoder-decoderCNNset of box predictionsbipartite matching lossno object (ø)no object (ø)set of image featuresEnd-to-End Object Detection with Transformers

3

from auxiliary decoding losses in the transformer. We thoroughly explore what
components are crucial for the demonstrated performance.

The design ethos of DETR easily extend to more complex tasks. In our
experiments, we show that a simple segmentation head trained on top of a pre-
trained DETR outperfoms competitive baselines on Panoptic Segmentation [19],
a challenging pixel-level recognition task that has recently gained popularity.

2 Related work

Our work build on prior work in several domains: bipartite matching losses for
set prediction, encoder-decoder architectures based on the transformer, parallel
decoding, and object detection methods.

2.1 Set Prediction

There is no canonical deep learning model to directly predict sets. The basic set
prediction task is multilabel classiﬁcation (see e.g., [40,33] for references in the
context of computer vision) for which the baseline approach, one-vs-rest, does
not apply to problems such as detection where there is an underlying structure
between elements (i.e., near-identical boxes). The ﬁrst diﬃculty in these tasks
is to avoid near-duplicates. Most current detectors use postprocessings such as
non-maximal suppression to address this issue, but direct set prediction are
postprocessing-free. They need global inference schemes that model interactions
between all predicted elements to avoid redundancy. For constant-size set pre-
diction, dense fully connected networks [9] are suﬃcient but costly. A general
approach is to use auto-regressive sequence models such as recurrent neural net-
works [48]. In all cases, the loss function should be invariant by a permutation of
the predictions. The usual solution is to design a loss based on the Hungarian al-
gorithm [20], to ﬁnd a bipartite matching between ground-truth and prediction.
This enforces permutation-invariance, and guarantees that each target element
has a unique match. We follow the bipartite matching loss approach. In contrast
to most prior work however, we step away from autoregressive models and use
transformers with parallel decoding, which we describe below.

2.2 Transformers and Parallel Decoding

Transformers were introduced by Vaswani et al . [47] as a new attention-based
building block for machine translation. Attention mechanisms [2] are neural net-
work layers that aggregate information from the entire input sequence. Trans-
formers introduced self-attention layers, which, similarly to Non-Local Neural
Networks [49], scan through each element of a sequence and update it by ag-
gregating information from the whole sequence. One of the main advantages of
attention-based models is their global computations and perfect memory, which
makes them more suitable than RNNs on long sequences. Transformers are now

4

Carion et al.

replacing RNNs in many problems in natural language processing, speech pro-
cessing and computer vision [8,27,45,34,31].

Transformers were ﬁrst used in auto-regressive models, following early sequence-

to-sequence models [44], generating output tokens one by one. However, the pro-
hibitive inference cost (proportional to output length, and hard to batch) lead
to the development of parallel sequence generation, in the domains of audio [29],
machine translation [12,10], word representation learning [8], and more recently
speech recognition [6]. We also combine transformers and parallel decoding for
their suitable trade-oﬀ between computational cost and the ability to perform
the global computations required for set prediction.

2.3 Object detection

Most modern object detection methods make predictions relative to some ini-
tial guesses. Two-stage detectors [37,5] predict boxes w.r.t. proposals, whereas
single-stage methods make predictions w.r.t. anchors [23] or a grid of possible
object centers [53,46]. Recent work [52] demonstrate that the ﬁnal performance
of these systems heavily depends on the exact way these initial guesses are set.
In our model we are able to remove this hand-crafted process and streamline the
detection process by directly predicting the set of detections with absolute box
prediction w.r.t. the input image rather than an anchor.

Set-based loss. Several object detectors [9,25,35] used the bipartite matching
loss. However, in these early deep learning models, the relation between diﬀerent
prediction was modeled with convolutional or fully-connected layers only and a
hand-designed NMS post-processing can improve their performance. More recent
detectors [37,23,53] use non-unique assignment rules between ground truth and
predictions together with an NMS.

Learnable NMS methods [16,4] and relation networks [17] explicitly model
relations between diﬀerent predictions with attention. Using direct set losses,
they do not require any post-processing steps. However, these methods employ
additional hand-crafted context features like proposal box coordinates to model
relations between detections eﬃciently, while we look for solutions that reduce
the prior knowledge encoded in the model.

Recurrent detectors. Closest to our approach are end-to-end set predictions
for object detection [43] and instance segmentation [41,30,36,42]. Similarly to us,
they use bipartite-matching losses with encoder-decoder architectures based on
CNN activations to directly produce a set of bounding boxes. These approaches,
however, were only evaluated on small datasets and not against modern baselines.
In particular, they are based on autoregressive models (more precisely RNNs),
so they do not leverage the recent transformers with parallel decoding.

3 The DETR model

Two ingredients are essential for direct set predictions in detection: (1) a set
prediction loss that forces unique matching between predicted and ground truth

End-to-End Object Detection with Transformers

5

boxes; (2) an architecture that predicts (in a single pass) a set of objects and
models their relation. We describe our architecture in detail in Figure 2.

3.1 Object detection set prediction loss

DETR infers a ﬁxed-size set of N predictions, in a single pass through the
decoder, where N is set to be signiﬁcantly larger than the typical number of
objects in an image. One of the main diﬃculties of training is to score predicted
objects (class, position, size) with respect to the ground truth. Our loss produces
an optimal bipartite matching between predicted and ground truth objects, and
then optimize object-speciﬁc (bounding box) losses.

Let us denote by y the ground truth set of objects, and ˆy = {ˆyi}N
i=1 the
set of N predictions. Assuming N is larger than the number of objects in the
image, we consider y also as a set of size N padded with ∅ (no object). To ﬁnd
a bipartite matching between these two sets we search for a permutation of N
elements σ ∈ SN with the lowest cost:

ˆσ = arg min

N
(cid:88)

σ∈SN

i

Lmatch(yi, ˆyσ(i)),

(1)

where Lmatch(yi, ˆyσ(i)) is a pair-wise matching cost between ground truth yi and
a prediction with index σ(i). This optimal assignment is computed eﬃciently
with the Hungarian algorithm, following prior work (e.g. [43]).

The matching cost takes into account both the class prediction and the sim-
ilarity of predicted and ground truth boxes. Each element i of the ground truth
set can be seen as a yi = (ci, bi) where ci is the target class label (which
may be ∅) and bi ∈ [0, 1]4 is a vector that deﬁnes ground truth box cen-
ter coordinates and its height and width relative to the image size. For the
prediction with index σ(i) we deﬁne probability of class ci as ˆpσ(i)(ci) and
the predicted box as ˆbσ(i). With these notations we deﬁne Lmatch(yi, ˆyσ(i)) as
−1{ci(cid:54)=∅} ˆpσ(i)(ci) + 1{ci(cid:54)=∅}Lbox(bi, ˆbσ(i)).

This procedure of ﬁnding matching plays the same role as the heuristic assign-
ment rules used to match proposal [37] or anchors [22] to ground truth objects
in modern detectors. The main diﬀerence is that we need to ﬁnd one-to-one
matching for direct set prediction without duplicates.

The second step is to compute the loss function, the Hungarian loss for all
pairs matched in the previous step. We deﬁne the loss similarly to the losses of
common object detectors, i.e. a linear combination of a negative log-likelihood
for class prediction and a box loss deﬁned later:

LHungarian(y, ˆy) =

N
(cid:88)

(cid:104)

i=1

− log ˆpˆσ(i)(ci) + 1{ci(cid:54)=∅}Lbox(bi, ˆbˆσ(i))

(cid:105)

,

(2)

where ˆσ is the optimal assignment computed in the ﬁrst step (1). In practice, we
down-weight the log-probability term when ci = ∅ by a factor 10 to account for

6

Carion et al.

class imbalance. This is analogous to how Faster R-CNN training procedure bal-
ances positive/negative proposals by subsampling [37]. Notice that the matching
cost between an object and ∅ doesn’t depend on the prediction, which means
that in that case the cost is a constant. In the matching cost we use probabil-
ities ˆpˆσ(i)(ci) instead of log-probabilities. This makes the class prediction term
commensurable to Lbox(·, ·) (described below), and we observed better empirical
performances.

Bounding box loss. The second part of the matching cost and the Hungarian
loss is Lbox(·) that scores the bounding boxes. Unlike many detectors that do box
predictions as a ∆ w.r.t. some initial guesses, we make box predictions directly.
While such approach simplify the implementation it poses an issue with relative
scaling of the loss. The most commonly-used (cid:96)1 loss will have diﬀerent scales for
small and large boxes even if their relative errors are similar. To mitigate this
issue we use a linear combination of the (cid:96)1 loss and the generalized IoU loss [38]
Liou(·, ·) that is scale-invariant. Overall, our box loss is Lbox(bi, ˆbσ(i)) deﬁned as
λiouLiou(bi, ˆbσ(i)) + λL1||bi − ˆbσ(i)||1 where λiou, λL1 ∈ R are hyperparameters.
These two losses are normalized by the number of objects inside the batch.

3.2 DETR architecture

The overall DETR architecture is surprisingly simple and depicted in Figure 2. It
contains three main components, which we describe below: a CNN backbone to
extract a compact feature representation, an encoder-decoder transformer, and
a simple feed forward network (FFN) that makes the ﬁnal detection prediction.
Unlike many modern detectors, DETR can be implemented in any deep learn-
ing framework that provides a common CNN backbone and a transformer archi-
tecture implementation with just a few hundred lines. Inference code for DETR
can be implemented in less than 50 lines in PyTorch [32]. We hope that the sim-
plicity of our method will attract new researchers to the detection community.
Backbone. Starting from the initial image ximg ∈ R3×H0×W0 (with 3 color
channels2), a conventional CNN backbone generates a lower-resolution activation
map f ∈ RC×H×W . Typical values we use are C = 2048 and H, W = H0

32 , W0
32 .

Transformer encoder. First, a 1x1 convolution reduces the channel dimension
of the high-level activation map f from C to a smaller dimension d. creating a
new feature map z0 ∈ Rd×H×W . The encoder expects a sequence as input, hence
we collapse the spatial dimensions of z0 into one dimension, resulting in a d×HW
feature map. Each encoder layer has a standard architecture and consists of a
multi-head self-attention module and a feed forward network (FFN). Since the
transformer architecture is permutation-invariant, we supplement it with ﬁxed
positional encodings [31,3] that are added to the input of each attention layer. We
defer to the supplementary material the detailed deﬁnition of the architecture,
which follows the one described in [47].

2 The input images are batched together, applying 0-padding adequately to ensure

they all have the same dimensions (H0, W0) as the largest image of the batch.

End-to-End Object Detection with Transformers

7

Fig. 2: DETR uses a conventional CNN backbone to learn a 2D representation of an
input image. The model ﬂattens it and supplements it with a positional encoding before
passing it into a transformer encoder. A transformer decoder then takes as input a
small ﬁxed number of learned positional embeddings, which we call object queries, and
additionally attends to the encoder output. We pass each output embedding of the
decoder to a shared feed forward network (FFN) that predicts either a detection (class
and bounding box) or a “no object” class.

Transformer decoder. The decoder follows the standard architecture of the
transformer, transforming N embeddings of size d using multi-headed self- and
encoder-decoder attention mechanisms. The diﬀerence with the original trans-
former is that our model decodes the N objects in parallel at each decoder layer,
while Vaswani et al. [47] use an autoregressive model that predicts the output
sequence one element at a time. We refer the reader unfamiliar with the concepts
to the supplementary material. Since the decoder is also permutation-invariant,
the N input embeddings must be diﬀerent to produce diﬀerent results. These in-
put embeddings are learnt positional encodings that we refer to as object queries,
and similarly to the encoder, we add them to the input of each attention layer.
The N object queries are transformed into an output embedding by the decoder.
They are then independently decoded into box coordinates and class labels by
a feed forward network (described in the next subsection), resulting N ﬁnal
predictions. Using self- and encoder-decoder attention over these embeddings,
the model globally reasons about all objects together using pair-wise relations
between them, while being able to use the whole image as context.

Prediction feed-forward networks (FFNs). The ﬁnal prediction is com-
puted by a 3-layer perceptron with ReLU activation function and hidden dimen-
sion d, and a linear projection layer. The FFN predicts the normalized center
coordinates, height and width of the box w.r.t. the input image, and the lin-
ear layer predicts the class label using a softmax function. Since we predict a
ﬁxed-size set of N bounding boxes, where N is usually much larger than the
actual number of objects of interest in an image, an additional special class la-
bel ∅ is used to represent that no object is detected within a slot. This class
plays a similar role to the “background” class in the standard object detection
approaches.

Auxiliary decoding losses. We found helpful to use auxiliary losses [1] in
decoder during training, especially to help the model output the correct number

CNNset of image featurestransformer encoder……positional encoding+transformer decoderclass,boxclass,boxno objectno objectFFNFFNFFNFFNobject queriesbackboneencoderdecoderprediction heads8

Carion et al.

of objects of each class. We add prediction FFNs and Hungarian loss after each
decoder layer. All predictions FFNs share their parameters. We use an additional
shared layer-norm to normalize the input to the prediction FFNs from diﬀerent
decoder layers.

4 Experiments

We show that DETR achieves competitive results compared to Faster R-CNN
in quantitative evaluation on COCO. Then, we provide a detailed ablation
study of the architecture and loss, with insights and qualitative results. Fi-
nally, to show that DETR is a versatile and extensible model, we present results
on panoptic segmentation, training only a small extension on a ﬁxed DETR
model. We provide code and pretrained models to reproduce our experiments at
https://github.com/facebookresearch/detr.

Dataset. We perform experiments on COCO 2017 detection and panoptic seg-
mentation datasets [24,18], containing 118k training images and 5k validation
images. Each image is annotated with bounding boxes and panoptic segmenta-
tion. There are 7 instances per image on average, up to 63 instances in a single
image in training set, ranging from small to large on the same images. If not
speciﬁed, we report AP as bbox AP, the integral metric over multiple thresholds.
For comparison with Faster R-CNN we report validation AP at the last training
epoch, for ablations we report median over validation results from the last 10
epochs.

Technical details. We train DETR with AdamW [26] setting the initial trans-
former’s learning rate to 10−4, the backbone’s to 10−5, and weight decay to 10−4.
All transformer weights are initialized with Xavier init [11], and the backbone
is with ImageNet-pretrained ResNet model [15] from torchvision with frozen
batchnorm layers. We report results with two diﬀerent backbones: a ResNet-
50 and a ResNet-101. The corresponding models are called respectively DETR
and DETR-R101. Following [21], we also increase the feature resolution by
adding a dilation to the last stage of the backbone and removing a stride from
the ﬁrst convolution of this stage. The corresponding models are called respec-
tively DETR-DC5 and DETR-DC5-R101 (dilated C5 stage). This modiﬁcation
increases the resolution by a factor of two, thus improving performance for small
objects, at the cost of a 16x higher cost in the self-attentions of the encoder,
leading to an overall 2x increase in computational cost. A full comparison of
FLOPs of these models and Faster R-CNN is given in Table 1.

We use scale augmentation, resizing the input images such that the shortest
side is at least 480 and at most 800 pixels while the longest at most 1333 [50].
To help learning global relationships through the self-attention of the encoder,
we also apply random crop augmentations during training, improving the per-
formance by approximately 1 AP. Speciﬁcally, a train image is cropped with
probability 0.5 to a random rectangular patch which is then resized again to
800-1333. The transformer is trained with default dropout of 0.1. At inference

End-to-End Object Detection with Transformers

9

Table 1: Comparison with Faster R-CNN with a ResNet-50 and ResNet-101 backbones
on the COCO validation set. The top section shows results for Faster R-CNN models
in Detectron2 [50], the middle section shows results for Faster R-CNN models with
GIoU [38], random crops train-time augmentation, and the long 9x training schedule.
DETR models achieve comparable results to heavily tuned Faster R-CNN baselines,
having lower APS but greatly improved APL. We use torchscript Faster R-CNN and
DETR models to measure FLOPS and FPS. Results without R101 in the name corre-
spond to ResNet-50.

Model

GFLOPS/FPS #params AP AP50 AP75 APS APM APL

Faster RCNN-DC5
Faster RCNN-FPN
Faster RCNN-R101-FPN

Faster RCNN-DC5+
Faster RCNN-FPN+
Faster RCNN-R101-FPN+

DETR
DETR-DC5
DETR-R101
DETR-DC5-R101

320/16
180/26
246/20

320/16
180/26
246/20

86/28
187/12
152/20
253/10

166M 39.0 60.5 42.3 21.4 43.5 52.5
42M 40.2 61.0 43.8 24.2 43.5 52.0
60M 42.0 62.5 45.9 25.2 45.6 54.6

166M 41.1 61.4 44.3 22.9 45.9 55.0
42M 42.0 62.1 45.5 26.6 45.4 53.4
60M 44.0 63.9 47.8 27.2 48.1 56.0

41M 42.0 62.4 44.2 20.5 45.8 61.1
41M 43.3 63.1 45.9 22.5 47.3 61.1
60M 43.5 63.8 46.4 21.9 48.0 61.8
60M 44.9 64.7 47.7 23.7 49.5 62.3

time, some slots predict empty class. To optimize for AP, we override the predic-
tion of these slots with the second highest scoring class, using the corresponding
conﬁdence. This improves AP by 2 points compared to ﬁltering out empty slots.
Other training hyperparameters can be found in section A.4. For our ablation
experiments we use training schedule of 300 epochs with a learning rate drop
by a factor of 10 after 200 epochs, where a single epoch is a pass over all train-
ing images once. Training the baseline model for 300 epochs on 16 V100 GPUs
takes 3 days, with 4 images per GPU (hence a total batch size of 64). For the
longer schedule used to compare with Faster R-CNN we train for 500 epochs
with learning rate drop after 400 epochs. This schedule adds 1.5 AP compared
to the shorter schedule.

4.1 Comparison with Faster R-CNN

Transformers are typically trained with Adam or Adagrad optimizers with very
long training schedules and dropout, and this is true for DETR as well. Faster
R-CNN, however, is trained with SGD with minimal data augmentation and
we are not aware of successful applications of Adam or dropout. Despite these
diﬀerences we attempt to make a Faster R-CNN baseline stronger. To align it
with DETR, we add generalized IoU [38] to the box loss, the same random
crop augmentation and long training known to improve results [13]. Results
are presented in Table 1. In the top section we show Faster R-CNN results
from Detectron2 Model Zoo [50] for models trained with the 3x schedule. In the
middle section we show results (with a “+”) for the same models but trained

10

Carion et al.

Table 2: Eﬀect of encoder size. Each row corresponds to a model with varied number
of encoder layers and ﬁxed number of decoder layers. Performance gradually improves
with more encoder layers.

#layers

GFLOPS/FPS

#params

0
3
6
12

76/28
81/25
86/23
95/20

33.4M
37.4M
41.3M
49.2M

AP

36.7
40.1
40.6
41.6

AP50

57.4
60.6
61.6
62.1

APS

16.8
18.5
19.9
19.8

APM

APL

39.6
43.8
44.3
44.9

54.2
58.6
60.2
61.9

with the 9x schedule (109 epochs) and the described enhancements, which in
total adds 1-2 AP. In the last section of Table 1 we show the results for multiple
DETR models. To be comparable in the number of parameters we choose a
model with 6 transformer and 6 decoder layers of width 256 with 8 attention
heads. Like Faster R-CNN with FPN this model has 41.3M parameters, out of
which 23.5M are in ResNet-50, and 17.8M are in the transformer. Even though
both Faster R-CNN and DETR are still likely to further improve with longer
training, we can conclude that DETR can be competitive with Faster R-CNN
with the same number of parameters, achieving 42 AP on the COCO val subset.
The way DETR achieves this is by improving APL (+7.8), however note that the
model is still lagging behind in APS (-5.5). DETR-DC5 with the same number
of parameters and similar FLOP count has higher AP, but is still signiﬁcantly
behind in APS too. Faster R-CNN and DETR with ResNet-101 backbone show
comparable results as well.

4.2 Ablations

Attention mechanisms in the transformer decoder are the key components which
model relations between feature representations of diﬀerent detections. In our
ablation analysis, we explore how other components of our architecture and loss
inﬂuence the ﬁnal performance. For the study we choose ResNet-50-based DETR
model with 6 encoder, 6 decoder layers and width 256. The model has 41.3M
parameters, achieves 40.6 and 42.0 AP on short and long schedules respectively,
and runs at 28 FPS, similarly to Faster R-CNN-FPN with the same backbone.

Number of encoder layers. We evaluate the importance of global image-
level self-attention by changing the number of encoder layers (Table 2). Without
encoder layers, overall AP drops by 3.9 points, with a more signiﬁcant drop of
6.0 AP on large objects. We hypothesize that, by using global scene reasoning,
the encoder is important for disentangling objects. In Figure 3, we visualize the
attention maps of the last encoder layer of a trained model, focusing on a few
points in the image. The encoder seems to separate instances already, which
likely simpliﬁes object extraction and localization for the decoder.

Number of decoder layers. We apply auxiliary losses after each decoding
layer (see Section 3.2), hence, the prediction FFNs are trained by design to pre-

End-to-End Object Detection with Transformers

11

Fig. 3: Encoder self-attention for a set of reference points. The encoder is able to sep-
arate individual instances. Predictions are made with baseline DETR model on a vali-
dation set image.

dict objects out of the outputs of every decoder layer. We analyze the importance
of each decoder layer by evaluating the objects that would be predicted at each
stage of the decoding (Fig. 4). Both AP and AP50 improve after every layer,
totalling into a very signiﬁcant +8.2/9.5 AP improvement between the ﬁrst and
the last layer. With its set-based loss, DETR does not need NMS by design. To
verify this we run a standard NMS procedure with default parameters [50] for
the outputs after each decoder. NMS improves performance for the predictions
from the ﬁrst decoder. This can be explained by the fact that a single decoding
layer of the transformer is not able to compute any cross-correlations between
the output elements, and thus it is prone to making multiple predictions for the
same object. In the second and subsequent layers, the self-attention mechanism
over the activations allows the model to inhibit duplicate predictions. We ob-
serve that the improvement brought by NMS diminishes as depth increases. At
the last layers, we observe a small loss in AP as NMS incorrectly removes true
positive predictions.

Similarly to visualizing encoder attention, we visualize decoder attentions in
Fig. 6, coloring attention maps for each predicted object in diﬀerent colors. We
observe that decoder attention is fairly local, meaning that it mostly attends to
object extremities such as heads or legs. We hypothesise that after the encoder
has separated instances via global attention, the decoder only needs to attend
to the extremities to extract the class and object boundaries.

Importance of FFN. FFN inside tranformers can be seen as 1 × 1 convo-
lutional layers, making encoder similar to attention augmented convolutional
networks [3]. We attempt to remove it completely leaving only attention in the
transformer layers. By reducing the number of network parameters from 41.3M
to 28.7M, leaving only 10.8M in the transformer, performance drops by 2.3 AP,
we thus conclude that FFN are important for achieving good results.

Importance of positional encodings. There are two kinds of positional en-
codings in our model: spatial positional encodings and output positional encod-

self-attention(430, 600)self-attention(520, 450)self-attention(450, 830)self-attention(440, 1200)12

Carion et al.

Fig. 4: AP and AP50 performance after each de-
coder layer. A single long schedule baseline model
is evaluated. DETR does not need NMS by de-
sign, which is validated by this ﬁgure. NMS lowers
AP in the ﬁnal layers, removing TP predictions,
but improves AP in the ﬁrst decoder layers, re-
moving double predictions, as there is no commu-
nication in the ﬁrst layer, and slightly improves
AP50.

Fig. 5: Out of distribution gen-
eralization for
rare classes.
Even though no image in the
training set has more than 13
giraﬀes, DETR has no diﬃ-
culty generalizing to 24 and
more instances of the same
class.

ings (object queries). We experiment with various combinations of ﬁxed and
learned encodings, results can be found in table 3. Output positional encodings
are required and cannot be removed, so we experiment with either passing them
once at decoder input or adding to queries at every decoder attention layer. In
the ﬁrst experiment we completely remove spatial positional encodings and pass
output positional encodings at input and, interestingly, the model still achieves
more than 32 AP, losing 7.8 AP to the baseline. Then, we pass ﬁxed sine spatial
positional encodings and the output encodings at input once, as in the original
transformer [47], and ﬁnd that this leads to 1.4 AP drop compared to passing
the positional encodings directly in attention. Learned spatial encodings passed
to the attentions give similar results. Surprisingly, we ﬁnd that not passing any
spatial encodings in the encoder only leads to a minor AP drop of 1.3 AP. When
we pass the encodings to the attentions, they are shared across all layers, and
the output encodings (object queries) are always learned.

Given these ablations, we conclude that transformer components: the global
self-attention in encoder, FFN, multiple decoder layers, and positional encodings,
all signiﬁcantly contribute to the ﬁnal object detection performance.

Loss ablations. To evaluate the importance of diﬀerent components of the
matching cost and the loss, we train several models turning them on and oﬀ.
There are three components to the loss: classiﬁcation loss, (cid:96)1 bounding box
distance loss, and GIoU [38] loss. The classiﬁcation loss is essential for training
and cannot be turned oﬀ, so we train a model without bounding box distance
loss, and a model without the GIoU loss, and compare with baseline, trained with
all three losses. Results are presented in table 4. GIoU loss on its own accounts

1234563436384042decoderlayerAPAPNoNMSAPNMS=0.71234565456586062decoderlayerAP50APNoNMSAPNMS=0.7AP50NoNMSAP50NMS=0.7End-to-End Object Detection with Transformers

13

Fig. 6: Visualizing decoder attention for every predicted object (images from COCO
val set). Predictions are made with DETR-DC5 model. Attention scores are coded with
diﬀerent colors for diﬀerent objects. Decoder typically attends to object extremities,
such as legs and heads. Best viewed in color.

Table 3: Results for diﬀerent positional encodings compared to the baseline (last row),
which has ﬁxed sine pos. encodings passed at every attention layer in both the encoder
and the decoder. Learned embeddings are shared between all layers. Not using spatial
positional encodings leads to a signiﬁcant drop in AP. Interestingly, passing them in
decoder only leads to a minor AP drop. All these models use learned output positional
encodings.

spatial pos. enc.

encoder

decoder

none
sine at input
learned at attn.
none
sine at attn.

none
sine at input
learned at attn.
sine at attn.
sine at attn.

output pos. enc.
decoder

learned at input
learned at input
learned at attn.
learned at attn.
learned at attn.

AP

∆

AP50

∆

32.8
39.2
39.6
39.3
40.6

-7.8
-1.4
-1.0
-1.3
-

55.2
60.0
60.7
60.3
61.6

-6.5
-1.6
-0.9
-1.4
-

Table 4: Eﬀect of loss components on AP. We train two models turning oﬀ (cid:96)1 loss, and
GIoU loss, and observe that (cid:96)1 gives poor results on its own, but when combined with
GIoU improves APM and APL. Our baseline (last row) combines both losses.

class

(cid:88)
(cid:88)
(cid:88)

(cid:96)1

(cid:88)

(cid:88)

GIoU

AP

(cid:88)
(cid:88)

35.8
39.9
40.6

∆

-4.8
-0.7
-

AP50

57.3
61.6
61.6

∆

-4.4
0
-

APS

13.7
19.9
19.9

APM

39.8
43.2
44.3

APL

57.9
57.9
60.2

for most of the model performance, losing only 0.7 AP to the baseline with
combined losses. Using (cid:96)1 without GIoU shows poor results. We only studied

14

Carion et al.

Fig. 7: Visualization of all box predictions on all images from COCO 2017 val set
for 20 out of total N = 100 prediction slots in DETR decoder. Each box prediction is
represented as a point with the coordinates of its center in the 1-by-1 square normalized
by each image size. The points are color-coded so that green color corresponds to small
boxes, red to large horizontal boxes and blue to large vertical boxes. We observe that
each slot learns to specialize on certain areas and box sizes with several operating
modes. We note that almost all slots have a mode of predicting large image-wide boxes
that are common in COCO dataset.

simple ablations of diﬀerent losses (using the same weighting every time), but
other means of combining them may achieve diﬀerent results.

4.3 Analysis

Decoder output slot analysis In Fig. 7 we visualize the boxes predicted
by diﬀerent slots for all images in COCO 2017 val set. DETR learns diﬀerent
specialization for each query slot. We observe that each slot has several modes of
operation focusing on diﬀerent areas and box sizes. In particular, all slots have
the mode for predicting image-wide boxes (visible as the red dots aligned in the
middle of the plot). We hypothesize that this is related to the distribution of
objects in COCO.

Generalization to unseen numbers of instances. Some classes in COCO
are not well represented with many instances of the same class in the same
image. For example, there is no image with more than 13 giraﬀes in the training
set. We create a synthetic image3 to verify the generalization ability of DETR
(see Figure 5). Our model is able to ﬁnd all 24 giraﬀes on the image which
is clearly out of distribution. This experiment conﬁrms that there is no strong
class-specialization in each object query.

4.4 DETR for panoptic segmentation

Panoptic segmentation [19] has recently attracted a lot of attention from the
computer vision community. Similarly to the extension of Faster R-CNN [37] to
Mask R-CNN [14], DETR can be naturally extended by adding a mask head on
top of the decoder outputs. In this section we demonstrate that such a head can
be used to produce panoptic segmentation [19] by treating stuﬀ and thing classes

3 Base picture credit: https://www.piqsels.com/en/public-domain-photo-jzlwu

End-to-End Object Detection with Transformers

15

Fig. 8: Illustration of the panoptic head. A binary mask is generated in parallel for each
detected object, then the masks are merged using pixel-wise argmax.

Fig. 9: Qualitative results for panoptic segmentation generated by DETR-R101. DETR
produces aligned mask predictions in a uniﬁed manner for things and stuﬀ.

in a uniﬁed way. We perform our experiments on the panoptic annotations of the
COCO dataset that has 53 stuﬀ categories in addition to 80 things categories.

We train DETR to predict boxes around both stuﬀ and things classes on
COCO, using the same recipe. Predicting boxes is required for the training to
be possible, since the Hungarian matching is computed using distances between
boxes. We also add a mask head which predicts a binary mask for each of the
predicted boxes, see Figure 8. It takes as input the output of transformer decoder
for each object and computes multi-head (with M heads) attention scores of this
embedding over the output of the encoder, generating M attention heatmaps
per object in a small resolution. To make the ﬁnal prediction and increase the
resolution, an FPN-like architecture is used. We describe the architecture in
more details in the supplement. The ﬁnal resolution of the masks has stride 4
and each mask is supervised independently using the DICE/F-1 loss [28] and
Focal loss [23].

The mask head can be trained either jointly, or in a two steps process, where
we train DETR for boxes only, then freeze all the weights and train only the mask
head for 25 epochs. Experimentally, these two approaches give similar results, we
report results using the latter method since it results in a shorter total wall-clock
time training.

Multi head attentionInput image(3 x H x W)Box embeddings(d x N)Encoded image(d x H/32 x W/32)Attention maps(N x M x H/32 x W/32)Masks logits(N x H/4 x W/4)Pixel-wise argmaxConcatenate2 x (Conv 3x3 + GN + ReLU)2x up + addConv 3x3 + GN + ReLU2x up + addConv 3x3 + GN + ReLU2x up + addConv 3x3 + GN + ReLU + Conv 3x3FPN-style CNNResnet featuresRes5Res4Res3Res216

Carion et al.

Table 5: Comparison with the state-of-the-art methods UPSNet [51] and Panoptic
FPN [18] on the COCO val dataset We retrained PanopticFPN with the same data-
augmentation as DETR, on a 18x schedule for fair comparison. UPSNet uses the 1x
schedule, UPSNet-M is the version with multiscale test-time augmentations.

Model

Backbone PQ SQ RQ PQth SQth RQth PQst SQst RQst AP

PanopticFPN++
UPSnet
UPSnet-M
PanopticFPN++
DETR
DETR-DC5
DETR-R101

R50
R50
R50
R101
R50
R50
R101

82.4
79.4
79.7

49.2
48.6
48.9

32.3 74.8 40.6 37.7
58.8
42.4 79.3 51.6
33.4 75.9 41.7 34.3
59.6
42.5 78.0 52.5
34.1 78.2 42.3 34.3
43.0 79.1 52.8
59.7
33.6 74.0 42.1 39.7
44.1 79.5 53.3 51.0 83.2 60.6
79.8
59.5
43.4 79.3 53.8
36.3 78.5 45.3 31.1
48.2
60.6 37.3 78.7 46.5 31.9
80.5
49.4
44.6 79.8 55.0
80.9 61.7 37.0 78.5 46.0 33.0
45.1 79.9 55.5 50.5

To predict the ﬁnal panoptic segmentation we simply use an argmax over
the mask scores at each pixel, and assign the corresponding categories to the
resulting masks. This procedure guarantees that the ﬁnal masks have no overlaps
and, therefore, DETR does not require a heuristic [19] that is often used to align
diﬀerent masks.

Training details. We train DETR, DETR-DC5 and DETR-R101 models fol-
lowing the recipe for bounding box detection to predict boxes around stuﬀ and
things classes in COCO dataset. The new mask head is trained for 25 epochs
(see supplementary for details). During inference we ﬁrst ﬁlter out the detection
with a conﬁdence below 85%, then compute the per-pixel argmax to determine
in which mask each pixel belongs. We then collapse diﬀerent mask predictions
of the same stuﬀ category in one, and ﬁlter the empty ones (less than 4 pixels).

Main results. Qualitative results are shown in Figure 9. In table 5 we compare
our uniﬁed panoptic segmenation approach with several established methods
that treat things and stuﬀ diﬀerently. We report the Panoptic Quality (PQ) and
the break-down on things (PQth) and stuﬀ (PQst). We also report the mask
AP (computed on the things classes), before any panoptic post-treatment (in
our case, before taking the pixel-wise argmax). We show that DETR outper-
forms published results on COCO-val 2017, as well as our strong PanopticFPN
baseline (trained with same data-augmentation as DETR, for fair comparison).
The result break-down shows that DETR is especially dominant on stuﬀ classes,
and we hypothesize that the global reasoning allowed by the encoder attention
is the key element to this result. For things class, despite a severe deﬁcit of
up to 8 mAP compared to the baselines on the mask AP computation, DETR
obtains competitive PQth. We also evaluated our method on the test set of the
COCO dataset, and obtained 46 PQ. We hope that our approach will inspire the
exploration of fully uniﬁed models for panoptic segmentation in future work.

End-to-End Object Detection with Transformers

17

5 Conclusion

We presented DETR, a new design for object detection systems based on trans-
formers and bipartite matching loss for direct set prediction. The approach
achieves comparable results to an optimized Faster R-CNN baseline on the chal-
lenging COCO dataset. DETR is straightforward to implement and has a ﬂexible
architecture that is easily extensible to panoptic segmentation, with competitive
results. In addition, it achieves signiﬁcantly better performance on large objects
than Faster R-CNN, likely thanks to the processing of global information per-
formed by the self-attention.

This new design for detectors also comes with new challenges, in particular
regarding training, optimization and performances on small objects. Current
detectors required several years of improvements to cope with similar issues,
and we expect future work to successfully address them for DETR.

6 Acknowledgements

We thank Sainbayar Sukhbaatar, Piotr Bojanowski, Natalia Neverova, David
Lopez-Paz, Guillaume Lample, Danielle Rothermel, Kaiming He, Ross Girshick,
Xinlei Chen and the whole Facebook AI Research Paris team for discussions and
advices without which this work would not be possible.

References

1. Al-Rfou, R., Choe, D., Constant, N., Guo, M., Jones, L.: Character-level language
modeling with deeper self-attention. In: AAAI Conference on Artiﬁcial Intelligence
(2019)

2. Bahdanau, D., Cho, K., Bengio, Y.: Neural machine translation by jointly learning

to align and translate. In: ICLR (2015)

3. Bello, I., Zoph, B., Vaswani, A., Shlens, J., Le, Q.V.: Attention augmented convo-

lutional networks. In: ICCV (2019)

4. Bodla, N., Singh, B., Chellappa, R., Davis, L.S.: Soft-NMS improving object

detection with one line of code. In: ICCV (2017)

5. Cai, Z., Vasconcelos, N.: Cascade R-CNN: High quality object detection and in-

stance segmentation. PAMI (2019)

6. Chan, W., Saharia, C., Hinton, G., Norouzi, M., Jaitly, N.: Imputer: Sequence
modelling via imputation and dynamic programming. arXiv:2002.08926 (2020)
7. Cordonnier, J.B., Loukas, A., Jaggi, M.: On the relationship between self-attention

and convolutional layers. In: ICLR (2020)

8. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: BERT: Pre-training of deep
bidirectional transformers for language understanding. In: NAACL-HLT (2019)
9. Erhan, D., Szegedy, C., Toshev, A., Anguelov, D.: Scalable object detection using

deep neural networks. In: CVPR (2014)

10. Ghazvininejad, M., Levy, O., Liu, Y., Zettlemoyer, L.: Mask-predict: Parallel de-

coding of conditional masked language models. arXiv:1904.09324 (2019)

11. Glorot, X., Bengio, Y.: Understanding the diﬃculty of training deep feedforward

neural networks. In: AISTATS (2010)

18

Carion et al.

12. Gu, J., Bradbury, J., Xiong, C., Li, V.O., Socher, R.: Non-autoregressive neural

machine translation. In: ICLR (2018)

13. He, K., Girshick, R., Doll´ar, P.: Rethinking imagenet pre-training. In: ICCV (2019)
14. He, K., Gkioxari, G., Doll´ar, P., Girshick, R.B.: Mask R-CNN. In: ICCV (2017)
15. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition.

In: CVPR (2016)

16. Hosang, J.H., Benenson, R., Schiele, B.: Learning non-maximum suppression. In:

CVPR (2017)

17. Hu, H., Gu, J., Zhang, Z., Dai, J., Wei, Y.: Relation networks for object detection.

In: CVPR (2018)

18. Kirillov, A., Girshick, R., He, K., Doll´ar, P.: Panoptic feature pyramid networks.

In: CVPR (2019)

19. Kirillov, A., He, K., Girshick, R., Rother, C., Dollar, P.: Panoptic segmentation.

In: CVPR (2019)

20. Kuhn, H.W.: The hungarian method for the assignment problem (1955)
21. Li, Y., Qi, H., Dai, J., Ji, X., Wei, Y.: Fully convolutional instance-aware semantic

segmentation. In: CVPR (2017)

22. Lin, T.Y., Doll´ar, P., Girshick, R., He, K., Hariharan, B., Belongie, S.: Feature

pyramid networks for object detection. In: CVPR (2017)

23. Lin, T.Y., Goyal, P., Girshick, R.B., He, K., Doll´ar, P.: Focal loss for dense object

detection. In: ICCV (2017)

24. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Doll´ar, P.,
Zitnick, C.L.: Microsoft COCO: Common objects in context. In: ECCV (2014)
25. Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S.E., Fu, C.Y., Berg, A.C.:

Ssd: Single shot multibox detector. In: ECCV (2016)

26. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: ICLR (2017)
27. L¨uscher, C., Beck, E., Irie, K., Kitza, M., Michel, W., Zeyer, A., Schl¨uter, R., Ney,
H.: Rwth asr systems for librispeech: Hybrid vs attention - w/o data augmentation.
arXiv:1905.03072 (2019)

28. Milletari, F., Navab, N., Ahmadi, S.A.: V-net: Fully convolutional neural networks

for volumetric medical image segmentation. In: 3DV (2016)

29. Oord, A.v.d., Li, Y., Babuschkin, I., Simonyan, K., Vinyals, O., Kavukcuoglu, K.,
Driessche, G.v.d., Lockhart, E., Cobo, L.C., Stimberg, F., et al.: Parallel wavenet:
Fast high-ﬁdelity speech synthesis. arXiv:1711.10433 (2017)

30. Park, E., Berg, A.C.: Learning to decompose for object detection and instance

segmentation. arXiv:1511.06449 (2015)

31. Parmar, N., Vaswani, A., Uszkoreit, J., Kaiser, L., Shazeer, N., Ku, A., Tran, D.:

Image transformer. In: ICML (2018)

32. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T.,
Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z.,
Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., Chintala, S.:
Pytorch: An imperative style, high-performance deep learning library. In: NeurIPS
(2019)

33. Pineda, L., Salvador, A., Drozdzal, M., Romero, A.: Elucidating image-to-set pre-
diction: An analysis of models, losses and datasets. arXiv:1904.05709 (2019)
34. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I.: Language

models are unsupervised multitask learners (2019)

35. Redmon, J., Divvala, S., Girshick, R., Farhadi, A.: You only look once: Uniﬁed,

real-time object detection. In: CVPR (2016)

36. Ren, M., Zemel, R.S.: End-to-end instance segmentation with recurrent attention.

In: CVPR (2017)

End-to-End Object Detection with Transformers

19

37. Ren, S., He, K., Girshick, R.B., Sun, J.: Faster R-CNN: Towards real-time object

detection with region proposal networks. PAMI (2015)

38. Rezatoﬁghi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I., Savarese, S.: General-

ized intersection over union. In: CVPR (2019)

39. Rezatoﬁghi, S.H., Kaskman, R., Motlagh, F.T., Shi, Q., Cremers, D., Leal-Taix´e,
L., Reid, I.: Deep perm-set net: Learn to predict sets with unknown permutation
and cardinality using deep neural networks. arXiv:1805.00613 (2018)

40. Rezatoﬁghi, S.H., Milan, A., Abbasnejad, E., Dick, A., Reid, I., Kaskman, R.,
Cremers, D., Leal-Taix, l.: Deepsetnet: Predicting sets with deep neural networks.
In: ICCV (2017)

41. Romera-Paredes, B., Torr, P.H.S.: Recurrent instance segmentation. In: ECCV

(2015)

42. Salvador, A., Bellver, M., Baradad, M., Marqu´es, F., Torres, J., Gir´o, X.: Recurrent
neural networks for semantic instance segmentation. arXiv:1712.00617 (2017)
43. Stewart, R.J., Andriluka, M., Ng, A.Y.: End-to-end people detection in crowded

scenes. In: CVPR (2015)

44. Sutskever, I., Vinyals, O., Le, Q.V.: Sequence to sequence learning with neural

networks. In: NeurIPS (2014)

45. Synnaeve, G., Xu, Q., Kahn, J., Grave, E., Likhomanenko, T., Pratap, V., Sri-
ram, A., Liptchinsky, V., Collobert, R.: End-to-end ASR: from supervised to semi-
supervised learning with modern architectures. arXiv:1911.08460 (2019)

46. Tian, Z., Shen, C., Chen, H., He, T.: FCOS: Fully convolutional one-stage object

detection. In: ICCV (2019)

47. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser,

L., Polosukhin, I.: Attention is all you need. In: NeurIPS (2017)

48. Vinyals, O., Bengio, S., Kudlur, M.: Order matters: Sequence to sequence for sets.

In: ICLR (2016)

49. Wang, X., Girshick, R.B., Gupta, A., He, K.: Non-local neural networks. In: CVPR

(2018)

50. Wu, Y., Kirillov, A., Massa, F., Lo, W.Y., Girshick, R.: Detectron2. https://

github.com/facebookresearch/detectron2 (2019)

51. Xiong, Y., Liao, R., Zhao, H., Hu, R., Bai, M., Yumer, E., Urtasun, R.: Upsnet: A

uniﬁed panoptic segmentation network. In: CVPR (2019)

52. Zhang, S., Chi, C., Yao, Y., Lei, Z., Li, S.Z.: Bridging the gap between anchor-based
and anchor-free detection via adaptive training sample selection. arXiv:1912.02424
(2019)

53. Zhou, X., Wang, D., Kr¨ahenb¨uhl, P.: Objects as points. arXiv:1904.07850 (2019)

20

Carion et al.

A Appendix

A.1 Preliminaries: Multi-head attention layers

Since our model is based on the Transformer architecture, we remind here the
general form of attention mechanisms we use for exhaustivity. The attention
mechanism follows [47], except for the details of positional encodings (see Equa-
tion 8) that follows [7].

Multi-head The general form of multi-head attention with M heads of dimen-
sion d is a function with the following signature (using d(cid:48) = d
M , and giving
matrix/tensors sizes in underbrace)

mh-attn : Xq
(cid:124)(cid:123)(cid:122)(cid:125)
d×Nq

, Xkv
(cid:124)(cid:123)(cid:122)(cid:125)
d×Nkv

,

T
(cid:124)(cid:123)(cid:122)(cid:125)
M ×3×d(cid:48)×d

, L
(cid:124)(cid:123)(cid:122)(cid:125)
d×d

(cid:55)→ ˜Xq
(cid:124)(cid:123)(cid:122)(cid:125)
d×Nq

(3)

where Xq is the query sequence of length Nq, Xkv is the key-value sequence of
length Nkv (with the same number of channels d for simplicity of exposition), T
is the weight tensor to compute the so-called query, key and value embeddings,
and L is a projection matrix. The output is the same size as the query sequence.
To ﬁx the vocabulary before giving details, multi-head self-attention (mh-s-attn)
is the special case Xq = Xkv, i.e.

mh-s-attn(X, T, L) = mh-attn(X, X, T, L) .

(4)

The multi-head attention is simply the concatenation of M single attention
heads followed by a projection with L. The common practice [47] is to use residual
connections, dropout and layer normalization. In other words, denoting ˜Xq =
mh-attn(Xq, Xkv, T, L) and ¯¯X (q) the concatenation of attention heads, we have

q = [attn(Xq, Xkv, T1); ...; attn(Xq, Xkv, TM )]

X (cid:48)
˜Xq = layernorm(cid:0)Xq + dropout(LX (cid:48)

q)(cid:1) ,

(5)

(6)

where [;] denotes concatenation on the channel axis.
Single head An attention head with weight tensor T (cid:48) ∈ R3×d(cid:48)×d, denoted by
attn(Xq, Xkv, T (cid:48)), depends on additional positional encoding Pq ∈ Rd×Nq and
Pkv ∈ Rd×Nkv . It starts by computing so-called query, key and value embeddings
after adding the query and key positional encodings [7]:

[Q; K; V ] = [T (cid:48)

2(Xkv + Pkv); T (cid:48)

3Xkv]

(7)

where T (cid:48) is the concatenation of T (cid:48)
3. The attention weights α are then
computed based on the softmax of dot products between queries and keys, so
that each element of the query sequence attends to all elements of the key-value
sequence (i is a query index and j a key-value index):

1(Xq + Pq); T (cid:48)
2, T (cid:48)
1, T (cid:48)

αi,j =

1
√d(cid:48)

e

QT

i Kj

Zi

where Zi =

Nkv(cid:88)

j=1

1
√d(cid:48)

QT
i Kj .

e

(8)

End-to-End Object Detection with Transformers

21

In our case, the positional encodings may be learnt or ﬁxed, but are shared
across all attention layers for a given query/key-value sequence, so we do not
explicitly write them as parameters of the attention. We give more details on
their exact value when describing the encoder and the decoder. The ﬁnal output
is the aggregation of values weighted by attention weights: The i-th row is given
by attni(Xq, Xkv, T (cid:48)) = (cid:80)Nkv
Feed-forward network (FFN) layers The original transformer alternates
multi-head attention and so-called FFN layers [47], which are eﬀectively multi-
layer 1x1 convolutions, which have M d input and output channels in our case.
The FFN we consider is composed of two-layers of 1x1 convolutions with ReLU
activations. There is also a residual connection/dropout/layernorm after the two
layers, similarly to equation 6.

j=1 αi,jVj.

A.2 Losses

For completeness, we present in detail the losses used in our approach. All losses
are normalized by the number of objects inside the batch. Extra care must be
taken for distributed training: since each GPU receives a sub-batch, it is not
suﬃcient to normalize by the number of objects in the local batch, since in
general the sub-batches are not balanced across GPUs. Instead, it is important
to normalize by the total number of objects in all sub-batches.

Box loss Similarly to [41,36], we use a soft version of Intersection over Union
in our loss, together with a (cid:96)1 loss on ˆb:

Lbox(bσ(i), ˆbi) = λiouLiou(bσ(i), ˆbi) + λL1||bσ(i) − ˆbi||1 ,

(9)

where λiou, λL1 ∈ R are hyperparameters and Liou(·) is the generalized IoU [38]:

Liou(bσ(i), ˆbi) = 1 −

(cid:18) |bσ(i) ∩ ˆbi|
|bσ(i) ∪ ˆbi|

−

|B(bσ(i), ˆbi) \ bσ(i) ∪ ˆbi|
|B(bσ(i), ˆbi)|

(cid:19)

.

(10)

|.| means “area”, and the union and intersection of box coordinates are used
as shorthands for the boxes themselves. The areas of unions or intersections
are computed by min / max of the linear functions of bσ(i) and ˆbi, which makes
the loss suﬃciently well-behaved for stochastic gradients. B(bσ(i), ˆbi) means the
largest box containing bσ(i), ˆbi (the areas involving B are also computed based
on min / max of linear functions of the box coordinates).

DICE/F-1 loss [28] The DICE coeﬃcient is closely related to the Intersection
over Union. If we denote by ˆm the raw mask logits prediction of the model, and
m the binary target mask, the loss is deﬁned as:

LDICE(m, ˆm) = 1 −

2mσ( ˆm) + 1
σ( ˆm) + m + 1

(11)

where σ is the sigmoid function. This loss is normalized by the number of objects.

22

Carion et al.

A.3 Detailed architecture

The detailed description of the transformer used in DETR, with positional en-
codings passed at every attention layer, is given in Fig. 10. Image features from
the CNN backbone are passed through the transformer encoder, together with
spatial positional encoding that are added to queries and keys at every multi-
head self-attention layer. Then, the decoder receives queries (initially set to zero),
output positional encoding (object queries), and encoder memory, and produces
the ﬁnal set of predicted class labels and bounding boxes through multiple multi-
head self-attention and decoder-encoder attention. The ﬁrst self-attention layer
in the ﬁrst decoder layer can be skipped.

Fig. 10: Architecture of DETR’s transformer. Please, see Section A.3 for details.

Computational complexity Every self-attention in the encoder has complex-
ity O(d2HW +d(HW )2): O(d(cid:48)d) is the cost of computing a single query/key/value
embeddings (and M d(cid:48) = d), while O(d(cid:48)(HW )2) is the cost of computing the at-
tention weights for one head. Other computations are negligible. In the decoder,
each self-attention is in O(d2N +dN 2), and cross-attention between encoder and
decoder is in O(d2(N + HW ) + dN HW ), which is much lower than the encoder
since N (cid:28) HW in practice.

Add&NormFFNAdd&NormMulti-HeadSelf-Attention++KQVN×ImagefeaturesEncoderMulti-HeadSelf-AttentionAdd&NormMulti-HeadAttentionAdd&NormFFNAdd&Norm++KQV++KQM×DecoderVSpatialpositionalencodingObjectqueriesFFNFFNClassBoundingBoxEnd-to-End Object Detection with Transformers

23

FLOPS computation Given that the FLOPS for Faster R-CNN depends on
the number of proposals in the image, we report the average number of FLOPS
for the ﬁrst 100 images in the COCO 2017 validation set. We compute the
FLOPS with the tool flop count operators from Detectron2 [50]. We use it
without modiﬁcations for Detectron2 models, and extend it to take batch matrix
multiply (bmm) into account for DETR models.

A.4 Training hyperparameters

We train DETR using AdamW [26] with improved weight decay handling, set to
10−4. We also apply gradient clipping, with a maximal gradient norm of 0.1. The
backbone and the transformers are treated slightly diﬀerently, we now discuss
the details for both.

Backbone ImageNet pretrained backbone ResNet-50 is imported from Torchvi-
sion, discarding the last classiﬁcation layer. Backbone batch normalization weights
and statistics are frozen during training, following widely adopted practice in ob-
ject detection. We ﬁne-tune the backbone using learning rate of 10−5. We observe
that having the backbone learning rate roughly an order of magnitude smaller
than the rest of the network is important to stabilize training, especially in the
ﬁrst few epochs.
Transformer We train the transformer with a learning rate of 10−4. Additive
dropout of 0.1 is applied after every multi-head attention and FFN before layer
normalization. The weights are randomly initialized with Xavier initialization.

Losses We use linear combination of (cid:96)1 and GIoU losses for bounding box re-
gression with λL1 = 5 and λiou = 2 weights respectively. All models were trained
with N = 100 decoder query slots.

Baseline Our enhanced Faster-RCNN+ baselines use GIoU [38] loss along with
the standard (cid:96)1 loss for bounding box regression. We performed a grid search
to ﬁnd the best weights for the losses and the ﬁnal models use only GIoU loss
with weights 20 and 1 for box and proposal regression tasks respectively. For the
baselines we adopt the same data augmentation as used in DETR and train it
with 9× schedule (approximately 109 epochs). All other settings are identical to
the same models in the Detectron2 model zoo [50].

Spatial positional encoding Encoder activations are associated with corre-
sponding spatial positions of image features. In our model we use a ﬁxed absolute
encoding to represent these spatial positions. We adopt a generalization of the
original Transformer [47] encoding to the 2D case [31]. Speciﬁcally, for both
spatial coordinates of each embedding we independently use d
2 sine and cosine
functions with diﬀerent frequencies. We then concatenate them to get the ﬁnal
d channel positional encoding.

A.5 Additional results

Some extra qualitative results for the panoptic prediction of the DETR-R101
model are shown in Fig.11.

24

Carion et al.

(a) Failure case with overlapping objects. PanopticFPN misses one plane entirely, while
DETR fails to accurately segment 3 of them.

(b) Things masks are predicted at full resolution, which allows sharper boundaries than
PanopticFPN

Fig. 11: Comparison of panoptic predictions. From left to right: Ground truth, Panop-
ticFPN with ResNet 101, DETR with ResNet 101

Increasing the number of instances By design, DETR cannot predict more
objects than it has query slots, i.e. 100 in our experiments. In this section,
we analyze the behavior of DETR when approaching this limit. We select a
canonical square image of a given class, repeat it on a 10 × 10 grid, and compute
the percentage of instances that are missed by the model. To test the model with
less than 100 instances, we randomly mask some of the cells. This ensures that
the absolute size of the objects is the same no matter how many are visible. To
account for the randomness in the masking, we repeat the experiment 100 times
with diﬀerent masks. The results are shown in Fig.12. The behavior is similar
across classes, and while the model detects all instances when up to 50 are
visible, it then starts saturating and misses more and more instances. Notably,
when the image contains all 100 instances, the model only detects 30 on average,
which is less than if the image contains only 50 instances that are all detected.
The counter-intuitive behavior of the model is likely because the images and the
detections are far from the training distribution.

Note that this test is a test of generalization out-of-distribution by design,
since there are very few example images with a lot of instances of a single class.
It is diﬃcult to disentangle, from the experiment, two types of out-of-domain
generalization: the image itself vs the number of object per class. But since few
to no COCO images contain only a lot of objects of the same class, this type
of experiment represents our best eﬀort to understand whether query objects
overﬁt the label and position distribution of the dataset. Overall, the experiments
suggests that the model does not overﬁt on these distributions since it yields
near-perfect detections up to 50 objects.

End-to-End Object Detection with Transformers

25

Fig. 12: Analysis of the number of instances of various classes missed by DETR de-
pending on how many are present in the image. We report the mean and the standard
deviation. As the number of instances gets close to 100, DETR starts saturating and
misses more and more objects

A.6 PyTorch inference code

To demonstrate the simplicity of the approach, we include inference code with
PyTorch and Torchvision libraries in Listing 1. The code runs with Python 3.6+,
PyTorch 1.4 and Torchvision 0.5. Note that it does not support batching, hence
it is suitable only for inference or training with DistributedDataParallel with
one image per GPU. Also note that for clarity, this code uses learnt positional
encodings in the encoder instead of ﬁxed, and positional encodings are added
to the input only instead of at each transformer layer. Making these changes
requires going beyond PyTorch implementation of transformers, which hampers
readability. The entire code to reproduce the experiments will be made available
before the conference.

20406080100010203040506070Numberofvisibleinstances%ofmissedinstancesdogpersonapple26

Carion et al.

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32

33

34

35

36

import torch
from torch import nn
from torchvision.models import resnet50

class DETR(nn.Module):

def __init__(self, num_classes, hidden_dim, nheads,

num_encoder_layers, num_decoder_layers):

super().__init__()
# We take only convolutional layers from ResNet-50 model
self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
self.conv = nn.Conv2d(2048, hidden_dim, 1)
self.transformer = nn.Transformer(hidden_dim, nheads,

num_encoder_layers, num_decoder_layers)

self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
self.linear_bbox = nn.Linear(hidden_dim, 4)
self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

def forward(self, inputs):

x = self.backbone(inputs)
h = self.conv(x)
H, W = h.shape[-2:]
pos = torch.cat([

self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),

], dim=-1).flatten(0, 1).unsqueeze(1)
h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),

self.query_pos.unsqueeze(1))

return self.linear_class(h), self.linear_bbox(h).sigmoid()

detr = DETR(num_classes=91, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6)
detr.eval()
inputs = torch.randn(1, 3, 800, 1200)
logits, bboxes = detr(inputs)

Listing 1: DETR PyTorch inference code. For clarity it uses learnt positional encod-
ings in the encoder instead of ﬁxed, and positional encodings are added to the input
only instead of at each transformer layer. Making these changes requires going beyond
PyTorch implementation of transformers, which hampers readability. The entire code
to reproduce the experiments will be made available before the conference.


