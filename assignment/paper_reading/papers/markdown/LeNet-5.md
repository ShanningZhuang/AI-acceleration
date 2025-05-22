PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:4)

Gradient(cid:2)Based Learning Applied to Document

Recognition

Yann LeCun(cid:2) L(cid:3)eon Bottou(cid:2) Yoshua Bengio(cid:2) and Patrick Ha(cid:4)ner

Abstract(cid:0)

I(cid:2) Introduction

Multilayer Neural Networks trained with the backpropa(cid:2)

gation algorithm constitute the best example of a successful

Over the last several years(cid:4) machine learning techniques(cid:4)

Gradient(cid:2)Based Learning technique(cid:3) Given an appropriate

particularly when applied to neural networks(cid:4) have played

network architecture(cid:4) Gradient(cid:2)Based Learning algorithms

can be used to synthesize a complex decision surface that can

an increasingly important role in the design of pattern

classify high(cid:2)dimensional patterns such as handwritten char(cid:2)

recognition systems(cid:2) In fact(cid:4) it could be argued that the

acters(cid:4) with minimal preprocessing(cid:3) This paper reviews var(cid:2)

availability of learning techniques has been a crucial fac(cid:3)

ious methods applied to handwritten character recognition

and compares them on a standard handwritten digit recog(cid:2)

tor in the recent success of pattern recognition applica(cid:3)

nition task(cid:3) Convolutional Neural Networks(cid:4) that are specif(cid:2)

tions such as continuous speech recognition and handwrit(cid:3)

ically designed to deal with the variability of (cid:5)D shapes(cid:4) are

ing recognition(cid:2)

shown to outperform all other techniques(cid:3)

Real(cid:2)life document recognition systems are composed

The main message of this paper is that better pattern

of multiple modules including (cid:6)eld extraction(cid:4) segmenta(cid:2)

recognition systems can be built by relying more on auto(cid:3)

tion(cid:4) recognition(cid:4) and language modeling(cid:3) A new learning

matic learning(cid:4) and less on hand(cid:3)designed heuristics(cid:2) This

paradigm(cid:4) called Graph Transformer Networks (cid:7)GTN(cid:8)(cid:4) al(cid:2)

is made possible by recent progress in machine learning

lows such multi(cid:2)module systems to be trained globally using

Gradient(cid:2)Based methods so as to minimize an overall per(cid:2)

and computer technology(cid:2) Using character recognition as

formance measure(cid:3)

a case study(cid:4) we show that hand(cid:3)crafted feature extrac(cid:3)

Two systems for on(cid:2)line handwriting recognition are de(cid:2)

tion can be advantageously replaced by carefully designed

scribed(cid:3) Experiments demonstrate the advantage of global

learning machines that operate directly on pixel images(cid:2)

training(cid:4) and the (cid:9)exibility of Graph Transformer Networks(cid:3)

A Graph Transformer Network for reading bank check is

Using document understanding as a case study(cid:4) we show

also described(cid:3) It uses Convolutional Neural Network char(cid:2)

that the traditional way of building recognition systems by

acter recognizers combined with global training techniques

manually integrating individually designed modules can be

to provides record accuracy on business and personal checks(cid:3)

It is deployed commercially and reads several million checks

replaced by a uni(cid:5)ed and well(cid:3)principled design paradigm(cid:4)

per day(cid:3)

called Graph Transformer Networks(cid:4) that allows training

Keywords(cid:0) Neural Networks(cid:4) OCR(cid:4) Document Recogni(cid:2)

all the modules to optimize a global performance criterion(cid:2)

tion(cid:4) Machine Learning(cid:4) Gradient(cid:2)Based Learning(cid:4) Convo(cid:2)

Since the early days of pattern recognition it has been

lutional Neural Networks(cid:4) Graph Transformer Networks(cid:4) Fi(cid:2)

known that the variability and richness of natural data(cid:4)

nite State Transducers(cid:3)

Nomenclature

entirely by hand(cid:2) Consequently(cid:4) most pattern recognition

be it speech(cid:4) glyphs(cid:4) or other types of patterns(cid:4) make it

almost impossible to build an accurate recognition system

(cid:0)

(cid:0)

(cid:0)

(cid:0)

(cid:0)

(cid:0)

(cid:0)

(cid:0)

(cid:0)

(cid:0)

(cid:0)

(cid:0)

(cid:0)

(cid:0)

GT Graph transformer(cid:2)

GTN Graph transformer network(cid:2)

HMM Hidden Markov model(cid:2)

HOS Heuristic oversegmentation(cid:2)

K(cid:3)NN K(cid:3)nearest neighbor(cid:2)

NN Neural network(cid:2)

OCR Optical character recognition(cid:2)

PCA Principal component analysis(cid:2)

RBF Radial basis function(cid:2)

RS(cid:3)SVM Reduced(cid:3)set support vector method(cid:2)

SDNN Space displacement neural network(cid:2)

SVM Support vector method(cid:2)

TDNN Time delay neural network(cid:2)

V(cid:3)SVM Virtual support vector method(cid:2)

systems are built using a combination of automatic learn(cid:3)

ing techniques and hand(cid:3)crafted algorithms(cid:2) The usual

method of recognizing individual patterns consists in divid(cid:3)

ing the system into two main modules shown in (cid:5)gure (cid:6)(cid:2)

The (cid:5)rst module(cid:4) called the feature extractor(cid:4) transforms

the input patterns so that they can be represented by low(cid:3)

dimensional vectors or short strings of symbols that (cid:7)a(cid:8) can

be easily matched or compared(cid:4) and (cid:7)b(cid:8) are relatively in(cid:3)

variant with respect to transformations and distortions of

the input patterns that do not change their nature(cid:2) The

feature extractor contains most of the prior knowledge and

is rather speci(cid:5)c to the task(cid:2) It is also the focus of most of

the design e(cid:9)ort(cid:4) because it is often entirely hand(cid:3)crafted(cid:2)

The classi(cid:5)er(cid:4) on the other hand(cid:4) is often general(cid:3)purpose

and trainable(cid:2) One of the main problems with this ap(cid:3)

proach is that the recognition accuracy is largely deter(cid:3)

The

authors

are with

the

Speech

and

Image Pro(cid:2)

cessing

Services

Research

Laboratory(cid:3)

AT(cid:4)T

Labs(cid:2)

mined by the ability of the designer to come up with an

Research(cid:3) (cid:5)(cid:6)(cid:6) Schulz Drive Red Bank(cid:3) NJ (cid:6)(cid:7)(cid:7)(cid:6)(cid:5)(cid:8)

E(cid:2)mail(cid:9)

appropriate set of features(cid:2) This turns out to be a daunt(cid:3)

fyann(cid:3)leonb(cid:3)yoshua(cid:3)ha(cid:10)nerg(cid:11)research(cid:8)att(cid:8)com(cid:8)

Yoshua Bengio

ing task which(cid:4) unfortunately(cid:4) must be redone for each new

is also with the D(cid:12)epartement d(cid:13)Informatique et de Recherche

Op(cid:12)erationelle(cid:3) Universit(cid:12)e de Montr(cid:12)eal(cid:3) C(cid:8)P(cid:8) (cid:14)(cid:5)(cid:15)(cid:16) Succ(cid:8) Centre(cid:2)Ville(cid:3)

problem(cid:2) A large amount of the pattern recognition liter(cid:3)

(cid:15)(cid:17)(cid:15)(cid:6) Chemin de la Tour(cid:3) Montr(cid:12)eal(cid:3) Qu(cid:12)ebec(cid:3) Canada H(cid:18)C (cid:18)J(cid:7)(cid:8)

ature is devoted to describing and comparing the relative

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:7)

Class scores

manipulate directed graphs(cid:2) This leads to the concept of

TRAINABLE CLASSIFIER MODULE

duced in Section IV(cid:2) Section V describes the now clas(cid:3)

trainable Graph Transformer Network (cid:7)GTN(cid:8) also intro(cid:3)

Feature vector

ing words or other character strings(cid:2) Discriminative and

sical method of heuristic over(cid:3)segmentation for recogniz(cid:3)

non(cid:3)discriminative gradient(cid:3)based techniques for training

FEATURE EXTRACTION MODULE

a recognizer at the word level without requiring manual

segmentation and labeling are presented in Section VI(cid:2) Sec(cid:3)

Raw input

tion VII presents the promising Space(cid:3)Displacement Neu(cid:3)

ral Network approach that eliminates the need for seg(cid:3)

Fig(cid:8) (cid:5)(cid:8) Traditional pattern recognition is performed with two mod(cid:2)

mentation heuristics by scanning a recognizer at all pos(cid:3)

ules(cid:9) a (cid:19)xed feature extractor(cid:3) and a trainable classi(cid:19)er(cid:8)

sible locations on the input(cid:2) In section VIII(cid:4) it is shown

that trainable Graph Transformer Networks can be for(cid:3)

mulated as multiple generalized transductions(cid:4) based on a

merits of di(cid:9)erent feature sets for particular tasks(cid:2)

general graph composition algorithm(cid:2) The connections be(cid:3)

Historically(cid:4) the need for appropriate feature extractors

tween GTNs and Hidden Markov Models(cid:4) commonly used

was due to the fact that the learning techniques used by

in speech recognition is also treated(cid:2) Section IX describes

the classi(cid:5)ers were limited to low(cid:3)dimensional spaces with

a globally trained GTN system for recognizing handwrit(cid:3)

easily separable classes (cid:10)(cid:6)(cid:11)(cid:2) A combination of three factors

ing entered in a pen computer(cid:2) This problem is known as

have changed this vision over the last decade(cid:2) First(cid:4) the

(cid:12)on(cid:3)line(cid:13) handwriting recognition(cid:4) since the machine must

availability of low(cid:3)cost machines with fast arithmetic units

produce immediate feedback as the user writes(cid:2) The core of

allows to rely more on brute(cid:3)force (cid:12)numerical(cid:13) methods

the system is a Convolutional Neural Network(cid:2) The results

than on algorithmic re(cid:5)nements(cid:2) Second(cid:4) the availability

clearly demonstrate the advantages of training a recognizer

of large databases for problems with a large market and

at the word level(cid:4) rather than training it on pre(cid:3)segmented(cid:4)

wide interest(cid:4) such as handwriting recognition(cid:4) has enabled

hand(cid:3)labeled(cid:4) isolated characters(cid:2) Section X describes a

designers to rely more on real data and less on hand(cid:3)crafted

complete GTN(cid:3)based system for reading handwritten and

feature extraction to build recognition systems(cid:2) The third

machine(cid:3)printed bank checks(cid:2) The core of the system is the

and very important factor is the availability of powerful ma(cid:3)

Convolutional Neural Network called LeNet(cid:3)(cid:15) described in

chine learning techniques that can handle high(cid:3)dimensional

Section II(cid:2) This system is in commercial use in the NCR

inputs and can generate intricate decision functions when

Corporation line of check recognition systems for the bank(cid:3)

fed with these large data sets(cid:2) It can be argued that the

ing industry(cid:2) It is reading millions of checks per month in

recent progress in the accuracy of speech and handwriting

several banks across the United States(cid:2)

recognition systems can be attributed in large part to an

increased reliance on learning techniques and large training

A(cid:2) Learning from Data

data sets(cid:2) As evidence to this fact(cid:4) a large proportion of

There are several approaches to automatic machine

modern commercial OCR systems use some form of multi(cid:3)

learning(cid:4) but one of the most successful approaches(cid:4) pop(cid:3)

layer Neural Network trained with back(cid:3)propagation(cid:2)

ularized in recent years by the neural network community(cid:4)

In this study(cid:4) we consider the tasks of handwritten char(cid:3)

can be called (cid:12)numerical(cid:13) or gradient(cid:3)based learning(cid:2) The

acter recognition (cid:7)Sections I and II(cid:8) and compare the per(cid:3)

learning machine computes a function Y

(cid:16) F (cid:7)Z

(cid:2) W (cid:8)

p

p

formance of several learning techniques on a benchmark

where Z

is the p(cid:3)th input pattern(cid:4) and W represents the

p

data set for handwritten digit recognition (cid:7)Section III(cid:8)(cid:2)

collection of adjustable parameters in the system(cid:2)

In a

While more automatic learning is bene(cid:5)cial(cid:4) no learning

pattern recognition setting(cid:4) the output Y

may be inter(cid:3)

p

technique can succeed without a minimal amount of prior

preted as the recognized class label of pattern Z

(cid:4) or as

p

knowledge about the task(cid:2) In the case of multi(cid:3)layer neu(cid:3)

scores or probabilities associated with each class(cid:2) A loss

ral networks(cid:4) a good way to incorporate knowledge is to

function E

(cid:16) D(cid:7)D

(cid:2) F (cid:7)W(cid:2) Z

(cid:8)(cid:8)(cid:4) measures the discrep(cid:3)

p

p

p

tailor its architecture to the task(cid:2) Convolutional Neu(cid:3)

ancy between D

(cid:4) the (cid:12)correct(cid:13) or desired output for pat(cid:3)

p

ral Networks (cid:10)(cid:14)(cid:11) introduced in Section II are an exam(cid:3)

tern Z

(cid:4) and the output produced by the system(cid:2) The

p

ple of specialized neural network architectures which in(cid:3)

average loss function E

(cid:7)W (cid:8) is the average of the er(cid:3)

train

corporate knowledge about the invariances of (cid:14)D shapes

rors E

over a set of labeled examples called the training

p

by using local connection patterns(cid:4) and by imposing con(cid:3)

set f(cid:7)Z

(cid:2) D

(cid:8)(cid:2) (cid:3)(cid:3)(cid:3)(cid:3)(cid:7)Z

(cid:2) D

(cid:8)g(cid:2)

In the simplest setting(cid:4) the

(cid:4)

(cid:4)

P

P

straints on the weights(cid:2) A comparison of several methods

learning problem consists in (cid:5)nding the value of W that

for isolated handwritten digit recognition is presented in

minimizes E

(cid:7)W (cid:8)(cid:2) In practice(cid:4) the performance of the

train

section III(cid:2) To go from the recognition of individual char(cid:3)

system on a training set is of little interest(cid:2) The more rel(cid:3)

acters to the recognition of words and sentences in docu(cid:3)

evant measure is the error rate of the system in the (cid:5)eld(cid:4)

ments(cid:4) the idea of combining multiple modules trained to

where it would be used in practice(cid:2) This performance is

reduce the overall error is introduced in Section IV(cid:2) Rec(cid:3)

estimated by measuring the accuracy on a set of samples

ognizing variable(cid:3)length ob jects such as handwritten words

disjoint from the training set(cid:4) called the test set(cid:2) Much

using multi(cid:3)module systems is best done if the modules

theoretical and experimental work (cid:10)(cid:17)(cid:11)(cid:4) (cid:10)(cid:18)(cid:11)(cid:4) (cid:10)(cid:15)(cid:11) has shown

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:8)

that the gap between the expected error rate on the test

Hessian matrix as in Newton or Quasi(cid:3)Newton methods(cid:2)

set E

and the error rate on the training set E

de(cid:3)

The Conjugate Gradient method (cid:10)(cid:25)(cid:11) can also be used(cid:2)

test

train

creases with the number of training samples approximately

However(cid:4) Appendix B shows that despite many claims

as

to the contrary in the literature(cid:4) the usefulness of these

E

(cid:2) E

(cid:16) k(cid:7)h(cid:4)P (cid:8)

(cid:7)(cid:6)(cid:8)

test

train

second(cid:3)order methods to large learning machines is very

(cid:2)

where P is the number of training samples(cid:4) h is a measure of

A popular minimization procedure is the stochastic gra(cid:3)

(cid:12)e(cid:9)ective capacity(cid:13) or complexity of the machine (cid:10)(cid:19)(cid:11)(cid:4) (cid:10)(cid:20)(cid:11)(cid:4) (cid:5)

dient algorithm(cid:4) also called the on(cid:3)line update(cid:2) It consists

is a number between (cid:21)(cid:3)(cid:15) and (cid:6)(cid:3)(cid:21)(cid:4) and k is a constant(cid:2) This

in updating the parameter vector using a noisy(cid:4) or approx(cid:3)

gap always decreases when the number of training samples

imated(cid:4) version of the average gradient(cid:2) In the most com(cid:3)

increases(cid:2) Furthermore(cid:4) as the capacity h increases(cid:4) E

train

mon instance of it(cid:4) W is updated on the basis of a single

limited(cid:2)

decreases(cid:2) Therefore(cid:4) when increasing the capacity h(cid:4) there

sample(cid:24)

is a trade(cid:3)o(cid:9) between the decrease of E

and the in(cid:3)

train

crease of the gap(cid:4) with an optimal value of the capacity h

that achieves the lowest generalization error E

(cid:2) Most

test

p

k

(cid:8)E

(cid:7)W (cid:8)

W

(cid:16) W

(cid:2) (cid:7)

(cid:7)(cid:17)(cid:8)

(cid:2)

k

k

(cid:4)

(cid:8)W

learning algorithms attempt to minimize E

as well as

train

With this procedure the parameter vector (cid:26)uctuates

some estimate of the gap(cid:2) A formal version of this is called

around an average tra jectory(cid:4) but usually converges consid(cid:3)

structural risk minimization (cid:10)(cid:19)(cid:11)(cid:4) (cid:10)(cid:20)(cid:11)(cid:4) and is based on de(cid:5)n(cid:3)

erably faster than regular gradient descent and second or(cid:3)

ing a sequence of learning machines of increasing capacity(cid:4)

der methods on large training sets with redundant samples

corresponding to a sequence of subsets of the parameter

(cid:7)such as those encountered in speech or character recogni(cid:3)

space such that each subset is a superset of the previous

tion(cid:8)(cid:2) The reasons for this are explained in Appendix B(cid:2)

subset(cid:2) In practical terms(cid:4) Structural Risk Minimization

The properties of such algorithms applied to learning have

is implemented by minimizing E

(cid:22) (cid:6)H (cid:7)W (cid:8)(cid:4) where the

train

been studied theoretically since the (cid:6)(cid:27)(cid:19)(cid:21)(cid:28)s (cid:10)(cid:27)(cid:11)(cid:4) (cid:10)(cid:6)(cid:21)(cid:11)(cid:4) (cid:10)(cid:6)(cid:6)(cid:11)(cid:4)

but practical successes for non(cid:3)trivial tasks did not occur

function H (cid:7)W (cid:8) is called a regularization function(cid:4) and (cid:6) is

a constant(cid:2) H (cid:7)W (cid:8) is chosen such that it takes large val(cid:3)

until the mid eighties(cid:2)

ues on parameters W that belong to high(cid:3)capacity subsets

C(cid:2) Gradient Back(cid:3)Propagation

of the parameter space(cid:2) Minimizing H (cid:7)W (cid:8) in e(cid:9)ect lim(cid:3)

its the capacity of the accessible subset of the parameter

Gradient(cid:3)Based Learning procedures have been used

space(cid:4) thereby controlling the tradeo(cid:9) between minimiz(cid:3)

since the late (cid:6)(cid:27)(cid:15)(cid:21)(cid:28)s(cid:4) but they were mostly limited to lin(cid:3)

ing the training error and minimizing the expected gap

ear systems (cid:10)(cid:6)(cid:11)(cid:2) The surprising usefulness of such sim(cid:3)

between the training error and test error(cid:2)

ple gradient descent techniques for complex machine learn(cid:3)

ing tasks was not widely realized until the following three

B(cid:2) Gradient(cid:3)Based Learning

events occurred(cid:2) The (cid:5)rst event was the realization that(cid:4)

The general problem of minimizing a function with re(cid:3)

of local minima in the loss function does not seem to

despite early warnings to the contrary (cid:10)(cid:6)(cid:14)(cid:11)(cid:4) the presence

spect to a set of parameters is at the root of many issues in

computer science(cid:2) Gradient(cid:3)Based Learning draws on the

fact that it is generally much easier to minimize a reason(cid:3)

ably smooth(cid:4) continuous function than a discrete (cid:7)combi(cid:3)

be a ma jor problem in practice(cid:2) This became apparent

when it was noticed that local minima did not seem to

be a ma jor impediment to the success of early non(cid:3)linear

gradient(cid:3)based Learning techniques such as Boltzmann ma(cid:3)

natorial(cid:8) function(cid:2) The loss function can be minimized by

chines (cid:10)(cid:6)(cid:17)(cid:11)(cid:4) (cid:10)(cid:6)(cid:18)(cid:11)(cid:2) The second event was the popularization

estimating the impact of small variations of the parame(cid:3)

ter values on the loss function(cid:2) This is measured by the

gradient of the loss function with respect to the param(cid:3)

eters(cid:2) E(cid:23)cient learning algorithms can be devised when

by Rumelhart(cid:4) Hinton and Williams (cid:10)(cid:6)(cid:15)(cid:11) and others of a

simple and e(cid:23)cient procedure(cid:4) the back(cid:3)propagation al(cid:3)

gorithm(cid:4) to compute the gradient in a non(cid:3)linear system

composed of several layers of processing(cid:2) The third event

the gradient vector can be computed analytically (cid:7)as op(cid:3)

was the demonstration that the back(cid:3)propagation proce(cid:3)

posed to numerically through perturbations(cid:8)(cid:2) This is the

basis of numerous gradient(cid:3)based learning algorithms with

continuous(cid:3)valued parameters(cid:2) In the procedures described

dure applied to multi(cid:3)layer neural networks with sigmoidal

units can solve complicated learning tasks(cid:2) The basic idea

of back(cid:3)propagation is that gradients can be computed e(cid:23)(cid:3)

in this article(cid:4) the set of parameters W is a real(cid:3)valued vec(cid:3)

ciently by propagation from the output to the input(cid:2) This

tor(cid:4) with respect to which E (cid:7)W (cid:8) is continuous(cid:4) as well as

idea was described in the control theory literature of the

di(cid:9)erentiable almost everywhere(cid:2) The simplest minimiza(cid:3)

tion procedure in such a setting is the gradient descent

algorithm where W is iteratively adjusted as follows(cid:24)

early sixties (cid:10)(cid:6)(cid:19)(cid:11)(cid:4) but its application to machine learning

was not generally realized then(cid:2)

Interestingly(cid:4) the early

derivations of back(cid:3)propagation in the context of neural

(cid:8)E (cid:7)W (cid:8)

network learning did not use gradients(cid:4) but (cid:12)virtual tar(cid:3)

W

(cid:16) W

(cid:2) (cid:7)

(cid:3)

(cid:7)(cid:14)(cid:8)

(cid:2)

k

k

(cid:4)

gets(cid:13) for units in intermediate layers (cid:10)(cid:6)(cid:20)(cid:11)(cid:4) (cid:10)(cid:6)(cid:25)(cid:11)(cid:4) or minimal

(cid:8)W

disturbance arguments (cid:10)(cid:6)(cid:27)(cid:11)(cid:2) The Lagrange formalism used

In the simplest case(cid:4) (cid:7) is a scalar constant(cid:2) More sophisti(cid:3)

in the control theory literature provides perhaps the best

cated procedures use variable (cid:7)(cid:4) or substitute it for a diag(cid:3)

rigorous method for deriving back(cid:3)propagation (cid:10)(cid:14)(cid:21)(cid:11)(cid:4) and for

onal matrix(cid:4) or substitute it for an estimate of the inverse

deriving generalizations of back(cid:3)propagation to recurrent

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:9)

networks (cid:10)(cid:14)(cid:6)(cid:11)(cid:4) and networks of heterogeneous modules (cid:10)(cid:14)(cid:14)(cid:11)(cid:2)

ferentiable(cid:4) and therefore lends itself to the use of Gradient(cid:3)

A simple derivation for generic multi(cid:3)layer systems is given

Based Learning methods(cid:2) Section V introduces the use of

in Section I(cid:3)E(cid:2)

directed acyclic graphs whose arcs carry numerical infor(cid:3)

The fact that local minima do not seem to be a problem

mation as a way to represent the alternative hypotheses(cid:4)

for multi(cid:3)layer neural networks is somewhat of a theoretical

and introduces the idea of GTN(cid:2)

mystery(cid:2) It is conjectured that if the network is oversized

The second solution described in Section VII is to elim(cid:3)

for the task (cid:7)as is usually the case in practice(cid:8)(cid:4) the presence

inate segmentation altogether(cid:2) The idea is to sweep the

of (cid:12)extra dimensions(cid:13) in parameter space reduces the risk

recognizer over every possible location on the input image(cid:4)

of unattainable regions(cid:2) Back(cid:3)propagation is by far the

and to rely on the (cid:12)character spotting(cid:13) property of the rec(cid:3)

most widely used neural(cid:3)network learning algorithm(cid:4) and

ognizer(cid:4) i(cid:2)e(cid:2) its ability to correctly recognize a well(cid:3)centered

probably the most widely used learning algorithm of any

character in its input (cid:5)eld(cid:4) even in the presence of other

form(cid:2)

characters besides it(cid:4) while rejecting images containing no

D(cid:2) Learning in Real Handwriting Recognition Systems

outputs obtained by sweeping the recognizer over the in(cid:3)

centered characters (cid:10)(cid:14)(cid:19)(cid:11)(cid:4) (cid:10)(cid:14)(cid:20)(cid:11)(cid:2) The sequence of recognizer

Isolated handwritten character recognition has been ex(cid:3)

put is then fed to a Graph Transformer Network that takes

tensively studied in the literature (cid:7)see (cid:10)(cid:14)(cid:17)(cid:11)(cid:4) (cid:10)(cid:14)(cid:18)(cid:11) for reviews(cid:8)(cid:4)

linguistic constraints into account and (cid:5)nally extracts the

and was one of the early successful applications of neural

most likely interpretation(cid:2) This GTN is somewhat similar

networks (cid:10)(cid:14)(cid:15)(cid:11)(cid:2) Comparative experiments on recognition of

to Hidden Markov Models (cid:7)HMM(cid:8)(cid:4) which makes the ap(cid:3)

individual handwritten digits are reported in Section III(cid:2)

proach reminiscent of the classical speech recognition (cid:10)(cid:14)(cid:25)(cid:11)(cid:4)

They show that neural networks trained with Gradient(cid:3)

(cid:10)(cid:14)(cid:27)(cid:11)(cid:2) While this technique would be quite expensive in

Based Learning perform better than all other methods

the general case(cid:4) the use of Convolutional Neural Networks

tested here on the same data(cid:2) The best neural networks(cid:4)

makes it particularly attractive because it allows signi(cid:5)cant

called Convolutional Networks(cid:4) are designed to learn to

savings in computational cost(cid:2)

extract relevant features directly from pixel images (cid:7)see

Section II(cid:8)(cid:2)

E(cid:2) Global ly Trainable Systems

One of the most di(cid:23)cult problems in handwriting recog(cid:3)

As stated earlier(cid:4) most practical pattern recognition sys(cid:3)

nition(cid:4) however(cid:4) is not only to recognize individual charac(cid:3)

tems are composed of multiple modules(cid:2) For example(cid:4) a

ters(cid:4) but also to separate out characters from their neigh(cid:3)

document recognition system is composed of a (cid:5)eld locator(cid:4)

bors within the word or sentence(cid:4) a process known as seg(cid:3)

which extracts regions of interest(cid:4) a (cid:5)eld segmenter(cid:4) which

mentation(cid:2) The technique for doing this that has become

cuts the input image into images of candidate characters(cid:4) a

the (cid:12)standard(cid:13) is called Heuristic Over(cid:3)Segmentation(cid:2) It

recognizer(cid:4) which classi(cid:5)es and scores each candidate char(cid:3)

consists in generating a large number of potential cuts

acter(cid:4) and a contextual post(cid:3)processor(cid:4) generally based on

between characters using heuristic image processing tech(cid:3)

a stochastic grammar(cid:4) which selects the best grammatically

niques(cid:4) and subsequently selecting the best combination of

correct answer from the hypotheses generated by the recog(cid:3)

cuts based on scores given for each candidate character by

nizer(cid:2) In most cases(cid:4) the information carried from module

the recognizer(cid:2) In such a model(cid:4) the accuracy of the sys(cid:3)

to module is best represented as graphs with numerical in(cid:3)

tem depends upon the quality of the cuts generated by the

formation attached to the arcs(cid:2) For example(cid:4) the output

heuristics(cid:4) and on the ability of the recognizer to distin(cid:3)

of the recognizer module can be represented as an acyclic

guish correctly segmented characters from pieces of char(cid:3)

graph where each arc contains the label and the score of

acters(cid:4) multiple characters(cid:4) or otherwise incorrectly seg(cid:3)

a candidate character(cid:4) and where each path represent a

mented characters(cid:2) Training a recognizer to perform this

alternative interpretation of the input string(cid:2) Typically(cid:4)

task poses a ma jor challenge because of the di(cid:23)culty in cre(cid:3)

each module is manually optimized(cid:4) or sometimes trained(cid:4)

ating a labeled database of incorrectly segmented charac(cid:3)

outside of its context(cid:2) For example(cid:4) the character recog(cid:3)

ters(cid:2) The simplest solution consists in running the images

nizer would be trained on labeled images of pre(cid:3)segmented

of character strings through the segmenter(cid:4) and then man(cid:3)

characters(cid:2) Then the complete system is assembled(cid:4) and

ually labeling all the character hypotheses(cid:2) Unfortunately(cid:4)

a subset of the parameters of the modules is manually ad(cid:3)

not only is this an extremely tedious and costly task(cid:4) it is

justed to maximize the overall performance(cid:2) This last step

also di(cid:23)cult to do the labeling consistently(cid:2) For example(cid:4)

is extremely tedious(cid:4) time(cid:3)consuming(cid:4) and almost certainly

should the right half of a cut up (cid:18) be labeled as a (cid:6) or as

suboptimal(cid:2)

a non(cid:3)character(cid:29) should the right half of a cut up (cid:25) be

A better alternative would be to somehow train the en(cid:3)

labeled as a (cid:17)(cid:29)

tire system so as to minimize a global error measure such as

The (cid:5)rst solution(cid:4) described in Section V consists in

the probability of character misclassi(cid:5)cations at the docu(cid:3)

training the system at the level of whole strings of char(cid:3)

ment level(cid:2) Ideally(cid:4) we would want to (cid:5)nd a good minimum

acters(cid:4) rather than at the character level(cid:2) The notion of

of this global loss function with respect to all the param(cid:3)

Gradient(cid:3)Based Learning can be used for this purpose(cid:2) The

eters in the system(cid:2) If the loss function E measuring the

system is trained to minimize an overall loss function which

performance can be made di(cid:9)erentiable with respect to the

measures the probability of an erroneous answer(cid:2) Section V

system(cid:28)s tunable parameters W (cid:4) we can (cid:5)nd a local min(cid:3)

explores various ways to ensure that the loss function is dif(cid:3)

imum of E using Gradient(cid:3)Based Learning(cid:2) However(cid:4) at

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:10)

(cid:5)rst glance(cid:4) it appears that the sheer size and complexity

tion system is best represented by graphs with numerical

of the system would make this intractable(cid:2)

information attached to the arcs(cid:2) In this case(cid:4) each module(cid:4)

To ensure that the global loss function E

(cid:7)Z

(cid:2) W (cid:8) is dif(cid:3)

called a Graph Transformer(cid:4) takes one or more graphs as

p

p

ferentiable(cid:4) the overall system is built as a feed(cid:3)forward net(cid:3)

input(cid:4) and produces a graph as output(cid:2) Networks of such

work of di(cid:9)erentiable modules(cid:2) The function implemented

modules are called Graph Transformer Networks (cid:7)GTN(cid:8)(cid:2)

by each module must be continuous and di(cid:9)erentiable al(cid:3)

Sections IV(cid:4) VI and VIII develop the concept of GTNs(cid:4)

most everywhere with respect to the internal parameters of

and show that Gradient(cid:3)Based Learning can be used to

the module (cid:7)e(cid:2)g(cid:2) the weights of a Neural Net character rec(cid:3)

train all the parameters in all the modules so as to mini(cid:3)

ognizer in the case of a character recognition module(cid:8)(cid:4) and

mize a global loss function(cid:2) It may seem paradoxical that

with respect to the module(cid:28)s inputs(cid:2) If this is the case(cid:4) a

gradients can be computed when the state information is

simple generalization of the well(cid:3)known back(cid:3)propagation

represented by essentially discrete ob jects such as graphs(cid:4)

procedure can be used to e(cid:23)ciently compute the gradients

but that di(cid:23)culty can be circumvented(cid:4) as shown later(cid:2)

of the loss function with respect to all the parameters in

the system (cid:10)(cid:14)(cid:14)(cid:11)(cid:2) For example(cid:4) let us consider a system

II(cid:2) Convolutional Neural Networks for

built as a cascade of modules(cid:4) each of which implements a

Isolated Character Recognition

function X

(cid:16) F

(cid:7)W

(cid:2) X

(cid:8)(cid:4) where X

is a vector rep(cid:3)

n

n

n

n

n

(cid:4)

(cid:2)

The ability of multi(cid:3)layer networks trained with gradi(cid:3)

resenting the output of the module(cid:4) W

is the vector of

n

ent descent to learn complex(cid:4) high(cid:3)dimensional(cid:4) non(cid:3)linear

tunable parameters in the module (cid:7)a subset of W (cid:8)(cid:4) and

mappings from large collections of examples makes them

X

is the module(cid:28)s input vector (cid:7)as well as the previous

(cid:2)

n

(cid:4)

obvious candidates for image recognition tasks(cid:2) In the tra(cid:3)

module(cid:28)s output vector(cid:8)(cid:2) The input X

to the (cid:5)rst module

(cid:11)

is the input pattern Z

(cid:2) If the partial derivative of E

with

p

p

ditional model of pattern recognition(cid:4) a hand(cid:3)designed fea(cid:3)

ture extractor gathers relevant information from the input

p

respect to X

is known(cid:4) then the partial derivatives of E

n

and eliminates irrelevant variabilities(cid:2) A trainable classi(cid:5)er

with respect to W

and X

can be computed using the

(cid:2)

n

n

(cid:4)

then categorizes the resulting feature vectors into classes(cid:2)

backward recurrence

In this scheme(cid:4) standard(cid:4) fully(cid:3)connected multi(cid:3)layer net(cid:3)

p

p

(cid:8)E

(cid:8)F

(cid:8)E

works can be used as classi(cid:5)ers(cid:2) A potentially more inter(cid:3)

(cid:16)

(cid:7)W

(cid:2) X

(cid:8)

(cid:2)

n

n

(cid:4)

(cid:8)W

(cid:8)W

(cid:8)X

n

n

esting scheme is to rely on as much as possible on learning

p

p

(cid:8)E

(cid:8)F

(cid:8)E

in the feature extractor itself(cid:2)

In the case of character

(cid:16)

(cid:7)W

(cid:2) X

(cid:8)

(cid:7)(cid:18)(cid:8)

(cid:2)

n

n

(cid:4)

(cid:8)X

(cid:8)X

(cid:8)X

n

(cid:4)

n

(cid:2)

recognition(cid:4) a network could be fed with almost raw in(cid:3)

(cid:3)F

where

(cid:7)W

(cid:2) X

(cid:8) is the Jacobian of F with respect to

(cid:2)

(cid:3)W

n

n

(cid:4)

with an ordinary fully connected feed(cid:3)forward network with

W evaluated at the point (cid:7)W

(cid:2) X

(cid:8)(cid:4) and

(cid:7)W

(cid:2) X

(cid:8)

n

n

n

n

(cid:4)

(cid:4)

(cid:3)X

(cid:2)

(cid:2)

some success for tasks such as character recognition(cid:4) there

(cid:3)F

puts (cid:7)e(cid:2)g(cid:2) size(cid:3)normalized images(cid:8)(cid:2) While this can be done

is the Jacobian of F with respect to X (cid:2) The Jacobian of

are problems(cid:2)

a vector function is a matrix containing the partial deriva(cid:3)

Firstly(cid:4) typical images are large(cid:4) often with several hun(cid:3)

tives of all the outputs with respect to all the inputs(cid:2)

dred variables (cid:7)pixels(cid:8)(cid:2) A fully(cid:3)connected (cid:5)rst layer with(cid:4)

The (cid:5)rst equation computes some terms of the gradient

say one hundred hidden units in the (cid:5)rst layer(cid:4) would al(cid:3)

p

of E

(cid:7)W (cid:8)(cid:4) while the second equation generates a back(cid:3)

ready contain several tens of thousands of weights(cid:2) Such

ward recurrence(cid:4) as in the well(cid:3)known back(cid:3)propagation

a large number of parameters increases the capacity of the

procedure for neural networks(cid:2) We can average the gradi(cid:3)

system and therefore requires a larger training set(cid:2) In ad(cid:3)

ents over the training patterns to obtain the full gradient(cid:2)

dition(cid:4) the memory requirement to store so many weights

It is interesting to note that in many instances there is

may rule out certain hardware implementations(cid:2) But(cid:4) the

no need to explicitly compute the Jacobian matrix(cid:2) The

main de(cid:5)ciency of unstructured nets for image or speech

above formula uses the product of the Jacobian with a vec(cid:3)

applications is that they have no built(cid:3)in invariance with

tor of partial derivatives(cid:4) and it is often easier to compute

respect to translations(cid:4) or local distortions of the inputs(cid:2)

this product directly without computing the Jacobian be(cid:3)

Before being sent to the (cid:5)xed(cid:3)size input layer of a neural

forehand(cid:2) In By analogy with ordinary multi(cid:3)layer neural

net(cid:4) character images(cid:4) or other (cid:14)D or (cid:6)D signals(cid:4) must be

networks(cid:4) all but the last module are called hidden layers

approximately size(cid:3)normalized and centered in the input

because their outputs are not observable from the outside(cid:2)

(cid:5)eld(cid:2) Unfortunately(cid:4) no such preprocessing can be perfect(cid:24)

more complex situations than the simple cascade of mod(cid:3)

handwriting is often normalized at the word level(cid:4) which

ules described above(cid:4) the partial derivative notation be(cid:3)

can cause size(cid:4) slant(cid:4) and position variations for individual

comes somewhat ambiguous and awkward(cid:2) A completely

characters(cid:2) This(cid:4) combined with variability in writing style(cid:4)

rigorous derivation in more general cases can be done using

will cause variations in the position of distinctive features

Lagrange functions (cid:10)(cid:14)(cid:21)(cid:11)(cid:4) (cid:10)(cid:14)(cid:6)(cid:11)(cid:4) (cid:10)(cid:14)(cid:14)(cid:11)(cid:2)

in input ob jects(cid:2) In principle(cid:4) a fully(cid:3)connected network of

Traditional multi(cid:3)layer neural networks are a special case

su(cid:23)cient size could learn to produce outputs that are in(cid:3)

of the above where the state information X

is represented

variant with respect to such variations(cid:2) However(cid:4) learning

n

with (cid:5)xed(cid:3)sized vectors(cid:4) and where the modules are al(cid:3)

such a task would probably result in multiple units with

ternated layers of matrix multiplications (cid:7)the weights(cid:8) and

similar weight patterns positioned at various locations in

component(cid:3)wise sigmoid functions (cid:7)the neurons(cid:8)(cid:2) However(cid:4)

the input so as to detect distinctive features wherever they

as stated earlier(cid:4) the state information in complex recogni(cid:3)

appear on the input(cid:2) Learning these weight con(cid:5)gurations

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:12)

requires a very large number of training instances to cover

planes(cid:4) each of which is a feature map(cid:2) A unit in a feature

the space of possible variations(cid:2) In convolutional networks(cid:4)

map has (cid:14)(cid:15) inputs connected to a (cid:15) by (cid:15) area in the input(cid:4)

described below(cid:4) shift invariance is automatically obtained

called the receptive (cid:4)eld of the unit(cid:2) Each unit has (cid:14)(cid:15) in(cid:3)

by forcing the replication of weight con(cid:5)gurations across

puts(cid:4) and therefore (cid:14)(cid:15) trainable coe(cid:23)cients plus a trainable

space(cid:2)

bias(cid:2) The receptive (cid:5)elds of contiguous units in a feature

Secondly(cid:4) a de(cid:5)ciency of fully(cid:3)connected architectures is

map are centered on correspondingly contiguous units in

that the topology of the input is entirely ignored(cid:2) The in(cid:3)

the previous layer(cid:2) Therefore receptive (cid:5)elds of neighbor(cid:3)

put variables can be presented in any (cid:7)(cid:5)xed(cid:8) order without

ing units overlap(cid:2) For example(cid:4) in the (cid:5)rst hidden layer

a(cid:9)ecting the outcome of the training(cid:2) On the contrary(cid:4)

of LeNet(cid:3)(cid:15)(cid:4) the receptive (cid:5)elds of horizontally contiguous

images (cid:7)or time(cid:3)frequency representations of speech(cid:8) have

units overlap by (cid:18) columns and (cid:15) rows(cid:2) As stated earlier(cid:4)

a strong (cid:14)D local structure(cid:24) variables (cid:7)or pixels(cid:8) that are

all the units in a feature map share the same set of (cid:14)(cid:15)

spatially or temporally nearby are highly correlated(cid:2) Local

weights and the same bias so they detect the same feature

correlations are the reasons for the well(cid:3)known advantages

at all possible locations on the input(cid:2) The other feature

of extracting and combining local features before recogniz(cid:3)

maps in the layer use di(cid:9)erent sets of weights and biases(cid:4)

ing spatial or temporal ob jects(cid:4) because con(cid:5)gurations of

thereby extracting di(cid:9)erent types of local features(cid:2) In the

neighboring variables can be classi(cid:5)ed into a small number

case of LeNet(cid:3)(cid:15)(cid:4) at each input location six di(cid:9)erent types

of categories (cid:7)e(cid:2)g(cid:2) edges(cid:4) corners(cid:2)(cid:2)(cid:2)(cid:8)(cid:2) Convolutional Net(cid:3)

of features are extracted by six units in identical locations

works force the extraction of local features by restricting

in the six feature maps(cid:2) A sequential implementation of

the receptive (cid:5)elds of hidden units to be local(cid:2)

a feature map would scan the input image with a single

A(cid:2) Convolutional Networks

of this unit at corresponding locations in the feature map(cid:2)

unit that has a local receptive (cid:5)eld(cid:4) and store the states

Convolutional Networks combine three architectural

This operation is equivalent to a convolution(cid:4) followed by

ideas to ensure some degree of shift(cid:4) scale(cid:4) and distor(cid:3)

an additive bias and squashing function(cid:4) hence the name

tion invariance(cid:24)

local receptive (cid:4)elds(cid:4) shared weights (cid:7)or

convolutional network(cid:2) The kernel of the convolution is the

weight replication(cid:8)(cid:4) and spatial or temporal sub(cid:3)sampling(cid:2)

set of connection weights used by the units in the feature

A typical convolutional network for recognizing characters(cid:4)

map(cid:2) An interesting property of convolutional layers is that

dubbed LeNet(cid:3)(cid:15)(cid:4) is shown in (cid:5)gure (cid:14)(cid:2) The input plane

if the input image is shifted(cid:4) the feature map output will

receives images of characters that are approximately size(cid:3)

be shifted by the same amount(cid:4) but will be left unchanged

normalized and centered(cid:2) Each unit in a layer receives in(cid:3)

otherwise(cid:2) This property is at the basis of the robustness

of convolutional networks to shifts and distortions of the

puts from a set of units located in a small neighborhood

in the previous layer(cid:2) The idea of connecting units to local

input(cid:2)

receptive (cid:5)elds on the input goes back to the Perceptron in

Once a feature has been detected(cid:4)

its exact location

the early (cid:19)(cid:21)s(cid:4) and was almost simultaneous with Hubel and

becomes less important(cid:2) Only its approximate position

Wiesel(cid:28)s discovery of locally(cid:3)sensitive(cid:4) orientation(cid:3)selective

relative to other features is relevant(cid:2) For example(cid:4) once

neurons in the cat(cid:28)s visual system (cid:10)(cid:17)(cid:21)(cid:11)(cid:2) Local connections

we know that the input image contains the endpoint of a

have been used many times in neural models of visual learn(cid:3)

roughly horizontal segment in the upper left area(cid:4) a corner

ing (cid:10)(cid:17)(cid:6)(cid:11)(cid:4)

(cid:10)(cid:17)(cid:14)(cid:11)(cid:4)

(cid:10)(cid:6)(cid:25)(cid:11)(cid:4) (cid:10)(cid:17)(cid:17)(cid:11)(cid:4)

(cid:10)(cid:17)(cid:18)(cid:11)(cid:4)

(cid:10)(cid:14)(cid:11)(cid:2) With local receptive

in the upper right area(cid:4) and the endpoint of a roughly ver(cid:3)

(cid:5)elds(cid:4) neurons can extract elementary visual features such

tical segment in the lower portion of the image(cid:4) we can tell

as oriented edges(cid:4) end(cid:3)points(cid:4) corners (cid:7)or similar features in

the input image is a (cid:20)(cid:2) Not only is the precise position of

other signals such as speech spectrograms(cid:8)(cid:2) These features

each of those features irrelevant for identifying the pattern(cid:4)

are then combined by the subsequent layers in order to de(cid:3)

it is potentially harmful because the positions are likely to

tect higher(cid:3)order features(cid:2) As stated earlier(cid:4) distortions or

vary for di(cid:9)erent instances of the character(cid:2) A simple way

shifts of the input can cause the position of salient features

to reduce the precision with which the position of distinc(cid:3)

to vary(cid:2) In addition(cid:4) elementary feature detectors that are

tive features are encoded in a feature map is to reduce the

useful on one part of the image are likely to be useful across

spatial resolution of the feature map(cid:2) This can be achieved

the entire image(cid:2) This knowledge can be applied by forcing

with a so(cid:3)called sub(cid:3)sampling layers which performs a local

a set of units(cid:4) whose receptive (cid:5)elds are located at di(cid:9)erent

averaging and a sub(cid:3)sampling(cid:4) reducing the resolution of

places on the image(cid:4) to have identical weight vectors (cid:10)(cid:17)(cid:14)(cid:11)(cid:4)

the feature map(cid:4) and reducing the sensitivity of the output

(cid:10)(cid:6)(cid:15)(cid:11)(cid:4) (cid:10)(cid:17)(cid:18)(cid:11)(cid:2) Units in a layer are organized in planes within

to shifts and distortions(cid:2) The second hidden layer of LeNet(cid:3)

which all the units share the same set of weights(cid:2) The set

(cid:15) is a sub(cid:3)sampling layer(cid:2) This layer comprises six feature

of outputs of the units in such a plane is called a feature

maps(cid:4) one for each feature map in the previous layer(cid:2) The

map(cid:2) Units in a feature map are all constrained to per(cid:3)

receptive (cid:5)eld of each unit is a (cid:14) by (cid:14) area in the previous

form the same operation on di(cid:9)erent parts of the image(cid:2)

layer(cid:28)s corresponding feature map(cid:2) Each unit computes the

A complete convolutional layer is composed of several fea(cid:3)

average of its four inputs(cid:4) multiplies it by a trainable coef(cid:3)

ture maps (cid:7)with di(cid:9)erent weight vectors(cid:8)(cid:4) so that multiple

(cid:5)cient(cid:4) adds a trainable bias(cid:4) and passes the result through

features can be extracted at each location(cid:2) A concrete ex(cid:3)

a sigmoid function(cid:2) Contiguous units have non(cid:3)overlapping

ample of this is the (cid:5)rst layer of LeNet(cid:3)(cid:15) shown in Figure (cid:14)(cid:2)

contiguous receptive (cid:5)elds(cid:2) Consequently(cid:4) a sub(cid:3)sampling

Units in the (cid:5)rst hidden layer of LeNet(cid:3)(cid:15) are organized in (cid:19)

layer feature map has half the number of rows and columns

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:13)

INPUT
32x32

C1: feature maps
6@28x28

C3: f. maps 16@10x10

S4: f. maps 16@5x5

S2: f. maps
6@14x14

C5: layer
120

F6: layer
 84

OUTPUT
 10

Convolutions

Subsampling

Convolutions

Subsampling

Full connection

Full connection

Gaussian connections

Fig(cid:8) (cid:15)(cid:8) Architecture of LeNet(cid:2)(cid:20)(cid:3) a Convolutional Neural Network(cid:3) here for digits recognition(cid:8) Each plane is a feature map(cid:3) i(cid:8)e(cid:8) a set of units

whose weights are constrained to be identical(cid:8)

as the feature maps in the previous layer(cid:2) The trainable

B(cid:2) LeNet(cid:3)(cid:5)

coe(cid:23)cient and bias control the e(cid:9)ect of the sigmoid non(cid:3)

linearity(cid:2) If the coe(cid:23)cient is small(cid:4) then the unit operates

This section describes in more detail the architecture of

in a quasi(cid:3)linear mode(cid:4) and the sub(cid:3)sampling layer merely

LeNet(cid:3)(cid:15)(cid:4) the Convolutional Neural Network used in the

blurs the input(cid:2)

If the coe(cid:23)cient is large(cid:4) sub(cid:3)sampling

experiments(cid:2) LeNet(cid:3)(cid:15) comprises (cid:20) layers(cid:4) not counting the

units can be seen as performing a (cid:12)noisy OR(cid:13) or a (cid:12)noisy

input(cid:4) all of which contain trainable parameters (cid:7)weights(cid:8)(cid:2)

AND(cid:13) function depending on the value of the bias(cid:2) Succes(cid:3)

The input is a (cid:17)(cid:14)x(cid:17)(cid:14) pixel image(cid:2) This is signi(cid:5)cantly larger

sive layers of convolutions and sub(cid:3)sampling are typically

than the largest character in the database (cid:7)at most (cid:14)(cid:21)x(cid:14)(cid:21)

alternated(cid:4) resulting in a (cid:12)bi(cid:3)pyramid(cid:13)(cid:24) at each layer(cid:4) the

pixels centered in a (cid:14)(cid:25)x(cid:14)(cid:25) (cid:5)eld(cid:8)(cid:2) The reason is that it is

number of feature maps is increased as the spatial resolu(cid:3)

desirable that potential distinctive features such as stroke

tion is decreased(cid:2) Each unit in the third hidden layer in (cid:5)g(cid:3)

end(cid:3)points or corner can appear in the center of the recep(cid:3)

ure (cid:14) may have input connections from several feature maps

tive (cid:5)eld of the highest(cid:3)level feature detectors(cid:2) In LeNet(cid:3)(cid:15)

in the previous layer(cid:2) The convolution(cid:30)sub(cid:3)sampling com(cid:3)

the set of centers of the receptive (cid:5)elds of the last convolu(cid:3)

bination(cid:4) inspired by Hubel and Wiesel(cid:28)s notions of (cid:12)sim(cid:3)

tional layer (cid:7)C(cid:17)(cid:4) see below(cid:8) form a (cid:14)(cid:21)x(cid:14)(cid:21) area in the center

ple(cid:13) and (cid:12)complex(cid:13) cells(cid:4) was implemented in Fukushima(cid:28)s

of the (cid:17)(cid:14)x(cid:17)(cid:14) input(cid:2) The values of the input pixels are nor(cid:3)

Neocognitron (cid:10)(cid:17)(cid:14)(cid:11)(cid:4) though no globally supervised learning

malized so that the background level (cid:7)white(cid:8) corresponds

procedure such as back(cid:3)propagation was available then(cid:2) A

to a value of (cid:3)(cid:21)(cid:2)(cid:6) and the foreground (cid:7)black(cid:8) corresponds

large degree of invariance to geometric transformations of

to (cid:6)(cid:2)(cid:6)(cid:20)(cid:15)(cid:2) This makes the mean input roughly (cid:21)(cid:4) and the

the input can be achieved with this progressive reduction

variance roughly (cid:6) which accelerates learning (cid:10)(cid:18)(cid:19)(cid:11)(cid:2)

of spatial resolution compensated by a progressive increase

In the following(cid:4) convolutional layers are labeled Cx(cid:4) sub(cid:3)

of the richness of the representation (cid:7)the number of feature

sampling layers are labeled Sx(cid:4) and fully(cid:3)connected layers

maps(cid:8)(cid:2)

are labeled Fx(cid:4) where x is the layer index(cid:2)

Since all the weights are learned with back(cid:3)propagation(cid:4)

Layer C(cid:6) is a convolutional layer with (cid:19) feature maps(cid:2)

convolutional networks can be seen as synthesizing their

Each unit in each feature map is connected to a (cid:15)x(cid:15) neigh(cid:3)

own feature extractor(cid:2) The weight sharing technique has

borhood in the input(cid:2) The size of the feature maps is (cid:14)(cid:25)x(cid:14)(cid:25)

the interesting side e(cid:9)ect of reducing the number of free

which prevents connection from the input from falling o(cid:9)

parameters(cid:4) thereby reducing the (cid:12)capacity(cid:13) of the ma(cid:3)

the boundary(cid:2) C(cid:6) contains (cid:6)(cid:15)(cid:19) trainable parameters(cid:4) and

chine and reducing the gap between test error and training

(cid:6)(cid:14)(cid:14)(cid:4)(cid:17)(cid:21)(cid:18) connections(cid:2)

error (cid:10)(cid:17)(cid:18)(cid:11)(cid:2) The network in (cid:5)gure (cid:14) contains (cid:17)(cid:18)(cid:21)(cid:4)(cid:27)(cid:21)(cid:25) con(cid:3)

Layer S(cid:14) is a sub(cid:3)sampling layer with (cid:19) feature maps of

nections(cid:4) but only (cid:19)(cid:21)(cid:4)(cid:21)(cid:21)(cid:21) trainable free parameters because

size (cid:6)(cid:18)x(cid:6)(cid:18)(cid:2) Each unit in each feature map is connected to a

of the weight sharing(cid:2)

(cid:14)x(cid:14) neighborhood in the corresponding feature map in C(cid:6)(cid:2)

Fixed(cid:3)size Convolutional Networks have been applied

The four inputs to a unit in S(cid:14) are added(cid:4) then multiplied

to many applications(cid:4) among other handwriting recogni(cid:3)

by a trainable coe(cid:23)cient(cid:4) and added to a trainable bias(cid:2)

tion (cid:10)(cid:17)(cid:15)(cid:11)(cid:4) (cid:10)(cid:17)(cid:19)(cid:11)(cid:4) machine(cid:3)printed character recognition (cid:10)(cid:17)(cid:20)(cid:11)(cid:4)

The result is passed through a sigmoidal function(cid:2) The

on(cid:3)line handwriting recognition (cid:10)(cid:17)(cid:25)(cid:11)(cid:4) and face recogni(cid:3)

(cid:14)x(cid:14) receptive (cid:5)elds are non(cid:3)overlapping(cid:4) therefore feature

tion (cid:10)(cid:17)(cid:27)(cid:11)(cid:2) Fixed(cid:3)size convolutional networks that share

maps in S(cid:14) have half the number of rows and column as

weights along a single temporal dimension are known as

feature maps in C(cid:6)(cid:2) Layer S(cid:14) has (cid:6)(cid:14) trainable parameters

Time(cid:3)Delay Neural Networks (cid:7)TDNNs(cid:8)(cid:2) TDNNs have been

and (cid:15)(cid:4)(cid:25)(cid:25)(cid:21) connections(cid:2)

used in phoneme recognition (cid:7)without sub(cid:3)sampling(cid:8) (cid:10)(cid:18)(cid:21)(cid:11)(cid:4)

Layer C(cid:17) is a convolutional layer with (cid:6)(cid:19) feature maps(cid:2)

(cid:10)(cid:18)(cid:6)(cid:11)(cid:4) spoken word recognition (cid:7)with sub(cid:3)sampling(cid:8) (cid:10)(cid:18)(cid:14)(cid:11)(cid:4)

Each unit in each feature map is connected to several (cid:15)x(cid:15)

(cid:10)(cid:18)(cid:17)(cid:11)(cid:4) on(cid:3)line recognition of isolated handwritten charac(cid:3)

neighborhoods at identical locations in a subset of S(cid:14)(cid:28)s

ters (cid:10)(cid:18)(cid:18)(cid:11)(cid:4) and signature veri(cid:5)cation (cid:10)(cid:18)(cid:15)(cid:11)(cid:2)

feature maps(cid:2) Table I shows the set of S(cid:14) feature maps

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:6)

(cid:21) (cid:6) (cid:14) (cid:17) (cid:18) (cid:15) (cid:19) (cid:20) (cid:25) (cid:27) (cid:6)(cid:21) (cid:6)(cid:6) (cid:6)(cid:14) (cid:6)(cid:17) (cid:6)(cid:18) (cid:6)(cid:15)

where A is the amplitude of the function and S determines

(cid:21) X

X X X

X X X X

X X

its slope at the origin(cid:2) The function f is odd(cid:4) with horizon(cid:3)

(cid:6) X X

X X X

X X X X

X

tal asymptotes at (cid:22)A and (cid:2)A(cid:2) The constant A is chosen

(cid:14) X X X

X X X

X

X X X

to be (cid:6)(cid:3)(cid:20)(cid:6)(cid:15)(cid:27)(cid:2) The rationale for this choice of a squashing

(cid:17)

X X X

X X X X

X

X X

function is given in Appendix A(cid:2)

(cid:18)

X X X

X X X X

X X

X

Finally(cid:4) the output layer is composed of Euclidean Radial

(cid:15)

X X X

X X X X

X X X

Basis Function units (cid:7)RBF(cid:8)(cid:4) one for each class(cid:4) with (cid:25)(cid:18)

Each column indicates which feature map in S(cid:2) are combined

as follows(cid:24)

by the units in a particular feature map of C(cid:3)(cid:4)

y

(cid:16)

(cid:7)x

(cid:2) w

(cid:8)

(cid:3)

(cid:7)(cid:20)(cid:8)

i

j

ij

(cid:7)

TABLE I

inputs each(cid:2) The outputs of each RBF unit y

is computed

i

X

j

In other words(cid:4) each output RBF unit computes the Eu(cid:3)

clidean distance between its input vector and its parameter

combined by each C(cid:17) feature map(cid:2) Why not connect ev(cid:3)

vector(cid:2) The further away is the input from the parameter

ery S(cid:14) feature map to every C(cid:17) feature map(cid:29) The rea(cid:3)

vector(cid:4) the larger is the RBF output(cid:2) The output of a

son is twofold(cid:2) First(cid:4) a non(cid:3)complete connection scheme

particular RBF can be interpreted as a penalty term mea(cid:3)

keeps the number of connections within reasonable bounds(cid:2)

suring the (cid:5)t between the input pattern and a model of the

More importantly(cid:4) it forces a break of symmetry in the net(cid:3)

class associated with the RBF(cid:2) In probabilistic terms(cid:4) the

work(cid:2) Di(cid:9)erent feature maps are forced to extract di(cid:9)erent

RBF output can be interpreted as the unnormalized nega(cid:3)

(cid:7)hopefully complementary(cid:8) features because they get dif(cid:3)

tive log(cid:3)likelihood of a Gaussian distribution in the space

ferent sets of inputs(cid:2) The rationale behind the connection

of con(cid:5)gurations of layer F(cid:19)(cid:2) Given an input pattern(cid:4) the

scheme in table I is the following(cid:2) The (cid:5)rst six C(cid:17) feature

loss function should be designed so as to get the con(cid:5)gu(cid:3)

maps take inputs from every contiguous subsets of three

ration of F(cid:19) as close as possible to the parameter vector

feature maps in S(cid:14)(cid:2) The next six take input from every

of the RBF that corresponds to the pattern(cid:28)s desired class(cid:2)

contiguous subset of four(cid:2) The next three take input from

The parameter vectors of these units were chosen by hand

some discontinuous subsets of four(cid:2) Finally the last one

and kept (cid:5)xed (cid:7)at least initially(cid:8)(cid:2) The components of those

takes input from all S(cid:14) feature maps(cid:2) Layer C(cid:17) has (cid:6)(cid:4)(cid:15)(cid:6)(cid:19)

parameters vectors were set to (cid:3)(cid:6) or (cid:22)(cid:6)(cid:2) While they could

trainable parameters and (cid:6)(cid:15)(cid:6)(cid:4)(cid:19)(cid:21)(cid:21) connections(cid:2)

have been chosen at random with equal probabilities for (cid:3)(cid:6)

Layer S(cid:18) is a sub(cid:3)sampling layer with (cid:6)(cid:19) feature maps of

and (cid:22)(cid:6)(cid:4) or even chosen to form an error correcting code

size (cid:15)x(cid:15)(cid:2) Each unit in each feature map is connected to a

as suggested by (cid:10)(cid:18)(cid:20)(cid:11)(cid:4) they were instead designed to repre(cid:3)

(cid:14)x(cid:14) neighborhood in the corresponding feature map in C(cid:17)(cid:4)

sent a stylized image of the corresponding character class

in a similar way as C(cid:6) and S(cid:14)(cid:2) Layer S(cid:18) has (cid:17)(cid:14) trainable

drawn on a (cid:20)x(cid:6)(cid:14) bitmap (cid:7)hence the number (cid:25)(cid:18)(cid:8)(cid:2) Such a

parameters and (cid:14)(cid:4)(cid:21)(cid:21)(cid:21) connections(cid:2)

representation is not particularly useful for recognizing iso(cid:3)

Layer C(cid:15) is a convolutional layer with (cid:6)(cid:14)(cid:21) feature maps(cid:2)

lated digits(cid:4) but it is quite useful for recognizing strings of

Each unit is connected to a (cid:15)x(cid:15) neighborhood on all (cid:6)(cid:19)

characters taken from the full printable ASCII set(cid:2) The

of S(cid:18)(cid:28)s feature maps(cid:2) Here(cid:4) because the size of S(cid:18) is also

rationale is that characters that are similar(cid:4) and therefore

(cid:15)x(cid:15)(cid:4) the size of C(cid:15)(cid:28)s feature maps is (cid:6)x(cid:6)(cid:24) this amounts

confusable(cid:4) such as uppercase O(cid:4) lowercase O(cid:4) and zero(cid:4) or

to a full connection between S(cid:18) and C(cid:15)(cid:2) C(cid:15) is labeled

lowercase l(cid:4) digit (cid:6)(cid:4) square brackets(cid:4) and uppercase I(cid:4) will

as a convolutional layer(cid:4) instead of a fully(cid:3)connected layer(cid:4)

have similar output codes(cid:2) This is particularly useful if the

because if LeNet(cid:3)(cid:15) input were made bigger with everything

system is combined with a linguistic post(cid:3)processor that

else kept constant(cid:4) the feature map dimension would be

can correct such confusions(cid:2) Because the codes for confus(cid:3)

larger than (cid:6)x(cid:6)(cid:2) This process of dynamically increasing the

able classes are similar(cid:4) the output of the corresponding

size of a convolutional network is described in the section

RBFs for an ambiguous character will be similar(cid:4) and the

Section VII(cid:2) Layer C(cid:15) has (cid:18)(cid:25)(cid:4)(cid:6)(cid:14)(cid:21) trainable connections(cid:2)

post(cid:3)processor will be able to pick the appropriate interpre(cid:3)

Layer F(cid:19)(cid:4) contains (cid:25)(cid:18) units (cid:7)the reason for this number

tation(cid:2) Figure (cid:17) gives the output codes for the full ASCII

comes from the design of the output layer(cid:4) explained be(cid:3)

set(cid:2)

low(cid:8) and is fully connected to C(cid:15)(cid:2) It has (cid:6)(cid:21)(cid:4)(cid:6)(cid:19)(cid:18) trainable

parameters(cid:2)

than the more common (cid:12)(cid:6) of N(cid:13) code (cid:7)also called place

Another reason for using such distributed codes(cid:4) rather

As in classical neural networks(cid:4) units in layers up to F(cid:19)

code(cid:4) or grand(cid:3)mother cell code(cid:8) for the outputs is that

compute a dot product between their input vector and their

non distributed codes tend to behave badly when the num(cid:3)

weight vector(cid:4) to which a bias is added(cid:2) This weighted sum(cid:4)

ber of classes is larger than a few dozens(cid:2) The reason is

denoted a

for unit i(cid:4) is then passed through a sigmoid

i

that output units in a non(cid:3)distributed code must be o(cid:9)

squashing function to produce the state of unit i(cid:4) denoted

most of the time(cid:2) This is quite di(cid:23)cult to achieve with

by x

(cid:24)

i

sigmoid units(cid:2) Yet another reason is that the classi(cid:5)ers are

x

(cid:16) f (cid:7)a

(cid:8)

(cid:7)(cid:15)(cid:8)

i

i

often used to not only recognize characters(cid:4) but also to re(cid:3)

The squashing function is a scaled hyperbolic tangent(cid:24)

ject non(cid:3)characters(cid:2) RBFs with distributed codes are more

f (cid:7)a(cid:8) (cid:16) A tanh(cid:7)S a(cid:8)

(cid:7)(cid:19)(cid:8)

are activated within a well circumscribed region of their in(cid:3)

appropriate for that purpose because unlike sigmoids(cid:4) they

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:5)

!

1

A

Q

a

q

"

2

B

R

b

r

#

3

C

S

c

s

$

4

D

T

d

t

%

5

E

U

e

u

&

6

F

V

f

v

’

7

G

W

g

w

(

8

H

X

h

x

)

9

I

Y

i

y

*

:

J

Z

j

z

+

;

K

[

k

{

,

<

L

\

l

|

−

=

M

]

m

}

.

>

N

^

n

~

0

@

P

‘

p

/

?

O

_

o

penalties(cid:4) it means that in addition to pushing down the

penalty of the correct class like the MSE criterion(cid:4) this

criterion also pulls up the penalties of the incorrect classes(cid:24)

P

E (cid:7)W (cid:8) (cid:16)

(cid:7)y

(cid:7)Z

(cid:2) W (cid:8) (cid:22) log(cid:7)e

(cid:22)

e

(cid:8)(cid:8)

D

X

X

p

(cid:6)

(cid:2)

(cid:2)

p

j

y

Z

(cid:4)W

i

(cid:15)

(cid:16)

p

P

p

(cid:14)(cid:4)

i

(cid:7)(cid:27)(cid:8)

The negative of the second term plays a (cid:12)competitive(cid:13) role(cid:2)

It is necessarily smaller than (cid:7)or equal to(cid:8) the (cid:5)rst term(cid:4)

therefore this loss function is positive(cid:2) The constant j is

Fig(cid:8) (cid:18)(cid:8)

Initial parameters of the output RBFs for recognizing the

positive(cid:4) and prevents the penalties of classes that are al(cid:3)

full ASCII set(cid:8)

ready very large from being pushed further up(cid:2) The pos(cid:3)

put space that non(cid:3)typical patterns are more likely to fall

terior probability of this rubbish class label would be the

(cid:2)

(cid:2)

(cid:2)

j

j

y

Z

(cid:4)W

i

(cid:15)

(cid:16)

p

ratio of e

and e

(cid:22)

e

(cid:2) This discrimina(cid:3)

i

P

tive criterion prevents the previously mentioned (cid:12)collaps(cid:3)

outside of(cid:2)

ing e(cid:9)ect(cid:13) when the RBF parameters are learned because

The parameter vectors of the RBFs play the role of target

it keeps the RBF centers apart from each other(cid:2) In Sec(cid:3)

vectors for layer F(cid:19)(cid:2) It is worth pointing out that the com(cid:3)

tion VI(cid:4) we present a generalization of this criterion for

ponents of those vectors are (cid:22)(cid:6) or (cid:3)(cid:6)(cid:4) which is well within

systems that learn to classify multiple ob jects in the input

the range of the sigmoid of F(cid:19)(cid:4) and therefore prevents those

(cid:7)e(cid:2)g(cid:2)(cid:4) characters in words or in documents(cid:8)(cid:2)

sigmoids from getting saturated(cid:2) In fact(cid:4) (cid:22)(cid:6) and (cid:3)(cid:6) are the

Computing the gradient of the loss function with respect

points of maximum curvature of the sigmoids(cid:2) This forces

to all the weights in all the layers of the convolutional

the F(cid:19) units to operate in their maximally non(cid:3)linear range(cid:2)

network is done with back(cid:3)propagation(cid:2) The standard al(cid:3)

Saturation of the sigmoids must be avoided because it is

gorithm must be slightly modi(cid:5)ed to take account of the

known to lead to slow convergence and ill(cid:3)conditioning of

weight sharing(cid:2) An easy way to implement it is to (cid:5)rst com(cid:3)

the loss function(cid:2)

C(cid:2) Loss Function

pute the partial derivatives of the loss function with respect

to each connection(cid:4) as if the network were a conventional

multi(cid:3)layer network without weight sharing(cid:2) Then the par(cid:3)

The simplest output loss function that can be used with

tial derivatives of all the connections that share a same

the above network is the Maximum Likelihood Estimation

parameter are added to form the derivative with respect to

criterion (cid:7)MLE(cid:8)(cid:4) which in our case is equivalent to the Min(cid:3)

that parameter(cid:2)

imum Mean Squared Error (cid:7)MSE(cid:8)(cid:2) The criterion for a set

Such a large architecture can be trained very e(cid:23)ciently(cid:4)

of training samples is simply(cid:24)

but doing so requires the use of a few techniques that are

P

(cid:6)

p

p

E (cid:7)W (cid:8) (cid:16)

y

(cid:7)Z

(cid:2) W (cid:8)

(cid:7)(cid:25)(cid:8)

X

D

P

p

(cid:14)(cid:4)

described in the appendix(cid:2) Section A of the appendix

describes details such as the particular sigmoid used(cid:4) and

the weight initialization(cid:2) Section B and C describe the

minimization procedure used(cid:4) which is a stochastic version

where y

is the output of the D

(cid:3)th RBF unit(cid:4) i(cid:2)e(cid:2) the

D

p

p

of a diagonal approximation to the Levenberg(cid:3)Marquardt

one that corresponds to the correct class of input pattern

procedure(cid:2)

p

Z

(cid:2) While this cost function is appropriate for most cases(cid:4)

it lacks three important properties(cid:2) First(cid:4) if we allow the

Methods

parameters of the RBF to adapt(cid:4) E (cid:7)W (cid:8) has a trivial(cid:4) but

III(cid:2) Results and Comparison with Other

totally unacceptable(cid:4) solution(cid:2) In this solution(cid:4) all the RBF

While recognizing individual digits is only one of many

parameter vectors are equal(cid:4) and the state of F(cid:19) is constant

problems involved in designing a practical recognition sys(cid:3)

and equal to that parameter vector(cid:2) In this case the net(cid:3)

tem(cid:4)

it is an excellent benchmark for comparing shape

work happily ignores the input(cid:4) and all the RBF outputs

recognition methods(cid:2) Though many existing method com(cid:3)

are equal to zero(cid:2) This collapsing phenomenon does not

bine a hand(cid:3)crafted feature extractor and a trainable clas(cid:3)

occur if the RBF weights are not allowed to adapt(cid:2) The

si(cid:5)er(cid:4) this study concentrates on adaptive methods that

second problem is that there is no competition between

operate directly on size(cid:3)normalized images(cid:2)

the classes(cid:2) Such a competition can be obtained by us(cid:3)

ing a more discriminative training criterion(cid:4) dubbed the

A(cid:2) Database(cid:6) the Modi(cid:4)ed NIST set

MAP (cid:7)maximum a posteriori(cid:8) criterion(cid:4) similar to Maxi(cid:3)

The database used to train and test the systems de(cid:3)

mum Mutual Information criterion sometimes used to train

scribed in this paper was constructed from the NIST(cid:28)s Spe(cid:3)

HMMs (cid:10)(cid:18)(cid:25)(cid:11)(cid:4) (cid:10)(cid:18)(cid:27)(cid:11)(cid:4) (cid:10)(cid:15)(cid:21)(cid:11)(cid:2)

It corresponds to maximizing the

cial Database (cid:17) and Special Database (cid:6) containing binary

posterior probability of the correct class D

(cid:7)or minimiz(cid:3)

images of handwritten digits(cid:2) NIST originally designated

p

ing the logarithm of the probability of the correct class(cid:8)(cid:4)

SD(cid:3)(cid:17) as their training set and SD(cid:3)(cid:6) as their test set(cid:2) How(cid:3)

given that the input image can come from one of the classes

ever(cid:4) SD(cid:3)(cid:17) is much cleaner and easier to recognize than SD(cid:3)

or from a background (cid:12)rubbish(cid:13) class label(cid:2)

In terms of

(cid:6)(cid:2) The reason for this can be found on the fact that SD(cid:3)(cid:17)

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:4)(cid:11)

was collected among Census Bureau employees(cid:4) while SD(cid:3)(cid:6)

was collected among high(cid:3)school students(cid:2) Drawing sensi(cid:3)

ble conclusions from learning experiments requires that the

result be independent of the choice of training set and test

among the complete set of samples(cid:2) Therefore it was nec(cid:3)

essary to build a new database by mixing NIST(cid:28)s datasets(cid:2)

SD(cid:3)(cid:6) contains (cid:15)(cid:25)(cid:4)(cid:15)(cid:14)(cid:20) digit images written by (cid:15)(cid:21)(cid:21) dif(cid:3)

ferent writers(cid:2) In contrast to SD(cid:3)(cid:17)(cid:4) where blocks of data

from each writer appeared in sequence(cid:4) the data in SD(cid:3)(cid:6) is

scrambled(cid:2) Writer identities for SD(cid:3)(cid:6) are available and we

used this information to unscramble the writers(cid:2) We then

split SD(cid:3)(cid:6) in two(cid:24) characters written by the (cid:5)rst (cid:14)(cid:15)(cid:21) writers

went into our new training set(cid:2) The remaining (cid:14)(cid:15)(cid:21) writers

were placed in our test set(cid:2) Thus we had two sets with

nearly (cid:17)(cid:21)(cid:4)(cid:21)(cid:21)(cid:21) examples each(cid:2) The new training set was

completed with enough examples from SD(cid:3)(cid:17)(cid:4) starting at

pattern (cid:31) (cid:21)(cid:4) to make a full set of (cid:19)(cid:21)(cid:4)(cid:21)(cid:21)(cid:21) training patterns(cid:2)

Similarly(cid:4) the new test set was completed with SD(cid:3)(cid:17) exam(cid:3)

ples starting at pattern (cid:31) (cid:17)(cid:15)(cid:4)(cid:21)(cid:21)(cid:21) to make a full set with

Fig(cid:8) (cid:21)(cid:8) Size(cid:2)normalized examples from the MNIST database(cid:8)

(cid:19)(cid:21)(cid:4)(cid:21)(cid:21)(cid:21) test patterns(cid:2) In the experiments described here(cid:4) we

only used a subset of (cid:6)(cid:21)(cid:4)(cid:21)(cid:21)(cid:21) test images (cid:7)(cid:15)(cid:4)(cid:21)(cid:21)(cid:21) from SD(cid:3)(cid:6)

and (cid:15)(cid:4)(cid:21)(cid:21)(cid:21) from SD(cid:3)(cid:17)(cid:8)(cid:4) but we used the full (cid:19)(cid:21)(cid:4)(cid:21)(cid:21)(cid:21) training

three(cid:4) (cid:21)(cid:2)(cid:21)(cid:21)(cid:21)(cid:6) for the next three(cid:4) (cid:21)(cid:2)(cid:21)(cid:21)(cid:21)(cid:21)(cid:15) for the next (cid:18)(cid:4)

samples(cid:2) The resulting database was called the Modi(cid:5)ed

and (cid:21)(cid:2)(cid:21)(cid:21)(cid:21)(cid:21)(cid:6) thereafter(cid:2) Before each iteration(cid:4) the diagonal

NIST(cid:4) or MNIST(cid:4) dataset(cid:2)

Hessian approximation was reevaluated on (cid:15)(cid:21)(cid:21) samples(cid:4) as

The original black and white (cid:7)bilevel(cid:8) images were size

described in Appendix C and kept (cid:5)xed during the entire

normalized to (cid:5)t in a (cid:14)(cid:21)x(cid:14)(cid:21) pixel box while preserving

iteration(cid:2) The parameter (cid:10) was set to (cid:21)(cid:2)(cid:21)(cid:14)(cid:2) The resulting

their aspect ratio(cid:2) The resulting images contain grey lev(cid:3)

e(cid:9)ective learning rates during the (cid:5)rst pass varied between

(cid:2)

(cid:10)

els as result of the anti(cid:3)aliasing (cid:7)image interpolation(cid:8) tech(cid:3)

approximately (cid:20) (cid:3) (cid:6)(cid:21)

and (cid:21)(cid:3)(cid:21)(cid:6)(cid:19) over the set of parame(cid:3)

nique used by the normalization algorithm(cid:2) Three ver(cid:3)

ters(cid:2) The test error rate stabilizes after around (cid:6)(cid:21) passes

sions of the database were used(cid:2)

In the (cid:5)rst version(cid:4)

through the training set at (cid:21)(cid:2)(cid:27)(cid:15)!(cid:2) The error rate on the

the images were centered in a (cid:14)(cid:25)x(cid:14)(cid:25) image by comput(cid:3)

training set reaches (cid:21)(cid:2)(cid:17)(cid:15)! after (cid:6)(cid:27) passes(cid:2) Many authors

ing the center of mass of the pixels(cid:4) and translating the

have reported observing the common phenomenon of over(cid:3)

image so as to position this point at the center of the

training when training neural networks or other adaptive

(cid:14)(cid:25)x(cid:14)(cid:25) (cid:5)eld(cid:2)

In some instances(cid:4) this (cid:14)(cid:25)x(cid:14)(cid:25) (cid:5)eld was ex(cid:3)

algorithms on various tasks(cid:2) When over(cid:3)training occurs(cid:4)

tended to (cid:17)(cid:14)x(cid:17)(cid:14) with background pixels(cid:2) This version of

the training error keeps decreasing over time(cid:4) but the test

the database will be referred to as the regular database(cid:2)

error goes through a minimum and starts increasing after

In the second version of the database(cid:4) the character im(cid:3)

a certain number of iterations(cid:2) While this phenomenon is

ages were deslanted and cropped down to (cid:14)(cid:21)x(cid:14)(cid:21) pixels im(cid:3)

very common(cid:4) it was not observed in our case as the learn(cid:3)

ages(cid:2) The deslanting computes the second moments of in(cid:3)

ing curves in (cid:5)gure (cid:15) show(cid:2) A possible reason is that the

ertia of the pixels (cid:7)counting a foreground pixel as (cid:6) and a

learning rate was kept relatively large(cid:2) The e(cid:9)ect of this is

background pixel as (cid:21)(cid:8)(cid:4) and shears the image by horizon(cid:3)

that the weights never settle down in the local minimum

tally shifting the lines so that the principal axis is verti(cid:3)

but keep oscillating randomly(cid:2) Because of those (cid:26)uctua(cid:3)

cal(cid:2) This version of the database will be referred to as the

tions(cid:4) the average cost will be lower in a broader minimum(cid:2)

deslanted database(cid:2) In the third version of the database(cid:4)

Therefore(cid:4) stochastic gradient will have a similar e(cid:9)ect as

used in some early experiments(cid:4) the images were reduced

a regularization term that favors broader minima(cid:2) Broader

to (cid:6)(cid:19)x(cid:6)(cid:19) pixels(cid:2) The regular database (cid:7)(cid:19)(cid:21)(cid:4)(cid:21)(cid:21)(cid:21) training

minima correspond to solutions with large entropy of the

examples(cid:4) (cid:6)(cid:21)(cid:4)(cid:21)(cid:21)(cid:21) test examples size(cid:3)normalized to (cid:14)(cid:21)x(cid:14)(cid:21)(cid:4)

parameter distribution(cid:4) which is bene(cid:5)cial to the general(cid:3)

and centered by center of mass in (cid:14)(cid:25)x(cid:14)(cid:25) (cid:5)elds(cid:8) is avail(cid:3)

ization error(cid:2)

able at

(cid:2)

http(cid:2)(cid:3)(cid:3)www(cid:4)research(cid:4)att(cid:4)com(cid:3)

yann(cid:3)ocr(cid:3)mnist

The in(cid:26)uence of the training set size was measured by

Figure (cid:18) shows examples randomly picked from the test set(cid:2)

training the network with (cid:6)(cid:15)(cid:4)(cid:21)(cid:21)(cid:21)(cid:4) (cid:17)(cid:21)(cid:4)(cid:21)(cid:21)(cid:21)(cid:4) and (cid:19)(cid:21)(cid:4)(cid:21)(cid:21)(cid:21) exam(cid:3)

B(cid:2) Results

in (cid:5)gure (cid:19)(cid:2) It is clear that(cid:4) even with specialized architec(cid:3)

ples(cid:2) The resulting training error and test error are shown

Several versions of LeNet(cid:3)(cid:15) were trained on the regular

tures such as LeNet(cid:3)(cid:15)(cid:4) more training data would improve

MNIST database(cid:2) (cid:14)(cid:21) iterations through the entire train(cid:3)

the accuracy(cid:2)

ing data were performed for each session(cid:2) The values of

To verify this hypothesis(cid:4) we arti(cid:5)cially generated more

the global learning rate (cid:9) (cid:7)see Equation (cid:14)(cid:6) in Appendix C

training examples by randomly distorting the original

for a de(cid:5)nition(cid:8) was decreased using the following sched(cid:3)

training images(cid:2) The increased training set was composed

ule(cid:24) (cid:21)(cid:2)(cid:21)(cid:21)(cid:21)(cid:15) for the (cid:5)rst two passes(cid:4) (cid:21)(cid:2)(cid:21)(cid:21)(cid:21)(cid:14) for the next

of the (cid:19)(cid:21)(cid:4)(cid:21)(cid:21)(cid:21) original patterns plus (cid:15)(cid:18)(cid:21)(cid:4)(cid:21)(cid:21)(cid:21) instances of

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:4)(cid:4)

Error Rate (%)

5%

4%

3%

2%

1%

0%

0

4

8

12

16

20
Training set Iterations

Test
Training

Fig(cid:8) (cid:20)(cid:8) Training and test error of LeNet(cid:2)(cid:20) as a function of the num(cid:2)

ber of passes through the (cid:14)(cid:6)(cid:3)(cid:6)(cid:6)(cid:6) pattern training set (cid:22)without

distortions(cid:23)(cid:8) The average training error is measured on(cid:2)the(cid:2)(cid:24)y as

training proceeds(cid:8) This explains why the training error appears

to be larger than the test error(cid:8) Convergence is attained after (cid:5)(cid:6)

to (cid:5)(cid:15) passes through the training set(cid:8)

Error Rate (%)

Fig(cid:8) (cid:7)(cid:8) Examples of distortions of ten training patterns(cid:8)

1.8

1.6

1.4

1.2

1

0.8

0.6

0.4

0.2

Test error (no distortions)

9−>4 8−>0 7−>8 5−>3 8−>7 0−>6 3−>7 2−>7 8−>3 9−>4

4−>6 3−>5 8−>2 2−>1 5−>3 4−>8 2−>8 3−>5 6−>5 7−>3

8−>2 5−>3 4−>8 3−>9 6−>0 9−>8 4−>9 6−>1 9−>4 9−>1

9−>4 2−>0 6−>1 3−>5 3−>2 9−>5 6−>0 6−>0 6−>0 6−>8

Test error
(with distortions)

4−>6 7−>3 9−>4 4−>6 2−>7 9−>7 4−>3 9−>4 9−>4 9−>4

8−>7 4−>2 8−>4 3−>5 8−>4 6−>5 8−>5 3−>8 3−>8 9−>8

Training error (no distortions)

1−>5 9−>8 6−>3 0−>2 6−>5 9−>5 0−>7 1−>6 4−>9 2−>1

0

0

10

20

30

40

50

60

70

80

90

100

2−>8 8−>5 4−>9 7−>2 7−>2 6−>5 9−>7 6−>1 5−>6 5−>0

Training Set Size (x1000)

4−>9 2−>8

Fig(cid:8) (cid:14)(cid:8) Training and test errors of LeNet(cid:2)(cid:20) achieved using training

sets of various sizes(cid:8) This graph suggests that a larger training

Fig(cid:8) (cid:16)(cid:8) The (cid:16)(cid:15) test patterns misclassi(cid:19)ed by LeNet(cid:2)(cid:20)(cid:8) Below each

set could improve the performance of LeNet(cid:2)(cid:20)(cid:8) The hollow square

image is displayed the correct answers (cid:22)left(cid:23) and the network an(cid:2)

show the test error when more training patterns are arti(cid:19)cially

swer (cid:22)right(cid:23)(cid:8) These errors are mostly caused either by genuinely

generated using random distortions(cid:8) The test patterns are not

ambiguous patterns(cid:3) or by digits written in a style that are under(cid:2)

distorted(cid:8)

represented in the training set(cid:8)

distorted patterns with randomly picked distortion param(cid:3)

perfectly identi(cid:5)able by humans(cid:4) although they are writ(cid:3)

eters(cid:2) The distortions were combinations of the follow(cid:3)

ten in an under(cid:3)represented style(cid:2) This shows that further

ing planar a(cid:23)ne transformations(cid:24) horizontal and verti(cid:3)

improvements are to be expected with more training data(cid:2)

cal translations(cid:4) scaling(cid:4) squeezing (cid:7)simultaneous horizon(cid:3)

C(cid:2) Comparison with Other Classi(cid:4)ers

tal compression and vertical elongation(cid:4) or the reverse(cid:8)(cid:4)

and horizontal shearing(cid:2) Figure (cid:20) shows examples of dis(cid:3)

For the sake of comparison(cid:4) a variety of other trainable

torted patterns used for training(cid:2) When distorted data was

classi(cid:5)ers was trained and tested on the same database(cid:2) An

used for training(cid:4) the test error rate dropped to (cid:21)(cid:2)(cid:25)! (cid:7)from

early subset of these results was presented in (cid:10)(cid:15)(cid:6)(cid:11)(cid:2) The error

(cid:21)(cid:2)(cid:27)(cid:15)! without deformation(cid:8)(cid:2) The same training parame(cid:3)

rates on the test set for the various methods are shown in

ters were used as without deformations(cid:2) The total length of

(cid:5)gure (cid:27)(cid:2)

the training session was left unchanged (cid:7)(cid:14)(cid:21) passes of (cid:19)(cid:21)(cid:4)(cid:21)(cid:21)(cid:21)

patterns each(cid:8)(cid:2) It is interesting to note that the network

C(cid:2)(cid:6) Linear Classi(cid:5)er(cid:4) and Pairwise Linear Classi(cid:5)er

e(cid:9)ectively sees each individual sample only twice over the

Possibly the simplest classi(cid:5)er that one might consider is

course of these (cid:14)(cid:21) passes(cid:2)

a linear classi(cid:5)er(cid:2) Each input pixel value contributes to a

Figure (cid:25) shows all (cid:25)(cid:14) misclassi(cid:5)ed test examples(cid:2) some

weighted sum for each output unit(cid:2) The output unit with

of those examples are genuinely ambiguous(cid:4) but several are

the highest sum (cid:7)including the contribution of a bias con(cid:3)

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:4)(cid:7)

Linear

 −−−− 12.0 −−−−>

[deslant] Linear

 −−−− 8.4 −−−−>

Pairwise

 −−−− 7.6 −−−−>

K−NN Euclidean

[deslant] K−NN Euclidean

40 PCA + quadratic

1000 RBF + linear

[16x16] Tangent Distance

SVM poly 4

RS−SVM poly 5

[dist] V−SVM poly 9

28x28−300−10

[dist] 28x28−300−10

[deslant] 20x20−300−10

28x28−1000−10

[dist] 28x28−1000−10

28x28−300−100−10

[dist] 28x28−300−100−10

28x28−500−150−10

[dist] 28x28−500−150−10

[16x16] LeNet−1

LeNet−4

LeNet−4 / Local

LeNet−4 / K−NN

LeNet−5

[dist] LeNet−5

[dist] Boosted LeNet−4

1.1

1.1

1

0.8

1.1

1.1

1.1

0.95

0.8

0.7

2.4

3.3

3.6

5

3.6

3.8

4.7

4.5

1.6

1.7

3.05

2.95

2.5

2.45

0

0.5

1

1.5

2

2.5

3

3.5

4

4.5

5

Fig(cid:8) (cid:17)(cid:8) Error rate on the test set (cid:22)(cid:25)(cid:23) for various classi(cid:19)cation methods(cid:8)

(cid:26)deslant(cid:27) indicates that the classi(cid:19)er was trained and tested on

the deslanted version of the database(cid:8) (cid:26)dist(cid:27) indicates that the training set was augmented with arti(cid:19)cially distorted examples(cid:8) (cid:26)(cid:5)(cid:14)x(cid:5)(cid:14)(cid:27)

indicates that the system used the (cid:5)(cid:14)x(cid:5)(cid:14) pixel images(cid:8) The uncertainty in the quoted error rates is about (cid:6)(cid:8)(cid:5)(cid:25)(cid:8)

stant(cid:8) indicates the class of the input character(cid:2) On the

However(cid:4) the memory requirement and recognition time are

regular data(cid:4) the error rate is (cid:6)(cid:14)!(cid:2) The network has (cid:20)(cid:25)(cid:15)(cid:21)

large(cid:24) the complete (cid:19)(cid:21)(cid:4)(cid:21)(cid:21)(cid:21) twenty by twenty pixel training

free parameters(cid:2) On the deslanted images(cid:4) the test error

images (cid:7)about (cid:14)(cid:18) Megabytes at one byte per pixel(cid:8) must be

rate is (cid:25)(cid:2)(cid:18)! The network has (cid:18)(cid:21)(cid:6)(cid:21) free parameters(cid:2) The

available at run time(cid:2) Much more compact representations

de(cid:5)ciencies of the linear classi(cid:5)er are well documented (cid:10)(cid:6)(cid:11)

could be devised with modest increase in error rate(cid:2) On the

and it is included here simply to form a basis of comparison

regular test set the error rate was (cid:15)(cid:2)(cid:21)!(cid:2) On the deslanted

for more sophisticated classi(cid:5)ers(cid:2) Various combinations of

data(cid:4) the error rate was (cid:14)(cid:2)(cid:18)!(cid:4) with k (cid:16) (cid:17)(cid:2) Naturally(cid:4) a

sigmoid units(cid:4) linear units(cid:4) gradient descent learning(cid:4) and

realistic Euclidean distance nearest(cid:3)neighbor system would

learning by directly solving linear systems gave similar re(cid:3)

operate on feature vectors rather than directly on the pix(cid:3)

sults(cid:2)

els(cid:4) but since all of the other systems presented in this

A simple improvement of the basic linear classi(cid:5)er was

study operate directly on the pixels(cid:4) this result is useful for

tested (cid:10)(cid:15)(cid:14)(cid:11)(cid:2) The idea is to train each unit of a single(cid:3)layer

a baseline comparison(cid:2)

network to separate each class from each other class(cid:2) In our

case this layer comprises (cid:18)(cid:15) units labeled (cid:21)(cid:30)(cid:6)(cid:4) (cid:21)(cid:30)(cid:14)(cid:4)(cid:2)(cid:2)(cid:2)(cid:21)(cid:30)(cid:27)(cid:4)

C(cid:2)(cid:17) Principal Component Analysis (cid:7)PCA(cid:8) and Polynomial

(cid:6)(cid:30)(cid:14)(cid:2)(cid:2)(cid:2)(cid:2)(cid:25)(cid:30)(cid:27)(cid:2) Unit i(cid:4)j is trained to produce (cid:22)(cid:6) on patterns

Classi(cid:5)er

of class i(cid:4) (cid:3)(cid:6) on patterns of class j (cid:4) and is not trained on

Following (cid:10)(cid:15)(cid:17)(cid:11)(cid:4)

(cid:10)(cid:15)(cid:18)(cid:11)(cid:4) a preprocessing stage was con(cid:3)

other patterns(cid:2) The (cid:5)nal score for class i is the sum of

structed which computes the pro jection of the input pat(cid:3)

the outputs all the units labeled i(cid:4)x minus the sum of the

tern on the (cid:18)(cid:21) principal components of the set of training

output of all the units labeled y(cid:4)i(cid:4) for all x and y (cid:2) The

vectors(cid:2) To compute the principal components(cid:4) the mean of

error rate on the regular test set was (cid:20)(cid:2)(cid:19)!(cid:2)

each input component was (cid:5)rst computed and subtracted

C(cid:2)(cid:14) Baseline Nearest Neighbor Classi(cid:5)er

sulting vectors was then computed and diagonalized using

from the training vectors(cid:2) The covariance matrix of the re(cid:3)

Another simple classi(cid:5)er is a K(cid:3)nearest neighbor classi(cid:3)

Singular Value Decomposition(cid:2) The (cid:18)(cid:21)(cid:3)dimensional feature

(cid:5)er with a Euclidean distance measure between input im(cid:3)

vector was used as the input of a second degree polynomial

ages(cid:2) This classi(cid:5)er has the advantage that no training

classi(cid:5)er(cid:2) This classi(cid:5)er can be seen as a linear classi(cid:5)er

time(cid:4) and no brain on the part of the designer(cid:4) are required(cid:2)

with (cid:25)(cid:14)(cid:6) inputs(cid:4) preceded by a module that computes all

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:4)(cid:8)

products of pairs of input variables(cid:2) The error on the reg(cid:3)

only marginally improved error rates(cid:24) (cid:14)(cid:2)(cid:27)(cid:15)!(cid:2) Training

ular test set was (cid:17)(cid:2)(cid:17)!(cid:2)

with distorted patterns improved the performance some(cid:3)

C(cid:2)(cid:18) Radial Basis Function Network

(cid:14)(cid:2)(cid:18)(cid:15)! for the (cid:14)(cid:25)x(cid:14)(cid:25)(cid:3)(cid:6)(cid:21)(cid:21)(cid:21)(cid:3)(cid:6)(cid:15)(cid:21)(cid:3)(cid:6)(cid:21) network(cid:2)

what(cid:24) (cid:14)(cid:2)(cid:15)(cid:21)! error for the (cid:14)(cid:25)x(cid:14)(cid:25)(cid:3)(cid:17)(cid:21)(cid:21)(cid:3)(cid:6)(cid:21)(cid:21)(cid:3)(cid:6)(cid:21) network(cid:4) and

Following (cid:10)(cid:15)(cid:15)(cid:11)(cid:4) an RBF network was constructed(cid:2) The

(cid:5)rst layer was composed of (cid:6)(cid:4)(cid:21)(cid:21)(cid:21) Gaussian RBF units with

C(cid:2)(cid:20) A Small Convolutional Network(cid:24) LeNet(cid:3)(cid:6)

(cid:14)(cid:25)x(cid:14)(cid:25) inputs(cid:4) and the second layer was a simple (cid:6)(cid:21)(cid:21)(cid:21) inputs

Convolutional Networks are an attempt to solve the

(cid:30) (cid:6)(cid:21) outputs linear classi(cid:5)er(cid:2) The RBF units were divided

dilemma between small networks

that cannot

learn

into (cid:6)(cid:21) groups of (cid:6)(cid:21)(cid:21)(cid:2) Each group of units was trained

the training set(cid:4) and large networks that seem over(cid:3)

on all the training examples of one of the (cid:6)(cid:21) classes using

parameterized(cid:2) LeNet(cid:3)(cid:6) was an early embodiment of the

the adaptive K(cid:3)means algorithm(cid:2) The second layer weights

Convolutional Network architecture which is included here

were computed using a regularized pseudo(cid:3)inverse method(cid:2)

for comparison purposes(cid:2) The images were down(cid:3)sampled

The error rate on the regular test set was (cid:17)(cid:2)(cid:19)!

to (cid:6)(cid:19)x(cid:6)(cid:19) pixels and centered in the (cid:14)(cid:25)x(cid:14)(cid:25) input layer(cid:2) Al(cid:3)

C(cid:2)(cid:15) One(cid:3)Hidden Layer Fully Connected Multilayer Neural

evaluate LeNet(cid:3)(cid:6)(cid:4) its convolutional nature keeps the num(cid:3)

though about (cid:6)(cid:21)(cid:21)(cid:4)(cid:21)(cid:21)(cid:21) multiply(cid:30)add steps are required to

Network

ber of free parameters to only about (cid:14)(cid:19)(cid:21)(cid:21)(cid:2) The LeNet(cid:3)

Another classi(cid:5)er that we tested was a fully connected

(cid:6) architecture was developed using our own version of

multi(cid:3)layer neural network with two layers of weights (cid:7)one

the USPS (cid:7)US Postal Service zip codes(cid:8) database and its

hidden layer(cid:8) trained with the version of back(cid:3)propagation

size was tuned to match the available data (cid:10)(cid:17)(cid:15)(cid:11)(cid:2) LeNet(cid:3)(cid:6)

described in Appendix C(cid:2) Error on the regular test set was

achieved (cid:6)(cid:2)(cid:20)! test error(cid:2) The fact that a network with such

(cid:18)(cid:2)(cid:20)! for a network with (cid:17)(cid:21)(cid:21) hidden units(cid:4) and (cid:18)(cid:2)(cid:15)! for a

a small number of parameters can attain such a good error

network with (cid:6)(cid:21)(cid:21)(cid:21) hidden units(cid:2) Using arti(cid:5)cial distortions

rate is an indication that the architecture is appropriate

to generate more training data brought only marginal im(cid:3)

for the task(cid:2)

provement(cid:24) (cid:17)(cid:2)(cid:19)! for (cid:17)(cid:21)(cid:21) hidden units(cid:4) and (cid:17)(cid:2)(cid:25)! for (cid:6)(cid:21)(cid:21)(cid:21)

hidden units(cid:2) When deslanted images were used(cid:4) the test

C(cid:2)(cid:25) LeNet(cid:3)(cid:18)

error jumped down to (cid:6)(cid:2)(cid:19)! for a network with (cid:17)(cid:21)(cid:21) hidden

Experiments with LeNet(cid:3)(cid:6) made it clear that a larger

units(cid:2)

convolutional network was needed to make optimal use of

It remains somewhat of a mystery that networks with

the large size of the training set(cid:2) LeNet(cid:3)(cid:18) and later LeNet(cid:3)

such a large number of free parameters manage to achieve

(cid:15) were designed to address this problem(cid:2) LeNet(cid:3)(cid:18) is very

reasonably low testing errors(cid:2) We conjecture that the dy(cid:3)

similar to LeNet(cid:3)(cid:15)(cid:4) except for the details of the architec(cid:3)

namics of gradient descent learning in multilayer nets has

ture(cid:2)

It contains (cid:18) (cid:5)rst(cid:3)level feature maps(cid:4) followed by

a (cid:12)self(cid:3)regularization(cid:13) e(cid:9)ect(cid:2) Because the origin of weight

(cid:25) subsampling maps connected in pairs to each (cid:5)rst(cid:3)layer

space is a saddle point that is attractive in almost every

feature maps(cid:4) then (cid:6)(cid:19) feature maps(cid:4) followed by (cid:6)(cid:19) sub(cid:3)

direction(cid:4) the weights invariably shrink during the (cid:5)rst

sampling map(cid:4) followed by a fully connected layer with

few epochs (cid:7)recent theoretical analysis seem to con(cid:5)rm

(cid:6)(cid:14)(cid:21) units(cid:4) followed by the output layer (cid:7)(cid:6)(cid:21) units(cid:8)(cid:2) LeNet(cid:3)(cid:18)

this (cid:10)(cid:15)(cid:19)(cid:11)(cid:8)(cid:2) Small weights cause the sigmoids to operate

contains about (cid:14)(cid:19)(cid:21)(cid:4)(cid:21)(cid:21)(cid:21) connections and has about (cid:6)(cid:20)(cid:4)(cid:21)(cid:21)(cid:21)

in the quasi(cid:3)linear region(cid:4) making the network essentially

free parameters(cid:2) Test error was (cid:6)(cid:2)(cid:6)!(cid:2)

In a series of ex(cid:3)

equivalent to a low(cid:3)capacity(cid:4) single(cid:3)layer network(cid:2) As the

periments(cid:4) we replaced the last layer of LeNet(cid:3)(cid:18) with a

learning proceeds(cid:4) the weights grow(cid:4) which progressively

Euclidean Nearest Neighbor classi(cid:5)er(cid:4) and with the (cid:12)local

increases the e(cid:9)ective capacity of the network(cid:2) This seems

learning(cid:13) method of Bottou and Vapnik (cid:10)(cid:15)(cid:25)(cid:11)(cid:4) in which a lo(cid:3)

to be an almost perfect(cid:4) if fortuitous(cid:4) implementation of

cal linear classi(cid:5)er is retrained each time a new test pattern

Vapnik(cid:28)s (cid:12)Structural Risk Minimization(cid:13) principle (cid:10)(cid:19)(cid:11)(cid:2) A

is shown(cid:2) Neither of those methods improved the raw error

better theoretical understanding of these phenomena(cid:4) and

rate(cid:4) although they did improve the rejection performance(cid:2)

more empirical evidence(cid:4) are de(cid:5)nitely needed(cid:2)

C(cid:2)(cid:27) Boosted LeNet(cid:3)(cid:18)

C(cid:2)(cid:19) Two(cid:3)Hidden Layer Fully Connected Multilayer Neural

Following theoretical work by R(cid:2) Schapire (cid:10)(cid:15)(cid:27)(cid:11)(cid:4) Drucker

Network

et al(cid:2) (cid:10)(cid:19)(cid:21)(cid:11) developed the (cid:12)boosting(cid:13) method for combining

To see the e(cid:9)ect of the architecture(cid:4) several two(cid:3)hidden

multiple classi(cid:5)ers(cid:2) Three LeNet(cid:3)(cid:18)s are combined(cid:24) the (cid:5)rst

layer multilayer neural networks were trained(cid:2) Theoreti(cid:3)

one is trained the usual way(cid:2) the second one is trained on

cal results have shown that any function can be approxi(cid:3)

patterns that are (cid:5)ltered by the (cid:5)rst net so that the second

mated by a one(cid:3)hidden layer neural network (cid:10)(cid:15)(cid:20)(cid:11)(cid:2) However(cid:4)

machine sees a mix of patterns(cid:4) (cid:15)(cid:21)! of which the (cid:5)rst net

several authors have observed that two(cid:3)hidden layer archi(cid:3)

got right(cid:4) and (cid:15)(cid:21)! of which it got wrong(cid:2) Finally(cid:4) the

tectures sometimes yield better performance in practical

third net is trained on new patterns on which the (cid:5)rst and

situations(cid:2) This phenomenon was also observed here(cid:2) The

the second nets disagree(cid:2) During testing(cid:4) the outputs of

test error rate of a (cid:14)(cid:25)x(cid:14)(cid:25)(cid:3)(cid:17)(cid:21)(cid:21)(cid:3)(cid:6)(cid:21)(cid:21)(cid:3)(cid:6)(cid:21) network was (cid:17)(cid:2)(cid:21)(cid:15)!(cid:4)

the three nets are simply added(cid:2) Because the error rate of

a much better result than the one(cid:3)hidden layer network(cid:4)

LeNet(cid:3)(cid:18) is very low(cid:4) it was necessary to use the arti(cid:5)cially

obtained using marginally more weights and connections(cid:2)

distorted images (cid:7)as with LeNet(cid:3)(cid:15)(cid:8) in order to get enough

Increasing the network size to (cid:14)(cid:25)x(cid:14)(cid:25)(cid:3)(cid:6)(cid:21)(cid:21)(cid:21)(cid:3)(cid:6)(cid:15)(cid:21)(cid:3)(cid:6)(cid:21) yielded

samples to train the second and third nets(cid:2) The test error

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:4)(cid:9)

rate was (cid:21)(cid:2)(cid:20)!(cid:4) the best of any of our classi(cid:5)ers(cid:2) At (cid:5)rst

glance(cid:4) boosting appears to be three times more expensive

[deslant] K−NN Euclidean

[16x16] Tangent Distance

SVM poly 4

1.9

1.8

8.1

as a single net(cid:2)

In fact(cid:4) when the (cid:5)rst net produces a

[deslant] 20x20−300−10

3.2

high con(cid:5)dence answer(cid:4) the other nets are not called(cid:2) The

average computational cost is about (cid:6)(cid:2)(cid:20)(cid:15) times that of a

single net(cid:2)

[16x16] LeNet−1

LeNet−4

LeNet−4 / Local

LeNet−4 / K−NN

1.4

1.8

1.6

[dist] Boosted LeNet−4

0.5

3.7

C(cid:2)(cid:6)(cid:21) Tangent Distance Classi(cid:5)er (cid:7)TDC(cid:8)

0

1

2

3

4

5

6

7

8

9

The Tangent Distance classi(cid:5)er (cid:7)TDC(cid:8) is a nearest(cid:3)

neighbor method where the distance function is made in(cid:3)

sensitive to small distortions and translations of the input

Fig(cid:8) (cid:5)(cid:6)(cid:8) Rejection Performance(cid:9) percentage of test patterns that

image (cid:10)(cid:19)(cid:6)(cid:11)(cid:2) If we consider an image as a point in a high

must be rejected to achieve (cid:6)(cid:8)(cid:20)(cid:25) error for some of the systems(cid:8)

dimensional pixel space (cid:7)where the dimensionality equals

the number of pixels(cid:8)(cid:4) then an evolving distortion of a char(cid:3)

acter traces out a curve in pixel space(cid:2) Taken together(cid:4)

all these distortions de(cid:5)ne a low(cid:3)dimensional manifold in

pixel space(cid:2) For small distortions(cid:4) in the vicinity of the

original image(cid:4) this manifold can be approximated by a

plane(cid:4) known as the tangent plane(cid:2) An excellent measure

of (cid:13)closeness(cid:13) for character images is the distance between

their tangent planes(cid:4) where the set of distortions used to

generate the planes includes translations(cid:4) scaling(cid:4) skewing(cid:4)

squeezing(cid:4) rotation(cid:4) and line thickness variations(cid:2) A test

error rate of (cid:6)(cid:2)(cid:6)! was achieved using (cid:6)(cid:19)x(cid:6)(cid:19) pixel images(cid:2)

Pre(cid:5)ltering techniques using simple Euclidean distance at

multiple resolutions allowed to reduce the number of nec(cid:3)

essary Tangent Distance calculations(cid:2)

Linear

4

Pairwise

36

[deslant] K−NN Euclidean

−−−− 24,000 −−−−>

40 PCA+quadratic

39

1000 RBF

[16x16] Tangent Distance

−−−− 20,000 −−−−>

SVM poly 4

−−−− 14,000 −−−−>

RS−SVM poly 5

[dist] V−SVM poly 9

−−−− 28,000 −−−−>

[deslant] 20x20−300−10

123

28x28−1000−10

28x28−300−100−10

28x28−500−150−10

[16x16] LeNet−1

100

LeNet−4

LeNet−4 / Local

−−−− 20,000 −−−−>

LeNet−4 / K−NN

−−−− 10,000 −−−−>

LeNet−5

Boosted LeNet−4

267

260

469

401

460

650

794

795

C(cid:2)(cid:6)(cid:6) Support Vector Machine (cid:7)SVM(cid:8)

0

300

600

900

Polynomial classi(cid:5)ers are well(cid:3)studied methods for gen(cid:3)

Fig(cid:8) (cid:5)(cid:5)(cid:8) Number of multiply(cid:2)accumulate operations for the recogni(cid:2)

erating complex decision surfaces(cid:2) Unfortunately(cid:4) they

tion of a single character starting with a size(cid:2)normalized image(cid:8)

are impractical for high(cid:3)dimensional problems(cid:4) because the

number of product terms is prohibitive(cid:2) The Support Vec(cid:3)

tor technique is an extremely economical way of represent(cid:3)

has reached (cid:21)(cid:2)(cid:25)! using a modi(cid:5)ed version of the V(cid:3)SVM(cid:2)

ing complex surfaces in high(cid:3)dimensional spaces(cid:4) including

Unfortunately(cid:4) V(cid:3)SVM is extremely expensive(cid:24) about twice

polynomials and many other types of surfaces (cid:10)(cid:19)(cid:11)(cid:2)

as much as regular SVM(cid:2) To alleviate this problem(cid:4) Burges

A particularly interesting subset of decision surfaces is

has proposed the Reduced Set Support Vector technique

the ones that correspond to hyperplanes that are at a max(cid:3)

(cid:7)RS(cid:3)SVM(cid:8)(cid:4) which attained (cid:6)(cid:2)(cid:6)! on the regular test set (cid:10)(cid:19)(cid:17)(cid:11)(cid:4)

imum distance from the convex hulls of the two classes in

with a computational cost of only (cid:19)(cid:15)(cid:21)(cid:4)(cid:21)(cid:21)(cid:21) multiply(cid:3)adds

the high(cid:3)dimensional space of the product terms(cid:2) Boser(cid:4)

per recognition(cid:4) i(cid:2)e(cid:2) only about (cid:19)(cid:21)! more expensive than

Guyon(cid:4) and Vapnik (cid:10)(cid:19)(cid:14)(cid:11) realized that any polynomial of

LeNet(cid:3)(cid:15)(cid:2)

degree k in this (cid:12)maximum margin(cid:13) set can be computed

by (cid:5)rst computing the dot product of the input image with

D(cid:2) Discussion

a subset of the training samples (cid:7)called the (cid:12)support vec(cid:3)

A summary of the performance of the classi(cid:5)ers is shown

tors(cid:13)(cid:8)(cid:4) elevating the result to the k (cid:3)th power(cid:4) and linearly

in Figures (cid:27) to (cid:6)(cid:14)(cid:2) Figure (cid:27) shows the raw error rate of the

combining the numbers thereby obtained(cid:2) Finding the sup(cid:3)

classi(cid:5)ers on the (cid:6)(cid:21)(cid:4)(cid:21)(cid:21)(cid:21) example test set(cid:2) Boosted LeNet(cid:3)(cid:18)

port vectors and the coe(cid:23)cients amounts to solving a high(cid:3)

performed best(cid:4) achieving a score of (cid:21)(cid:2)(cid:20)!(cid:4) closely followed

dimensional quadratic minimization problem with linear

by LeNet(cid:3)(cid:15) at (cid:21)(cid:2)(cid:25)!(cid:2)

inequality constraints(cid:2) For the sake of comparison(cid:4) we in(cid:3)

Figure (cid:6)(cid:21) shows the number of patterns in the test set

clude here the results obtained by Burges and Sch"olkopf

that must be rejected to attain a (cid:21)(cid:2)(cid:15)! error for some of

reported in (cid:10)(cid:19)(cid:17)(cid:11)(cid:2) With a regular SVM(cid:4) their error rate

the methods(cid:2) Patterns are rejected when the value of cor(cid:3)

on the regular test set was (cid:6)(cid:2)(cid:18)!(cid:2) Cortes and Vapnik had

responding output is smaller than a prede(cid:5)ned threshold(cid:2)

reported an error rate of (cid:6)(cid:2)(cid:6)! with SVM on the same

In many applications(cid:4) rejection performance is more signif(cid:3)

data using a slightly di(cid:9)erent technique(cid:2) The computa(cid:3)

icant than raw error rate(cid:2) The score used to decide upon

tional cost of this technique is very high(cid:24) about (cid:6)(cid:18) million

the rejection of a pattern was the di(cid:9)erence between the

multiply(cid:3)adds per recognition(cid:2) Using Sch"olkopf (cid:28)s Virtual

scores of the top two classes(cid:2) Again(cid:4) Boosted LeNet(cid:3)(cid:18) has

Support Vectors technique (cid:7)V(cid:3)SVM(cid:8)(cid:4) (cid:6)(cid:2)(cid:21)! error was at(cid:3)

the best performance(cid:2) The enhanced versions of LeNet(cid:3)(cid:18)

tained(cid:2) More recently(cid:4) Sch"olkopf (cid:7)personal communication(cid:8)

did better than the original LeNet(cid:3)(cid:18)(cid:4) even though the raw

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:4)(cid:10)

Linear

4

Pairwise

35

[deslant] K−NN Euclidean

−−− 24,000 −−−>

40 PCA+quadratic

40

1000 RBF

[16x16] Tangent Distance

−−− 25,000 −−−>

SVM poly 4

−−−− 14,000 −−−−>

RS−SVM poly 5

[dist] V−SVM poly 5

−−−− 28,000 −−−−>

[deslant] 20x20−300−10

123

28x28−1000−10

28x28−300−100−10

28x28−500−150−10

[16x16] LeNet 1

LeNet 4

3

17

LeNet 4 / Local

−−− 24,000 −−−>

LeNet 4 / K−NN

−−− 24,000 −−−>

LeNet 5

Boosted LeNet 4

60

51

ing the template images(cid:2) Not surprisingly(cid:4) neural networks

require much less memory than memory(cid:3)based methods(cid:2)

The Overall performance depends on many factors in(cid:3)

cluding accuracy(cid:4) running time(cid:4) and memory requirements(cid:2)

As computer technology improves(cid:4) larger(cid:3)capacity recog(cid:3)

nizers become feasible(cid:2) Larger recognizers in turn require

larger training sets(cid:2) LeNet(cid:3)(cid:6) was appropriate to the avail(cid:3)

able technology in (cid:6)(cid:27)(cid:25)(cid:27)(cid:4) just as LeNet(cid:3)(cid:15) is appropriate now(cid:2)

650

794

795

267

469

In (cid:6)(cid:27)(cid:25)(cid:27) a recognizer as complex as LeNet(cid:3)(cid:15) would have re(cid:3)

quired several weeks(cid:28) training(cid:4) and more data than was

available(cid:4) and was therefore not even considered(cid:2) For quite

a long time(cid:4) LeNet(cid:3)(cid:6) was considered the state of the art(cid:2)

The local learning classi(cid:5)er(cid:4) the optimal margin classi(cid:5)er(cid:4)

and the tangent distance classi(cid:5)er were developed to im(cid:3)

prove upon LeNet(cid:3)(cid:6) # and they succeeded at that(cid:2) How(cid:3)

0

300

600

900

ever(cid:4) they in turn motivated a search for improved neural

network architectures(cid:2) This search was guided in part by

Fig(cid:8) (cid:5)(cid:15)(cid:8) Memory requirements(cid:3) measured in number of variables(cid:3) for

estimates of the capacity of various learning machines(cid:4) de(cid:3)

each of the methods(cid:8) Most of the methods only require one byte

rived from measurements of the training and test error as

per variable for adequate performance(cid:8)

a function of the number of training examples(cid:2) We dis(cid:3)

covered that more capacity was needed(cid:2) Through a series

accuracies were identical(cid:2)

of experiments in architecture(cid:4) combined with an analy(cid:3)

Figure (cid:6)(cid:6) shows the number of multiply(cid:3)accumulate op(cid:3)

sis of the characteristics of recognition errors(cid:4) LeNet(cid:3)(cid:18) and

erations necessary for the recognition of a single size(cid:3)

LeNet(cid:3)(cid:15) were crafted(cid:2)

normalized image for each method(cid:2) Expectedly(cid:4) neural

We (cid:5)nd that boosting gives a substantial improvement in

networks are much less demanding than memory(cid:3)based

accuracy(cid:4) with a relatively modest penalty in memory and

methods(cid:2) Convolutional Neural Networks are particu(cid:3)

computing expense(cid:2) Also(cid:4) distortion models can be used

larly well suited to hardware implementations because of

to increase the e(cid:9)ective size of a data set without actually

their regular structure and their low memory requirements

requiring to collect more data(cid:2)

for the weights(cid:2) Single chip mixed analog(cid:3)digital imple(cid:3)

The Support Vector Machine has excellent accuracy(cid:4)

mentations of LeNet(cid:3)(cid:15)(cid:28)s predecessors have been shown to

which is most remarkable(cid:4) because unlike the other high

operate at speeds in excess of (cid:6)(cid:21)(cid:21)(cid:21) characters per sec(cid:3)

performance classi(cid:5)ers(cid:4) it does not include a priori knowl(cid:3)

ond (cid:10)(cid:19)(cid:18)(cid:11)(cid:2) However(cid:4) the rapid progress of mainstream com(cid:3)

edge about the problem(cid:2) In fact(cid:4) this classi(cid:5)er would do

puter technology renders those exotic technologies quickly

just as well if the image pixels were permuted with a (cid:5)xed

obsolete(cid:2) Cost(cid:3)e(cid:9)ective implementations of memory(cid:3)based

mapping and lost their pictorial structure(cid:2) However(cid:4) reach(cid:3)

techniques are more elusive(cid:4) due to their enormous memory

ing levels of performance comparable to the Convolutional

requirements(cid:4) and computational requirements(cid:2)

Neural Networks can only be done at considerable expense

Training time was also measured(cid:2) K(cid:3)nearest neighbors

in memory and computational requirements(cid:2) The reduced(cid:3)

and TDC have essentially zero training time(cid:2) While the

set SVM requirements are within a factor of two of the

single(cid:3)layer net(cid:4) the pairwise net(cid:4) and PCA(cid:22)quadratic net

Convolutional Networks(cid:4) and the error rate is very close(cid:2)

could be trained in less than an hour(cid:4) the multilayer net

Improvements of those results are expected(cid:4) as the tech(cid:3)

training times were expectedly much longer(cid:4) but only re(cid:3)

nique is relatively new(cid:2)

quired (cid:6)(cid:21) to (cid:14)(cid:21) passes through the training set(cid:2) This

When plenty of data is available(cid:4) many methods can at(cid:3)

amounts to (cid:14) to (cid:17) days of CPU to train LeNet(cid:3)(cid:15) on a Sil(cid:3)

tain respectable accuracy(cid:2) The neural(cid:3)net methods run

icon Graphics Origin (cid:14)(cid:21)(cid:21)(cid:21) server(cid:4) using a single (cid:14)(cid:21)(cid:21)MHz

much faster and require much less space than memory(cid:3)

R(cid:6)(cid:21)(cid:21)(cid:21)(cid:21) processor(cid:2) It is important to note that while the

based techniques(cid:2) The neural nets(cid:28) advantage will become

training time is somewhat relevant to the designer(cid:4) it is of

more striking as training databases continue to increase in

little interest to the (cid:5)nal user of the system(cid:2) Given the

size(cid:2)

choice between an existing technique(cid:4) and a new technique

that brings marginal accuracy improvements at the price

E(cid:2) Invariance and Noise Resistance

of considerable training time(cid:4) any (cid:5)nal user would chose

Convolutional networks are particularly well suited for

the latter(cid:2)

recognizing or rejecting shapes with widely varying size(cid:4)

Figure (cid:6)(cid:14) shows the memory requirements(cid:4) and therefore

position(cid:4) and orientation(cid:4) such as the ones typically pro(cid:3)

the number of free parameters(cid:4) of the various classi(cid:5)ers

duced by heuristic segmenters in real(cid:3)world string recogni(cid:3)

measured in terms of the number of variables that need

tion systems(cid:2)

to be stored(cid:2) Most methods require only about one byte

In an experiment like the one described above(cid:4) the im(cid:3)

per variable for adequate performance(cid:2) However(cid:4) Nearest(cid:3)

portance of noise resistance and distortion invariance is

Neighbor methods may get by with (cid:18) bits per pixel for stor(cid:3)

not obvious(cid:2) The situation in most real applications is

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:4)(cid:12)

quite di(cid:9)erent(cid:2) Characters must generally be segmented

out of their context prior to recognition(cid:2) Segmentation

algorithms are rarely perfect and often leave extraneous

marks in character images (cid:7)noise(cid:4) underlines(cid:4) neighboring

characters(cid:8)(cid:4) or sometimes cut characters too much and pro(cid:3)

duce incomplete characters(cid:2) Those images cannot be re(cid:3)

liably size(cid:3)normalized and centered(cid:2) Normalizing incom(cid:3)

plete characters can be very dangerous(cid:2) For example(cid:4) an

enlarged stray mark can look like a genuine (cid:6)(cid:2) Therefore

many systems have resorted to normalizing the images at

the level of (cid:5)elds or words(cid:2) In our case(cid:4) the upper and lower

W1

W2

tInput

Z

F1(X0,X1,W1)

X3

F3(X3,X4)

X5

F0(X0)

X1

X2

F2(X2,W2)

Loss
Function

E

X4

D
Desired Output

pro(cid:5)les of entire (cid:5)elds (cid:7)amounts in a check(cid:8) are detected

Fig(cid:8) (cid:5)(cid:21)(cid:8) A trainable system composed of heterogeneous modules(cid:8)

and used to normalize the image to a (cid:5)xed height(cid:2) While

this guarantees that stray marks will not be blown up into

character(cid:3)looking images(cid:4) this also creates wide variations

words(cid:4) that can be trained to simultaneously segment and

of the size and vertical position of characters after segmen(cid:3)

recognize words(cid:4) without ever being given the correct seg(cid:3)

tation(cid:2) Therefore it is preferable to use a recognizer that is

mentation(cid:2)

robust to such variations(cid:2) Figure (cid:6)(cid:17) shows several exam(cid:3)

Figure (cid:6)(cid:18) shows an example of a trainable multi(cid:3)modular

ples of distorted characters that are correctly recognized by

system(cid:2) A multi(cid:3)module system is de(cid:5)ned by the function

LeNet(cid:3)(cid:15)(cid:2) It is estimated that accurate recognition occurs

implemented by each of the modules(cid:4) and by the graph of

for scale variations up to about a factor of (cid:14)(cid:4) vertical shift

interconnection of the modules to each other(cid:2) The graph

variations of plus or minus about half the height of the

implicitly de(cid:5)nes a partial order according to which the

character(cid:4) and rotations up to plus or minus (cid:17)(cid:21) degrees(cid:2)

modules must be updated in the forward pass(cid:2) For exam(cid:3)

While fully invariant recognition of complex shapes is still

ple in Figure (cid:6)(cid:18)(cid:4) module (cid:21) is (cid:5)rst updated(cid:4) then modules (cid:6)

an elusive goal(cid:4) it seems that Convolutional Networks o(cid:9)er

and (cid:14) are updated (cid:7)possibly in parallel(cid:8)(cid:4) and (cid:5)nally mod(cid:3)

a partial answer to the problem of invariance or robustness

ule (cid:17)(cid:2) Modules may or may not have trainable parameters(cid:2)

with respect to geometrical distortions(cid:2)

Loss functions(cid:4) which measure the performance of the sys(cid:3)

Figure (cid:6)(cid:17) includes examples of the robustness of LeNet(cid:3)

tem(cid:4) are implemented as module (cid:18)(cid:2) In the simplest case(cid:4)

(cid:15) under extremely noisy conditions(cid:2) Processing those

the loss function module receives an external input that

images would pose unsurmountable problems of segmen(cid:3)

carries the desired output(cid:2) In this framework(cid:4) there is no

tation and feature extraction to many methods(cid:4) but

qualitative di(cid:9)erence between trainable parameters (cid:7)

W(cid:6)(cid:7)W(cid:8)

LeNet(cid:3)(cid:15) seems able to robustly extract salient features

in the (cid:5)gure(cid:8)(cid:4) external inputs and outputs (cid:7)

(cid:8)(cid:4) and

Z(cid:7)D(cid:7)E

from these cluttered images(cid:2) The training set used for

intermediate state variables(cid:7)

(cid:8)(cid:2)

X(cid:6)(cid:7)X(cid:8)(cid:7)X(cid:9)(cid:7)X(cid:10)(cid:7)X(cid:11)

the network shown here was the MNIST training set

with salt and pepper noise added(cid:2) Each pixel was ran(cid:3)

A(cid:2) An Object(cid:3)Oriented Approach

domly inverted with probability (cid:21)(cid:2)(cid:6)(cid:2) More examples

Ob ject(cid:3)Oriented programming o(cid:9)ers a particularly con(cid:3)

of LeNet(cid:3)(cid:15) in action are available on the Internet at

venient way of implementing multi(cid:3)module systems(cid:2) Each

http(cid:2)(cid:3)(cid:3)www(cid:4)research(cid:4)att(cid:4)com(cid:3)(cid:5)yann(cid:3)ocr

(cid:2)

module is an instance of a class(cid:2) Module classes have a (cid:12)for(cid:3)

IV(cid:2) Multi(cid:3)Module Systems and Graph

ward propagation(cid:13) method (cid:7)or member function(cid:8) called

Transformer Networks

fprop

whose arguments are the inputs and outputs of the

module(cid:2) For example(cid:4) computing the output of module (cid:17)

The classical back(cid:3)propagation algorithm(cid:4) as described

in Figure (cid:6)(cid:18) can be done by calling the method

on

fprop

and used in the previous sections(cid:4)

is a simple form of

module (cid:17) with the arguments

(cid:2) Complex mod(cid:3)

X(cid:9)(cid:7)X(cid:10)(cid:7)X(cid:11)

Gradient(cid:3)Based Learning(cid:2) However(cid:4) it is clear that the

ules can be constructed from simpler modules by simply

gradient back(cid:3)propagation algorithm given by Equation (cid:18)

de(cid:5)ning a new class whose slots will contain the member

describes a more general situation than simple multi(cid:3)layer

modules and the intermediate state variables between those

feed(cid:3)forward networks composed of alternated linear trans(cid:3)

modules(cid:2) The

method for the class simply calls the

fprop

formations and sigmoidal functions(cid:2)

In principle(cid:4) deriva(cid:3)

fprop

methods of the member modules(cid:4) with the appro(cid:3)

tives can be back(cid:3)propagated through any arrangement of

priate intermediate state variables or external input and

functional modules(cid:4) as long as we can compute the prod(cid:3)

outputs as arguments(cid:2) Although the algorithms are eas(cid:3)

uct of the Jacobians of those modules by any vector(cid:2) Why

ily generalizable to any network of such modules(cid:4) including

would we want to train systems composed of multiple het(cid:3)

those whose in(cid:26)uence graph has cycles(cid:4) we will limit the dis(cid:3)

erogeneous modules(cid:29) The answer is that large and complex

cussion to the case of directed acyclic graphs (cid:7)feed(cid:3)forward

trainable systems need to be built out of simple(cid:4) specialized

networks(cid:8)(cid:2)

modules(cid:2) The simplest example is LeNet(cid:3)(cid:15)(cid:4) which mixes

Computing derivatives in a multi(cid:3)module system is just

convolutional layers(cid:4) sub(cid:3)sampling layers(cid:4) fully(cid:3)connected

as simple(cid:2) A (cid:12)backward propagation(cid:13) method(cid:4) called

layers(cid:4) and RBF layers(cid:2) Another less trivial example(cid:4) de(cid:3)

(cid:4) for each module class can be de(cid:5)ned for that pur(cid:3)

bprop

scribed in the next two sections(cid:4) is a system for recognizing

pose(cid:2) The

method of a module takes the same ar(cid:3)

bprop

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:4)(cid:13)

4

4

4

C1 S2 C3 S4 C5

4

Output

F6

8

3

4

3

3

Fig(cid:8) (cid:5)(cid:18)(cid:8) Examples of unusual(cid:3) distorted(cid:3) and noisy characters correctly recognized by LeNet(cid:2)(cid:20)(cid:8) The grey(cid:2)level of the output label represents

the penalty (cid:22)lighter for higher penalties(cid:23)(cid:8)

guments as the

method(cid:2) All the derivatives in the

used to extend the procedures to networks with recurrent

fprop

system can be computed by calling the

method on all

connections(cid:2)

bprop

the modules in reverse order compared to the forward prop(cid:3)

agation phase(cid:2) The state variables are assumed to contain

B(cid:2) Special Modules

slots for storing the gradients computed during the back(cid:3)

Neural networks and many other standard pattern recog(cid:3)

ward pass(cid:4) in addition to storage for the states computed in

nition techniques can be formulated in terms of multi(cid:3)

the forward pass(cid:2) The backward pass e(cid:9)ectively computes

modular systems trained with Gradient(cid:3)Based Learning(cid:2)

the partial derivatives of the loss E with respect to all the

Commonly used modules include matrix multiplications

state variables and all the parameters in the system(cid:2) There

and sigmoidal modules(cid:4) the combination of which can be

is an interesting duality property between the forward and

used to build conventional neural networks(cid:2) Other mod(cid:3)

backward functions of certain modules(cid:2) For example(cid:4) a

ules include convolutional layers(cid:4) sub(cid:3)sampling layers(cid:4) RBF

sum of several variables in the forward direction is trans(cid:3)

layers(cid:4) and (cid:12)softmax(cid:13) layers (cid:10)(cid:19)(cid:15)(cid:11)(cid:2) Loss functions are also

formed into a simple fan(cid:3)out (cid:7)replication(cid:8) in the backward

represented as modules whose single output produces the

direction(cid:2) Conversely(cid:4) a fan(cid:3)out in the forward direction

value of the loss(cid:2) Commonly used modules have simple

is transformed into a sum in the backward direction(cid:2) The

bprop

bprop

methods(cid:2) In general(cid:4) the

method of a func(cid:3)

software environment used to obtain the results described

tion F is a multiplication by the Jacobian of F (cid:2) Here are

in this paper(cid:4) called SN(cid:17)(cid:2)(cid:6)(cid:4) uses the above concepts(cid:2) It is

a few commonly used examples(cid:2) The

method of a

bprop

based on a home(cid:3)grown ob ject(cid:3)oriented dialect of Lisp with

fanout (cid:7)a (cid:12)Y(cid:13) connection(cid:8) is a sum(cid:4) and vice versa(cid:2) The

a compiler to C(cid:2)

bprop

method of a multiplication by a coe(cid:23)cient is a mul(cid:3)

The fact that derivatives can be computed by propaga(cid:3)

tiplication by the same coe(cid:23)cient(cid:2) The

method of a

bprop

tion in the reverse graph is easy to understand intuitively(cid:2)

multiplication by a matrix is a multiplication by the trans(cid:3)

The best way to justify it theoretically is through the use of

pose of that matrix(cid:2) The

method of an addition with

bprop

Lagrange functions (cid:10)(cid:14)(cid:6)(cid:11)(cid:4) (cid:10)(cid:14)(cid:14)(cid:11)(cid:2) The same formalism can be

a constant is the identity(cid:2)

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:4)(cid:6)

Layer
Layer

Graph
Transformer

serious de(cid:5)ciency for many applications(cid:4) notably for tasks

that deal with variable length inputs (cid:7)e(cid:2)g continuous speech

tween the modules(cid:4) are all (cid:5)xed(cid:3)size vectors(cid:2) The limited

(cid:26)exibility of (cid:5)xed(cid:3)size vectors for data representation is a

Layer
Layer

Graph
Transformer

(a)

(b)

recognition and handwritten word recognition(cid:8)(cid:4) or for tasks

that require encoding relationships between ob jects or fea(cid:3)

tures whose number and nature can vary (cid:7)invariant per(cid:3)

ception(cid:4) scene analysis(cid:4) recognition of composite ob jects(cid:8)(cid:2)

An important special case is the recognition of strings of

characters or words(cid:2)

More generally(cid:4) (cid:5)xed(cid:3)size vectors lack (cid:26)exibility for tasks

in which the state must encode probability distributions

over sequences of vectors or symbols as is the case in lin(cid:3)

Fig(cid:8) (cid:5)(cid:20)(cid:8) Traditional neural networks(cid:3) and multi(cid:2)module systems com(cid:2)

guistic processing(cid:2) Such distributions over sequences are

municate (cid:19)xed(cid:2)size vectors between layer(cid:8) Multi(cid:2)Layer Graph

Transformer Networks are composed of trainable modules that

best represented by stochastic grammars(cid:4) or(cid:4) in the more

operate on and produce graphs whose arcs carry numerical in(cid:2)

general case(cid:4) directed graphs in which each arc contains a

formation(cid:8)

vector (cid:7)stochastic grammars are special cases in which the

Interestingly(cid:4) certain non(cid:3)di(cid:9)erentiable modules can be

inserted in a multi(cid:3)module system without adverse e(cid:9)ect(cid:2)

An interesting example of that is the multiplexer module(cid:2)

It has two (cid:7)or more(cid:8) regular inputs(cid:4) one switching input(cid:4)

and one output(cid:2) The module selects one of its inputs(cid:4) de(cid:3)

pending upon the (cid:7)discrete(cid:8) value of the switching input(cid:4)

and copies it on its output(cid:2) While this module is not dif(cid:3)

ferentiable with respect to the switching input(cid:4) it is di(cid:9)er(cid:3)

entiable with respect to the regular inputs(cid:2) Therefore the

overall function of a system that includes such modules will

be di(cid:9)erentiable with respect to its parameters as long as

the switching input does not depend upon the parameters(cid:2)

For example(cid:4) the switching input can be an external input(cid:2)

vector contains probabilities and symbolic information(cid:8)(cid:2)

Each path in the graph represents a di(cid:9)erent sequence of

vectors(cid:2) Distributions over sequences can be represented

by interpreting elements of the data associated with each

arc as parameters of a probability distribution or simply

as a penalty(cid:2) Distributions over sequences are particularly

handy for modeling linguistic knowledge in speech or hand(cid:3)

writing recognition systems(cid:24) each sequence(cid:4) i(cid:2)e(cid:2)(cid:4) each path

in the graph(cid:4) represents an alternative interpretation of the

input(cid:2) Successive processing modules progressively re(cid:5)ne

the interpretation(cid:2) For example(cid:4) a speech recognition sys(cid:3)

tem might start with a single sequence of acoustic vectors(cid:4)

transform it into a lattice of phonemes (cid:7)distribution over

phoneme sequences(cid:8)(cid:4) then into a lattice of words (cid:7)distribu(cid:3)

tion over word sequences(cid:8)(cid:4) and then into a single sequence

Another interesting case is the

module(cid:2) This mod(cid:3)

min

of words representing the best interpretation(cid:2)

ule has two (cid:7)or more(cid:8) inputs and one output(cid:2) The output

of the module is the minimum of the inputs(cid:2) The func(cid:3)

tion of this module is di(cid:9)erentiable everywhere(cid:4) except on

the switching surface which is a set of measure zero(cid:2) In(cid:3)

terestingly(cid:4) this function is continuous and reasonably reg(cid:3)

ular(cid:4) and that is su(cid:23)cient to ensure the convergence of a

Gradient(cid:3)Based Learning algorithm(cid:2)

In our work on building large(cid:3)scale handwriting recog(cid:3)

nition systems(cid:4) we have found that these systems could

much more easily and quickly be developed and designed

by viewing the system as a networks of modules that take

one or several graphs as input and produce graphs as out(cid:3)

put(cid:2) Such modules are called Graph Transformers(cid:4) and the

complete systems are called Graph Transformer Networks(cid:4)

The ob ject(cid:3)oriented implementation of the multi(cid:3)module

or GTN(cid:2) Modules in a GTN communicate their states and

idea can easily be extended to include a

method

bbprop

gradients in the form of directed graphs whose arcs carry

that propagates Gauss(cid:3)Newton approximations of the sec(cid:3)

numerical information (cid:7)scalars or vectors(cid:8) (cid:10)(cid:19)(cid:19)(cid:11)(cid:2)

ond derivatives(cid:2) This leads to a direct generalization for

modular systems of the second(cid:3)derivative back(cid:3)propagation

Equation (cid:14)(cid:14) given in the Appendix(cid:2)

From the statistical point of view(cid:4) the (cid:5)xed(cid:3)size state

vectors of conventional networks can be seen as represent(cid:3)

ing the means of distributions in state space(cid:2) In variable(cid:3)

The multiplexer module is a special case of a much

size networks such as the Space(cid:3)Displacement Neural Net(cid:3)

more general situation(cid:4) described at length in Section VIII(cid:4)

works described in section VII(cid:4) the states are variable(cid:3)

where the architecture of the system changes dynamically

length sequences of (cid:5)xed size vectors(cid:2) They can be seen

with the input data(cid:2) Multiplexer modules can be used to

as representing the mean of a probability distribution over

dynamically rewire (cid:7)or recon(cid:5)gure(cid:8) the architecture of the

variable(cid:3)length sequences of (cid:5)xed(cid:3)size vectors(cid:2) In GTNs(cid:4)

system for each new input pattern(cid:2)

the states are represented as graphs(cid:4) which can be seen

C(cid:2) Graph Transformer Networks

structured collections (cid:7)possibly sequences(cid:8) of vectors (cid:7)Fig(cid:3)

as representing mixtures of probability distributions over

Multi(cid:3)module systems are a very (cid:26)exible tool for build(cid:3)

ure (cid:6)(cid:15)(cid:8)(cid:2)

ing large trainable system(cid:2) However(cid:4) the descriptions in

One of the main points of the next several sections is

the previous sections implicitly assumed that the set of

to show that Gradient(cid:3)Based Learning procedures are not

parameters(cid:4) and the state information communicated be(cid:3)

limited to networks of simple modules that communicate

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:4)(cid:5)

through (cid:5)xed(cid:3)size vectors(cid:4) but can be generalized to GTNs(cid:2)

Gradient back(cid:3)propagation through a Graph Transformer

takes gradients with respect to the numerical informa(cid:3)

tion in the output graph(cid:4) and computes gradients with re(cid:3)

spect to the numerical information attached to the input

graphs(cid:4) and with respect to the module(cid:28)s internal param(cid:3)

eters(cid:2) Gradient(cid:3)Based Learning can be applied as long as

Segmentation(cid:8)

di(cid:9)erentiable functions are used to produce the numerical

data in the output graph from the numerical data in the

Fig(cid:8) (cid:5)(cid:14)(cid:8)

Building a segmentation graph with Heuristic Over(cid:2)

input graph and the functions parameters(cid:2)

that it avoids making hard decisions about the segmenta(cid:3)

The second point of the next several sections is to show

tion by taking a large number of di(cid:9)erent segmentations

that the functions implemented by many of the modules

into consideration(cid:2) The idea is to use heuristic image pro(cid:3)

used in typical document processing systems (cid:7)and other

cessing techniques to (cid:5)nd candidate cuts of the word or

image recognition systems(cid:8)(cid:4) though commonly thought to

string(cid:4) and then to use the recognizer to score the alter(cid:3)

be combinatorial in nature(cid:4) are indeed di(cid:9)erentiable with

native segmentations thereby generated(cid:2) The process is

respect to their internal parameters as well as with respect

depicted in Figure (cid:6)(cid:19)(cid:2) First(cid:4) a number of candidate cuts

to their inputs(cid:4) and are therefore usable as part of a globally

are generated(cid:2) Good candidate locations for cuts can be

trainable system(cid:2)

found by locating minima in the vertical pro jection pro(cid:5)le(cid:4)

In most of the following(cid:4) we will purposely avoid making

or minima of the distance between the upper and lower

references to probability theory(cid:2) All the quantities manip(cid:3)

contours of the word(cid:2) Better segmentation heuristics are

ulated are viewed as penalties(cid:4) or costs(cid:4) which if necessary

described in section X(cid:2) The cut generation heuristic is de(cid:3)

can be transformed into probabilities by taking exponen(cid:3)

signed so as to generate more cuts than necessary(cid:4) in the

tials and normalizing(cid:2)

hope that the (cid:12)correct(cid:13) set of cuts will be included(cid:2) Once

V(cid:2) Multiple Object Recognition(cid:4) Heuristic

best represented by a graph(cid:4) called the segmentation graph(cid:2)

the cuts have been generated(cid:4) alternative segmentations are

Over(cid:3)Segmentation

The segmentation graph is a Directed Acyclic Graph (cid:7)DAG(cid:8)

One of the most di(cid:23)cult problems of handwriting recog(cid:3)

with a start node and an end node(cid:2) Each internal node is

nition is to recognize not just isolated characters(cid:4) but

associated with a candidate cut produced by the segmen(cid:3)

strings of characters(cid:4) such as zip codes(cid:4) check amounts(cid:4)

tation algorithm(cid:2) Each arc between a source node and a

or words(cid:2) Since most recognizers can only deal with one

destination node is associated with an image that contains

character at a time(cid:4) we must (cid:5)rst segment the string into

all the ink between the cut associated with the source node

individual character images(cid:2) However(cid:4) it is almost impos(cid:3)

and the cut associated with the destination node(cid:2) An arc

sible to devise image analysis techniques that will infallibly

is created between two nodes if the segmentor decided that

segment naturally written sequences of characters into well

the ink between the corresponding cuts could form a can(cid:3)

formed characters(cid:2)

didate character(cid:2) Typically(cid:4) each individual piece of ink

The recent history of automatic speech recognition (cid:10)(cid:14)(cid:25)(cid:11)(cid:4)

would be associated with an arc(cid:2) Pairs of successive pieces

(cid:10)(cid:19)(cid:20)(cid:11) is here to remind us that training a recognizer by opti(cid:3)

of ink would also be included(cid:4) unless they are separated by

mizing a global criterion (cid:7)at the word or sentence level(cid:8) is

a wide gap(cid:4) which is a clear indication that they belong

much preferable to merely training it on hand(cid:3)segmented

to di(cid:9)erent characters(cid:2) Each complete path through the

phonemes or other units(cid:2) Several recent works have shown

graph contains each piece of ink once and only once(cid:2) Each

that the same is true for handwriting recognition (cid:10)(cid:17)(cid:25)(cid:11)(cid:24) op(cid:3)

path corresponds to a di(cid:9)erent way of associating pieces of

timizing a word(cid:3)level criterion is preferable to solely train(cid:3)

ink together so as to form characters(cid:2)

ing a recognizer on pre(cid:3)segmented characters because the

recognizer can learn not only to recognize individual char(cid:3)

B(cid:2) Recognition Transformer and Viterbi Transformer

acters(cid:4) but also to reject mis(cid:3)segmented characters thereby

A simple GTN to recognize character strings is shown

minimizing the overall word error(cid:2)

in Figure (cid:6)(cid:20)(cid:2)

It is composed of two graph transformers

This section and the next describe in detail a simple ex(cid:3)

called the recognition transformer T

(cid:4) and the Viterbi

rec

ample of GTN to address the problem of reading strings of

transformer T

(cid:2) The goal of the recognition transformer

vit

characters(cid:4) such as words or check amounts(cid:2) The method

is to generate a graph(cid:4) called the interpretation graph or

avoids the expensive and unreliable task of hand(cid:3)truthing

recognition graph G

(cid:4) that contains all the possible inter(cid:3)

int

the result of the segmentation often required in more tra(cid:3)

pretations for all the possible segmentations of the input(cid:2)

ditional systems trained on individually labeled character

Each path in G

represents one possible interpretation of

int

images(cid:2)

one particular segmentation of the input(cid:2) The role of the

A(cid:2) Segmentation Graph

from the interpretation graph(cid:2)

Viterbi transformer is to extract the best interpretation

A now classical method for word segmentation and recog(cid:3)

The recognition transformer T

takes the segmentation

rec

nition is called Heuristic Over(cid:3)Segmentation (cid:10)(cid:19)(cid:25)(cid:11)(cid:4) (cid:10)(cid:19)(cid:27)(cid:11)(cid:2) Its

graph G

as input(cid:4) and applies the recognizer for single

seg

main advantages over other approaches to segmentation are

characters to the images associated with each of the arcs

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:7)(cid:11)

Viterbi Penalty

Σ

4

3

Gvit

T

vit

Viterbi
Transformer

2

Viterbi
Path

G

int

3

2

3

4

3

4

1

4

1

4

2
3
4

Interpretation
    Graph

recT

NN

NN

NN

NN

NN

NN

Recognition
Transformer

class
label

character recognizer
penalty for each class

    PIECE OF THE
INTERPRETATION
        GRAPH

"0"

6.7

"1"

10.3

"8"

0.3

"9"

12.5

"0"

"1"

"2"

"3"

7.9

11.2

6.8

0.2

"8"
"9"

13.5

8.4

W

 Character
Recognizer

 Character
Recognizer

8

0.1
  PIECE OF THE
SEGMENTATION
       GRAPH

3

0.5

candidate
 segment
   image

penalty given by
the segmentor

Gseg

Segmentation
      Graph

Fig(cid:8) (cid:5)(cid:16)(cid:8) The recognition transformer re(cid:19)nes each arc of the segmen(cid:2)

tation arc into a set of arcs in the interpretation graph(cid:3) one per

character class(cid:3) with attached penalties and labels(cid:8)

Fig(cid:8) (cid:5)(cid:7)(cid:8) Recognizing a character string with a GTN(cid:8) For readability(cid:3)

only the arcs with low penalties are shown(cid:8)

famous Viterbi algorithm (cid:10)(cid:20)(cid:21)(cid:11)(cid:4) an application of the prin(cid:3)

ciple of dynamic programming to (cid:5)nd the shortest path

in a graph e(cid:23)ciently(cid:2) Let c

be the penalty associated to

i

in the segmentation graph(cid:2) The interpretation graph G

int

arc i(cid:4) with source node s

(cid:4) and destination node d

(cid:7)note

i

i

has almost the same structure as the segmentation graph(cid:4)

that there can be multiple arcs between two nodes(cid:8)(cid:2)

In

except that each arc is replaced by a set of arcs from and

the interpretation graph(cid:4) arcs also have a label l

(cid:2) The

i

to the same node(cid:2) In this set of arcs(cid:4) there is one arc for

Viterbi algorithm proceeds as follows(cid:2) Each node n is as(cid:3)

each possible class for the image associated with the cor(cid:3)

sociated with a cumulated Viterbi penalty v

(cid:2) Those cu(cid:3)

n

responding arc in G

(cid:2) As shown in Figure (cid:6)(cid:25)(cid:4) to each

seg

mulated penalties are computed in any order that satis(cid:5)es

arc is attached a class label(cid:4) and the penalty that the im(cid:3)

the partial order de(cid:5)ned by the interpretation graph (cid:7)which

age belongs to this class as produced by the recognizer(cid:2) If

is directed and acyclic(cid:8)(cid:2) The start node is initialized with

the segmentor has computed penalties for the candidate

the cumulated penalty v

(cid:16) (cid:21)(cid:2) The other nodes cu(cid:3)

start

segments(cid:4) these penalties are combined with the penalties

mulated penalties v

are computed recursively from the v

n

computed by the character recognizer(cid:4) to obtain the penal(cid:3)

values of their parent nodes(cid:4) through the upstream arcs

ties on the arcs of the interpretation graph(cid:2) Although com(cid:3)

U

(cid:16) farc i with destination d

(cid:16) ng(cid:24)

n

i

bining penalties of di(cid:9)erent nature seems highly heuristic(cid:4)

the GTN training procedure will tune the penalties and

take advantage of this combination anyway(cid:2) Each path in

the interpretation graph corresponds to a possible inter(cid:3)

(cid:3)

i

U

n

v

(cid:16) min

(cid:7)c

(cid:22) v

(cid:8)(cid:3)

(cid:7)(cid:6)(cid:21)(cid:8)

n

i

s

i

pretation of the input word(cid:2) The penalty of a particular

interpretation for a particular segmentation is given by the

sum of the arc penalties along the corresponding path in

Furthermore(cid:4) the value of i for each node n which minimizes

the interpretation graph(cid:2) Computing the penalty of an in(cid:3)

the right hand side is noted m

(cid:4) the minimizing entering

n

terpretation independently of the segmentation requires to

arc(cid:2) When the end node is reached we obtain in v

the

end

combine the penalties of all the paths with that interpre(cid:3)

total penalty of the path with the smallest total penalty(cid:2)

tation(cid:2) An appropriate rule for combining the penalties of

We call this penalty the Viterbi penalty(cid:4) and this sequence

parallel paths is given in section VI(cid:3)C(cid:2)

of arcs and nodes the Viterbi path(cid:2) To obtain the Viterbi

The Viterbi transformer produces a graph G

with a

path with nodes n

(cid:3) (cid:3) (cid:3) n

and arcs i

(cid:3) (cid:3) (cid:3) i

(cid:4) we trace back

(cid:2)

vit

T

T

(cid:4)

(cid:4)

(cid:4)

single path(cid:2) This path is the path of least cumulated

these nodes and arcs as follows(cid:4) starting with n

(cid:16) the end

T

penalty in the Interpretation graph(cid:2) The result of the

node(cid:4) and recursively using the minimizing entering arc(cid:24)

recognition can be produced by reading o(cid:9) the labels of

i

(cid:16) m

(cid:4) and n

(cid:16) s

until the start node is reached(cid:2)

t

n

t

i

t

(cid:0)(cid:2)

t

the arcs along the graph G

extracted by the Viterbi

The label sequence can then be read o(cid:9) the arcs of the

vit

transformer(cid:2) The Viterbi transformer owes its name to the

Viterbi path(cid:2)

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:7)(cid:4)

VI(cid:2) Global Training for Graph Transformer

Networks

The previous section describes the process of recognizing

a string using Heuristic Over(cid:3)Segmentation(cid:4) assuming that

the recognizer is trained so as to give low penalties for the

correct class label of correctly segmented characters(cid:4) high

penalties for erroneous categories of correctly segmented

characters(cid:4) and high penalties for all categories for badly

formed characters(cid:2) This section explains how to train the

system at the string level to do the above without requiring

manual labeling of character segments(cid:2) This training will

be performed with a GTN whose architecture is slightly

di(cid:9)erent from the recognition architecture described in the

previous section(cid:2)

Constrained Viterbi Penalty

Ccvit

Σ

Best Constrained Path

Gcvit

Viterbi Transformer

Constrained
Interpretation Graph

Gc

Desired Sequence

Path Selector

Interpretation Graph

Gint

Recognition
Transformer

In many applications(cid:4) there is enough a priori knowl(cid:3)

Fig(cid:8) (cid:5)(cid:17)(cid:8) Viterbi Training GTN Architecture for a character string

edge about what is expected from each of the modules in

recognizer based on Heuristic Over(cid:2)Segmentation(cid:8)

order to train them separately(cid:2) For example(cid:4) with Heuris(cid:3)

tic Over(cid:3)Segmentation one could individually label single(cid:3)

character images and train a character recognizer on them(cid:4)

Neural Networks (cid:7)RNN(cid:8)(cid:2) Unfortunately(cid:4) despite early en(cid:3)

but it might be di(cid:23)cult to obtain an appropriate set of

thusiasm(cid:4) the training of RNNs with gradient(cid:3)based tech(cid:3)

non(cid:3)character images to train the model to reject wrongly

niques has proved very di(cid:23)cult in practice (cid:10)(cid:20)(cid:27)(cid:11)(cid:2)

segmented candidates(cid:2) Although separate training is sim(cid:3)

ple(cid:4) it requires additional supervision information that is

The GTN techniques presented below simplify and gen(cid:3)

eralize the global training methods developed for speech

often lacking or incomplete (cid:7)the correct segmentation and

recognition(cid:2)

the labels of incorrect candidate segments(cid:8)(cid:2) Furthermore

it can be shown that separate training is sub(cid:3)optimal (cid:10)(cid:19)(cid:20)(cid:11)(cid:2)

The following section describes three di(cid:9)erent gradient(cid:3)

A(cid:2) Viterbi Training

based methods for training GTN(cid:3)based handwriting recog(cid:3)

nizers at the string level(cid:24) Viterbi training(cid:4) discriminative

During recognition(cid:4) we select the path in the Interpre(cid:3)

Viterbi training(cid:4) forward training(cid:4) and discriminative for(cid:3)

tation Graph that has the lowest penalty with the Viterbi

ward training(cid:2) The last one is a generalization to graph(cid:3)

algorithm(cid:2) Ideally(cid:4) we would like this path of lowest penalty

based systems of the MAP criterion introduced in Sec(cid:3)

to be associated with the correct label sequence as often as

tion II(cid:3)C(cid:2) Discriminative forward training is somewhat

possible(cid:2) An obvious loss function to minimize is therefore

similar to the so(cid:3)called Maximum Mutual Information cri(cid:3)

the average over the training set of the penalty of the path

terion used to train HMM in speech recognition(cid:2) However(cid:4)

associated with the correct label sequence that has the low(cid:3)

our rationale di(cid:9)ers from the classical one(cid:2) We make no

est penalty(cid:2) The goal of training will be to (cid:5)nd the set of

recourse to a probabilistic interpretation(cid:4) but show that(cid:4)

recognizer parameters (cid:7)the weights(cid:4) if the recognizer is a

within the Gradient(cid:3)Based Learning approach(cid:4) discrimina(cid:3)

neural network(cid:8) that minimize the average penalty of this

tive training is a simple instance of the pervasive principle

(cid:12)correct(cid:13) lowest penalty path(cid:2) The gradient of this loss

of error correcting learning(cid:2)

function can be computed by back(cid:3)propagation through

Training methods for graph(cid:3)based sequence recognition

the GTN architecture shown in (cid:5)gure (cid:6)(cid:27)(cid:2) This training

systems such as HMMs have been extensively studied in

architecture is almost identical to the recognition archi(cid:3)

the context of speech recognition (cid:10)(cid:14)(cid:25)(cid:11)(cid:2) Those methods re(cid:3)

tecture described in the previous section(cid:4) except that an

quire that the system be based on probabilistic generative

extra graph transformer called a path selector is inserted

models of the data(cid:4) which provide normalized likelihoods

between the Interpretation Graph and the Viterbi Trans(cid:3)

over the space of possible input sequences(cid:2) Popular HMM

former(cid:2) This transformer takes the interpretation graph

learning methods(cid:4) such as the the Baum(cid:3)Welsh algorithm(cid:4)

and the desired label sequence as input(cid:2) It extracts from

rely on this normalization(cid:2) The normalization cannot be

the interpretation graph those paths that contain the cor(cid:3)

preserved when non(cid:3)generative models such as neural net(cid:3)

rect (cid:7)desired(cid:8) label sequence(cid:2) Its output graph G

is called

c

works are integrated into the system(cid:2) Other techniques(cid:4)

the constrained interpretation graph (cid:7)also known as forced

such as discriminative training methods(cid:4) must be used in

alignment in the HMM literature(cid:8)(cid:4) and contains all the

this case(cid:2) Several authors have proposed such methods to

paths that correspond to the correct label sequence(cid:2) The

train neural network(cid:30)HMM speech recognizers at the word

constrained interpretation graph is then sent to the Viterbi

or sentence level (cid:10)(cid:20)(cid:6)(cid:11)(cid:4) (cid:10)(cid:20)(cid:14)(cid:11)(cid:4) (cid:10)(cid:20)(cid:17)(cid:11)(cid:4) (cid:10)(cid:20)(cid:18)(cid:11)(cid:4) (cid:10)(cid:20)(cid:15)(cid:11)(cid:4) (cid:10)(cid:20)(cid:19)(cid:11)(cid:4) (cid:10)(cid:20)(cid:20)(cid:11)(cid:4) (cid:10)(cid:20)(cid:25)(cid:11)(cid:4)

transformer which produces a graph G

with a single

cvit

(cid:10)(cid:14)(cid:27)(cid:11)(cid:4) (cid:10)(cid:19)(cid:20)(cid:11)(cid:2)

path(cid:2) This path is the (cid:12)correct(cid:13) path with the lowest

Other globally trainable sequence recognition systems

penalty(cid:2) Finally(cid:4) a path scorer transformer takes G

(cid:4) and

cvit

avoid the di(cid:23)culties of statistical modeling by not resorting

simply computes its cumulated penalty C

by adding up

cvit

to graph(cid:3)based techniques(cid:2) The best example is Recurrent

the penalties along the path(cid:2) The output of this GTN is

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:7)(cid:7)

the loss function for the current pattern(cid:24)

that integrate neural networks with time alignment (cid:10)(cid:20)(cid:6)(cid:11)(cid:4)

E

(cid:16) C

(cid:7)(cid:6)(cid:6)(cid:8)

vit

cvit

(cid:10)(cid:20)(cid:15)(cid:11)(cid:2)

(cid:10)(cid:20)(cid:14)(cid:11)(cid:4) (cid:10)(cid:20)(cid:19)(cid:11) or hybrid neural(cid:3)network(cid:30)HMM systems (cid:10)(cid:14)(cid:27)(cid:11)(cid:4) (cid:10)(cid:20)(cid:18)(cid:11)(cid:4)

The only label information that is required by the above

While it seems simple and satisfying(cid:4) this training ar(cid:3)

system is the sequence of desired character labels(cid:2) No

chitecture has a (cid:26)aw that can potentially be fatal(cid:2) The

knowledge of the correct segmentation is required on the

problem was already mentioned in Section II(cid:3)C(cid:2)

If the

part of the supervisor(cid:4) since it chooses among the segmen(cid:3)

recognizer is a simple neural network with sigmoid out(cid:3)

tations in the interpretation graph the one that yields the

put units(cid:4) the minimum of the loss function is attained(cid:4)

lowest penalty(cid:2)

not when the recognizer always gives the right answer(cid:4) but

The process of back(cid:3)propagating gradients through the

when it ignores the input(cid:4) and sets its output to a constant

Viterbi training GTN is now described(cid:2) As explained in

vector with small values for all the components(cid:2) This is

section IV(cid:4) the gradients must be propagated backwards

known as the col lapse problem(cid:2) The collapse only occurs if

through all modules of the GTN(cid:4) in order to compute gra(cid:3)

the recognizer outputs can simultaneously take their min(cid:3)

dients in preceding modules and thereafter tune their pa(cid:3)

imum value(cid:2)

If on the other hand the recognizer(cid:28)s out(cid:3)

rameters(cid:2) Back(cid:3)propagating gradients through the path

put layer contains RBF units with (cid:5)xed parameters(cid:4) then

scorer is quite straightforward(cid:2) The partial derivatives of

there is no such trivial solution(cid:2) This is due to the fact

the loss function with respect to the individual penalties on

that a set of RBF with (cid:5)xed distinct parameter vectors

the constrained Viterbi path G

are equal to (cid:6)(cid:4) since the

cvit

cannot simultaneously take their minimum value(cid:2) In this

loss function is simply the sum of those penalties(cid:2) Back(cid:3)

case(cid:4) the complete collapse described above does not occur(cid:2)

propagating through the Viterbi Transformer is equally

However(cid:4) this does not totally prevent the occurrence of a

simple(cid:2) The partial derivatives of E

with respect to the

vit

milder collapse because the loss function still has a (cid:12)(cid:26)at

penalties on the arcs of the constrained graph G

are (cid:6)

c

spot(cid:13) for a trivial solution with constant recognizer out(cid:3)

for those arcs that appear in the constrained Viterbi path

put(cid:2) This (cid:26)at spot is a saddle point(cid:4) but it is attractive in

G

(cid:4) and (cid:21) for those that do not(cid:2) Why is it legitimate

cvit

almost all directions and is very di(cid:23)cult to get out of using

to back(cid:3)propagate through an essentially discrete function

gradient(cid:3)based minimization procedures(cid:2) If the parameters

such as the Viterbi Transformer(cid:29) The answer is that the

of the RBFs are allowed to adapt(cid:4) then the collapse prob(cid:3)

Viterbi Transformer is nothing more than a collection of

lems reappears because the RBF centers can all converge

min

functions and adders put together(cid:2)

It was shown in

to a single vector(cid:4) and the underlying neural network can

Section IV that gradients can be back(cid:3)propagated through

learn to produce that vector(cid:4) and ignore the input(cid:2) A dif(cid:3)

min

functions without adverse e(cid:9)ects(cid:2) Back(cid:3)propagation

ferent kind of collapse occurs if the width of the RBFs are

through the path selector transformer is similar to back(cid:3)

also allowed to adapt(cid:2) The collapse only occurs if a train(cid:3)

propagation through the Viterbi transformer(cid:2) Arcs in G

int

able module such as a neural network feeds the RBFs(cid:2) The

that appear in G

have the same gradient as the corre(cid:3)

c

collapse does not occur in HMM(cid:3)based speech recognition

sponding arc in G

(cid:4) i(cid:2)e(cid:2) (cid:6) or (cid:21)(cid:4) depending on whether the

c

systems because they are generative systems that produce

arc appear in G

(cid:2) The other arcs(cid:4) i(cid:2)e(cid:2)

those that do

cvit

normalized likelihoods for the input data (cid:7)more on this

not have an alter ego in G

because they do not contain

c

later(cid:8)(cid:2) Another way to avoid the collapse is to train the

the right label have a gradient of (cid:21)(cid:2) During the forward

whole system with respect to a discriminative training cri(cid:3)

propagation through the recognition transformer(cid:4) one in(cid:3)

terion(cid:4) such as maximizing the conditional probability of

stance of the recognizer for single character was created

the correct interpretations (cid:7)correct sequence of class labels(cid:8)

for each arc in the segmentation graph(cid:2) The state of rec(cid:3)

given the input image(cid:2)

ognizer instances was stored(cid:2) Since each arc penalty in

Another problem with Viterbi training is that the

G

is produced by an individual output of a recognizer

int

penalty of the answer cannot be used reliably as a mea(cid:3)

instance(cid:4) we now have a gradient (cid:7)(cid:6) or (cid:21)(cid:8) for each out(cid:3)

sure of con(cid:5)dence because it does not take low(cid:3)penalty (cid:7)or

put of each instance of the recognizer(cid:2) Recognizer outputs

high(cid:3)scoring(cid:8) competing answers into account(cid:2)

that have a non zero gradient are part of the correct an(cid:3)

swer(cid:4) and will therefore have their value pushed down(cid:2) The

B(cid:2) Discriminative Viterbi Training

gradients present on the recognizer outputs can be back(cid:3)

A modi(cid:5)cation of the training criterion can circumvent

propagated through each recognizer instance(cid:2) For each rec(cid:3)

the collapse problem described above and at the same time

ognizer instance(cid:4) we obtain a vector of partial derivatives

produce more reliable con(cid:5)dence values(cid:2) The idea is to not

of the loss function with respect to the recognizer instance

only minimize the cumulated penalty of the lowest penalty

parameters(cid:2) All the recognizer instances share the same pa(cid:3)

path with the correct interpretation(cid:4) but also to somehow

rameter vector(cid:4) since they are merely clones of each other(cid:4)

increase the penalty of competing and possibly incorrect

therefore the full gradient of the loss function with respect

paths that have a dangerously low penalty(cid:2) This type of

to the recognizer(cid:28)s parameter vector is simply the sum of

criterion is called discriminative(cid:4) because it plays the good

the gradient vectors produced by each recognizer instance(cid:2)

answers against the bad ones(cid:2) Discriminative training pro(cid:3)

Viterbi training(cid:4) though formulated di(cid:9)erently(cid:4) is often use

cedures can be seen as attempting to build appropriate

in HMM(cid:3)based speech recognition systems (cid:10)(cid:14)(cid:25)(cid:11)(cid:2) Similar al(cid:3)

separating surfaces between classes rather than to model

gorithms have been applied to speech recognition systems

individual classes independently of each other(cid:2) For exam(cid:3)

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:7)(cid:8)

Loss Function

[0.1](+1)

Σ

+

−

[0.7](+1)

+

[0.6](−1)

+

3 [0.1](+1)

Gcvit

4 [0.6](+1)

3 [0.1](−1)

4 [0.4](−1)

1 [0.1](−1)

Gvit

Gc

"34"
Desired
Answer

Viterbi Tansformer

3 [0.1](+1)

4 [2.4](0)

3 [3.4](0)

4 [0.6](+1)

Path Selector

Viterbi Transformer

3 [0.1](0)
5 [2.3](0)

4 [0.4](−1)
2 [1.3](0)

1 [0.1](−1)
4 [2.4](0)

3 [3.4](0)
4 [4.4](0)

4 [0.6](+1)
9 [1.2](0)

Interpretation
    Graph
Gint

(−1)

(+1)

(−1)

4

NN

NN

NN

4

NN

1

NN

Recognition
Transfomer

recT

W
Neural Net
Weights

Segmentation
      Graph

Gseg

Segmenter

Fig(cid:8) (cid:15)(cid:6)(cid:8) Discriminative Viterbi Training GTN Architecture for a character string recognizer based on Heuristic Over(cid:2)Segmentation(cid:8) Quantities

in square brackets are penalties computed during the forward propagation(cid:8) Quantities in parentheses are partial derivatives computed

during the backward propagation(cid:8)

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:7)(cid:9)

ple(cid:4) modeling the conditional distribution of the classes

low penalty(cid:4) but should have had a higher penalty since it

given the input image is more discriminative (cid:7)focus(cid:3)sing

is not part of the desired answer(cid:2)

more on the classi(cid:5)cation surface(cid:8) than having a separate

Variations of this technique have been used for the speech

generative model of the input data associated to each class

recognition(cid:2) Driancourt and Bottou (cid:10)(cid:20)(cid:19)(cid:11) used a version of

(cid:7)which(cid:4) with class priors(cid:4) yields the whole joint distribu(cid:3)

it where the loss function is saturated to a (cid:5)xed value(cid:2)

tion of classes and inputs(cid:8)(cid:2) This is because the conditional

This can be seen as a generalization of the Learning Vector

approach does not need to assume a particular form for the

Quantization (cid:14) (cid:7)LVQ(cid:3)(cid:14)(cid:8) loss function (cid:10)(cid:25)(cid:21)(cid:11)(cid:2) Other variations

distribution of the input data(cid:2)

of this method use not only the Viterbi path(cid:4) but the K(cid:3)

One example of discriminative criterion is the di(cid:9)erence

best paths(cid:2) The Discriminative Viterbi algorithm does not

between the penalty of the Viterbi path in the constrained

have the (cid:26)aws of the non(cid:3)discriminative version(cid:4) but there

graph(cid:4) and the penalty of the Viterbi path in the (cid:7)uncon(cid:3)

are problems nonetheless(cid:2) The main problem is that the

strained(cid:8) interpretation graph(cid:4) i(cid:2)e(cid:2) the di(cid:9)erence between

criterion does not build a margin between the classes(cid:2) The

the penalty of the best correct path(cid:4) and the penalty of

gradient is zero as soon as the penalty of the constrained

the best path (cid:7)correct or incorrect(cid:8)(cid:2) The corresponding

Viterbi path is equal to that of the Viterbi path(cid:2) It would

GTN training architecture is shown in (cid:5)gure (cid:14)(cid:21)(cid:2) The left

be desirable to push up the penalties of the wrong paths

side of the diagram is identical to the GTN used for non(cid:3)

when they are dangerously close to the good one(cid:2) The

discriminative Viterbi training(cid:2) This loss function reduces

following section presents a solution to this problem(cid:2)

the risk of collapse because it forces the recognizer to in(cid:3)

creases the penalty of wrongly recognized ob jects(cid:2) Dis(cid:3)

C(cid:2) Forward Scoring(cid:7) and Forward Training

criminative training can also be seen as another example

of error correction procedure(cid:4) which tends to minimize the

While the penalty of the Viterbi path is perfectly appro(cid:3)

di(cid:9)erence between the desired output computed in the left

priate for the purpose of recognition(cid:4) it gives only a partial

half of the GTN in (cid:5)gure (cid:14)(cid:21) and the actual output com(cid:3)

picture of the situation(cid:2) Imagine the lowest penalty paths

puted in the right half of (cid:5)gure (cid:14)(cid:21)(cid:2)

corresponding to several di(cid:8)erent segmentations produced

Let the discriminative Viterbi loss function be denoted

the same answer (cid:7)the same label sequence(cid:8)(cid:2) Then it could

E

(cid:4) and let us call C

the penalty of the Viterbi path in

dvit

cvit

be argued that the overall penalty for the interpretation

the constrained graph(cid:4) and C

the penalty of the Viterbi

vit

should be smaller than the penalty obtained when only one

path in the unconstrained interpretation graph(cid:24)

path produced that interpretation(cid:4) because multiple paths

E

(cid:16) C

(cid:2) C

(cid:7)(cid:6)(cid:14)(cid:8)

dvit

cvit

vit

label sequence is correct(cid:2) Several rules can be used com(cid:3)

with identical label sequences are more evidence that the

E

is always positive since the constrained graph is a

dvit

pute the penalty associated to a graph that contains several

subset of the paths in the interpretation graph(cid:4) and the

parallel paths(cid:2) We use a combination rule borrowed from

Viterbi algorithm selects the path with the lowest total

a probabilistic interpretation of the penalties as negative

penalty(cid:2)

In the ideal case(cid:4) the two paths C

and C

cvit

vit

log posteriors(cid:2) In a probabilistic framework(cid:4) the posterior

coincide(cid:4) and E

is zero(cid:2)

dvit

probability for the interpretation should be the sum of the

Back(cid:3)propagating gradients through the discriminative

posteriors for all the paths that produce that interpreta(cid:3)

Viterbi GTN adds some (cid:12)negative(cid:13) training to the pre(cid:3)

tion(cid:2) Translated in terms of penalties(cid:4) the penalty of an

viously described non(cid:3)discriminative training(cid:2) Figure (cid:14)(cid:21)

interpretation should be the negative logarithm of the sum

shows how the gradients are back(cid:3)propagated(cid:2) The left

of the negative exponentials of the penalties of the individ(cid:3)

half is identical to the non(cid:3)discriminative Viterbi training

ual paths(cid:2) The overall penalty will be smaller than all the

GTN(cid:4) therefore the back(cid:3)propagation is identical(cid:2) The gra(cid:3)

penalties of the individual paths(cid:2)

dients back(cid:3)propagated through the right half of the GTN

Given an interpretation(cid:4) there is a well known method(cid:4)

are multiplied by (cid:3)(cid:6)(cid:4) since C

contributes to the loss with

called the forward algorithm for computing the above quan(cid:3)

vit

a negative sign(cid:2) Otherwise the process is similar to the left

tity e(cid:23)ciently (cid:10)(cid:14)(cid:25)(cid:11)(cid:2) The penalty computed with this pro(cid:3)

half(cid:2) The gradients on arcs of G

get positive contribu(cid:3)

cedure for a particular interpretation is called the forward

int

tions from the left half and negative contributions from the

penalty(cid:2) Consider again the concept of constrained graph(cid:4)

right half(cid:2) The two contributions must be added(cid:4) since the

the subgraph of the interpretation graph which contains

penalties on G

arcs are sent to the two halves through

only the paths that are consistent with a particular label

int

a (cid:12)Y(cid:13) connection in the forward pass(cid:2) Arcs in G

that

sequence(cid:2) There is one constrained graph for each pos(cid:3)

int

appear neither in G

nor in G

have a gradient of zero(cid:2)

sible label sequence (cid:7)some may be empty graphs(cid:4) which

vit

cvit

They do not contribute to the cost(cid:2) Arcs that appear in

have in(cid:5)nite penalties(cid:8)(cid:2) Given an interpretation(cid:4) running

both G

and G

also have zero gradient(cid:2) The (cid:3)(cid:6) contri(cid:3)

the forward algorithm on the corresponding constrained

vit

cvit

bution from the right half cancels the the (cid:22)(cid:6) contribution

graph gives the forward penalty for that interpretation(cid:2)

from the left half(cid:2) In other words(cid:4) when an arc is rightfully

The forward algorithm proceeds in a way very similar to

part of the answer(cid:4) there is no gradient(cid:2) If an arc appears

the Viterbi algorithm(cid:4) except that the operation used at

in G

but not in G

(cid:4) the gradient is (cid:22)(cid:6)(cid:2) The arc should

each node to combine the incoming cumulated penalties(cid:4)

cvit

vit

have had a lower penalty to make it to G

(cid:2) If an arc is

instead of being the

function is the so(cid:3)called

min

logadd

vit

in G

but not in G

(cid:4) the gradient is (cid:3)(cid:6)(cid:2) The arc had a

operation(cid:4) which can be seen as a (cid:12)soft(cid:13) version of the

min

vit

cvit

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:7)(cid:10)

function(cid:24)

f

(cid:16) logadd

(cid:7)c

(cid:22) f

(cid:8)(cid:3)

(cid:7)(cid:6)(cid:17)(cid:8)

n

i

s

(cid:3)

i

U

n

i

where f

(cid:16) (cid:21)(cid:4) U

is the set of upstream arcs of node n(cid:4)

start

n

c

is the penalty on arc i(cid:4) and

i

Edforw

+

−

Cdforw

Cforw

Forward  Scorer

logadd(cid:7)x

(cid:2) x

(cid:2) (cid:3) (cid:3) (cid:3) (cid:2) x

(cid:8) (cid:16) (cid:2) log(cid:7)

e

(cid:8)

(cid:7)(cid:6)(cid:18)(cid:8)

(cid:4)

(cid:7)

n

X

i

(cid:14)(cid:4)

n

(cid:2)

x

i

Constrained
Interpretation Graph

Gc

Forward Scorer

Note that because of numerical inaccuracies(cid:4) it is better

to factorize the largest e

(cid:7)corresponding to the smallest

(cid:2)

x

i

penalty(cid:8) out of the logarithm(cid:2)

An interesting analogy can be drawn if we consider that

a graph on which we apply the forward algorithm is equiv(cid:3)

alent to a neural network on which we run a forward prop(cid:3)

Desired
Sequence

Path Selector

Interpretation Graph

Gint

Recognition
Transformer

agation(cid:4) except that multiplications are replaced by addi(cid:3)

Fig(cid:8) (cid:15)(cid:5)(cid:8)

Discriminative Forward Training GTN Architecture

tions(cid:4) the additions are replaced by logadds(cid:4) and there are

for a character string recognizer based on Heuristic Over(cid:2)

no sigmoids(cid:2)

Segmentation(cid:8)

One way to understand the forward algorithm is to think

about multiplicative scores (cid:7)e(cid:2)g(cid:2)(cid:4) probabilities(cid:8) instead of

G

c

additive penalties on the arcs(cid:24) score (cid:16) exp(cid:7)(cid:2) penalty (cid:8)(cid:2) In

(cid:2)

(cid:2)

(cid:8)E

(cid:8)E

that case the Viterbi algorithm selects the path with the

largest cumulative score (cid:7)with scores multiplied along the

f

n

f

c

d

i

i

(cid:16) e

e

(cid:7)(cid:6)(cid:15)(cid:8)

X

(cid:8) f

n

(cid:8) f

d

i

(cid:3)

i

D

n

path(cid:8)(cid:4) whereas the forward score is the sum of the cumula(cid:3)

where D

(cid:16) farc i with source s

(cid:16) ng is the set of down(cid:3)

n

i

tive scores associated to each of the possible paths from the

stream arcs from node n(cid:2) From the above derivatives(cid:4) the

start to the end node(cid:2) The forward penalty is always lower

derivatives with respect to the arc penalties are obtained(cid:24)

than the cumulated penalty on any of the paths(cid:4) but if one

path (cid:12)dominates(cid:13) (cid:7)with a much lower penalty(cid:8)(cid:4) its penalty

(cid:16)

e

(cid:7)(cid:6)(cid:19)(cid:8)

(cid:8)E

(cid:8)E

(cid:2)

(cid:2)

c

f

f

i

s

(cid:17)

d

i

i

is almost equal to the forward penalty(cid:2) The forward algo(cid:3)

(cid:8) c

(cid:8) f

i

d

i

rithm gets its name from the forward pass of the well(cid:3)known

This can be seen as a (cid:12)soft(cid:13) version of the back(cid:3)propagation

Baum(cid:3)Welsh algorithm for training Hidden Markov Mod(cid:3)

through a Viterbi scorer and transformer(cid:2) All the arcs in

els (cid:10)(cid:14)(cid:25)(cid:11)(cid:2) Section VIII(cid:3)E gives more details on the relation

G

have an in(cid:26)uence on the loss function(cid:2) The arcs that

c

between this work and HMMs(cid:2)

belong to low penalty paths have a larger in(cid:26)uence(cid:2) Back(cid:3)

The advantage of the forward penalty with respect to

propagation through the path selector is the same as before(cid:2)

the Viterbi penalty is that it takes into account all the

The derivative with respect to G

arcs that have an alter

int

di(cid:9)erent ways to produce an answer(cid:4) and not just the one

ego in G

are simply copied from the corresponding arc in

c

with the lowest penalty(cid:2) This is important if there is some

G

(cid:2) The derivatives with respect to the other arcs are (cid:21)(cid:2)

c

ambiguity in the segmentation(cid:4) since the combined forward

Several authors have applied the

idea of back(cid:3)

penalty of two paths C

and C

associated with the same

(cid:4)

(cid:7)

propagating gradients through a forward scorer to train

label sequence may be less than the penalty of a path C

(cid:8)

speech recognition systems(cid:4) including Bridle and his (cid:5)(cid:3)net

associated with another label sequence(cid:4) even though the

model (cid:10)(cid:20)(cid:17)(cid:11) and Ha(cid:9)ner and his (cid:5)(cid:6) (cid:3)TDNN model (cid:10)(cid:25)(cid:6)(cid:11)(cid:4) but

penalty of C

might be less than any one of C

or C

(cid:2)

(cid:8)

(cid:4)

(cid:7)

these authors recommended discriminative training as de(cid:3)

The Forward training GTN is only a slight modi(cid:5)ca(cid:3)

scribed in the next section(cid:2)

tion of the previously introduced Viterbi training GTN(cid:2) It

su(cid:23)ces to turn the Viterbi transformers in Figure (cid:6)(cid:27) into

D(cid:2) Discriminative Forward Training

Forward Scorers that take an interpretation graph as input

The information contained in the forward penalty can be

an produce the forward penalty of that graph on output(cid:2)

used in another discriminative training criterion which we

Then the penalties of all the paths that contain the correct

will call the discriminative forward criterion(cid:2) This criterion

answer are lowered(cid:4) instead of just that of the best one(cid:2)

corresponds to maximization of the posterior probability of

Back(cid:3)propagating through the forward penalty computa(cid:3)

choosing the paths associated with the correct interpreta(cid:3)

tion (cid:7)the forward transformer(cid:8) is quite di(cid:9)erent from back(cid:3)

tion(cid:2) This posterior probability is de(cid:5)ned as the exponen(cid:3)

propagating through a Viterbi transformer(cid:2) All the penal(cid:3)

tial of the minus the constrained forward penalty(cid:4) normal(cid:3)

ties of the input graph have an in(cid:26)uence on the forward

ized by the exponential of minus the unconstrained forward

penalty(cid:4) but penalties that belong to low(cid:3)penalty paths

penalty(cid:2) Note that the forward penalty of the constrained

have a stronger in(cid:26)uence(cid:2) Computing derivatives with re(cid:3)

graph is always larger or equal to the forward penalty of the

spect to the forward penalties f

computed at each n node

unconstrained interpretation graph(cid:2) Ideally(cid:4) we would like

n

of a graph is done by back(cid:3)propagation through the graph

the forward penalty of the constrained graph to be equal to

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:7)(cid:12)

the forward penalty of the complete interpretation graph(cid:2)

E(cid:2) Remarks on Discriminative Training

Equality between those two quantities is achieved when the

In the above discussion(cid:4) the global training criterion

combined penalties of the paths with the correct label se(cid:3)

was given a probabilistic interpretation(cid:4) but the individ(cid:3)

quence is negligibly small compared to the penalties of all

ual penalties on the arcs of the graphs were not(cid:2) There are

the other paths(cid:4) or that the posterior probability associ(cid:3)

good reasons for that(cid:2) For example(cid:4) if some penalties are

ated to the paths with the correct interpretation is almost

associated to the di(cid:9)erent class labels(cid:4) they would (cid:7)(cid:6)(cid:8) have

(cid:6)(cid:4) which is precisely what we want(cid:2) The corresponding

to sum to (cid:6) (cid:7)class posteriors(cid:8)(cid:4) or (cid:7)(cid:14)(cid:8) integrate to (cid:6) over the

GTN training architecture is shown in (cid:5)gure (cid:14)(cid:6)(cid:2)

input domain (cid:7)likelihoods(cid:8)(cid:2)

Let the di(cid:9)erence be denoted E

(cid:4) and let us call

dforw

Let us (cid:5)rst discuss the (cid:5)rst case (cid:7)class posteriors normal(cid:3)

C

the forward penalty of the constrained graph(cid:4) and

cforw

ization(cid:8)(cid:2) This local normalization of penalties may elimi(cid:3)

C

the forward penalty of the complete interpretation

forw

nate information that is important for locally rejecting all

graph(cid:24)

the classes (cid:10)(cid:25)(cid:14)(cid:11)(cid:4) e(cid:2)g(cid:2)(cid:4) when a piece of image does not cor(cid:3)

E

(cid:16) C

(cid:2) C

(cid:7)(cid:6)(cid:20)(cid:8)

dforw

cforw

forw

respond to a valid character class(cid:4) because some of the

E

is always positive since the constrained graph is a

segmentation candidates may be wrong(cid:2) Although an ex(cid:3)

dforw

subset of the paths in the interpretation graph(cid:4) and the

plicit (cid:12)garbage class(cid:13) can be introduced in a probabilistic

forward penalty of a graph is always larger than the for(cid:3)

framework to address that question(cid:4) some problems remain

ward penalty of a subgraph of this graph(cid:2) In the ideal case(cid:4)

because it is di(cid:23)cult to characterize such a class probabilis(cid:3)

the penalties of incorrect paths are in(cid:5)nitely large(cid:4) there(cid:3)

tically and to train a system in this way (cid:7)it would require

fore the two penalties coincide and E

is zero(cid:2) Readers

a density model of unseen or unlabeled samples(cid:8)(cid:2)

dforw

familiar with the Boltzmann machine connectionist model

The probabilistic interpretation of individual variables

might recognize the constrained and unconstrained graphs

plays an important role in the Baum(cid:3)Welsh algorithm

as analogous to the (cid:12)clamped(cid:13) (cid:7)constrained by the ob(cid:3)

in combination with the Expectation(cid:3)Maximization proce(cid:3)

served values of the output variable(cid:8) and (cid:12)free(cid:13) (cid:7)uncon(cid:3)

dure(cid:2) Unfortunately(cid:4) those methods cannot be applied to

strained(cid:8) phases of the Boltzmann machine algorithm (cid:10)(cid:6)(cid:17)(cid:11)(cid:2)

discriminative training criteria(cid:4) and one is reduced to us(cid:3)

Back(cid:3)propagating derivatives through the discriminative

ing gradient(cid:3)based methods(cid:2) Enforcing the normalization

Forward GTN distributes gradients more evenly than in the

of the probabilistic quantities while performing gradient(cid:3)

Viterbi case(cid:2) Derivatives are back(cid:3)propagated through the

based learning is complex(cid:4) ine(cid:23)cient(cid:4) time consuming(cid:4) and

left half of the the GTN in Figure (cid:14)(cid:6) down to the interpre(cid:3)

creates ill(cid:3)conditioning of the loss(cid:3)function(cid:2)

tation graph(cid:2) Derivatives are negated and back(cid:3)propagated

Following (cid:10)(cid:25)(cid:14)(cid:11)(cid:4) we therefore prefer to postpone normal(cid:3)

through the right(cid:3)half(cid:4) and the result for each arc is added

ization as far as possible (cid:7)in fact(cid:4) until the (cid:5)nal decision

to the contribution from the left half(cid:2) Each arc in G

stage of the system(cid:8)(cid:2) Without normalization(cid:4) the quanti(cid:3)

int

now has a derivative(cid:2) Arcs that are part of a correct path

ties manipulated in the system do not have a direct prob(cid:3)

have a positive derivative(cid:2) This derivative is very large if

abilistic interpretation(cid:2)

an incorrect path has a lower penalty than all the correct

Let us now discuss the second case (cid:7)using a generative

paths(cid:2) Similarly(cid:4) the derivatives with respect to arcs that

model of the input(cid:8)(cid:2) Generative models build the boundary

are part of a low(cid:3)penalty incorrect path have a large nega(cid:3)

indirectly(cid:4) by (cid:5)rst building an independent density model

tive derivative(cid:2) On the other hand(cid:4) if the penalty of a path

for each class(cid:4) and then performing classi(cid:5)cation decisions

associated with the correct interpretation is much smaller

on the basis of these models(cid:2) This is not a discriminative

than all other paths(cid:4) the loss function is very close to (cid:21)

approach in that it does not focus on the ultimate goal of

and almost no gradient is back(cid:3)propagated(cid:2) The training

learning(cid:4) which in this case is to learn the classi(cid:5)cation de(cid:3)

therefore concentrates on examples of images which yield a

cision surface(cid:2) Theoretical arguments (cid:10)(cid:19)(cid:11)(cid:4) (cid:10)(cid:20)(cid:11) suggest that

classi(cid:5)cation error(cid:4) and furthermore(cid:4) it concentrates on the

estimating input densities when the real goal is to obtain

pieces of the image which cause that error(cid:2) Discriminative

a discriminant function for classi(cid:5)cation is a suboptimal

forward training is an elegant and e(cid:23)cient way of solving

strategy(cid:2) In theory(cid:4) the problem of estimating densities in

the infamous credit assignment problem for learning ma(cid:3)

high(cid:3)dimensional spaces is much more ill(cid:3)posed than (cid:5)nd(cid:3)

chines that manipulate (cid:12)dynamic(cid:13) data structures such as

ing decision boundaries(cid:2)

graphs(cid:2) More generally(cid:4) the same idea can be used in all

Even though the internal variables of the system do not

situations where a learning machine must choose between

have a direct probabilistic interpretation(cid:4) the overall sys(cid:3)

discrete alternative interpretations(cid:2)

tem can still be viewed as producing posterior probabilities

As previously(cid:4) the derivatives on the interpretation graph

for the classes(cid:2) In fact(cid:4) assuming that a particular label se(cid:3)

penalties can then be back(cid:3)propagated into the character

quence is given as the (cid:12)desired sequence(cid:13) to the GTN in

recognizer instances(cid:2) Back(cid:3)propagation through the char(cid:3)

(cid:5)gure (cid:14)(cid:6)(cid:4) the exponential of minus E

can be inter(cid:3)

dforw

acter recognizer gives derivatives on its parameters(cid:2) All the

preted as an estimate of the posterior probability of that

gradient contributions for the di(cid:9)erent candidate segments

label sequence given the input(cid:2) The sum of those posteriors

are added up to obtain the total gradient associated to one

for all the possible label sequences is (cid:6)(cid:2) Another approach

pair (cid:7)input image(cid:4) correct label sequence(cid:8)(cid:4) that is(cid:4) one ex(cid:3)

would consists of directly minimizing an approximation of

ample in the training set(cid:2) A step of stochastic gradient

the number of misclassi(cid:5)cations (cid:10)(cid:25)(cid:17)(cid:11) (cid:10)(cid:20)(cid:19)(cid:11)(cid:2) We prefer to use

descent can then be applied to update the parameters(cid:2)

the discriminative forward loss function because it causes

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:7)(cid:13)

"U"

Recognizer

Fig(cid:8) (cid:15)(cid:15)(cid:8) Explicit segmentation can be avoided by sweeping a recog(cid:2)

nizer at every possible location in the input (cid:19)eld(cid:8)

less numerical problems during the optimization(cid:2) We will

see in Section X(cid:3)C that this is a good way to obtain scores

$

on which to base a rejection strategy(cid:2) The important point

Fig(cid:8) (cid:15)(cid:18)(cid:8) A Space Displacement Neural Network is a convolutional

being made here is that one is free to choose any param(cid:3)

network that has been replicated over a wide input (cid:19)eld(cid:8)

eterization deemed appropriate for a classi(cid:5)cation model(cid:2)

The fact that a particular parameterization uses internal

variables with no clear probabilistic interpretation does not

characters within a string may have widely varying sizes

make the model any less legitimate than models that ma(cid:3)

and baseline positions(cid:2) Therefore the recognizer must be

nipulate normalized quantities(cid:2)

very robust to shifts and size variations(cid:2)

An important advantage of global and discriminative

These three problems are elegantly circumvented if a

training is that learning focuses on the most important

convolutional network is replicated over the input (cid:5)eld(cid:2)

errors(cid:4) and the system learns to integrate the ambigui(cid:3)

First of all(cid:4) as shown in section III(cid:4) convolutional neu(cid:3)

ties from the segmentation algorithm with the ambigui(cid:3)

ral networks are very robust to shifts and scale varia(cid:3)

ties of the character recognizer(cid:2) In Section IX we present

tions of the input image(cid:4) as well as to noise and extra(cid:3)

experimental results with an on(cid:3)line handwriting recogni(cid:3)

neous marks in the input(cid:2) These properties take care of

tion system that con(cid:5)rm the advantages of using global

the latter two problems mentioned in the previous para(cid:3)

training versus separate training(cid:2) Experiments in speech

graph(cid:2) Second(cid:4) convolutional networks provide a drastic

recognition with hybrids of neural networks and HMMs

saving in computational requirement when replicated over

also showed marked improvements brought by global train(cid:3)

large input (cid:5)elds(cid:2) A replicated convolutional network(cid:4) also

ing (cid:10)(cid:20)(cid:20)(cid:11)(cid:4) (cid:10)(cid:14)(cid:27)(cid:11)(cid:4) (cid:10)(cid:19)(cid:20)(cid:11)(cid:4) (cid:10)(cid:25)(cid:18)(cid:11)(cid:2)

called a Space Displacement Neural Network or SDNN (cid:10)(cid:14)(cid:20)(cid:11)(cid:4)

VII(cid:2) Multiple Object Recognition(cid:4) Space

be prohibitively expensive in general(cid:4) convolutional net(cid:3)

is shown in Figure (cid:14)(cid:17)(cid:2) While scanning a recognizer can

Displacement Neural Network

works can be scanned or replicated very e(cid:23)ciently over

(cid:2)

large(cid:4) variable(cid:3)size input (cid:5)elds(cid:2) Consider one instance of

There is a simple alternative to explicitly segmenting im(cid:3)

a convolutional net and its alter ego at a nearby location(cid:2)

ages of character strings using heuristics(cid:2) The idea is to

Because of the convolutional nature of the network(cid:4) units

sweep a recognizer at all possible locations across a nor(cid:3)

in the two instances that look at identical locations on the

malized image of the entire word or string as shown in

input have identical outputs(cid:4) therefore their states do not

Figure (cid:14)(cid:14)(cid:2) With this technique(cid:4) no segmentation heuris(cid:3)

need to be computed twice(cid:2) Only a thin (cid:12)slice(cid:13) of new

tics are required since the system essentially examines al l

states that are not shared by the two network instances

the possible segmentations of the input(cid:2) However(cid:4) there

needs to be recomputed(cid:2) When all the slices are put to(cid:3)

are problems with this approach(cid:2) First(cid:4) the method is in

gether(cid:4) the result is simply a larger convolutional network

general quite expensive(cid:2) The recognizer must be applied

whose structure is identical to the original network(cid:4) except

at every possible location on the input(cid:4) or at least at a

that the feature maps are larger in the horizontal dimen(cid:3)

large enough subset of locations so that misalignments of

sion(cid:2) In other words(cid:4) replicating a convolutional network

characters in the (cid:5)eld of view of the recognizers are small

can be done simply by increasing the size of the (cid:5)elds over

enough to have no e(cid:9)ect on the error rate(cid:2) Second(cid:4) when

which the convolutions are performed(cid:4) and by replicating

the recognizer is centered on a character to be recognized(cid:4)

the output layer accordingly(cid:2) The output layer e(cid:9)ectively

the neighbors of the center character will be present in the

becomes a convolutional layer(cid:2) An output whose receptive

(cid:5)eld of view of the recognizer(cid:4) possibly touching the cen(cid:3)

(cid:5)eld is centered on an elementary ob ject will produce the

ter character(cid:2) Therefore the recognizer must be able to

class of this ob ject(cid:4) while an in(cid:3)between output may indi(cid:3)

correctly recognize the character in the center of its input

cate no character or contain rubbish(cid:2) The outputs can be

(cid:5)eld(cid:4) even if neighboring characters are very close to(cid:4) or

interpreted as evidences for the presence of ob jects at all

touching the central character(cid:2) Third(cid:4) a word or charac(cid:3)

possible positions in the input (cid:5)eld(cid:2)

ter string cannot be perfectly size normalized(cid:2) Individual

The SDNN architecture seems particularly attractive for

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:7)(cid:6)

recognizing cursive handwriting where no reliable segmen(cid:3)

tation heuristic exists(cid:2) Although the idea of SDNN is quite

old(cid:4) and very attractive by its simplicity(cid:4) it has not gener(cid:3)

ated wide interest until recently because(cid:4) as stated above(cid:4)

it puts enormous demands on the recognizer (cid:10)(cid:14)(cid:19)(cid:11)(cid:4) (cid:10)(cid:14)(cid:20)(cid:11)(cid:2) In

speech recognition(cid:4) where the recognizer is at least one

order of magnitude smaller(cid:4) replicated convolutional net(cid:3)

works are easier to implement(cid:4) for instance in Ha(cid:9)ner(cid:28)s

Multi(cid:3)State TDNN model (cid:10)(cid:20)(cid:25)(cid:11)(cid:4) (cid:10)(cid:25)(cid:15)(cid:11)(cid:2)

A(cid:2) Interpreting the Output of an SDNN with a GTN

The output of an SDNN is a sequence of vectors which

encode the likelihoods(cid:4) penalties(cid:4) or scores of (cid:5)nding char(cid:3)

acter of a particular class label at the corresponding lo(cid:3)

cation in the input(cid:2) A post(cid:3)processor is required to pull

out the best possible label sequence from this vector se(cid:3)

Viterbi Graph

Interpretation Graph

Character
Model
Transducer

SDNN Output

Viterbi Answer

Viterbi Transformer

Compose

S....c.....r......i....p....t
s....e.....n.....e.j...o.T
5......a...i...u......p.....f

SDNN
Transformer

quence(cid:2) An example of SDNN output is shown in Fig(cid:3)

Fig(cid:8) (cid:15)(cid:21)(cid:8) A Graph Transformer pulls out the best interpretation from

ure (cid:14)(cid:15)(cid:2) Very often(cid:4) individual characters are spotted by

the output of the SDNN(cid:8)

several neighboring instances of the recognizer(cid:4) a conse(cid:3)

quence of the robustness of the recognizer to horizontal

translations(cid:2) Also quite often(cid:4) characters are erroneously

C1

C3 C5

detected by recognizer instances that see only a piece of

a character(cid:2) For example a recognizer instance that only

sees the right third of a (cid:12)(cid:18)(cid:13) might output the label (cid:6)(cid:2) How

can we eliminate those extraneous characters from the out(cid:3)

put sequence and pull(cid:3)out the best interpretation(cid:29) This

can be done using a new type of Graph Transformer with

two input graphs as shown in Figure (cid:14)(cid:18)(cid:2) The sequence of

vectors produced by the SDNN is (cid:5)rst coded into a linear

graph with multiple arcs between pairs of successive nodes(cid:2)

2345
Compose + Viterbi

2

3 3 4 5

Answer

SDNN
Output
F6

Input

Each arc between a particular pair of nodes contains the

With SDNN(cid:3) no explicit segmentation is performed(cid:8)

Fig(cid:8) (cid:15)(cid:20)(cid:8) An example of multiple character recognition with SDNN(cid:8)

label of one of the possible categories(cid:4) together with the

penalty produced by the SDNN for that class label at that

location(cid:2) This graph is called the SDNN Output Graph(cid:2)

B(cid:2) Experiments with SDNN

The second input graph to the transformer is a grammar

transducer(cid:4) more speci(cid:5)cally a (cid:4)nite(cid:3)state transducer (cid:10)(cid:25)(cid:19)(cid:11)(cid:4)

In a series of experiments(cid:4) LeNet(cid:3)(cid:15) was trained with the

that encodes the relationship between input strings of class

goal of being replicated so as to recognize multiple char(cid:3)

labels and corresponding output strings of recognized char(cid:3)

acters without segmentations(cid:2) The data was generated

acters(cid:2)The transducer is a weighted (cid:5)nite state machine (cid:7)a

from the previously described Modi(cid:5)ed NIST set as fol(cid:3)

graph(cid:8) where each arc contains a pair of labels and possibly

lows(cid:2) Training images were composed of a central char(cid:3)

a penalty(cid:2) Like a (cid:5)nite(cid:3)state machine(cid:4) a transducer is in a

acter(cid:4) (cid:26)anked by two side characters picked at random in

state and follows an arc to a new state when an observed

the training set(cid:2) The separation between the bounding

input symbol matches the (cid:5)rst symbol in the symbol pair

boxes of the characters were chosen at random between (cid:3)(cid:6)

attached to the arc(cid:2) At this point the transducer emits the

and (cid:18) pixels(cid:2) In other instances(cid:4) no central character was

second symbol in the pair together with a penalty that com(cid:3)

present(cid:4) in which case the desired output of the network

bines the penalty of the input symbol and the penalty of

was the blank space class(cid:2)

In addition(cid:4) training images

the arc(cid:2) A transducer therefore transforms a weighted sym(cid:3)

were degraded with (cid:6)(cid:21)! salt and pepper noise (cid:7)random

bol sequence into another weighted symbol sequence(cid:2) The

pixel inversions(cid:8)(cid:2)

graph transformer shown in (cid:5)gure (cid:14)(cid:18) performs a composi(cid:3)

Figures (cid:14)(cid:15) and (cid:14)(cid:19) show a few examples of success(cid:3)

tion between the recognition graph and the grammar trans(cid:3)

ful recognitions of multiple characters by the LeNet(cid:3)(cid:15)

ducer(cid:2) This operation takes every possible sequence corre(cid:3)

SDNN(cid:2) Standard techniques based on Heuristic Over(cid:3)

sponding to every possible path in the recognition graph

Segmentation would fail miserably on many of those ex(cid:3)

and matches them with the paths in the grammar trans(cid:3)

amples(cid:2) As can be seen on these examples(cid:4) the network

ducer(cid:2) The composition produces the interpretation graph(cid:4)

exhibits striking invariance and noise resistance properties(cid:2)

which contains a path for each corresponding output label

While some authors have argued that invariance requires

sequence(cid:2) This composition operation may seem combina(cid:3)

more sophisticated models than feed(cid:3)forward neural net(cid:3)

torially intractable(cid:4) but it turns out there exists an e(cid:23)cient

works (cid:10)(cid:25)(cid:20)(cid:11)(cid:4) LeNet(cid:3)(cid:15) exhibits these properties to a large ex(cid:3)

algorithm for it described in more details in Section VIII(cid:2)

tent(cid:2)

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:7)(cid:5)

540

5 5

4 0

1114
1 1 1 4 4 1

678

6 7 7 7

8 8

3514
3 5 5 1 1 4

Answer
SDNN
output

F6

Input

Fig(cid:8) (cid:15)(cid:14)(cid:8) An SDNN applied to a noisy image of digit string(cid:8) The digits shown in the SDNN output represent the winning class labels(cid:3) with

a lighter grey level for high(cid:2)penalty answers(cid:8)

Similarly(cid:4) it has been suggested that accurate recognition

which they can be implemented on parallel hardware(cid:2) Spe(cid:3)

of multiple overlapping ob jects require explicit mechanisms

cialized analog(cid:30)digital chips have been designed and used

that would solve the so(cid:3)called feature binding problem (cid:10)(cid:25)(cid:20)(cid:11)(cid:2)

in character recognition(cid:4) and in image preprocessing appli(cid:3)

As can be seen on Figures (cid:14)(cid:15) and (cid:14)(cid:19)(cid:4) the network is able to

cations (cid:10)(cid:25)(cid:25)(cid:11)(cid:2) However the rapid progress of conventional

tell the characters apart(cid:4) even when they are closely inter(cid:3)

processor technology with reduced(cid:3)precision vector arith(cid:3)

twined(cid:4) a task that would be impossible to achieve with the

metic instructions (cid:7)such as Intel(cid:28)s MMX(cid:8) make the success

more classical Heuristic Over(cid:3)Segmentation technique(cid:2) The

of specialized hardware hypothetical at best(cid:2)

SDNN is also able to correctly group disconnected pieces

Short video clips of the LeNet(cid:3)(cid:15) SDNN can be viewed at

of ink that form characters(cid:2) Good examples of that are

http(cid:2)(cid:3)(cid:3)www(cid:4)research(cid:4)att(cid:4)com(cid:3)(cid:5)yann(cid:3)ocr

(cid:2)

shown in the upper half of (cid:5)gure (cid:14)(cid:19)(cid:2) In the top left ex(cid:3)

ample(cid:4) the (cid:18) and the (cid:21) are more connected to each other

C(cid:2) Global Training of SDNN

than they are connected with themselves(cid:4) yet the system

In the above experiments(cid:4) the string image were arti(cid:5)(cid:3)

correctly identi(cid:5)es the (cid:18) and the (cid:21) as separate ob jects(cid:2) The

cially generated from individual character(cid:2) The advantage

top right example is interesting for several reasons(cid:2) First

is that we know in advance the location and the label of

the system correctly identi(cid:5)es the three individual ones(cid:2)

the important character(cid:2) With real training data(cid:4) the cor(cid:3)

Second(cid:4) the left half and right half of disconnected (cid:18) are

rect sequence of labels for a string is generally available(cid:4)

correctly grouped(cid:4) even though no geometrical information

but the precise locations of each corresponding character

could decide to associate the left half to the vertical bar on

in the input image are unknown(cid:2)

its left or on its right(cid:2) The right half of the (cid:18) does cause

In the experiments described in the previous section(cid:4) the

the appearance of an erroneous (cid:6) on the SDNN output(cid:4)

best interpretation was extracted from the SDNN output

but this one is removed by the character model transducer

using a very simple graph transformer(cid:2) Global training of

which prevents characters from appearing on contiguous

an SDNN can be performed by back(cid:3)propagating gradients

outputs(cid:2)

through such graph transformers arranged in architectures

Another important advantage of SDNN is the ease with

similar to the ones described in section VI(cid:2)

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:8)(cid:11)

Edforw

+

−

Cdforw

Cforw

Forward  Scorer

Constrained
Interpretation Graph

Gc

Forward Scorer

Desired
Sequence

Path Selector

Interpretation Graph

Gint

Character
Model
Transducer

SDNN Output

Compose

S....c.....r......i....p....t
s....e.....n.....e.j...o.T
5......a...i...u......p.....f

SDNN
Transformer

zip code recognition (cid:10)(cid:27)(cid:6)(cid:11)(cid:4) and more recent experiments in

on(cid:3)line handwriting recognition (cid:10)(cid:17)(cid:25)(cid:11) have demonstrated the

idea of globally(cid:3)trained SDNN(cid:30)HMM hybrids(cid:2) SDNN is an

extremely promising and attractive technique for OCR(cid:4) but

so far it has not yielded better results than Heuristic Over(cid:3)

Segmentation(cid:2) We hope that these results will improve as

more experience is gained with these models(cid:2)

D(cid:2) Object Detection and Spotting with SDNN

An interesting application of SDNNs is ob ject detection

and spotting(cid:2) The invariance properties of Convolutional

Networks(cid:4) combined with the e(cid:23)ciency with which they

can be replicated over large (cid:5)elds suggest that they can

be used for (cid:12)brute force(cid:13) ob ject spotting and detection in

large images(cid:2) The main idea is to train a single Convolu(cid:3)

tional Network to distinguish images of the ob ject of inter(cid:3)

est from images present in the background(cid:2) In utilization

mode(cid:4) the network is replicated so as to cover the entire

image to be analyzed(cid:4) thereby forming a two(cid:3)dimensional

Space Displacement Neural Network(cid:2) The output of the

SDNN is a two(cid:3)dimensional plane in which activated units

Fig(cid:8) (cid:15)(cid:7)(cid:8) A globally trainable SDNN(cid:28)HMM hybrid system expressed

indicate the presence of the ob ject of interest in the corre(cid:3)

as a GTN(cid:8)

sponding receptive (cid:5)eld(cid:2) Since the sizes of the ob jects to

be detected within the image are unknown(cid:4) the image can

be presented to the network at multiple resolutions(cid:4) and

This is somewhat equivalent to modeling the output

the results at multiple resolutions combined(cid:2) The idea has

of an SDNN with a Hidden Markov Model(cid:2) Globally

been applied to face location(cid:4) (cid:10)(cid:27)(cid:17)(cid:11)(cid:4) address block location

trained(cid:4) variable(cid:3)size TDNN(cid:30)HMM hybrids have been used

on envelopes (cid:10)(cid:27)(cid:18)(cid:11)(cid:4) and hand tracking in video (cid:10)(cid:27)(cid:15)(cid:11)(cid:2)

for speech recognition and on(cid:3)line handwriting recogni(cid:3)

To illustrate the method(cid:4) we will consider the case of

tion (cid:10)(cid:20)(cid:20)(cid:11)(cid:4) (cid:10)(cid:25)(cid:27)(cid:11)(cid:4) (cid:10)(cid:27)(cid:21)(cid:11)(cid:4) (cid:10)(cid:19)(cid:20)(cid:11)(cid:2) Space Displacement Neural Net(cid:3)

face detection in images as described in (cid:10)(cid:27)(cid:17)(cid:11)(cid:2) First(cid:4) images

works have been used in combination with HMMs or other

containing faces at various scales are collected(cid:2) Those im(cid:3)

elastic matching methods for handwritten word recogni(cid:3)

ages are (cid:5)ltered through a zero(cid:3)mean Laplacian (cid:5)lter so as

tion (cid:10)(cid:27)(cid:6)(cid:11)(cid:4) (cid:10)(cid:27)(cid:14)(cid:11)(cid:2)

to remove variations in global illumination and low spatial

Figure (cid:14)(cid:20) shows the graph transformer architecture for

frequency illumination gradients(cid:2) Then(cid:4) training samples

training an SDNN(cid:30)HMM hybrid with the Discriminative

of faces and non(cid:3)faces are manually extracted from those

Forward Criterion(cid:2) The top part is comparable to the top

images(cid:2) The face sub(cid:3)images are then size normalized so

part of (cid:5)gure (cid:14)(cid:6)(cid:2) On the right side the composition of the

that the height of the entire face is approximately (cid:14)(cid:21) pixels

recognition graph with the grammar gives the interpreta(cid:3)

while keeping fairly large variations (cid:7)within a factor of two(cid:8)(cid:2)

tion graph with all the possible legal interpretations(cid:2) On

The scale of background sub(cid:3)images are picked at random(cid:2)

the left side the composition is performed with a grammar

A single convolutional network is trained on those samples

that only contains paths with the desired sequence of la(cid:3)

to classify face sub(cid:3)images from non(cid:3)face sub(cid:3)images(cid:2)

bels(cid:2) This has a somewhat similar function to the path

When a scene image is to be analyzed(cid:4) it is (cid:5)rst (cid:5)ltered

selector used in the previous section(cid:2) Like in Section VI(cid:3)D

through the Laplacian (cid:5)lter(cid:4) and sub(cid:3)sampled at powers(cid:3)

the loss function is the di(cid:9)erence between the forward score

of(cid:3)two resolutions(cid:2) The network is replicated over each of

obtained from the left half and the forward score obtained

multiple resolution images(cid:2) A simple voting technique is

from the right half(cid:2) To back(cid:3)propagate through the com(cid:3)

used to combine the results from multiple resolutions(cid:2)

position transformer(cid:4) we need to keep a record of which arc

A two(cid:3)dimensional version of the global training method

in the recognition graph originated which arcs in the inter(cid:3)

described in the previous section can be used to allevi(cid:3)

pretation graph(cid:2) The derivative with respect to an arc in

ate the need to manually locate faces when building the

the recognition graph is equal to the sum of the derivatives

training sample (cid:10)(cid:27)(cid:17)(cid:11)(cid:2) Each possible location is seen as an

with respect to all the arcs in the interpretation graph that

alternative interpretation(cid:4) i(cid:2)e(cid:2) one of several parallel arcs

originated from it(cid:2) Derivative can also be computed for the

in a simple graph that only contains a start node and an

penalties on the grammar graph(cid:4) allowing to learn them as

end node(cid:2)

well(cid:2) As in the previous example(cid:4) a discriminative criterion

Other authors have used Neural Networks(cid:4) or other clas(cid:3)

must be used(cid:4) because using a non(cid:3)discriminative criterion

si(cid:5)ers such as Support Vector Machines for face detection

could result in a collapse e(cid:9)ect if the network(cid:28)s output RBF

with great success (cid:10)(cid:27)(cid:19)(cid:11)(cid:4) (cid:10)(cid:27)(cid:20)(cid:11)(cid:2) Their systems are very similar

are adaptive(cid:2) The above training procedure can be equiv(cid:3)

to the one described above(cid:4) including the idea of presenting

alently formulated in term of HMM(cid:2) Early experiments in

the image to the network at multiple scales(cid:2) But since those

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:8)(cid:4)

systems do not use Convolutional Networks(cid:4) they cannot

path and a corresponding pair of input(cid:30)output sequences

take advantage of the speedup described here(cid:4) and have to

(cid:7)S

(cid:4)S

(cid:8) in the transducer graph(cid:2) The weights on the arcs

out

in

rely on other techniques(cid:4) such as pre(cid:3)(cid:5)ltering and real(cid:3)time

of the output graph are obtained by adding the weights

tracking(cid:4) to keep the computational requirement within

from the matching arcs in the input acceptor and trans(cid:3)

reasonable limits(cid:2) In addition(cid:4) because those classi(cid:5)ers are

ducer graphs(cid:2)

In the rest of the paper(cid:4) we will call this

much less invariant to scale variations than Convolutional

graph composition operation using transducers the (cid:9)stan(cid:3)

Networks(cid:4) it is necessary to multiply the number of scales

dard(cid:10) transduction operation(cid:2)

at which the images are presented to the classi(cid:5)er(cid:2)

A simple example of transduction is shown in Figure (cid:14)(cid:25)(cid:2)

VIII(cid:2) Graph Transformer Networks and

In this simple example(cid:4) the input and output symbols on

Transducers

ducer graph is called a grammar graph(cid:2) To better under(cid:3)

the transducer arcs are always identical(cid:2) This type of trans(cid:3)

In Section IV(cid:4) Graph Transformer Networks (cid:7)GTN(cid:8)

stand the transduction operation(cid:4) imagine two tokens sit(cid:3)

were introduced as a generalization of multi(cid:3)layer(cid:4) multi(cid:3)

ting each on the start nodes of the input acceptor graph

module networks where the state information is repre(cid:3)

and the transducer graph(cid:2) The tokens can freely follow

sented as graphs instead of (cid:5)xed(cid:3)size vectors(cid:2) This section

any arc labeled with a null input symbol(cid:2) A token can

re(cid:3)interprets the GTNs in the framework of Generalized

follow an arc labeled with a non(cid:3)null input symbol if the

Transduction(cid:4) and proposes a powerful Graph Composition

other token also follows an arc labeled with the same in(cid:3)

algorithm(cid:2)

put symbol(cid:2) We have an acceptable trajectory when both

A(cid:2) Previous Work

have reached the terminal con(cid:5)guration(cid:8)(cid:2) This tra jectory

tokens reach the end nodes of their graphs (cid:7)i(cid:2)e(cid:2) the tokens

Numerous authors in speech recognition have used

represents a sequence of input symbols that complies with

Gradient(cid:3)Based Learning methods that integrate graph(cid:3)

both the acceptor and the transducer(cid:2) We can then collect

based statistical models (cid:7)notably HMM(cid:8) with acoustic

the corresponding sequence of output symbols along the

recognition modules(cid:4) mainly Gaussian mixture models(cid:4) but

tra jectory of the transducer token(cid:2) The above procedure

also neural networks (cid:10)(cid:27)(cid:25)(cid:11)(cid:4) (cid:10)(cid:20)(cid:25)(cid:11)(cid:4) (cid:10)(cid:27)(cid:27)(cid:11)(cid:4) (cid:10)(cid:19)(cid:20)(cid:11)(cid:2) Similar ideas have

produces a tree(cid:4) but a simple technique described in Sec(cid:3)

been applied to handwriting recognition (cid:7)see (cid:10)(cid:17)(cid:25)(cid:11) for a re(cid:3)

tion VIII(cid:3)C can be used to avoid generating multiple copies

view(cid:8)(cid:2) However(cid:4) there has been no proposal for a system(cid:3)

of certain subgraphs by detecting when a particular output

atic approach to multi(cid:3)layer graph(cid:3)based trainable systems(cid:2)

state has already been seen(cid:2)

The idea of transforming graphs into other graphs has re(cid:3)

The transduction operation can be performed very e(cid:23)(cid:3)

ceived considerable interest in computer science(cid:4) through

ciently (cid:10)(cid:6)(cid:21)(cid:19)(cid:11)(cid:4) but presents complex book(cid:3)keeping problems

the concept of weighted (cid:4)nite(cid:3)state transducers (cid:10)(cid:25)(cid:19)(cid:11)(cid:2) Trans(cid:3)

concerning the handling of all combinations of null and non

ducers have been applied to speech recognition (cid:10)(cid:6)(cid:21)(cid:21)(cid:11) and

null symbols(cid:2) If the weights are interpreted as probabilities

language translation (cid:10)(cid:6)(cid:21)(cid:6)(cid:11)(cid:4) and proposals have been made

(cid:7)normalized appropriately(cid:8) then an acceptor graph repre(cid:3)

for handwriting recognition (cid:10)(cid:6)(cid:21)(cid:14)(cid:11)(cid:2) This line of work has

sents a probability distribution over the language de(cid:5)ned

been mainly focused on e(cid:23)cient search algorithms (cid:10)(cid:6)(cid:21)(cid:17)(cid:11)

by the set of label sequences associated to all possible paths

and on the algebraic aspects of combining transducers and

(cid:7)from the start to the end node(cid:8) in the graph(cid:2)

graphs (cid:7)called acceptors in this context(cid:8)(cid:4) but very little

An example of application of the transduction opera(cid:3)

e(cid:9)ort has been devoted to building globally trainable sys(cid:3)

tion is the incorporation of linguistic constraints (cid:7)a lexicon

tems out of transducers(cid:2) What is proposed in the follow(cid:3)

or a grammar(cid:8) when recognizing words or other character

ing sections is a systematic approach to automatic training

strings(cid:2) The recognition transformer produces the recog(cid:3)

in graph(cid:3)manipulating systems(cid:2) A di(cid:9)erent approach to

nition graph (cid:7)an acceptor graph(cid:8) by applying the neural

graph(cid:3)based trainable systems(cid:4) called Input(cid:3)Output HMM(cid:4)

network recognizer to each candidate segment(cid:2) This ac(cid:3)

was proposed in (cid:10)(cid:6)(cid:21)(cid:18)(cid:11)(cid:4) (cid:10)(cid:6)(cid:21)(cid:15)(cid:11)(cid:2)

ceptor graph is composed with a transducer graph for the

grammar(cid:2) The grammar transducer contains a path for

B(cid:2) Standard Transduction

each legal sequence of symbol(cid:4) possibly augmented with

In the established framework of (cid:5)nite(cid:3)state transduc(cid:3)

penalties to indicate the relative likelihoods of the possi(cid:3)

ers (cid:10)(cid:25)(cid:19)(cid:11)(cid:4) discrete symbols are attached to arcs in the graphs(cid:2)

ble sequences(cid:2) The arcs contain identical input and output

Acceptor graphs have a single symbol attached to each

symbols(cid:2) Another example of transduction was mentioned

arc whereas transducer graphs have two symbols (cid:7)an input

in Section V(cid:24) the path selector used in the heuristic over(cid:3)

symbol and an output symbol(cid:8)(cid:2) A special null symbol is

segmentation training GTN is implementable by a compo(cid:3)

absorbed by any other symbol (cid:7)when concatenating sym(cid:3)

sition(cid:2) The transducer graph is linear graph which con(cid:3)

bols to build a symbol sequence(cid:8)(cid:2) Weighted transducers

tains the correct label sequence(cid:2) The composition of the

and acceptors also have a scalar quantity attached to each

interpretation graph with this linear graph yields the con(cid:3)

arc(cid:2) In this framework(cid:4) the composition operation takes as

strained graph(cid:2)

input an acceptor graph and a transducer graph and builds

an output acceptor graph(cid:2) Each path in this output graph

C(cid:2) Generalized Transduction

(cid:7)with symbol sequence S

(cid:8) corresponds to one path (cid:7)with

If the data structures associated to each arc took only

out

symbol sequence S

(cid:8) in the input acceptor graph and one

a (cid:5)nite number of values(cid:4) composing the input graph and

in

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:8)(cid:7)

an appropriate transducer would be a sound solution(cid:2) For

our applications however(cid:4) the data structures attached to

the arcs of the graphs may be vectors(cid:4) images or other

high(cid:3)dimensional ob jects that are not readily enumerated(cid:2)

We present a new composition operation that solves this

problem(cid:2)

Instead of only handling graphs with discrete symbols

and penalties on the arcs(cid:4) we are interested in considering

graphs whose arcs may carry complex data structures(cid:4) in(cid:3)

cluding continuous(cid:3)valued data structures such as vectors

and images(cid:2) Composing such graphs requires additional

information(cid:24)

(cid:0)

When examining a pair of arcs (cid:7)one from each input

graph(cid:8)(cid:4) we need a criterion to decide whether to create cor(cid:3)

responding arc(cid:7)s(cid:8) and node(cid:7)s(cid:8) in the output graph(cid:4) based

on the information attached to the input arcs(cid:2) We can de(cid:3)

cide to build an arc(cid:4) several arcs(cid:4) or an entire sub(cid:3)graph

with several nodes and arcs(cid:2)

interpretation graph

0.8

"u"

"c"

0.4

"a"

0.2

0.8

"t"

"p"

0.2

"t"

0.8

match
& add

match
& add

match
& add

n
o
i
t
i
s
o
p
m
o
C
h
p
a
r
G

interpretations:
cut  (2.0)
cap (0.8)
cat  (1.4)

grammar graph

"r"

"n"

"a"

"b"

"u"

"t"

"c"

"u"

"a"

"t"

"e"

"e"

"r"

"p"

"t"

"r"

"d"

"c"

0.4

"x"

0.1

"o"

1.0

"a"

0.2

"d"

1.8

"u"

0.8

"p"

0.2

"t"

0.8

Recognition
Graph

(cid:0)

When that criterion is met(cid:4) we must build the corre(cid:3)

Fig(cid:8) (cid:15)(cid:16)(cid:8)

Example of composition of the recognition graph with

sponding arc(cid:7)s(cid:8) and node(cid:7)s(cid:8) in the output graph and com(cid:3)

consistent with both of them(cid:8) During the forward propagation

the grammar graph in order to build an interpretation that is

pute the information attached to the newly created arc(cid:7)s(cid:8)

(cid:22)dark arrows(cid:23)(cid:3) the methods

and

are used(cid:8) Gradients

check

fprop

as a function the the information attached to the input

(cid:22)dashed arrows(cid:23) are back(cid:2)propagated with the application of the

arcs(cid:2)

method

(cid:8)

bprop

These functions are encapsulated in an ob ject called a

Composition Transformer(cid:2) An instance of Composition

Transformer implements three methods(cid:24)

(cid:0)

check(cid:12)arc(cid:6)(cid:7) arc(cid:8)(cid:13)

jectory is acceptable (cid:7)i(cid:2)e(cid:2) both tokens simultaneously reach

the end nodes of their graphs(cid:8)(cid:2) The management of null

compares the data structures pointed to by arcs

(cid:7)from

arc(cid:6)

transitions is a straightforward modi(cid:5)cation of the token

the (cid:5)rst graph(cid:8) and

(cid:7)from the second graph(cid:8) and re(cid:3)

arc(cid:8)

simulation function(cid:2) Before enumerating the possible non

turns a boolean indicating whether corresponding arc(cid:7)s(cid:8)

null joint token transitions(cid:4) we loop on the possible null

should be created in the output graph(cid:2)

(cid:0)

fprop(cid:12)ngraph(cid:7) upnode(cid:7) downnode(cid:7) arc(cid:6)(cid:7) arc(cid:8)(cid:13)

transitions of each token(cid:4) recursively call the token sim(cid:3)

ulation function(cid:4) and (cid:5)nally call the method

(cid:2) The

fprop

is called when

returns true(cid:2) This

check(cid:12)arc(cid:6)(cid:7) arc(cid:8)(cid:13)

safest way for identifying acceptable tra jectories consists in

method creates new arcs and nodes between nodes

upnode

running a preliminary pass for identifying the token con(cid:3)

and

in the output graph

(cid:4) and computes

downnode

ngraph

(cid:5)gurations from which we can reach the terminal con(cid:5)gu(cid:3)

the information attached to these newly created arcs as a

ration (cid:7)i(cid:2)e(cid:2) both tokens on the end nodes(cid:8)(cid:2) This is easily

function of the attached information of the input arcs

arc(cid:6)

achieved by enumerating the tra jectories in the opposite

and

(cid:2)

arc(cid:8)

direction(cid:2) We start on the end nodes and follow the arcs

(cid:0)

bprop(cid:12)ngraph(cid:7) upnode(cid:7) downnode(cid:7) arc(cid:6)(cid:7) arc(cid:8)(cid:13)

upstream(cid:2) During the main pass(cid:4) we only build the nodes

is called during training in order to propagate gradient in(cid:3)

that allow the tokens to reach the terminal con(cid:5)guration(cid:2)

formation from the output sub(cid:3)graph between

and

upnode

Graph composition using transducers (cid:7)i(cid:2)e(cid:2) standard

downnode

arc(cid:6)

arc(cid:8)

into the data structures on the

and

(cid:4)

transduction(cid:8) is easily and e(cid:23)ciently implemented as a gen(cid:3)

as well as with respect to the parameters that were used in

eralized transduction(cid:2) The method

simply tests the

check

the

call with the same arguments(cid:2) This assumes that

fprop

equality of the input symbols on the two arcs(cid:4) and the

the function used by

to compute the values attached

fprop

method

creates a single arc whose symbol is the

fprop

to its output arcs is di(cid:9)erentiable(cid:2)

output symbol on the transducer(cid:28)s arc(cid:2)

The

method can be seen as constructing a dy(cid:3)

check

The composition between pairs of graphs is particularly

namic architecture of functional dependencies(cid:4) while the

useful for incorporating linguistic constraints in a hand(cid:3)

fprop

method performs a forward propagation through

writing recognizer(cid:2) Examples of its use are given in the

that architecture to compute the numerical information at(cid:3)

on(cid:3)line handwriting recognition system described in Sec(cid:3)

tached to the arcs(cid:2) The

method performs a back(cid:3)

bprop

tion IX(cid:8) and in the check reading system described in Sec(cid:3)

ward propagation through the same architecture to com(cid:3)

tion X(cid:8)(cid:2)

pute the partial derivatives of the loss function with respect

In the rest of the paper(cid:4) the term Composition Trans(cid:3)

to the information attached to the arcs(cid:2) This is illustrated

former will denote a Graph Transformer based on the gen(cid:3)

in Figure (cid:14)(cid:25)(cid:2)

eralized transductions of multiple graphs(cid:2) The concept of

Figure (cid:14)(cid:27) shows a simpli(cid:5)ed generalized graph composi(cid:3)

generalized transduction is a very general one(cid:2)

In fact(cid:4)

tion algorithm(cid:2) This simpli(cid:5)ed algorithm does not handle

many of the graph transformers described earlier in this

null transitions(cid:4) and does not check whether the tokens tra(cid:3)

paper(cid:4) such as the segmenter and the recognizer(cid:4) can be

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:8)(cid:8)

formulated in terms of generalized transduction(cid:2)

In this

case the(cid:4) the generalized transduction does not take two in(cid:3)

put graphs but a single input graph(cid:2) The method

of

fprop

the transformer may create several arcs or even a complete

Function generalized(cid:14)composition(cid:12)PGRAPH graph(cid:6)(cid:7)

subgraph for each arc of the initial graph(cid:2) In fact the pair

PGRAPH graph(cid:8)(cid:7)

(cid:12)check(cid:7) fprop(cid:13)

itself can be seen as procedurally de(cid:5)ning

PTRANS trans(cid:13)

a transducer(cid:2)

Returns PGRAPH

In addition(cid:4) It can be shown that the generalized trans(cid:3)

(cid:15)

duction of a single graph is theoretically equivalent to the

(cid:3)(cid:3) Create new graph

standard composition of this graph with a particular trans(cid:3)

PGRAPH ngraph (cid:16) new(cid:14)graph(cid:12)(cid:13)

ducer graph(cid:2) However(cid:4) implementing the operation this

(cid:3)(cid:3) Create map between token positions

very complicated(cid:2)

(cid:3)(cid:3) and nodes of the new graph

In practice(cid:4) the graph produced by a generalized trans(cid:3)

PNODE map(cid:17)PNODE(cid:7)PNODE(cid:18) (cid:16) new(cid:14)empty(cid:14)map(cid:12)(cid:13)

duction is represented procedurally(cid:4) in order to avoid build(cid:3)

map(cid:17)endnode(cid:12)graph(cid:6)(cid:13)(cid:7) endnode(cid:12)graph(cid:8)(cid:13)(cid:18) (cid:16)

ing the whole output graph (cid:7)which may be huge when for

endnode(cid:12)newgraph(cid:13)

example the interpretation graph is composed with the

way may be very ine(cid:23)cient since the transducer can be

(cid:3)(cid:3) Recursive subroutine for simulating tokens

are visited by the search algorithm during recognition (cid:7)e(cid:2)g(cid:2)

Function simtokens(cid:12)PNODE node(cid:6)(cid:7) PNODE node(cid:8)(cid:13)

Viterbi(cid:8)(cid:2) This strategy propagates the bene(cid:5)ts of pruning

grammar graph(cid:8)(cid:2) We only instantiate the nodes which

Returns PNODE

(cid:15)

algorithms (cid:7)e(cid:2)g(cid:2) Beam Search(cid:8) in all the Graph Transformer

Network(cid:2)

PNODE currentnode (cid:16) map(cid:17)node(cid:6)(cid:7) node(cid:8)(cid:18)

(cid:3)(cid:3) Check if already visited

If (cid:12)currentnode (cid:16)(cid:16) nil(cid:13)

D(cid:2) Notes on the Graph Structures

(cid:3)(cid:3) Record new configuration

currentnode (cid:16) ngraph(cid:19)(cid:20)create(cid:14)node(cid:12)(cid:13)

Section VI has discussed the idea of global training

map(cid:17)node(cid:6)(cid:7) node(cid:8)(cid:18) (cid:16) currentnode

by back(cid:3)propagating gradient through simple graph trans(cid:3)

(cid:3)(cid:3) Enumerate the possible non(cid:19)null

formers(cid:2) The

method is the basis of the back(cid:3)

bprop

(cid:3)(cid:3) joint token transitions

propagation algorithm for generic graph transformers(cid:2) A

For ARC arc(cid:6) in down(cid:14)arcs(cid:12)node(cid:6)(cid:13)

generalized composition transformer can be seen as dynam(cid:3)

For ARC arc(cid:8) in down(cid:14)arcs(cid:12)node(cid:8)(cid:13)

ically establishing functional relationships between the nu(cid:3)

If (cid:12)trans(cid:19)(cid:20)check(cid:12)arc(cid:6)(cid:7) arc(cid:8)(cid:13)(cid:13)

merical quantities on the input and output arcs(cid:2) Once the

PNODE newnode (cid:16)

simtokens(cid:12)down(cid:14)node(cid:12)arc(cid:6)(cid:13)(cid:7)

check

function has decided that a relationship should be es(cid:3)

tablished(cid:4) the

function implements the numerical re(cid:3)

fprop

down(cid:14)node(cid:12)arc(cid:8)(cid:13)(cid:13)

lationship(cid:2) The

function establishes the structure of

check

trans(cid:19)(cid:20)fprop(cid:12)ngraph(cid:7) currentnode(cid:7)

the ephemeral network inside the composition transformer(cid:2)

newnode(cid:7) arc(cid:6)(cid:7) arc(cid:8)(cid:13)

Since

is assumed to be di(cid:9)erentiable(cid:4) gradients can

fprop

(cid:3)(cid:3) Return node in composed graph

be back(cid:3)propagated through that structure(cid:2) Most param(cid:3)

Return currentnode

eters a(cid:9)ect the scores stored on the arcs of the successive

(cid:21)

graphs of the system(cid:2) A few threshold parameters may de(cid:3)

termine whether an arc appears or not in the graph(cid:2) Since

(cid:3)(cid:3) Perform token simulation

non existing arcs are equivalent to arcs with very large

simtokens(cid:12)startnode(cid:12)graph(cid:6)(cid:13)(cid:7) startnode(cid:12)graph(cid:8)(cid:13)(cid:13)

penalties(cid:4) we only consider the case of parameters a(cid:9)ect(cid:3)

Delete map

ing the penalties(cid:2)

Return ngraph

(cid:21)

In the kind of systems we have discussed until now (cid:7)and

the application described in Section X(cid:8)(cid:4) much of the knowl(cid:3)

edge about the structure of the graph that is produced by

Fig(cid:8) (cid:15)(cid:17)(cid:8) Pseudo(cid:2)code for a simpli(cid:19)ed generalized composition algo(cid:2)

a Graph Transformer is determined by the nature of the

rithm(cid:8) For simplifying the presentation(cid:3) we do not handle null

transitions nor implement dead end avoidance(cid:8) The two main

Graph Transformer(cid:4) but it may also depend on the value

component of the composition appear clearly here(cid:9) (cid:22)a(cid:23) the re(cid:2)

of the parameters and on the input(cid:2) It may also be interest(cid:3)

cursive function

enumerating the token tra jectories(cid:3)

simtoken(cid:2)(cid:3)

ing to consider Graph Transformer modules which attempt

and(cid:3) (cid:22)b(cid:23) the associative array

used for remembering which

map

nodes of the composed graph have been visited(cid:8)

to learn the structure of the output graph(cid:2) This might

be considered a combinatorial problem and not amenable

to Gradient(cid:3)Based Learning(cid:4) but a solution to this prob(cid:3)

lem is to generate a large graph that contains the graph

candidates as sub(cid:3)graphs(cid:4) and then select the appropriate

sub(cid:3)graph(cid:2)

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:8)(cid:9)

E(cid:2) GTN and Hidden Markov Models

arcs are simply added in order to obtain the complete out(cid:3)

GTNs can be seen as a generalization and an extension of

put graph(cid:2) The input values of the emission and transition

HMMs(cid:2) On the one hand(cid:4) the probabilistic interpretation

modules are read o(cid:9) the data structure on the input arcs

can be either kept (cid:7)with penalties being log(cid:3)probabilities(cid:8)(cid:4)

of the IOHMM Graph Transformer(cid:2) In practice(cid:4) the out(cid:3)

pushed to the (cid:5)nal decision stage (cid:7)with the di(cid:9)erence of the

put graph may be very large(cid:4) and needs not be completely

constrained forward penalty and the unconstrained forward

instantiated (cid:7)i(cid:2)e(cid:2)(cid:4) it is pruned(cid:24) only the low penalty paths

penalty being interpreted as negative log(cid:3)probabilities of

are created(cid:8)(cid:2)

label sequences(cid:8)(cid:4) or dropped altogether (cid:7)the network just

IX(cid:2) An On(cid:3)Line Handwriting Recognition System

represents a decision surface for label sequences in input

space(cid:8)(cid:2) On the other hand(cid:4) Graph Transformer Networks

Natural handwriting is often a mixture of di(cid:9)erent

extend HMMs by allowing to combine in a well(cid:3)principled

(cid:12)styles(cid:13)(cid:4) lower case printed(cid:4) upper case(cid:4) and cursive(cid:2) A

framework multiple levels of processing(cid:4) or multiple mod(cid:3)

reliable recognizer for such handwriting would greatly im(cid:3)

els (cid:7)e(cid:2)g(cid:2)(cid:4) Pereira et al(cid:2) have been using the transducer

prove interaction with pen(cid:3)based devices(cid:4) but its imple(cid:3)

framework for stacking HMMs representing di(cid:9)erent levels

mentation presents new technical challenges(cid:2) Characters

of processing in automatic speech recognition (cid:10)(cid:25)(cid:19)(cid:11)(cid:8)(cid:2)

taken in isolation can be very ambiguous(cid:4) but consider(cid:3)

Unfolding a HMM in time yields a graph that is very sim(cid:3)

able information is available from the context of the whole

ilar to our interpretation graph (cid:7)at the (cid:5)nal stage of pro(cid:3)

word(cid:2) We have built a word recognition system for pen(cid:3)

cessing of the Graph Transformer Network(cid:4) before Viterbi

based devices based on four main modules(cid:24) a preprocessor

recognition(cid:8)(cid:2) It has nodes n(cid:7)t(cid:2) i(cid:8) associated to each time

that normalizes a word(cid:4) or word group(cid:4) by (cid:5)tting a geomet(cid:3)

step t and state i in the model(cid:2) The penalty c

for an arc

i

rical model to the word structure$ a module that produces

from n(cid:7)t (cid:2) (cid:6)(cid:2) j (cid:8) to n(cid:7)t(cid:2) i(cid:8) then corresponds to the nega(cid:3)

an (cid:12)annotated image(cid:13) from the normalized pen tra jectory$

tive log(cid:3)probability of emitting observed data o

at posi(cid:3)

t

a replicated convolutional neural network that spots and

tion t and going from state j to state i in the time interval

recognizes characters$ and a GTN that interprets the net(cid:3)

(cid:7)t (cid:2) (cid:6)(cid:2) t(cid:8)(cid:2) With this probabilistic interpretation(cid:4) the for(cid:3)

works output by taking word(cid:3)level constraints into account(cid:2)

ward penalty is the negative logarithm of the likelihood of

The network and the GTN are jointly trained to minimize

whole observed data sequence (cid:7)given the model(cid:8)(cid:2)

an error measure de(cid:5)ned at the word level(cid:2)

In Section VI we mentioned that the collapsing phe(cid:3)

In this work(cid:4) we have compared a system based on

nomenon can occur when non(cid:3)discriminative loss functions

SDNNs (cid:7)such as described in Section VII(cid:8)(cid:4) and a system

are used to train neural networks(cid:30)HMM hybrid systems(cid:2)

based on Heuristic Over(cid:3)Segmentation (cid:7)such as described

With classical HMMs with (cid:5)xed preprocessing(cid:4) this prob(cid:3)

in Section V(cid:8)(cid:2) Because of the sequential nature of the infor(cid:3)

lem does not occur because the parameters of the emission

mation in the pen tra jectory (cid:7)which reveals more informa(cid:3)

and transition probability models are forced to satisfy cer(cid:3)

tion than the purely optical input from in image(cid:8)(cid:4) Heuristic

tain probabilistic constraints(cid:24) the sum or the integral of

Over(cid:3)Segmentation can be very e(cid:23)cient in proposing can(cid:3)

the probabilities of a random variable over its possible val(cid:3)

didate character cuts(cid:4) especially for non(cid:3)cursive script(cid:2)

ues must be (cid:6)(cid:2) Therefore(cid:4) when the probability of certain

events is increased(cid:4) the probability of other events must au(cid:3)

A(cid:2) Preprocessing

tomatically be decreased(cid:2) On the other hand(cid:4) if the prob(cid:3)

abilistic assumptions in an HMM (cid:7)or other probabilistic

Input normalization reduces intra(cid:3)character variability(cid:4)

model(cid:8) are not realistic(cid:4) discriminative training(cid:4) discussed

simplifying character recognition(cid:2) We have used a word

in Section VI(cid:4) can improve performance as this has been

normalization scheme (cid:10)(cid:27)(cid:14)(cid:11) based on (cid:5)tting a geometrical

clearly shown for speech recognition systems (cid:10)(cid:18)(cid:25)(cid:11)(cid:4) (cid:10)(cid:18)(cid:27)(cid:11)(cid:4) (cid:10)(cid:15)(cid:21)(cid:11)(cid:4)

model of the word structure(cid:2) Our model has four (cid:12)(cid:26)exi(cid:3)

(cid:10)(cid:6)(cid:21)(cid:20)(cid:11)(cid:4) (cid:10)(cid:6)(cid:21)(cid:25)(cid:11)(cid:2)

ble(cid:13) lines representing respectively the ascenders line(cid:4) the

The Input(cid:3)Output HMM model (cid:7)IOHMM(cid:8) (cid:10)(cid:6)(cid:21)(cid:15)(cid:11)(cid:4) (cid:10)(cid:6)(cid:21)(cid:27)(cid:11)(cid:4)

core line(cid:4) the base line and the descenders line(cid:2) The lines

is strongly related to graph transformers(cid:2) Viewed as a

are (cid:5)tted to local minima or maxima of the pen tra jectory(cid:2)

probabilistic model(cid:4) an IOHMM represents the conditional

The parameters of the lines are estimated with a modi(cid:5)ed

distribution of output sequences given input sequences (cid:7)of

version of the EM algorithm to maximize the joint prob(cid:3)

the same or a di(cid:9)erent length(cid:8)(cid:2) It is parameterized from

ability of observed points and parameter values(cid:4) using a

an emission probability module and a transition probabil(cid:3)

prior on parameters that prevents the lines from collapsing

ity module(cid:2) The emission probability module computes

on each other(cid:2)

the conditional emission probability of an output variable

The recognition of handwritten characters from a pen

(cid:7)given an input value and the value of discrete (cid:12)state(cid:13) vari(cid:3)

tra jectory on a digitizing surface is often done in the time

able(cid:8)(cid:2) The transition probability module computes condi(cid:3)

domain (cid:10)(cid:6)(cid:6)(cid:21)(cid:11)(cid:4) (cid:10)(cid:18)(cid:18)(cid:11)(cid:4) (cid:10)(cid:6)(cid:6)(cid:6)(cid:11)(cid:2) Typically(cid:4) tra jectories are nor(cid:3)

tional transition probabilities of a change in the value of

malized(cid:4) and local geometrical or dynamical features are

the (cid:12)state(cid:13) variable(cid:4) given the an input value(cid:2) Viewed as a

extracted(cid:2) The recognition may then be performed us(cid:3)

graph transformer(cid:4) it assigns an output graph (cid:7)representing

ing curve matching (cid:10)(cid:6)(cid:6)(cid:21)(cid:11)(cid:4) or other classi(cid:5)cation techniques

a probability distribution over the sequences of the output

such as TDNNs (cid:10)(cid:18)(cid:18)(cid:11)(cid:4) (cid:10)(cid:6)(cid:6)(cid:6)(cid:11)(cid:2) While these representations

variable(cid:8) to each path in the input graph(cid:2) All these output

have several advantages(cid:4) their dependence on stroke order(cid:3)

graphs have the same structure(cid:4) and the penalties on their

ing and individual writing styles makes them di(cid:23)cult to

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:8)(cid:10)

"Script"

"Script"

Viterbi Graph

Viterbi Graph

Beam Search
Transformer

Interpretation Graph

Language
Model

Compose

Recognition Graph

AMAP Graph

Recognition
Transformer

AMAP Computation

Interpretation Graph

Language
Model

Recognition Graph

Character
Model

SDNN Output

Beam Search
Transformer

Compose

Compose

SDNN
Transformer

Segmentation Graph

AMAP

Segmentation
Transformer

Normalized Word

AMAP Computation

Normalized Word

Word Normalization

Word Normalization

Fig(cid:8) (cid:18)(cid:6)(cid:8) An on(cid:2)line handwriting recognition GTN based on heuristic

Fig(cid:8) (cid:18)(cid:5)(cid:8) An on(cid:2)line handwriting recognition GTN based on Space(cid:2)

over(cid:2)segmentation

Displacement Neural Network

B(cid:2) Network Architecture

One of the best networks we found for both online and

use in high accuracy(cid:4) writer independent systems that in(cid:3)

o%ine character recognition is a (cid:15)(cid:3)layer convolutional net(cid:3)

tegrate the segmentation with the recognition(cid:2)

work somewhat similar to LeNet(cid:3)(cid:15) (cid:7)Figure (cid:14)(cid:8)(cid:4) but with

Since the intent of the writer is to produce a legible im(cid:3)

last two layers$ layer (cid:6)(cid:24) convolution with (cid:25) kernels of size

multiple input planes and di(cid:9)erent numbers of units on the

age(cid:4) it seems natural to preserve as much of the pictorial

(cid:17)x(cid:17)(cid:4) layer (cid:14)(cid:24) (cid:14)x(cid:14) sub(cid:3)sampling(cid:4) layer (cid:17)(cid:24) convolution with

nature of the signal as possible(cid:4) while at the same time ex(cid:3)

(cid:14)(cid:15) kernels of size (cid:15)x(cid:15)(cid:4) layer (cid:18) convolution with (cid:25)(cid:18) kernels

ploit the sequential information in the tra jectory(cid:2) For this

of size (cid:18)x(cid:18)(cid:4) layer (cid:15)(cid:24) (cid:14)x(cid:6) sub(cid:3)sampling(cid:4) classi(cid:5)cation layer(cid:24)

purpose we have designed a representation scheme(cid:4) called

(cid:27)(cid:15) RBF units (cid:7)one per class in the full printable ASCII

AMAP (cid:10)(cid:17)(cid:25)(cid:11)(cid:4) where pen tra jectories are represented by low(cid:3)

set(cid:8)(cid:2) The distributed codes on the output are the same as

resolution images in which each picture element contains

for LeNet(cid:3)(cid:15)(cid:4) except they are adaptive unlike with LeNet(cid:3)(cid:15)(cid:2)

information about the local properties of the tra jectory(cid:2) An

When used in the heuristic over(cid:3)segmentation system(cid:4) the

AMAP can be viewed as an (cid:12)annotated image(cid:13) in which

input to above network consisted of an AMAP with (cid:5)ve

each pixel is a (cid:15)(cid:3)element feature vector(cid:24) (cid:18) features are as(cid:3)

planes(cid:4) (cid:14)(cid:21) rows and (cid:6)(cid:25) columns(cid:2) It was determined that

sociated to four orientations of the pen tra jectory in the

this resolution was su(cid:23)cient for representing handwritten

area around the pixel(cid:4) and the (cid:5)fth one is associated to

characters(cid:2) In the SDNN version(cid:4) the number of columns

local curvature in the area around the pixel(cid:2) A particu(cid:3)

was varied according to the width of the input word(cid:2) Once

larly useful feature of the AMAP representation is that it

the number of sub(cid:3)sampling layers and the sizes of the ker(cid:3)

makes very few assumptions about the nature of the input

nels are chosen(cid:4) the sizes of all the layers(cid:4) including the

tra jectory(cid:2) It does not depend on stroke ordering or writ(cid:3)

input(cid:4) are determined unambiguously(cid:2) The only architec(cid:3)

ing speed(cid:4) and it can be used with all types of handwriting

tural parameters that remain to be selected are the num(cid:3)

(cid:7)capital(cid:4) lower case(cid:4) cursive(cid:4) punctuation(cid:4) symbols(cid:8)(cid:2) Un(cid:3)

ber of feature maps in each layer(cid:4) and the information as

like many other representations (cid:7)such as global features(cid:8)(cid:4)

to what feature map is connected to what other feature

AMAPs can be computed for complete words without re(cid:3)

map(cid:2) In our case(cid:4) the sub(cid:3)sampling rates were chosen as

quiring segmentation(cid:2)

small as possible (cid:7)(cid:14)x(cid:14)(cid:8)(cid:4) and the kernels as small as pos(cid:3)

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:8)(cid:12)

sible in the (cid:5)rst layer (cid:7)(cid:17)x(cid:17)(cid:8) to limit the total number of

In this application(cid:4) the language model simply constrains

connections(cid:2) Kernel sizes in the upper layers are chosen to

the (cid:5)nal output graph to represent sequences of character

be as small as possible while satisfying the size constraints

labels from a given dictionary(cid:2) Furthermore(cid:4) the interpre(cid:3)

mentioned above(cid:2) Larger architectures did not necessarily

tation graph is not actually completely instantiated(cid:24) the

perform better and required considerably more time to be

only nodes created are those that are needed by the Beam

trained(cid:2) A very small architecture with half the input (cid:5)eld

Search module(cid:2) The interpretation graph is therefore rep(cid:3)

also performed worse(cid:4) because of insu(cid:23)cient input resolu(cid:3)

resented procedurally rather than explicitly(cid:2)

tion(cid:2) Note that the input resolution is nonetheless much

A crucial contribution of this research was the joint train(cid:3)

less than for optical character recognition(cid:4) because the an(cid:3)

ing of all graph transformer modules within the network

gle and curvature provide more information than would a

with respect to a single criterion(cid:4) as explained in Sec(cid:3)

single grey level at each pixel(cid:2)

tions VI and VIII(cid:2) We used the Discriminative Forward loss

C(cid:2) Network Training

penalty of the constrained interpretation (cid:7)i(cid:2)e(cid:2)(cid:4) along all the

function on the (cid:5)nal output graph(cid:24) minimize the forward

Training proceeded in two phases(cid:2) First(cid:4) we kept the

(cid:12)correct(cid:13) paths(cid:8) while maximizing the forward penalty of

centers of the RBFs (cid:5)xed(cid:4) and trained the network weights

the whole interpretation graph (cid:7)i(cid:2)e(cid:2)(cid:4) along all the paths(cid:8)(cid:2)

so as to minimize the output distance of the RBF unit

During global training(cid:4) the loss function was optimized

corresponding to the correct class(cid:2) This is equivalent to

with the stochastic diagonal Levenberg(cid:3)Marquardt proce(cid:3)

minimizing the mean(cid:3)squared error between the previous

dure described in Appendix C(cid:4) that uses second derivatives

layer and the center of the correct(cid:3)class RBF(cid:2) This boot(cid:3)

to compute optimal learning rates(cid:2) This optimization op(cid:3)

strap phase was performed on isolated characters(cid:2) In the

erates on al l the parameters in the system(cid:4) most notably

second phase(cid:4) all the parameters(cid:4) network weights and RBF

the network weights and the RBF centers(cid:2)

centers were trained globally to minimize a discriminative

criterion at the word level(cid:2)

D(cid:2) Experimental Results

With the Heuristic Over(cid:3)Segmentation approach(cid:4) the

In the (cid:5)rst set of experiments(cid:4) we evaluated the general(cid:3)

GTN was composed of four main Graph Transformers(cid:24)

ization ability of the neural network classi(cid:5)er coupled with

(cid:6)(cid:2) The

performs

the

Segmentation Transformer

the word normalization preprocessing and AMAP input

Heuristic Over(cid:3)Segmentation(cid:4) and outputs the segmenta(cid:3)

representation(cid:2) All results are in writer independent mode

tion graph(cid:2) An AMAP is then computed for each image

(cid:7)di(cid:9)erent writers in training and testing(cid:8)(cid:2)

Initial train(cid:3)

attached to the arcs of this graph(cid:2)

ing on isolated characters was performed on a database of

(cid:14)(cid:2) The

applies

Character Recognition Transformer

approximately (cid:6)(cid:21)(cid:21)(cid:4)(cid:21)(cid:21)(cid:21) hand printed characters (cid:7)(cid:27)(cid:15) classes

the the convolutional network character recognizer to each

of upper case(cid:4) lower case(cid:4) digits(cid:4) and punctuation(cid:8)(cid:2) Tests

candidate segment(cid:4) and outputs the recognition graph(cid:4)

on a database of isolated characters were performed sepa(cid:3)

with penalties and classes on each arc(cid:2)

rately on the four types of characters(cid:24) upper case (cid:7)(cid:14)(cid:2)(cid:27)(cid:27)!

(cid:17)(cid:2) The

composes the recog(cid:3)

Composition Transformer

error on (cid:27)(cid:6)(cid:14)(cid:14) patterns(cid:8)(cid:4) lower case (cid:7)(cid:18)(cid:2)(cid:6)(cid:15)! error on (cid:25)(cid:14)(cid:21)(cid:6)

nition graph with a grammar graph representing a language

patterns(cid:8)(cid:4) digits (cid:7)(cid:6)(cid:2)(cid:18)! error on (cid:14)(cid:27)(cid:17)(cid:25) patterns(cid:8)(cid:4) and punc(cid:3)

model incorporating lexical constraints(cid:2)

tuation (cid:7)(cid:18)(cid:2)(cid:17)! error on (cid:25)(cid:25)(cid:6) patterns(cid:8)(cid:2) Experiments were

(cid:18)(cid:2) The

extracts a good inter(cid:3)

Beam Search Transformer

performed with the network architecture described above(cid:2)

pretation from the interpretation graph(cid:2) This task could

To enhance the robustness of the recognizer to variations

have been achieved with the usual Viterbi Transformer(cid:2)

in position(cid:4) size(cid:4) orientation(cid:4) and other distortions(cid:4) addi(cid:3)

The Beam Search algorithm however implements pruning

tional training data was generated by applying local a(cid:23)ne

strategies which are appropriate for large interpretation

transformations to the original characters(cid:2)

graphs(cid:2)

The second and third set of experiments concerned the

With the SDNN approach(cid:4) the main Graph Transformers

recognition of lower case words (cid:7)writer independent(cid:8)(cid:2) The

are the following(cid:24)

tests were performed on a database of (cid:25)(cid:25)(cid:6) words(cid:2) First

(cid:6)(cid:2) The

replicates the convolutional

we evaluated the improvements brought by the word nor(cid:3)

SDNN Transformer

network over the a whole word image(cid:4) and outputs a recog(cid:3)

malization to the system(cid:2) For the SDNN(cid:30)HMM system

nition graph that is a linear graph with class penalties for

we have to use word(cid:3)level normalization since the net(cid:3)

every window centered at regular intervals on the input

work sees one whole word at a time(cid:2) With the Heuris(cid:3)

image(cid:2)

tic Over(cid:3)Segmentation system(cid:4) and before doing any word(cid:3)

(cid:14)(cid:2) The

level training(cid:4) we obtained with character(cid:3)level normaliza(cid:3)

Character(cid:2)Level Composition Transformer

composes the recognition graph with a left(cid:3)to(cid:3)right HMM

tion (cid:20)(cid:2)(cid:17)! and (cid:17)(cid:2)(cid:15)! word and character errors (cid:7)adding in(cid:3)

for each character class (cid:7)as in Figure (cid:14)(cid:20)(cid:8)(cid:2)

sertions(cid:4) deletions and substitutions(cid:8) when the search was

(cid:17)(cid:2) The

com(cid:3)

constrained within a (cid:14)(cid:15)(cid:18)(cid:19)(cid:6)(cid:3)word dictionary(cid:2) When using

Word(cid:2)Level Composition Transformer

poses the output of the previous transformer with a lan(cid:3)

the word normalization preprocessing instead of a charac(cid:3)

guage model incorporating lexical constraints(cid:4) and outputs

ter level normalization(cid:4) error rates dropped to (cid:18)(cid:2)(cid:19)! and

the interpretation graph(cid:2)

(cid:14)(cid:2)(cid:21)! for word and character errors respectively(cid:4) i(cid:2)e(cid:2)(cid:4) a rel(cid:3)

(cid:18)(cid:2) The

extracts a good in(cid:3)

ative drop of (cid:17)(cid:20)! and (cid:18)(cid:17)! in word and character error

Beam Search Transformer

terpretation from the interpretation graph(cid:2)

respectively(cid:2) This suggests that normalizing the word in

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:8)(cid:13)

its entirety is better than (cid:5)rst segmenting it and then nor(cid:3)

the system as (cid:5)(cid:11)(cid:12) correct (cid:13) (cid:14)(cid:15)(cid:12) reject (cid:13) (cid:16)(cid:12) error(cid:2) The

malizing and processing each of the segments(cid:2)

system presented here was one of the (cid:5)rst to cross that

SDNN/HMM

no global training

with global training

HOS
no global training

with global training

HOS
no global training

with global training

   No Language Model

   No Language Model

   25K Word Lexicon

2

1.4

8.2

8.5

6.3

12.4

sonal checks(cid:2)

threshold on representative mixtures of business and per(cid:3)

Checks contain at least two versions of the amount(cid:2) The

Courtesy amount is written with numerals(cid:4) while the Legal

amount is written with letters(cid:2) On business checks(cid:4) which

are generally machine(cid:3)printed(cid:4) these amounts are relatively

easy to read(cid:4) but quite di(cid:23)cult to (cid:5)nd due to the lack of

0

5

10

15

standard for business check layout(cid:2) On the other hand(cid:4)

these amounts on personal checks are easy to (cid:5)nd but much

Fig(cid:8) (cid:18)(cid:15)(cid:8) Comparative results (cid:22)character error rates(cid:23) showing the

harder to read(cid:2)

improvement brought by global training on the SDNN(cid:28)HMM

hybrid(cid:3) and on the Heuristic Over(cid:2)Segmentation system (cid:22)HOS(cid:23)(cid:3)

For simplicity (cid:7)and speed requirements(cid:8)(cid:4) our initial task

without and with a (cid:15)(cid:20)(cid:21)(cid:14)(cid:5) words dictionary(cid:8)

is to read the Courtesy amount only(cid:2) This task consists of

two main steps(cid:24)

In the third set of experiments(cid:4) we measured the im(cid:3)

(cid:0)

The system has to (cid:5)nd(cid:4) among all the (cid:5)elds (cid:7)lines of

provements obtained with the joint training of the neural

text(cid:8)(cid:4) the candidates that are the most likely to contain the

network and the post(cid:3)processor with the word(cid:3)level crite(cid:3)

courtesy amount(cid:2) This is obvious for many personal checks(cid:4)

rion(cid:4) in comparison to training based only on the errors

where the position of the amount is standardized(cid:2) However(cid:4)

performed at the character level(cid:2) After initial training on

as already noted(cid:4) (cid:5)nding the amount can be rather di(cid:23)cult

individual characters as above(cid:4) global word(cid:3)level discrim(cid:3)

in business checks(cid:4) even for the human eye(cid:2) There are

inative training was performed with a database of (cid:17)(cid:15)(cid:21)(cid:21)

many strings of digits(cid:4) such as the check number(cid:4) the date(cid:4)

lower case words(cid:2) For the SDNN(cid:30)HMM system(cid:4) without

or even (cid:12)not to exceed(cid:13) amounts(cid:4) that can be confused

any dictionary constraints(cid:4) the error rates dropped from

with the actual amount(cid:2) In many cases(cid:4) it is very di(cid:23)cult

(cid:17)(cid:25)! and (cid:6)(cid:14)(cid:2)(cid:18)! word and character error to (cid:14)(cid:19)! and (cid:25)(cid:2)(cid:14)!

to decide which candidate is the courtesy amount before

respectively after word(cid:3)level training(cid:4) i(cid:2)e(cid:2)(cid:4) a relative drop

performing a full recognition(cid:2)

of (cid:17)(cid:14)! and (cid:17)(cid:18)!(cid:2) For the Heuristic Over(cid:3)Segmentation sys(cid:3)

(cid:0)

In order to read (cid:7)and choose(cid:8) some Courtesy amount

tem and a slightly improved architecture(cid:4) without any dic(cid:3)

candidates(cid:4) the system has to segment the (cid:5)elds into char(cid:3)

tionary constraints(cid:4) the error rates dropped from (cid:14)(cid:14)(cid:2)(cid:15)!

acters(cid:4) read and score the candidate characters(cid:4) and (cid:5)nally

and (cid:25)(cid:2)(cid:15)! word and character error to (cid:6)(cid:20)! and (cid:19)(cid:2)(cid:17)! re(cid:3)

(cid:5)nd the best interpretation of the amount using contextual

spectively(cid:4) i(cid:2)e(cid:2)(cid:4) a relative drop of (cid:14)(cid:18)(cid:2)(cid:18)! and (cid:14)(cid:15)(cid:2)(cid:19)!(cid:2) With a

knowledge represented by a stochastic grammar for check

(cid:14)(cid:15)(cid:18)(cid:19)(cid:6)(cid:3)word dictionary(cid:4) errors dropped from (cid:18)(cid:2)(cid:19)! and (cid:14)(cid:2)(cid:21)!

amounts(cid:2)

word and character errors to (cid:17)(cid:2)(cid:14)! and (cid:6)(cid:2)(cid:18)! respectively

The GTN methodology was used to build a check amount

after word(cid:3)level training(cid:4) i(cid:2)e(cid:2)(cid:4) a relative drop of (cid:17)(cid:21)(cid:2)(cid:18)! and

reading system that handles both personal checks and busi(cid:3)

(cid:17)(cid:21)(cid:2)(cid:21)!(cid:2) Even lower error rates can be obtained by dras(cid:3)

ness checks(cid:2)

tically reducing the size of the dictionary to (cid:17)(cid:15)(cid:21) words(cid:4)

yielding (cid:6)(cid:2)(cid:19)! and (cid:21)(cid:2)(cid:27)(cid:18)! word and character errors(cid:2)

A(cid:2) A GTN for Check Amount Recognition

These results clearly demonstrate the usefulness of glob(cid:3)

ally trained Neural(cid:3)Net(cid:30)HMM hybrids for handwriting

We now describe the successive graph transformations

recognition(cid:2) This con(cid:5)rms similar results obtained earlier

that allow this network to read the check amount (cid:7)cf(cid:2) Fig(cid:3)

in speech recognition (cid:10)(cid:20)(cid:20)(cid:11)(cid:2)

ure (cid:17)(cid:17)(cid:8)(cid:2) Each Graph Transformer produces a graph whose

X(cid:2) A Check Reading System

at this stage of the system(cid:2)

paths encode and score the current hypotheses considered

This section describes a GTN based Check Reading Sys(cid:3)

The input to the system is a trivial graph with a single

tem(cid:4) intended for immediate industrial deployment(cid:2) It also

arc that carries the image of the whole check (cid:7)cf(cid:2) Figure (cid:17)(cid:17)(cid:8)(cid:2)

shows how the use of Gradient Based(cid:3)Learning and GTNs

T

(cid:5)rst performs

f ield

The (cid:3)eld location transformer

make this deployment fast and cost(cid:3)e(cid:9)ective while yielding

classical image analysis (cid:7)including connected component

an accurate and reliable solution(cid:2)

analysis(cid:4)

ink density histograms(cid:4) layout analysis(cid:4) etc(cid:2)(cid:2)(cid:2)(cid:8)

The veri(cid:5)cation of the amount on a check is a task that

and heuristically extracts rectangular zones that may con(cid:3)

is extremely time and money consuming for banks(cid:2) As a

tain the check amount(cid:2) T

produces an output graph(cid:4)

f ield

consequence(cid:4) there is a very high interest in automating the

called the (cid:4)eld graph (cid:7)cf(cid:2) Figure (cid:17)(cid:17)(cid:8) such that each can(cid:3)

process as much as possible (cid:7)see for example (cid:10)(cid:6)(cid:6)(cid:14)(cid:11)(cid:4) (cid:10)(cid:6)(cid:6)(cid:17)(cid:11)(cid:4)

didate zone is associated with one arc that links the start

(cid:10)(cid:6)(cid:6)(cid:18)(cid:11)(cid:8)(cid:2) Even a partial automation would result in consid(cid:3)

node to the end node(cid:2) Each arc contains the image of the

erable cost reductions(cid:2) The threshold of economic viability

zone(cid:4) and a penalty term computed from simple features

for automatic check readers(cid:4) as set by the bank(cid:4) is when

extracted from the zone (cid:7)absolute position(cid:4) size(cid:4) aspect ra(cid:3)

(cid:15)(cid:21)! of the checks are read with less than (cid:6)! error(cid:2) The

tio(cid:4) etc(cid:2)(cid:2)(cid:2)(cid:8)(cid:2) The penalty term is close to zero if the features

other (cid:15)(cid:21)! of the check being rejected and sent to human

suggest that the (cid:5)eld is a likely candidate(cid:4) and is large if

operators(cid:2) In such a case(cid:4) we describe the performance of

the (cid:5)eld is deemed less likely to be an amount(cid:2) The penalty

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:8)(cid:6)

Viterbi Answer

The segmenter uses a variety of heuristics to (cid:5)nd candi(cid:3)

date cut(cid:2) One of the most important ones is called (cid:12)hit and

de(cid:26)ect(cid:13) (cid:10)(cid:6)(cid:6)(cid:15)(cid:11)(cid:2) The idea is to cast lines downward from the

Viterbi Transformer

top of the (cid:5)eld image(cid:2) When a line hits a black pixel(cid:4) it is

de(cid:26)ected so as to follow the contour of the ob ject(cid:2) When a

Best Amount Graph

Interpretation Graph

"$" 0.2
"*" 0.4
"3" 0.1
.......

"$" 0.2
"*" 0.4
"3" 0.1
"B" 23.6
.......

Grammar

Compose

Recognition Graph

Segmentation Graph

Recognition
Transformer

$

*

**

3

45

Segmentation Transf.

Field Graph

45/xx

$ *** 3.45

$10,000.00

Check Graph

Field Location Transf.

2nd Nat. Bank

not to exceed $10,000.00

$ *** 3.45

three dollars and 45/xx

line hits a local minimum of the upper pro(cid:5)le(cid:4) i(cid:2)e(cid:2) when it

cannot continue downward without crossing a black pixel(cid:4)

it is just propagated vertically downward through the ink(cid:2)

When two such lines meet each other(cid:4) they are merged into

a single cut(cid:2) The procedure can be repeated from the bot(cid:3)

tom up(cid:2) This strategy allows the separation of touching

characters such as double zeros(cid:2)

The recognition transformer

T

iterates over all

rec

segment arcs in the segmentation graph and runs a charac(cid:3)

ter recognizer on the corresponding segment image(cid:2) In our

case(cid:4) the recognizer is LeNet(cid:3)(cid:15)(cid:4) the Convolutional Neural

Network described in Section II(cid:4) whose weights constitute

the largest and most important subset of tunable parame(cid:3)

ters(cid:2) The recognizer classi(cid:5)es segment images into one of

(cid:27)(cid:15) classes (cid:7)full printable ASCII set(cid:8) plus a rubbish class for

unknown symbols or badly(cid:3)formed characters(cid:2) Each arc in

the input graph T

is replaced by (cid:27)(cid:19) arcs in the output

rec

graph(cid:2) Each of those (cid:27)(cid:19) arcs contains the label of one of

Fig(cid:8) (cid:18)(cid:18)(cid:8) A complete check amount reader implemented as a single

the classes(cid:4) and a penalty that is the sum of the penalty

cascade of Graph Transformer modules(cid:8) Successive graph trans(cid:2)

of the corresponding arc in the input (cid:7)segmentation(cid:8) graph

formations progressively extract higher level information(cid:8)

and the penalty associated with classifying the image in

the corresponding class(cid:4) as computed by the recognizer(cid:2) In

other words(cid:4) the recognition graph represents a weighted

function is di(cid:9)erentiable(cid:4) therefore its parameter are glob(cid:3)

trellis of scored character classes(cid:2) Each path in this graph

ally tunable(cid:2)

represents a possible character string for the correspond(cid:3)

An arc may represent separate dollar and cent amounts

ing (cid:5)eld(cid:2) We can compute a penalty for this interpretation

as a sequence of (cid:5)elds(cid:2) In fact(cid:4) in handwritten checks(cid:4) the

by adding the penalties along the path(cid:2) This sequence of

cent amount may be written over a fractional bar(cid:4) and not

characters may or may not be a valid check amount(cid:2)

aligned at all with the dollar amount(cid:2) In the worst case(cid:4)

The composition transformer

T

selects the

gram

one may (cid:5)nd several cent amount candidates (cid:7)above and

paths of the recognition graph that represent valid char(cid:3)

below the fraction bar(cid:8) for the same dollar amount(cid:2)

acter sequences for check amounts(cid:2) This transformer takes

The segmentation transformer

T

(cid:4) similar to the

seg

two graphs as input(cid:24) the recognition graph(cid:4) and the gram(cid:3)

one described in Section VIII examines each zone contained

mar graph(cid:2) The grammar graph contains all possible se(cid:3)

in the (cid:5)eld graph(cid:4) and cuts each image into pieces of ink

quences of symbols that constitute a well(cid:3)formed amount(cid:2)

using heuristic image processing techniques(cid:2) Each piece

The output of the composition transformer(cid:4) called the in(cid:3)

of ink may be a whole character or a piece of character(cid:2)

terpretation graph(cid:4) contains all the paths in the recognition

Each arc in the (cid:5)eld graph is replaced by its correspond(cid:3)

graph that are compatible with the grammar(cid:2) The oper(cid:3)

ing segmentation graph that represents all possible group(cid:3)

ation that combines the two input graphs to produce the

ings of pieces of ink(cid:2) Each (cid:5)eld segmentation graph is ap(cid:3)

output is a generalized transduction (cid:7)see Section VIII(cid:8)(cid:2)A

pended to an arc that contains the penalty of the (cid:5)eld in

di(cid:9)erentiable function is used to compute the data attached

the (cid:5)eld graph(cid:2) Each arc carries the segment image(cid:4) to(cid:3)

to the output arc from the data attached to the input arcs(cid:2)

gether with a penalty that provides a (cid:5)rst evaluation of

In our case(cid:4) the output arc receives the class label of the

the likelihood that the segment actually contains a charac(cid:3)

two arcs(cid:4) and a penalty computed by simply summing the

ter(cid:2) This penalty is obtained with a di(cid:9)erentiable function

penalties of the two input arcs (cid:7)the recognizer penalty(cid:4) and

that combines a few simple features such as the space be(cid:3)

the arc penalty in the grammar graph(cid:8)(cid:2) Each path in the

tween the pieces of ink or the compliance of the segment

interpretation graph represents one interpretation of one

image with a global baseline(cid:4) and a few tunable parame(cid:3)

segmentation of one (cid:5)eld on the check(cid:2) The sum of the

ters(cid:2) The segmentation graph represents al l the possible

penalties along the path represents the (cid:12)badness(cid:13) of the

segmentations of al l the (cid:5)eld images(cid:2) We can compute the

corresponding interpretation and combines evidence from

penalty for one segmented (cid:5)eld by adding the arc penalties

each of the modules along the process(cid:4) as well as from the

along the corresponding path(cid:2) As before using a di(cid:9)eren(cid:3)

grammar(cid:2)

tiable function for computing the penalties will ensure that

(cid:5)nally selects the path with

The Viterbi transformer

the parameters can be optimized globally(cid:2)

the lowest accumulated penalty(cid:4) corresponding to the best

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:8)(cid:5)

Edforw

+

−

Cdforw

Cforw

Forward  Scorer

as described in Figure (cid:14)(cid:6)(cid:4) using as our desired sequence the

Viterbi answer(cid:2) This is summarized in Figure (cid:17)(cid:18)(cid:4) with(cid:24)

con(cid:5)dence (cid:16) exp(cid:7)E

(cid:8)

dforw

Forward Scorer

D(cid:2) Results

Viterbi
Answer

Path Selector

Interpretation Graph

A version of the above system was fully implemented

and tested on machine(cid:3)print business checks(cid:2) This sys(cid:3)

tem is basically a generic GTN engine with task speci(cid:5)c

heuristics encapsulated in the

and

method(cid:2)

check

fprop

As a consequence(cid:4) the amount of code to write was min(cid:3)

Fig(cid:8) (cid:18)(cid:21)(cid:8) Additional processing required to compute the con(cid:19)dence(cid:8)

imal(cid:24) mostly the adaptation of an earlier segmenter into

the segmentation transformer(cid:2) The system that deals with

hand(cid:3)written or personal checks was based on earlier im(cid:3)

grammatically correct interpretations(cid:2)

plementations that used the GTN concept in a restricted

B(cid:2) Gradient(cid:3)Based Learning

way(cid:2)

The neural network classi(cid:5)er was initially trained on

Each stage of this check reading system contains tun(cid:3)

(cid:15)(cid:21)(cid:21)(cid:4)(cid:21)(cid:21)(cid:21) images of character images from various origins

able parameters(cid:2) While some of these parameters could be

spanning the entire printable ASCII set(cid:2) This contained

manually adjusted(cid:4) for example the parameters of the (cid:5)eld

both handwritten and machine(cid:3)printed characters that had

locator and segmenter(cid:4) the vast ma jority of them must be

been previously size normalized at the string level(cid:2) Addi(cid:3)

learned(cid:4) particularly the weights of the neural net recog(cid:3)

tional images were generated by randomly distorting the

nizer(cid:2)

original images using simple a(cid:23)ne transformations of the

Prior to globally optimizing the system(cid:4) each module pa(cid:3)

images(cid:2) The network was then further trained on character

rameters are initialized with reasonable values(cid:2) The param(cid:3)

images that had been automatically segmented from check

eters of the (cid:5)eld locator and the segmenter are initialized

images and manually truthed(cid:2) The network was also ini(cid:3)

by hand(cid:4) while the parameters of the neural net charac(cid:3)

tially trained to reject non(cid:3)characters that resulted from

ter recognizer are initialized by training on a database of

segmentation errors(cid:2) The recognizer was then inserted in

pre(cid:3)segmented and labeled characters(cid:2) Then(cid:4) the entire

the check reading system and a small subset of the parame(cid:3)

system is trained globally from whole check images labeled

ters were trained globally (cid:7)at the (cid:5)eld level(cid:8) on whole check

with the correct amount(cid:2) No explicit segmentation of the

images(cid:2)

amounts is needed to train the system(cid:24) it is trained at the

On (cid:19)(cid:18)(cid:19) business checks that were automatically catego(cid:3)

check level(cid:2)

rized as machine printed the performance was (cid:25)(cid:14)! cor(cid:3)

The loss function E minimized by our global train(cid:3)

rectly recognized checks(cid:4) (cid:6)! errors(cid:4) and (cid:6)(cid:20)! rejects(cid:2) This

ing procedure is the Discriminative Forward criterion de(cid:3)

can be compared to the performance of the previous sys(cid:3)

scribed in Section VI(cid:24) the di(cid:9)erence between (cid:7)a(cid:8) the for(cid:3)

tem on the same test set(cid:24) (cid:19)(cid:25)! correct(cid:4) (cid:6)! errors(cid:4) and

ward penalty of the constrained interpretation graph (cid:7)con(cid:3)

(cid:17)(cid:6)! rejects(cid:2) A check is categorized as machine(cid:3)printed

strained by the correct label sequence(cid:8)(cid:4) and (cid:7)b(cid:8) the forward

when characters that are near a standard position Dollar

penalty of the unconstrained interpretation graph(cid:2) Deriva(cid:3)

sign are detected as machine printed(cid:4) or when(cid:4) if nothing

tives can be back(cid:3)propagated through the entire structure(cid:4)

is found in the standard position(cid:4) at least one courtesy

although it only practical to do it down to the segmenter(cid:2)

amount candidate is found somewhere else(cid:2) The improve(cid:3)

C(cid:2) Rejecting Low Con(cid:4)dence Checks

network recognizer was bigger(cid:4) and trained on more data(cid:2)

ment is attributed to three main causes(cid:2) First the neural

In order to be able to reject checks which are the most

Second(cid:4) because of the GTN architecture(cid:4) the new system

likely to carry erroneous Viterbi answers(cid:4) we must rate

could take advantage of grammatical constraints in a much

them with a con(cid:4)dence(cid:4) and reject the check if this con(cid:3)

more e(cid:23)cient way than the previous system(cid:2) Third(cid:4) the

(cid:5)dence is below a given threshold(cid:2) To compare the un(cid:3)

GTN architecture provided extreme (cid:26)exibility for testing

normalized Viterbi Penalties of two di(cid:9)erent checks would

heuristics(cid:4) adjusting parameters(cid:4) and tuning the system(cid:2)

be meaningless when it comes to decide which answer we

This last point is more important than it seems(cid:2) The GTN

trust the most(cid:2)

framework separates the (cid:12)algorithmic(cid:13) part of the system

The optimal measure of con(cid:5)dence is the probability of

from the (cid:12)knowledge(cid:3)based(cid:13) part of the system(cid:4) allowing

the Viterbi answer given the input image(cid:2) As seen in Sec(cid:3)

easy adjustments of the latter(cid:2) The importance of global

tion VI(cid:3)E(cid:4) given a target sequence (cid:7)which(cid:4) in this case(cid:4)

training was only minor in this task because the global

would be the Viterbi answer(cid:8)(cid:4) the discriminative forward

training only concerned a small subset of the parameters(cid:2)

loss function is an estimate of the logarithm of this prob(cid:3)

An independent test performed by systems integrators

ability(cid:2) Therefore(cid:4) a simple solution to obtain a good esti(cid:3)

in (cid:6)(cid:27)(cid:27)(cid:15) showed the superiority of this system over other

mate of the con(cid:5)dence is to reuse the interpretation graph

commercial Courtesy amount reading systems(cid:2) The system

(cid:7)see Figure (cid:17)(cid:17)(cid:8) to compute the discriminative forward loss

was integrated in NCR(cid:28)s line of check reading systems(cid:2) It

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:9)(cid:11)

has been (cid:5)elded in several banks across the US since June

Neural Networks allows to learn appropriate features from

(cid:6)(cid:27)(cid:27)(cid:19)(cid:4) and has been reading millions of checks per day since

examples(cid:2) The success of this approach was demonstrated

then(cid:2)

in extensive comparative digit recognition experiments on

XI(cid:2) Conclusions

the NIST database(cid:2)

(cid:14)(cid:2) Segmentation and recognition of ob jects in images can(cid:3)

During the short history of automatic pattern recogni(cid:3)

not be completely decoupled(cid:2) Instead of taking hard seg(cid:3)

tion(cid:4) increasing the role of learning seems to have invari(cid:3)

mentation decisions too early(cid:4) we have used Heuristic Over(cid:3)

ably improved the overall performance of recognition sys(cid:3)

Segmentation to generate and evaluate a large number of

tems(cid:2) The systems described in this paper are more ev(cid:3)

hypotheses in parallel(cid:4) postponing any decision until the

idence to this fact(cid:2) Convolutional Neural Networks have

overall criterion is minimized(cid:2)

been shown to eliminate the need for hand(cid:3)crafted fea(cid:3)

(cid:17)(cid:2) Hand truthing images to obtain segmented characters

ture extractors(cid:2) Graph Transformer Networks have been

for training a character recognizer is expensive and does

shown to reduce the need for hand(cid:3)crafted heuristics(cid:4) man(cid:3)

not take into account the way in which a whole document

ual labeling(cid:4) and manual parameter tuning in document

or sequence of characters will be recognized (cid:7)in particular

recognition systems(cid:2) As training data becomes plentiful(cid:4) as

the fact that some segmentation candidates may be wrong(cid:4)

computers get faster(cid:4) as our understanding of learning al(cid:3)

even though they may look like true characters(cid:8)(cid:2) Instead

gorithms improves(cid:4) recognition systems will rely more and

we train multi(cid:3)module systems to optimize a global mea(cid:3)

more of learning(cid:4) and their performance will improve(cid:2)

sure of performance(cid:4) which does not require time consum(cid:3)

Just as the back(cid:3)propagation algorithm elegantly solved

ing detailed hand(cid:3)truthing(cid:4) and yields signi(cid:5)cantly better

the credit assignment problem in multi(cid:3)layer neural net(cid:3)

recognition performance(cid:4) because it allows to train these

works(cid:4) the gradient(cid:3)based learning procedure for Graph

modules to cooperate towards a common goal(cid:2)

Transformer Networks introduced in this paper solves the

(cid:18)(cid:2) Ambiguities inherent in the segmentation(cid:4) character

credit assignment problem in systems whose functional ar(cid:3)

recognition(cid:4) and linguistic model should be integrated op(cid:3)

chitecture dynamically changes with each new input(cid:2) The

timally(cid:2)

Instead of using a sequence of task(cid:3)dependent

learning algorithms presented here are in a sense nothing

heuristics to combine these sources of information(cid:4) we

more than unusual forms of gradient descent in complex(cid:4)

have proposed a uni(cid:5)ed framework in which generalized

dynamic architectures(cid:4) with e(cid:23)cient back(cid:3)propagation al(cid:3)

transduction methods are applied to graphs representing a

gorithms to compute the gradient(cid:2) The results in this pa(cid:3)

weighted set of hypotheses about the input(cid:2) The success of

per help establish the usefulness and relevance of gradient(cid:3)

this approach was demonstrated with a commercially de(cid:3)

based minimization methods as a general organizing prin(cid:3)

ployed check reading system that reads millions of business

ciple for learning in large systems(cid:2)

and personal checks per day(cid:24) the generalized transduction

It was shown that all the steps of a document analysis

engine resides in only a few hundred lines of code(cid:2)

system can be formulated as graph transformers through

(cid:15)(cid:2) Traditional recognition systems rely on many hand(cid:3)

which gradients can be back(cid:3)propagated(cid:2) Even in the

crafted heuristics to isolate individually recognizable ob(cid:3)

non(cid:3)trainable parts of the system(cid:4) the design philosophy

jects(cid:2) The promising Space Displacement Neural Network

in terms of graph transformation provides a clear separa(cid:3)

approach draws on the robustness and e(cid:23)ciency of Con(cid:3)

tion between domain(cid:3)speci(cid:5)c heuristics (cid:7)e(cid:2)g(cid:2) segmentation

volutional Neural Networks to avoid explicit segmentation

heuristics(cid:8) and generic(cid:4) procedural knowledge (cid:7)the gener(cid:3)

altogether(cid:2) Simultaneous automatic learning of segmenta(cid:3)

alized transduction algorithm(cid:8)

tion and recognition can be achieved with Gradient(cid:3)Based

It is worth pointing out that data generating models

Learning methods(cid:2)

(cid:7)such as HMMs(cid:8) and the Maximum Likelihood Principle

This paper presents a small number of examples of graph

were not called upon to justify most of the architectures

transformer modules(cid:4) but it is clear that the concept can be

and the training criteria described in this paper(cid:2) Gradient

applied to many situations where the domain knowledge or

based learning applied to global discriminative loss func(cid:3)

the state information can be represented by graphs(cid:2) This is

tions guarantees optimal classi(cid:5)cation and rejection with(cid:3)

the case in many audio signal recognition tasks(cid:4) and visual

out the use of (cid:12)hard to justify(cid:13) principles that put strong

scene analysis applications(cid:2) Future work will attempt to

constraints on the system architecture(cid:4) often at the expense

apply Graph Transformer Networks to such problems(cid:4) with

of performances(cid:2)

the hope of allowing more reliance on automatic learning(cid:4)

More speci(cid:5)cally(cid:4) the methods and architectures pre(cid:3)

and less on detailed engineering(cid:2)

sented in this paper o(cid:9)er generic solutions to a large num(cid:3)

Appendices

ber of problems encountered in pattern recognition sys(cid:3)

tems(cid:24)

A(cid:2) Pre(cid:3)conditions for faster convergence

(cid:6)(cid:2) Feature extraction is traditionally a (cid:5)xed transform(cid:4)

As seen before(cid:4) the squashing function used in our Con(cid:3)

generally derived from some expert prior knowledge about

volutional Networks is f (cid:7)a(cid:8) (cid:16) A tanh(cid:7)S a(cid:8)(cid:2) Symmetric

the task(cid:2) This relies on the probably incorrect assumption

functions are believed to yield faster convergence(cid:4) although

that the human designer is able to capture all the rele(cid:3)

the learning can become extremely slow if the weights are

vant information in the input(cid:2) We have shown that the

too small(cid:2) The cause of this problem is that in weight space

application of Gradient(cid:3)Based Learning to Convolutional

the origin is a (cid:5)xed point of the learning dynamics(cid:4) and(cid:4)

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:9)(cid:4)

although it is a saddle point(cid:4) it is attractive in almost all

performing two complete learning iterations over the small

directions (cid:10)(cid:6)(cid:6)(cid:19)(cid:11)(cid:2) For our simulations(cid:4) we use A (cid:16) (cid:6)(cid:3)(cid:20)(cid:6)(cid:15)(cid:27)

subset(cid:2) This idea can be generalized to training sets where

and S (cid:16)

(cid:7)see (cid:10)(cid:14)(cid:21)(cid:11)(cid:4) (cid:10)(cid:17)(cid:18)(cid:11)(cid:8)(cid:2) With this choice of parame(cid:3)

there exist no precise repetition of the same pattern but

(cid:8)

(cid:7)

ters(cid:4) the equalities f (cid:7)(cid:6)(cid:8) (cid:16) (cid:6) and f (cid:7)(cid:2)(cid:6)(cid:8) (cid:16) (cid:2)(cid:6) are satis(cid:5)ed(cid:2)

where some redundancy is present(cid:2) In fact stochastic up(cid:3)

The rationale behind this is that the overall gain of the

date must be better when there is redundancy(cid:4) i(cid:2)e(cid:2)(cid:4) when a

squashing transformation is around (cid:6) in normal operat(cid:3)

certain level of generalization is expected(cid:2)

ing conditions(cid:4) and the interpretation of the state of the

Many authors have claimed that second(cid:3)order meth(cid:3)

network is simpli(cid:5)ed(cid:2) Moreover(cid:4) the absolute value of the

ods should be used in lieu of gradient descent for neu(cid:3)

second derivative of f is a maximum at (cid:22)(cid:6) and (cid:2)(cid:6)(cid:4) which

ral net training(cid:2) The literature abounds with recom(cid:3)

improves the convergence towards the end of the learning

mendations (cid:10)(cid:6)(cid:6)(cid:25)(cid:11) for classical second(cid:3)order methods such

session(cid:2) This particular choice of parameters is merely a

as the Gauss(cid:3)Newton or Levenberg(cid:3)Marquardt algorithms(cid:4)

convenience(cid:4) and does not a(cid:9)ect the result(cid:2)

for Quasi(cid:3)Newton methods such as the Broyden(cid:3)Fletcher(cid:3)

Before training(cid:4) the weights are initialized with random

Goldfarb(cid:3)Shanno method (cid:7)BFGS(cid:8)(cid:4) Limited(cid:3)storage BFGS(cid:4)

values using a uniform distribution between (cid:2)(cid:14)(cid:3)(cid:18)(cid:4)F

and

i

or for various versions of the Conjugate Gradients (cid:7)CG(cid:8)

(cid:14)(cid:3)(cid:18)(cid:4)F

where F

is the number of inputs (cid:7)fan(cid:3)in(cid:8) of the unit

i

i

method(cid:2) Unfortunately(cid:4) all of the above methods are un(cid:3)

which the connection belongs to(cid:2) Since several connections

suitable for training large neural networks on large data

share a weight(cid:4) this rule could be di(cid:23)cult to apply(cid:4) but in

sets(cid:2) The Gauss(cid:3)Newton and Levenberg(cid:3)Marquardt meth(cid:3)

our case(cid:4) all connections sharing a same weight belong to

ods require O(cid:7)N

(cid:8) operations per update(cid:4) where N is

(cid:8)

units with identical fan(cid:3)ins(cid:2) The reason for dividing by the

the number of parameters(cid:4) which makes them impracti(cid:3)

fan(cid:3)in is that we would like the initial standard deviation

cal for even moderate size networks(cid:2) Quasi(cid:3)Newton meth(cid:3)

of the weighted sums to be in the same range for each

ods require (cid:12)only(cid:13) O(cid:7)N

(cid:8) operations per update(cid:4) but that

(cid:7)

unit(cid:4) and to fall within the normal operating region of the

still makes them impractical for large networks(cid:2) Limited(cid:3)

sigmoid(cid:2) If the initial weights are too small(cid:4) the gradients

Storage BFGS and Conjugate Gradient require only O(cid:7)N (cid:8)

are very small and the learning is slow(cid:2)

If they are too

operations per update so they would appear appropriate(cid:2)

large(cid:4) the sigmoids are saturated and the gradient is also

Unfortunately(cid:4) their convergence speed relies on an accu(cid:3)

very small(cid:2) The standard deviation of the weighted sum

rate evaluation of successive (cid:12)conjugate descent directions(cid:13)

scales like the square root of the number of inputs when

which only makes sense in (cid:12)batch(cid:13) mode(cid:2) For large data

the inputs are independent(cid:4) and it scales linearly with the

sets(cid:4) the speed(cid:3)up brought by these methods over regular

number of inputs if the inputs are highly correlated(cid:2) We

batch gradient descent cannot match the enormous speed

chose to assume the second hypothesis since some units

up brought by the use of stochastic gradient(cid:2) Several au(cid:3)

receive highly correlated signals(cid:2)

thors have attempted to use Conjugate Gradient with small

B(cid:2) Stochastic Gradient vs Batch Gradient

attempts have not yet been demonstrated to surpass a care(cid:3)

batches(cid:4) or batches of increasing sizes (cid:10)(cid:6)(cid:6)(cid:27)(cid:11)(cid:4) (cid:10)(cid:6)(cid:14)(cid:21)(cid:11)(cid:4) but those

Gradient(cid:3)Based Learning algorithms can use one of two

fully tuned stochastic gradient(cid:2) Our experiments were per(cid:3)

classes of methods to update the parameters(cid:2) The (cid:5)rst

formed with a stochastic method that scales the parameter

method(cid:4) dubbed (cid:12)Batch Gradient(cid:13)(cid:4) is the classical one(cid:24) the

axes so as to minimize the eccentricity of the error surface(cid:2)

gradients are accumulated over the entire training set(cid:4) and

C(cid:2) Stochastic Diagonal Levenberg(cid:3)Marquardt

the parameters are updated after the exact gradient has

been so computed(cid:2) In the second method(cid:4) called (cid:12)Stochas(cid:3)

Owing to the reasons given in Appendix B(cid:4) we prefer to

tic Gradient(cid:13)(cid:4) a partial(cid:4) or noisy(cid:4) gradient is evaluated on

update the weights after each presentation of a single pat(cid:3)

the basis of one single training sample (cid:7)or a small num(cid:3)

tern in accordance with stochastic update methods(cid:2) The

ber of samples(cid:8)(cid:4) and the parameters are updated using

patterns are presented in a constant random order(cid:4) and the

this approximate gradient(cid:2) The training samples can be

training set is typically repeated (cid:14)(cid:21) times(cid:2)

selected randomly or according to a properly randomized

Our update algorithm is dubbed the Stochastic Diagonal

sequence(cid:2) In the stochastic version(cid:4) the gradient estimates

Levenberg(cid:3)Marquardt method where an individual learning

are noisy(cid:4) but the parameters are updated much more often

rate (cid:7)step size(cid:8) is computed for each parameter (cid:7)weight(cid:8)

than with the batch version(cid:2) An empirical result of con(cid:3)

before each pass through the training set (cid:10)(cid:14)(cid:21)(cid:11)(cid:4) (cid:10)(cid:6)(cid:14)(cid:6)(cid:11)(cid:4) (cid:10)(cid:17)(cid:18)(cid:11)(cid:2)

siderable practical importance is that on tasks with large(cid:4)

These learning rates are computed using the diagonal terms

redundant data sets(cid:4) the stochastic version is considerably

of an estimate of the Gauss(cid:3)Newton approximation to the

faster than the batch version(cid:4) sometimes by orders of mag(cid:3)

Hessian (cid:7)second derivative(cid:8) matrix(cid:2) This algorithm is not

nitude (cid:10)(cid:6)(cid:6)(cid:20)(cid:11)(cid:2) Although the reasons for this are not totally

believed to bring a tremendous increase in learning speed

understood theoretically(cid:4) an intuitive explanation can be

but it converges reliably without requiring extensive ad(cid:3)

found in the following extreme example(cid:2) Let us take an

justments of the learning parameters(cid:2) It corrects ma jor ill(cid:3)

example where the training database is composed of two

conditioning of the loss function that are due to the pecu(cid:3)

copies of the same subset(cid:2) Then accumulating the gradient

liarities of the network architecture and the training data(cid:2)

over the whole set would cause redundant computations

The additional cost of using this procedure over standard

to be performed(cid:2) On the other hand(cid:4) running Stochas(cid:3)

stochastic gradient descent is negligible(cid:2)

tic Gradient once on this training set would amount to

At each learning iteration a particular parameter w

is

k

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:9)(cid:7)

updated according to the following stochastic update rule

the total input to unit i (cid:7)denoted a

(cid:8)(cid:2) Interestingly(cid:4) there is

i

w

(cid:4) w

(cid:2) (cid:7)

(cid:3)

(cid:7)(cid:6)(cid:25)(cid:8)

k

k

k

which is very similar to the back(cid:3)propagation procedure

p

(cid:8)E

an e(cid:23)cient algorithm to compute those second derivatives

(cid:8)w

k

used to compute the (cid:5)rst derivatives (cid:10)(cid:14)(cid:21)(cid:11)(cid:4) (cid:10)(cid:6)(cid:14)(cid:6)(cid:11)(cid:24)

where E

is the instantaneous loss function for pattern p(cid:2)

(cid:7)

p

(cid:7)

p

p

p

(cid:8)

E

(cid:8)

E

(cid:8)E

(cid:4)

(cid:4)(cid:4)

(cid:7)

(cid:7)

In Convolutional Neural Networks(cid:4) because of the weight

(cid:7)

(cid:7)

X

ki

(cid:16) f

(cid:7)a

(cid:8)

u

(cid:22) f

(cid:7)a

(cid:8)

(cid:7)(cid:14)(cid:19)(cid:8)

i

i

p

(cid:3)E

i

k

(cid:8) a

(cid:8) a

(cid:8) x

i

k

sharing(cid:4) the partial derivative

is the sum of the partial

(cid:3)w

k

derivatives with respect to the connections that share the

Unfortunately(cid:4) using those derivatives leads to well(cid:3)known

parameter w

(cid:24)

k

problems associated with every Newton(cid:3)like algorithm(cid:24)

p

p

(cid:8)E

(cid:8)E

(cid:16)

(cid:7)(cid:6)(cid:27)(cid:8)

X

(cid:8)w

k

(cid:8) u

ij

(cid:15)

(cid:16)

i(cid:4)j

V

k

(cid:3)

these terms can be negative(cid:4) and can cause the gradient

algorithm to move uphill instead of downhill(cid:2) Therefore(cid:4)

our second approximation is a well(cid:3)known trick(cid:4) called the

where u

is the connection weight from unit j to unit i(cid:4) V

ij

k

Gauss(cid:3)Newton approximation(cid:4) which guarantees that the

is the set of unit index pairs (cid:7)i(cid:2) j (cid:8) such that the connection

second derivative estimates are non(cid:3)negative(cid:2) The Gauss(cid:3)

between i and j share the parameter w

(cid:4) i(cid:2)e(cid:2)(cid:24)

k

Newton approximation essentially ignores the non(cid:3)linearity

u

(cid:16) w

(cid:5)(cid:7)i(cid:2) j (cid:8) (cid:6) V

(cid:7)(cid:14)(cid:21)(cid:8)

ij

k

k

of the estimated function (cid:7)the Neural Network in our case(cid:8)(cid:4)

but not that of the loss function(cid:2) The back(cid:3)propagation

As stated previously(cid:4) the step sizes (cid:7)

are not constant but

k

equation for Gauss(cid:3)Newton approximations of the second

are function of the second derivative of the loss function

derivatives is(cid:24)

along the axis w

(cid:24)

k

(cid:7)

p

(cid:7)

p

(cid:8)

E

(cid:8)

E

(cid:4)

(cid:7)

(cid:7)

(cid:7)

(cid:7)

i

X

ki

(cid:16) f

(cid:7)a

(cid:8)

u

(cid:7)(cid:14)(cid:20)(cid:8)

(cid:9)

(cid:8) a

(cid:8) a

i

k

k

(cid:7)

(cid:16)

k

(cid:7)(cid:14)(cid:6)(cid:8)

(cid:10) (cid:22) h

kk

This is very similar to the formula for back(cid:3)propagating the

where (cid:10) is a hand(cid:3)picked constant and h

is an estimate

kk

(cid:5)rst derivatives(cid:4) except that the sigmoid(cid:28)s derivative and

of the second derivative of the loss function E with re(cid:3)

the weight values are squared(cid:2) The right(cid:3)hand side is a sum

spect to w

(cid:2) The larger h

(cid:4) the smaller the weight update(cid:2)

of products of non(cid:3)negative terms(cid:4) therefore the left(cid:3)hand

k

kk

The parameter (cid:10) prevents the step size from becoming too

side term is non(cid:3)negative(cid:2)

large when the second derivative is small(cid:4) very much like

The third approximation we make is that we do not run

the (cid:12)model(cid:3)trust(cid:13) methods(cid:4) and the Levenberg(cid:3)Marquardt

the average in Equation (cid:14)(cid:18) over the entire training set(cid:4) but

methods in non(cid:3)linear optimization (cid:10)(cid:25)(cid:11)(cid:2) The exact formula

run it on a small subset of the training set instead(cid:2)

In

to compute h

from the second derivatives with respect

addition the re(cid:3)estimation does not need to be done of(cid:3)

kk

to the connection weights is(cid:24)

ten since the second order properties of the error surface

h

(cid:16)

kk

(cid:7)(cid:14)(cid:14)(cid:8)

X

X

(cid:15)

(cid:16)

(cid:15)

(cid:16)

k

k

i(cid:4)j

V

k(cid:4)l

V

(cid:3)

(cid:3)

(cid:8) u

(cid:8) u

ij

kl

(cid:7)

(cid:8)

E

change rather slowly(cid:2) In the experiments described in this

paper(cid:4) we re(cid:3)estimate the h

on (cid:15)(cid:21)(cid:21) patterns before each

kk

training pass through the training set(cid:2) Since the size of the

training set is (cid:19)(cid:21)(cid:4)(cid:21)(cid:21)(cid:21)(cid:4) the additional cost of re(cid:3)estimating

However(cid:4) we make three approximations(cid:2) The (cid:5)rst approx(cid:3)

the h

is negligible(cid:2) The estimates are not particularly

kk

imation is to drop the o(cid:9)(cid:3)diagonal terms of the Hessian

sensitive to the particular subset of the training set used in

with respect to the connection weights in the above equa(cid:3)

the averaging(cid:2) This seems to suggest that the second(cid:3)order

tion(cid:24)

properties of the error surface are mainly determined by

h

(cid:16)

kk

(cid:7)(cid:14)(cid:17)(cid:8)

the structure of the network(cid:4) rather than by the detailed

(cid:7)

(cid:8)

E

X

(cid:7)

(cid:15)

(cid:16)

i(cid:4)j

V

k

(cid:3)

(cid:8) u

ij

(cid:3)

(cid:3)

E

(cid:3)

statistics of the samples(cid:2) This algorithm is particularly use(cid:3)

ful for shared(cid:3)weight networks because the weight sharing

Naturally(cid:4) the terms

are the average over the training

creates ill(cid:3)conditionning of the error surface(cid:2) Because of

(cid:3) u

ij

set of the local second derivatives(cid:24)

the sharing(cid:4) one single parameter in the (cid:5)rst few layers can

(cid:7)

(cid:7)

p

P

(cid:8)

E

(cid:6)

(cid:8)

E

(cid:16)

(cid:7)(cid:14)(cid:18)(cid:8)

(cid:7)

(cid:7)

X

(cid:8) u

P

(cid:8) u

ij

ij

p

(cid:14)(cid:4)

have an enormous in(cid:26)uence on the output(cid:2) Consequently(cid:4)

the second derivative of the error with respect to this pa(cid:3)

rameter may be very large(cid:4) while it can be quite small for

other parameters elsewhere in the network(cid:2) The above al(cid:3)

Those local second derivatives with respect to connection

gorithm compensates for that phenomenon(cid:2)

weights can be computed from local second derivatives with

Unlike most other second(cid:3)order acceleration methods for

respect to the total input of the downstream unit(cid:24)

back(cid:3)propagation(cid:4) the above method works in stochastic

(cid:7)

(cid:7)

p

p

(cid:8)

E

(cid:8)

E

(cid:7)

(cid:7)

(cid:7)

j

(cid:16)

x

(cid:7)(cid:14)(cid:15)(cid:8)

(cid:8) u

(cid:8) a

ij

i

mode(cid:2)

It uses a diagonal approximation of the Hessian(cid:2)

Like the classical Levenberg(cid:3)Marquardt algorithm(cid:4) it uses a

(cid:12)safety(cid:13) factor (cid:10) to prevent the step sizes from getting too

large if the second derivative estimates are small(cid:2) Hence

(cid:3)

p

(cid:3)

E

(cid:3)

where x

is the state of unit j and

is the second

j

(cid:3) a

i

the method is called the Stochastic Diagonal Levenberg(cid:3)

derivative of the instantaneous loss function with respect to

Marquardt method(cid:2)

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:9)(cid:8)

Acknowledgments

E(cid:8) Bienenstock(cid:3) F(cid:8) Fogelman(cid:2)Souli(cid:12)e(cid:3) and G(cid:8) Weisbuch(cid:3) Eds(cid:8)(cid:3)

Les Houches(cid:3) France(cid:3) (cid:5)(cid:17)(cid:16)(cid:14)(cid:3) pp(cid:8) (cid:15)(cid:18)(cid:18)(cid:31)(cid:15)(cid:21)(cid:6)(cid:3) Springer(cid:2)Verlag(cid:8)

Some of the systems described in this paper is the work

(cid:26)(cid:5)(cid:17)(cid:27) D(cid:8) B(cid:8) Parker(cid:3) (cid:29)Learning(cid:2)logic(cid:3)(cid:30) Tech(cid:8) Rep(cid:8)(cid:3) TR(cid:2)(cid:21)(cid:7)(cid:3) Sloan

of many researchers now at AT&T(cid:4) and Lucent Technolo(cid:3)

School of Management(cid:3) MIT(cid:3) Cambridge(cid:3) Mass(cid:8)(cid:3) April (cid:5)(cid:17)(cid:16)(cid:20)(cid:8)

gies(cid:2) In particular(cid:4) Christopher Burges(cid:4) Craig Nohl(cid:4) Troy

nectionist learning models(cid:14)(cid:3) Ph(cid:8)D(cid:8) thesis(cid:3) Universit(cid:12)e P(cid:8) et M(cid:8)

(cid:26)(cid:15)(cid:6)(cid:27) Y(cid:8) LeCun(cid:3) Mod(cid:11)eles connexionnistes de l(cid:12)apprentissage (cid:13)con(cid:8)

Cauble and Jane Bromley contributed much to the check

Curie (cid:22)Paris (cid:14)(cid:23)(cid:3) June (cid:5)(cid:17)(cid:16)(cid:7)(cid:8)

reading system(cid:2) Experimental results described in sec(cid:3)

(cid:26)(cid:15)(cid:5)(cid:27) Y(cid:8) LeCun(cid:3) (cid:29)A theoretical framework for back(cid:2)propagation(cid:3)(cid:30) in

tion III include contributions by Chris Burges(cid:4) Aymeric

D(cid:8) Touretzky(cid:3) G(cid:8) Hinton(cid:3) and T(cid:8) Sejnowski(cid:3) Eds(cid:8)(cid:3) CMU(cid:3) Pitts(cid:2)

Proceedings of the (cid:7)(cid:15)(cid:9)(cid:9) Connectionist Models Summer School(cid:3)

Brunot(cid:4) Corinna Cortes(cid:4) Harris Drucker(cid:4) Larry Jackel(cid:4) Urs

burgh(cid:3) Pa(cid:3) (cid:5)(cid:17)(cid:16)(cid:16)(cid:3) pp(cid:8) (cid:15)(cid:5)(cid:31)(cid:15)(cid:16)(cid:3) Morgan Kaufmann(cid:8)

M"uller(cid:4) Bernhard Sch"olkopf(cid:4) and Patrice Simard(cid:2) The au(cid:3)

(cid:26)(cid:15)(cid:15)(cid:27) L(cid:8) Bottou and P(cid:8) Gallinari(cid:3) (cid:29)A framework for the cooperation of

thors wish to thank Fernando Pereira(cid:4) Vladimir Vapnik(cid:4)

cessing Systems(cid:3) D(cid:8) Touretzky and R(cid:8) Lippmann(cid:3) Eds(cid:8)(cid:3) Denver(cid:3)

learning algorithms(cid:3)(cid:30) in Advances in Neural Information Pro(cid:8)

John Denker(cid:4) and Isabelle Guyon for helpful discussions(cid:4)

(cid:5)(cid:17)(cid:17)(cid:5)(cid:3) vol(cid:8) (cid:18)(cid:3) Morgan Kaufmann(cid:8)

Charles Stenard and Ray Higgins for providing the appli(cid:3)

(cid:26)(cid:15)(cid:18)(cid:27) C(cid:8) Y(cid:8) Suen(cid:3) C(cid:8) Nadal(cid:3) R(cid:8) Legault(cid:3) T(cid:8) A(cid:8) Mai(cid:3) and L(cid:8) Lam(cid:3)

cations that motivated some of this work(cid:4) and Lawrence R(cid:2)

als(cid:3)(cid:30) Proceedings of the IEEE(cid:16) Special issue on Optical Char(cid:8)

(cid:29)Computer recognition of unconstrained handwritten numer(cid:2)

Rabiner and Lawrence D(cid:2) Jackel for relentless support and

acter Recognition(cid:3) vol(cid:8) (cid:16)(cid:6)(cid:3) no(cid:8) (cid:7)(cid:3) pp(cid:8) (cid:5)(cid:5)(cid:14)(cid:15)(cid:31)(cid:5)(cid:5)(cid:16)(cid:6)(cid:3) July (cid:5)(cid:17)(cid:17)(cid:15)(cid:8)

encouragements(cid:2)

(cid:26)(cid:15)(cid:21)(cid:27) S(cid:8) N(cid:8) Srihari(cid:3) (cid:29)High(cid:2)performance reading machines(cid:3)(cid:30) Proceed(cid:8)

ings of the IEEE(cid:16) Special issue on Optical Character Recogni(cid:8)

References

tion(cid:3) vol(cid:8) (cid:16)(cid:6)(cid:3) no(cid:8) (cid:7)(cid:3) pp(cid:8) (cid:5)(cid:5)(cid:15)(cid:6)(cid:31)(cid:5)(cid:5)(cid:18)(cid:15)(cid:3) July (cid:5)(cid:17)(cid:17)(cid:15)(cid:8)

(cid:26)(cid:15)(cid:20)(cid:27) Y(cid:8) LeCun(cid:3) L(cid:8) D(cid:8) Jackel(cid:3) B(cid:8) Boser(cid:3) J(cid:8) S(cid:8) Denker(cid:3) H(cid:8) P(cid:8) Graf(cid:3)

(cid:26)(cid:5)(cid:27)

R(cid:8) O(cid:8) Duda and P(cid:8) E(cid:8) Hart(cid:3) Pattern Classi(cid:2)cation And Scene

I(cid:8) Guyon(cid:3) D(cid:8) Henderson(cid:3) R(cid:8) E(cid:8) Howard(cid:3) and W(cid:8) Hubbard(cid:3)

Analysis(cid:3) Wiley and Son(cid:3) (cid:5)(cid:17)(cid:7)(cid:18)(cid:8)

(cid:29)Handwritten digit recognition(cid:9) Applications of neural net

(cid:26)(cid:15)(cid:27)

Y(cid:8) LeCun(cid:3) B(cid:8) Boser(cid:3) J(cid:8) S(cid:8) Denker(cid:3) D(cid:8) Henderson(cid:3) R(cid:8) E(cid:8) Howard(cid:3)

chips and automatic learning(cid:3)(cid:30) IEEE Communication(cid:3) pp(cid:8) (cid:21)(cid:5)(cid:31)

W(cid:8) Hubbard(cid:3) and L(cid:8) D(cid:8) Jackel(cid:3) (cid:29)Backpropagation applied to

(cid:21)(cid:14)(cid:3) November (cid:5)(cid:17)(cid:16)(cid:17)(cid:3) invited paper(cid:8)

handwritten zip code recognition(cid:3)(cid:30) Neural Computation(cid:3) vol(cid:8)

(cid:26)(cid:15)(cid:14)(cid:27)

J(cid:8) Keeler(cid:3) D(cid:8) Rumelhart(cid:3) and W(cid:8) K(cid:8) Leow(cid:3) (cid:29)Integrated seg(cid:2)

(cid:5)(cid:3) no(cid:8) (cid:21)(cid:3) pp(cid:8) (cid:20)(cid:21)(cid:5)(cid:31)(cid:20)(cid:20)(cid:5)(cid:3) Winter (cid:5)(cid:17)(cid:16)(cid:17)(cid:8)

mentation and recognition of hand(cid:2)printed numerals(cid:3)(cid:30) in Neu(cid:8)

(cid:26)(cid:18)(cid:27)

S(cid:8) Seung(cid:3) H(cid:8) Sompolinsky(cid:3) and N(cid:8) Tishby(cid:3) (cid:29)Statistical mechan(cid:2)

ral Information Processing Systems(cid:3) R(cid:8) P(cid:8) Lippmann(cid:3) J(cid:8) M(cid:8)

ics of learning from examples(cid:3)(cid:30) Physical Review A(cid:3) vol(cid:8) (cid:21)(cid:20)(cid:3) pp(cid:8)

Moody(cid:3) and D(cid:8) S(cid:8) Touretzky(cid:3) Eds(cid:8)(cid:3) vol(cid:8) (cid:18)(cid:3) pp(cid:8) (cid:20)(cid:20)(cid:7)(cid:31)(cid:20)(cid:14)(cid:18)(cid:8) Morgan

(cid:14)(cid:6)(cid:20)(cid:14)(cid:31)(cid:14)(cid:6)(cid:17)(cid:5)(cid:3) (cid:5)(cid:17)(cid:17)(cid:15)(cid:8)

Kaufmann Publishers(cid:3) San Mateo(cid:3) CA(cid:3) (cid:5)(cid:17)(cid:17)(cid:5)(cid:8)

(cid:26)(cid:21)(cid:27)

V(cid:8) N(cid:8) Vapnik(cid:3) E(cid:8) Levin(cid:3) and Y(cid:8) LeCun(cid:3) (cid:29)Measuring the vc(cid:2)

(cid:26)(cid:15)(cid:7)(cid:27) Ofer Matan(cid:3) Christopher J(cid:8) C(cid:8) Burges(cid:3) Yann LeCun(cid:3) and

dimension of a learning machine(cid:3)(cid:30) Neural Computation(cid:3) vol(cid:8) (cid:14)(cid:3)

John S(cid:8) Denker(cid:3) (cid:29)Multi(cid:2)digit recognition using a space dis(cid:2)

no(cid:8) (cid:20)(cid:3) pp(cid:8) (cid:16)(cid:20)(cid:5)(cid:31)(cid:16)(cid:7)(cid:14)(cid:3) (cid:5)(cid:17)(cid:17)(cid:21)(cid:8)

placement neural network(cid:3)(cid:30) in Neural Information Processing

(cid:26)(cid:20)(cid:27)

C(cid:8) Cortes(cid:3) L(cid:8) Jackel(cid:3) S(cid:8) Solla(cid:3) V(cid:8) N(cid:8) Vapnik(cid:3) and J(cid:8) Denker(cid:3)

Systems(cid:3) J(cid:8) M(cid:8) Moody(cid:3) S(cid:8) J(cid:8) Hanson(cid:3) and R(cid:8) P(cid:8) Lippman(cid:3) Eds(cid:8)

(cid:29)Learning curves(cid:9) asymptotic values and rate of convergence(cid:3)(cid:30)

(cid:5)(cid:17)(cid:17)(cid:15)(cid:3) vol(cid:8) (cid:21)(cid:3) Morgan Kaufmann Publishers(cid:3) San Mateo(cid:3) CA(cid:8)

in Advances in Neural Information Processing Systems (cid:3)(cid:3) J(cid:8) D(cid:8)

(cid:26)(cid:15)(cid:16)(cid:27) L(cid:8) R(cid:8) Rabiner(cid:3) (cid:29)A tutorial on hidden Markov models and se(cid:2)

Cowan(cid:3) G(cid:8) Tesauro(cid:3) and J(cid:8) Alspector(cid:3) Eds(cid:8)(cid:3) San Mateo(cid:3) CA(cid:3)

lected applications in speech recognition(cid:3)(cid:30) Proceedings of the

(cid:5)(cid:17)(cid:17)(cid:21)(cid:3) pp(cid:8) (cid:18)(cid:15)(cid:7)(cid:31)(cid:18)(cid:18)(cid:21)(cid:3) Morgan Kaufmann(cid:8)

IEEE(cid:3) vol(cid:8) (cid:7)(cid:7)(cid:3) no(cid:8) (cid:15)(cid:3) pp(cid:8) (cid:15)(cid:20)(cid:7)(cid:31)(cid:15)(cid:16)(cid:14)(cid:3) February (cid:5)(cid:17)(cid:16)(cid:17)(cid:8)

(cid:26)(cid:14)(cid:27)

V(cid:8) N(cid:8) Vapnik(cid:3) The Nature of Statistical Learning Theory(cid:3)

(cid:26)(cid:15)(cid:17)(cid:27) H(cid:8) A(cid:8) Bourlard and N(cid:8) Morgan(cid:3) CONNECTIONIST SPEECH

Springer(cid:3) New(cid:2)York(cid:3) (cid:5)(cid:17)(cid:17)(cid:20)(cid:8)

RECOGNITION(cid:4) A Hybrid Approach(cid:3) Kluwer Academic Pub(cid:2)

(cid:26)(cid:7)(cid:27)

V(cid:8) N(cid:8) Vapnik(cid:3) Statistical Learning Theory(cid:3) John Wiley (cid:4) Sons(cid:3)

lisher(cid:3) Boston(cid:3) (cid:5)(cid:17)(cid:17)(cid:21)(cid:8)

New(cid:2)York(cid:3) (cid:5)(cid:17)(cid:17)(cid:16)(cid:8)

(cid:26)(cid:18)(cid:6)(cid:27) D(cid:8) H(cid:8) Hubel and T(cid:8) N(cid:8) Wiesel(cid:3) (cid:29)Receptive (cid:19)elds(cid:3) binocular

(cid:26)(cid:16)(cid:27) W(cid:8) H(cid:8) Press(cid:3) B(cid:8) P(cid:8) Flannery(cid:3) S(cid:8) A(cid:8) Teukolsky(cid:3) and W(cid:8) T(cid:8) Vet(cid:2)

interaction(cid:3) and functional architecture in the cat(cid:13)s visual cor(cid:2)

terling(cid:3) Numerical Recipes(cid:4) The Art of Scienti(cid:2)c Computing(cid:3)

tex(cid:3)(cid:30) Journal of Physiology (cid:13)London(cid:14)(cid:3) vol(cid:8) (cid:5)(cid:14)(cid:6)(cid:3) pp(cid:8) (cid:5)(cid:6)(cid:14)(cid:31)(cid:5)(cid:20)(cid:21)(cid:3)

Cambridge University Press(cid:3) Cambridge(cid:3) (cid:5)(cid:17)(cid:16)(cid:14)(cid:8)

(cid:5)(cid:17)(cid:14)(cid:15)(cid:8)

(cid:26)(cid:17)(cid:27)

S(cid:8) I(cid:8) Amari(cid:3) (cid:29)A theory of adaptive pattern classi(cid:19)ers(cid:3)(cid:30) IEEE

(cid:26)(cid:18)(cid:5)(cid:27) K(cid:8) Fukushima(cid:3) (cid:29)Cognitron(cid:9) A self(cid:2)organizing multilayered neu(cid:2)

Transactions on Electronic Computers(cid:3) vol(cid:8) EC(cid:2)(cid:5)(cid:14)(cid:3) pp(cid:8) (cid:15)(cid:17)(cid:17)(cid:31)

ral network(cid:3)(cid:30) Biological Cybernetics(cid:3) vol(cid:8) (cid:15)(cid:6)(cid:3) no(cid:8) (cid:14)(cid:3) pp(cid:8) (cid:5)(cid:15)(cid:5)(cid:31)(cid:5)(cid:18)(cid:14)(cid:3)

(cid:18)(cid:6)(cid:7)(cid:3) (cid:5)(cid:17)(cid:14)(cid:7)(cid:8)

November (cid:5)(cid:17)(cid:7)(cid:20)(cid:8)

(cid:26)(cid:5)(cid:6)(cid:27) Ya(cid:8) Tsypkin(cid:3) Adaptation and Learning in automatic systems(cid:3)

(cid:26)(cid:18)(cid:15)(cid:27) K(cid:8) Fukushima and S(cid:8) Miyake(cid:3) (cid:29)Neocognitron(cid:9) A new algorithm

Academic Press(cid:3) (cid:5)(cid:17)(cid:7)(cid:5)(cid:8)

for pattern recognition tolerant of deformations and shifts in

(cid:26)(cid:5)(cid:5)(cid:27) Ya(cid:8) Tsypkin(cid:3) Foundations of the theory of learning systems(cid:3)

position(cid:3)(cid:30) Pattern Recognition(cid:3) vol(cid:8) (cid:5)(cid:20)(cid:3) pp(cid:8) (cid:21)(cid:20)(cid:20)(cid:31)(cid:21)(cid:14)(cid:17)(cid:3) (cid:5)(cid:17)(cid:16)(cid:15)(cid:8)

Academic Press(cid:3) (cid:5)(cid:17)(cid:7)(cid:18)(cid:8)

(cid:26)(cid:18)(cid:18)(cid:27) M(cid:8) C(cid:8) Mozer(cid:3) The perception of multiple objects(cid:4) A connec(cid:8)

(cid:26)(cid:5)(cid:15)(cid:27) M(cid:8) Minsky and O(cid:8) Selfridge(cid:3) (cid:29)Learning in random nets(cid:3)(cid:30) in

tionist approach(cid:3) MIT Press(cid:2)Bradford Books(cid:3) Cambridge(cid:3) MA(cid:3)

(cid:5)th London symposium on Information Theory(cid:3) London(cid:3) (cid:5)(cid:17)(cid:14)(cid:5)(cid:3)

(cid:5)(cid:17)(cid:17)(cid:5)(cid:8)

pp(cid:8) (cid:18)(cid:18)(cid:20)(cid:31)(cid:18)(cid:21)(cid:7)(cid:8)

(cid:26)(cid:18)(cid:21)(cid:27) Y(cid:8) LeCun(cid:3) (cid:29)Generalization and network design strategies(cid:3)(cid:30) in

(cid:26)(cid:5)(cid:18)(cid:27) D(cid:8) H(cid:8) Ackley(cid:3) G(cid:8) E(cid:8) Hinton(cid:3) and T(cid:8) J(cid:8) Sejnowski(cid:3) (cid:29)A learning

Connectionism in Perspective(cid:3) R(cid:8) Pfeifer(cid:3) Z(cid:8) Schreter(cid:3) F(cid:8) Fogel(cid:2)

algorithm for boltzmann machines(cid:3)(cid:30) Cognitive Science(cid:3) vol(cid:8) (cid:17)(cid:3)

man(cid:3) and L(cid:8) Steels(cid:3) Eds(cid:8)(cid:3) Zurich(cid:3) Switzerland(cid:3) (cid:5)(cid:17)(cid:16)(cid:17)(cid:3) Elsevier(cid:3)

pp(cid:8) (cid:5)(cid:21)(cid:7)(cid:31)(cid:5)(cid:14)(cid:17)(cid:3) (cid:5)(cid:17)(cid:16)(cid:20)(cid:8)

an extended version was published as a technical report of the

(cid:26)(cid:5)(cid:21)(cid:27) G(cid:8) E(cid:8) Hinton and T(cid:8) J(cid:8) Sejnowski(cid:3) (cid:29)Learning and relearning

University of Toronto(cid:8)

in Boltzmann machines(cid:3)(cid:30) in Paral lel Distributed Processing(cid:4)

(cid:26)(cid:18)(cid:20)(cid:27) Y(cid:8) LeCun(cid:3) B(cid:8) Boser(cid:3) J(cid:8) S(cid:8) Denker(cid:3) D(cid:8) Henderson(cid:3) R(cid:8) E(cid:8) Howard(cid:3)

Explorations in the Microstructure of Cognition(cid:6) Volume (cid:7)(cid:4)

W(cid:8) Hubbard(cid:3) and L(cid:8) D(cid:8) Jackel(cid:3) (cid:29)Handwritten digit recognition

Foundations(cid:3) D(cid:8) E(cid:8) Rumelhart and J(cid:8) L(cid:8) McClelland(cid:3) Eds(cid:8) MIT

with a back(cid:2)propagation network(cid:3)(cid:30) in Advances in Neural In(cid:8)

Press(cid:3) Cambridge(cid:3) MA(cid:3) (cid:5)(cid:17)(cid:16)(cid:14)(cid:8)

formation Processing Systems (cid:17) (cid:13)NIPS(cid:18)(cid:9)(cid:15)(cid:14)(cid:3) David Touretzky(cid:3)

(cid:26)(cid:5)(cid:20)(cid:27) D(cid:8) E(cid:8) Rumelhart(cid:3) G(cid:8) E(cid:8) Hinton(cid:3) and R(cid:8) J(cid:8) Williams(cid:3) (cid:29)Learning

Ed(cid:8)(cid:3) Denver(cid:3) CO(cid:3) (cid:5)(cid:17)(cid:17)(cid:6)(cid:3) Morgan Kaufmann(cid:8)

internal representations by error propagation(cid:3)(cid:30) in Paral lel dis(cid:8)

(cid:26)(cid:18)(cid:14)(cid:27) G(cid:8) L(cid:8) Martin(cid:3) (cid:29)Centered(cid:2)ob ject integrated segmentation and

tributed processing(cid:4) Explorations in the microstructure of cog(cid:8)

recognition of overlapping hand(cid:2)printed characters(cid:3)(cid:30) Neural

nition(cid:3) vol(cid:8) I(cid:3) pp(cid:8) (cid:18)(cid:5)(cid:16)(cid:31)(cid:18)(cid:14)(cid:15)(cid:8) Bradford Books(cid:3) Cambridge(cid:3) MA(cid:3)

Computation(cid:3) vol(cid:8) (cid:20)(cid:3) no(cid:8) (cid:18)(cid:3) pp(cid:8) (cid:21)(cid:5)(cid:17)(cid:31)(cid:21)(cid:15)(cid:17)(cid:3) (cid:5)(cid:17)(cid:17)(cid:18)(cid:8)

(cid:5)(cid:17)(cid:16)(cid:14)(cid:8)

(cid:26)(cid:18)(cid:7)(cid:27)

J(cid:8) Wang and J Jean(cid:3) (cid:29)Multi(cid:2)resolution neural networks for om(cid:2)

(cid:26)(cid:5)(cid:14)(cid:27) A(cid:8) E(cid:8) Jr(cid:8) Bryson and Yu(cid:2)Chi Ho(cid:3) Applied Optimal Control(cid:3)

nifont character recognition(cid:3)(cid:30) in Proceedings of International

Blaisdell Publishing Co(cid:8)(cid:3) (cid:5)(cid:17)(cid:14)(cid:17)(cid:8)

Conference on Neural Networks(cid:3) (cid:5)(cid:17)(cid:17)(cid:18)(cid:3) vol(cid:8) III(cid:3) pp(cid:8) (cid:5)(cid:20)(cid:16)(cid:16)(cid:31)(cid:5)(cid:20)(cid:17)(cid:18)(cid:8)

(cid:26)(cid:5)(cid:7)(cid:27) Y(cid:8) LeCun(cid:3) (cid:29)A learning scheme for asymmetric threshold net(cid:2)

(cid:26)(cid:18)(cid:16)(cid:27) Y(cid:8) Bengio(cid:3) Y(cid:8) LeCun(cid:3) C(cid:8) Nohl(cid:3) and C(cid:8) Burges(cid:3) (cid:29)Lerec(cid:9) A

works(cid:3)(cid:30) in Proceedings of Cognitiva (cid:9)(cid:10)(cid:3) Paris(cid:3) France(cid:3) (cid:5)(cid:17)(cid:16)(cid:20)(cid:3)

NN(cid:28)HMM hybrid for on(cid:2)line handwriting recognition(cid:3)(cid:30) Neural

pp(cid:8) (cid:20)(cid:17)(cid:17)(cid:31)(cid:14)(cid:6)(cid:21)(cid:8)

Computation(cid:3) vol(cid:8) (cid:7)(cid:3) no(cid:8) (cid:20)(cid:3) (cid:5)(cid:17)(cid:17)(cid:20)(cid:8)

(cid:26)(cid:5)(cid:16)(cid:27) Y(cid:8) LeCun(cid:3) (cid:29)Learning processes in an asymmetric threshold

(cid:26)(cid:18)(cid:17)(cid:27) S(cid:8) Lawrence(cid:3) C(cid:8) Lee Giles(cid:3) A(cid:8) C(cid:8) Tsoi(cid:3) and A(cid:8) D(cid:8) Back(cid:3) (cid:29)Face

network(cid:3)(cid:30) in Disordered systems and biological organization(cid:3)

recognition(cid:9) A convolutional neural network approach(cid:3)(cid:30) IEEE

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:9)(cid:9)

Transactions on Neural Networks(cid:3) vol(cid:8) (cid:16)(cid:3) no(cid:8) (cid:5)(cid:3) pp(cid:8) (cid:17)(cid:16)(cid:31)(cid:5)(cid:5)(cid:18)(cid:3)

son(cid:3) J(cid:8) D(cid:8) Cowan(cid:3) and C(cid:8) L(cid:8) Giles(cid:3) Eds(cid:8)(cid:3) San Mateo(cid:3) CA(cid:3) (cid:5)(cid:17)(cid:17)(cid:18)(cid:3)

(cid:5)(cid:17)(cid:17)(cid:7)(cid:8)

pp(cid:8) (cid:21)(cid:15)(cid:31)(cid:21)(cid:17)(cid:3) Morgan Kaufmann(cid:8)

(cid:26)(cid:21)(cid:6)(cid:27) K(cid:8) J(cid:8) Lang and G(cid:8) E(cid:8) Hinton(cid:3) (cid:29)A time delay neural network

(cid:26)(cid:14)(cid:5)(cid:27) P(cid:8) Simard(cid:3) Y(cid:8) LeCun(cid:3) and Denker J(cid:8)(cid:3) (cid:29)E!cient pattern recog(cid:2)

architecture for speech recognition(cid:3)(cid:30) Tech(cid:8) Rep(cid:8) CMU(cid:2)CS(cid:2)(cid:16)(cid:16)(cid:2)

nition using a new transformation distance(cid:3)(cid:30) in Advances in

(cid:5)(cid:20)(cid:15)(cid:3) Carnegie(cid:2)Mellon University(cid:3) Pittsburgh PA(cid:3) (cid:5)(cid:17)(cid:16)(cid:16)(cid:8)

Neural Information Processing Systems(cid:3) S(cid:8) Hanson(cid:3) J(cid:8) Cowan(cid:3)

(cid:26)(cid:21)(cid:5)(cid:27) A(cid:8) H(cid:8) Waibel(cid:3) T(cid:8) Hanazawa(cid:3) G(cid:8) Hinton(cid:3) K(cid:8) Shikano(cid:3) and

and L(cid:8) Giles(cid:3) Eds(cid:8)(cid:3) vol(cid:8) (cid:20)(cid:8) Morgan Kaufmann(cid:3) (cid:5)(cid:17)(cid:17)(cid:18)(cid:8)

K(cid:8) Lang(cid:3) (cid:29)Phoneme recognition using time(cid:2)delay neural net(cid:2)

(cid:26)(cid:14)(cid:15)(cid:27) B(cid:8) Boser(cid:3) I(cid:8) Guyon(cid:3) and V(cid:8) Vapnik(cid:3) (cid:29)A training algorithm for

works(cid:3)(cid:30) IEEE Transactions on Acoustics(cid:16) Speech and Signal

optimal margin classi(cid:19)ers(cid:3)(cid:30) in Proceedings of the Fifth Annual

Processing(cid:3) vol(cid:8) (cid:18)(cid:7)(cid:3) pp(cid:8) (cid:18)(cid:15)(cid:16)(cid:31)(cid:18)(cid:18)(cid:17)(cid:3) March (cid:5)(cid:17)(cid:16)(cid:17)(cid:8)

Workshop on Computational Learning Theory(cid:3) (cid:5)(cid:17)(cid:17)(cid:15)(cid:3) vol(cid:8) (cid:20)(cid:3) pp(cid:8)

(cid:26)(cid:21)(cid:15)(cid:27) L(cid:8) Bottou(cid:3) F(cid:8) Fogelman(cid:3) P(cid:8) Blanchet(cid:3) and J(cid:8) S(cid:8) Lienard(cid:3)

(cid:5)(cid:21)(cid:21)(cid:31)(cid:5)(cid:20)(cid:15)(cid:8)

(cid:29)Speaker independent isolated digit recognition(cid:9) Multilayer

(cid:26)(cid:14)(cid:18)(cid:27) C(cid:8) J(cid:8) C(cid:8) Burges and B(cid:8) Schoelkopf(cid:3) (cid:29)Improving the accuracy

perceptron vs dynamic time warping(cid:3)(cid:30) Neural Networks(cid:3) vol(cid:8)

and speed of support vector machines(cid:3)(cid:30) in Advances in Neural

(cid:18)(cid:3) pp(cid:8) (cid:21)(cid:20)(cid:18)(cid:31)(cid:21)(cid:14)(cid:20)(cid:3) (cid:5)(cid:17)(cid:17)(cid:6)(cid:8)

Information Processing Systems (cid:15)(cid:3) M(cid:8) Jordan M(cid:8) Mozer and

(cid:26)(cid:21)(cid:18)(cid:27) P(cid:8) Ha(cid:10)ner and A(cid:8) H(cid:8) Waibel(cid:3) (cid:29)Time(cid:2)delay neural networks

T(cid:8) Petsche(cid:3) Eds(cid:8) (cid:5)(cid:17)(cid:17)(cid:7)(cid:3) The MIT Press(cid:3) Cambridge(cid:8)

embedding time alignment(cid:9) a performance analysis(cid:3)(cid:30) in EU(cid:8)

(cid:26)(cid:14)(cid:21)(cid:27) Eduard S ackinger(cid:3) Bernhard Boser(cid:3) Jane Bromley(cid:3) Yann Le(cid:2)

ROSPEECH(cid:12)(cid:15)(cid:7)(cid:16) (cid:17)nd European Conference on Speech Commu(cid:8)

Cun(cid:3) and Lawrence D(cid:8) Jackel(cid:3) (cid:29)Application of the ANNA neu(cid:2)

nication and Technology(cid:3) Genova(cid:3) Italy(cid:3) Sept(cid:8) (cid:5)(cid:17)(cid:17)(cid:5)(cid:8)

ral network chip to high(cid:2)speed character recognition(cid:3)(cid:30) IEEE

(cid:26)(cid:21)(cid:21)(cid:27)

I(cid:8) Guyon(cid:3) P(cid:8) Albrecht(cid:3) Y(cid:8) LeCun(cid:3) J(cid:8) S(cid:8) Denker(cid:3) and W(cid:8) Hub(cid:2)

Transaction on Neural Networks(cid:3) vol(cid:8) (cid:18)(cid:3) no(cid:8) (cid:15)(cid:3) pp(cid:8) (cid:21)(cid:17)(cid:16)(cid:31)(cid:20)(cid:6)(cid:20)(cid:3)

bard(cid:3) (cid:29)Design of a neural network character recognizer for a

March (cid:5)(cid:17)(cid:17)(cid:15)(cid:8)

touch terminal(cid:3)(cid:30) Pattern Recognition(cid:3) vol(cid:8) (cid:15)(cid:21)(cid:3) no(cid:8) (cid:15)(cid:3) pp(cid:8) (cid:5)(cid:6)(cid:20)(cid:31)

(cid:26)(cid:14)(cid:20)(cid:27)

J(cid:8) S(cid:8) Bridle(cid:3) (cid:29)Probabilistic interpretation of feedforward classi(cid:19)(cid:2)

(cid:5)(cid:5)(cid:17)(cid:3) (cid:5)(cid:17)(cid:17)(cid:5)(cid:8)

cation networks outputs(cid:3) with relationship to statistical pattern

(cid:26)(cid:21)(cid:20)(cid:27)

J(cid:8) Bromley(cid:3) J(cid:8) W(cid:8) Bentz(cid:3) L(cid:8) Bottou(cid:3) I(cid:8) Guyon(cid:3) Y(cid:8) LeCun(cid:3)

recognition(cid:3)(cid:30)

in Neurocomputing(cid:16) Algorithms(cid:16) Architectures

C(cid:8) Moore(cid:3) E(cid:8) S ackinger(cid:3) and R(cid:8) Shah(cid:3)

(cid:29)Signature veri(cid:19)ca(cid:2)

and Applications(cid:3) F(cid:8) Fogelman(cid:3) J(cid:8) Herault(cid:3) and Y(cid:8) Burnod(cid:3)

tion using a siamese time delay neural network(cid:3)(cid:30) International

Eds(cid:8)(cid:3) Les Arcs(cid:3) France(cid:3) (cid:5)(cid:17)(cid:16)(cid:17)(cid:3) Springer(cid:8)

Journal of Pattern Recognition and Arti(cid:2)cial Intel ligence(cid:3) vol(cid:8)

(cid:26)(cid:14)(cid:14)(cid:27) Y(cid:8) LeCun(cid:3) L(cid:8) Bottou(cid:3) and Y(cid:8) Bengio(cid:3) (cid:29)Reading checks with

(cid:7)(cid:3) no(cid:8) (cid:21)(cid:3) pp(cid:8) (cid:14)(cid:14)(cid:17)(cid:31)(cid:14)(cid:16)(cid:7)(cid:3) August (cid:5)(cid:17)(cid:17)(cid:18)(cid:8)

graph transformer networks(cid:3)(cid:30) in International Conference on

(cid:26)(cid:21)(cid:14)(cid:27) Y(cid:8) LeCun(cid:3) I(cid:8) Kanter(cid:3) and S(cid:8) Solla(cid:3) (cid:29)Eigenvalues of covariance

Acoustics(cid:16) Speech(cid:16) and Signal Processing(cid:3) Munich(cid:3) (cid:5)(cid:17)(cid:17)(cid:7)(cid:3) vol(cid:8) (cid:5)(cid:3)

matrices(cid:9) application to neural(cid:2)network learning(cid:3)(cid:30) Physical

pp(cid:8) (cid:5)(cid:20)(cid:5)(cid:31)(cid:5)(cid:20)(cid:21)(cid:3) IEEE(cid:8)

Review Letters(cid:3) vol(cid:8) (cid:14)(cid:14)(cid:3) no(cid:8) (cid:5)(cid:16)(cid:3) pp(cid:8) (cid:15)(cid:18)(cid:17)(cid:14)(cid:31)(cid:15)(cid:18)(cid:17)(cid:17)(cid:3) May (cid:5)(cid:17)(cid:17)(cid:5)(cid:8)

(cid:26)(cid:14)(cid:7)(cid:27) Y(cid:8) Bengio(cid:3) Neural Networks for Speech and Sequence Recogni(cid:8)

(cid:26)(cid:21)(cid:7)(cid:27) T(cid:8) G(cid:8) Dietterich and G(cid:8) Bakiri(cid:3) (cid:29)Solving multiclass learning

tion(cid:3) International Thompson Computer Press(cid:3) London(cid:3) UK(cid:3)

problems via error(cid:2)correcting output codes(cid:8)(cid:3)(cid:30) Journal of Arti(cid:8)

(cid:5)(cid:17)(cid:17)(cid:14)(cid:8)

(cid:2)cial Intel ligence Research(cid:3) vol(cid:8) (cid:15)(cid:3) pp(cid:8) (cid:15)(cid:14)(cid:18)(cid:31)(cid:15)(cid:16)(cid:14)(cid:3) (cid:5)(cid:17)(cid:17)(cid:20)(cid:8)

(cid:26)(cid:14)(cid:16)(cid:27) C(cid:8) Burges(cid:3) O(cid:8) Matan(cid:3) Y(cid:8) LeCun(cid:3) J(cid:8) Denker(cid:3) L(cid:8) Jackel(cid:3) C(cid:8) Ste(cid:2)

(cid:26)(cid:21)(cid:16)(cid:27) L(cid:8) R(cid:8) Bahl(cid:3) P(cid:8) F(cid:8) Brown(cid:3) P(cid:8) V(cid:8) de Souza(cid:3) and R(cid:8) L(cid:8) Mercer(cid:3)

nard(cid:3) C(cid:8) Nohl(cid:3) and J(cid:8) Ben(cid:3) (cid:29)Shortest path segmentation(cid:9) A

(cid:29)Maximum mutual information of hidden Markov model pa(cid:2)

method for training a neural network to recognize character

rameters for speech recognition(cid:3)(cid:30) in Proc(cid:6) Int(cid:6) Conf(cid:6) Acoust(cid:6)(cid:16)

strings(cid:3)(cid:30) in International Joint Conference on Neural Net(cid:8)

Speech(cid:16) Signal Processing(cid:3) (cid:5)(cid:17)(cid:16)(cid:14)(cid:3) pp(cid:8) (cid:21)(cid:17)(cid:31)(cid:20)(cid:15)(cid:8)

works(cid:3) Baltimore(cid:3) (cid:5)(cid:17)(cid:17)(cid:15)(cid:3) vol(cid:8) (cid:18)(cid:3) pp(cid:8) (cid:5)(cid:14)(cid:20)(cid:31)(cid:5)(cid:7)(cid:15)(cid:8)

(cid:26)(cid:21)(cid:17)(cid:27) L(cid:8) R(cid:8) Bahl(cid:3) P(cid:8) F(cid:8) Brown(cid:3) P(cid:8) V(cid:8) de Souza(cid:3) and R(cid:8) L(cid:8) Mercer(cid:3)

(cid:26)(cid:14)(cid:17)(cid:27) T(cid:8) M(cid:8) Breuel(cid:3) (cid:29)A system for the o(cid:10)(cid:2)line recognition of hand(cid:2)

(cid:29)Speech recognition with continuous(cid:2)parameter hidden Markov

written text(cid:3)(cid:30) in ICPR(cid:12)(cid:15)(cid:5)(cid:3) IEEE(cid:3) Ed(cid:8)(cid:3) Jerusalem (cid:5)(cid:17)(cid:17)(cid:21)(cid:3) (cid:5)(cid:17)(cid:17)(cid:21)(cid:3)

models(cid:3)(cid:30) Computer(cid:16) Speech and Language(cid:3) vol(cid:8) (cid:15)(cid:3) pp(cid:8) (cid:15)(cid:5)(cid:17)(cid:31)(cid:15)(cid:18)(cid:21)(cid:3)

pp(cid:8) (cid:5)(cid:15)(cid:17)(cid:31)(cid:5)(cid:18)(cid:21)(cid:8)

(cid:5)(cid:17)(cid:16)(cid:7)(cid:8)

(cid:26)(cid:7)(cid:6)(cid:27) A(cid:8) Viterbi(cid:3)

(cid:29)Error bounds for convolutional codes and an

(cid:26)(cid:20)(cid:6)(cid:27) B(cid:8) H(cid:8) Juang and S(cid:8) Katagiri(cid:3) (cid:29)Discriminative learning for min(cid:2)

asymptotically optimum decoding algorithm(cid:3)(cid:30) IEEE Trans(cid:8)

imum error classi(cid:19)cation(cid:3)(cid:30) IEEE Trans(cid:6) on Acoustics(cid:16) Speech(cid:16)

actions on Information Theory(cid:3) pp(cid:8) (cid:15)(cid:14)(cid:6)(cid:31)(cid:15)(cid:14)(cid:17)(cid:3) April (cid:5)(cid:17)(cid:14)(cid:7)(cid:8)

and Signal Processing(cid:3) vol(cid:8) (cid:21)(cid:6)(cid:3) no(cid:8) (cid:5)(cid:15)(cid:3) pp(cid:8) (cid:18)(cid:6)(cid:21)(cid:18)(cid:31)(cid:18)(cid:6)(cid:20)(cid:21)(cid:3) December

(cid:26)(cid:7)(cid:5)(cid:27) Lippmann R(cid:8) P(cid:8) and Gold B(cid:8)(cid:3) (cid:29)Neural(cid:2)net classi(cid:19)ers useful for

(cid:5)(cid:17)(cid:17)(cid:15)(cid:8)

speech recognition(cid:3)(cid:30) in Proceedings of the IEEE First Interna(cid:8)

(cid:26)(cid:20)(cid:5)(cid:27) Y(cid:8) LeCun(cid:3) L(cid:8) D(cid:8) Jackel(cid:3) L(cid:8) Bottou(cid:3) A(cid:8) Brunot(cid:3) C(cid:8) Cortes(cid:3) J(cid:8) S(cid:8)

tional Conference on Neural Networks(cid:3) San Diego(cid:3) June (cid:5)(cid:17)(cid:16)(cid:7)(cid:3)

Denker(cid:3) H(cid:8) Drucker(cid:3) I(cid:8) Guyon(cid:3) U(cid:8) A(cid:8) Muller(cid:3) E(cid:8) S ackinger(cid:3)

pp(cid:8) (cid:21)(cid:5)(cid:7)(cid:31)(cid:21)(cid:15)(cid:15)(cid:8)

P(cid:8) Simard(cid:3) and V(cid:8) N(cid:8) Vapnik(cid:3) (cid:29)Comparison of learning al(cid:2)

(cid:26)(cid:7)(cid:15)(cid:27) H(cid:8) Sakoe(cid:3) R(cid:8) Isotani(cid:3) K(cid:8) Yoshida(cid:3) K(cid:8) Iso(cid:3) and T(cid:8) Watan(cid:2)

gorithms for handwritten digit recognition(cid:3)(cid:30) in International

abe(cid:3) (cid:29)Speaker(cid:2)independent word recognition using dynamic

Conference on Arti(cid:2)cial Neural Networks(cid:3) F(cid:8) Fogelman and

programming neural networks(cid:3)(cid:30) in International Conference

P(cid:8) Gallinari(cid:3) Eds(cid:8)(cid:3) Paris(cid:3) (cid:5)(cid:17)(cid:17)(cid:20)(cid:3) pp(cid:8) (cid:20)(cid:18)(cid:31)(cid:14)(cid:6)(cid:3) EC(cid:15) (cid:4) Cie(cid:8)

on Acoustics(cid:16) Speech(cid:16) and Signal Processing(cid:3) Glasgow(cid:3) (cid:5)(cid:17)(cid:16)(cid:17)(cid:3)

(cid:26)(cid:20)(cid:15)(cid:27)

I Guyon(cid:3) I(cid:8) Poujaud(cid:3) L(cid:8) Personnaz(cid:3) G(cid:8) Dreyfus(cid:3) J(cid:8) Denker(cid:3) and

pp(cid:8) (cid:15)(cid:17)(cid:31)(cid:18)(cid:15)(cid:8)

Y(cid:8) LeCun(cid:3) (cid:29)Comparing di(cid:10)erent neural net architectures for

(cid:26)(cid:7)(cid:18)(cid:27)

J(cid:8) S(cid:8) Bridle(cid:3) (cid:29)Alphanets(cid:9) a recurrent "neural(cid:13) network archi(cid:2)

classifying handwritten digits(cid:3)(cid:30) in Proc(cid:6) of IJCNN(cid:16) Washing(cid:8)

tecture with a hidden markov model interpretation(cid:3)(cid:30) Speech

ton DC(cid:8) (cid:5)(cid:17)(cid:16)(cid:17)(cid:3) vol(cid:8) II(cid:3) pp(cid:8) (cid:5)(cid:15)(cid:7)(cid:31)(cid:5)(cid:18)(cid:15)(cid:3) IEEE(cid:8)

Communication(cid:3) vol(cid:8) (cid:17)(cid:3) no(cid:8) (cid:5)(cid:3) pp(cid:8) (cid:16)(cid:5)(cid:20)(cid:31)(cid:16)(cid:5)(cid:17)(cid:3) (cid:5)(cid:17)(cid:17)(cid:6)(cid:8)

(cid:26)(cid:20)(cid:18)(cid:27) R(cid:8) Ott(cid:3)

(cid:29)construction of quadratic polynomial classi(cid:19)ers(cid:3)(cid:30)

(cid:26)(cid:7)(cid:21)(cid:27) M(cid:8) A(cid:8) Franzini(cid:3) K(cid:8) F(cid:8) Lee(cid:3) and A(cid:8) H(cid:8) Waibel(cid:3) (cid:29)Connectionist

in Proc(cid:6) of International Conference on Pattern Recognition(cid:8)

viterbi training(cid:9) a new hybrid method for continuous speech

(cid:5)(cid:17)(cid:7)(cid:14)(cid:3) pp(cid:8) (cid:5)(cid:14)(cid:5)(cid:31)(cid:5)(cid:14)(cid:20)(cid:3) IEEE(cid:8)

recognition(cid:3)(cid:30) in International Conference on Acoustics(cid:16) Speech(cid:16)

(cid:26)(cid:20)(cid:21)(cid:27)

J(cid:8) Sch urmann(cid:3) (cid:29)A multi(cid:2)font word recognition system for postal

and Signal Processing(cid:3) Albuquerque(cid:3) NM(cid:3) (cid:5)(cid:17)(cid:17)(cid:6)(cid:3) pp(cid:8) (cid:21)(cid:15)(cid:20)(cid:31)(cid:21)(cid:15)(cid:16)(cid:8)

address reading(cid:3)(cid:30) IEEE Transactions on Computers(cid:3) vol(cid:8) C(cid:2)(cid:15)(cid:7)(cid:3)

(cid:26)(cid:7)(cid:20)(cid:27) L(cid:8) T(cid:8) Niles and H(cid:8) F(cid:8) Silverman(cid:3) (cid:29)Combining hidden markov

no(cid:8) (cid:16)(cid:3) pp(cid:8) (cid:7)(cid:15)(cid:5)(cid:31)(cid:7)(cid:18)(cid:15)(cid:3) August (cid:5)(cid:17)(cid:7)(cid:16)(cid:8)

models and neural network classi(cid:19)ers(cid:3)(cid:30) in International Con(cid:8)

(cid:26)(cid:20)(cid:20)(cid:27) Y(cid:8) Lee(cid:3) (cid:29)Handwritten digit recognition using k(cid:2)nearest neigh(cid:2)

ference on Acoustics(cid:16) Speech(cid:16) and Signal Processing(cid:3) Albu(cid:2)

bor(cid:3) radial(cid:2)basis functions(cid:3) and backpropagation neural net(cid:2)

querque(cid:3) NM(cid:3) (cid:5)(cid:17)(cid:17)(cid:6)(cid:3) pp(cid:8) (cid:21)(cid:5)(cid:7)(cid:31)(cid:21)(cid:15)(cid:6)(cid:8)

works(cid:3)(cid:30) Neural Computation(cid:3) vol(cid:8) (cid:18)(cid:3) no(cid:8) (cid:18)(cid:3) pp(cid:8) (cid:21)(cid:21)(cid:6)(cid:31)(cid:21)(cid:21)(cid:17)(cid:3) (cid:5)(cid:17)(cid:17)(cid:5)(cid:8)

(cid:26)(cid:7)(cid:14)(cid:27) X(cid:8) Driancourt and L(cid:8) Bottou(cid:3) (cid:29)MLP(cid:3) LVQ and DP(cid:9) Compari(cid:2)

(cid:26)(cid:20)(cid:14)(cid:27) D(cid:8) Saad and S(cid:8) A(cid:8) Solla(cid:3) (cid:29)Dynamics of on(cid:2)line gradient de(cid:2)

son (cid:4) cooperation(cid:3)(cid:30) in Proceedings of the International Joint

scent learning for multilayer neural networks(cid:3)(cid:30) in Advances in

Conference on Neural Networks(cid:3) Seattle(cid:3) (cid:5)(cid:17)(cid:17)(cid:5)(cid:3) vol(cid:8) (cid:15)(cid:3) pp(cid:8) (cid:16)(cid:5)(cid:20)(cid:31)

Neural Information Processing Systems(cid:3) David S(cid:8) Touretzky(cid:3)

(cid:16)(cid:5)(cid:17)(cid:8)

Michael C(cid:8) Mozer(cid:3) and Michael E(cid:8) Hasselmo(cid:3) Eds(cid:8) (cid:5)(cid:17)(cid:17)(cid:14)(cid:3) vol(cid:8) (cid:16)(cid:3)

(cid:26)(cid:7)(cid:7)(cid:27) Y(cid:8) Bengio(cid:3) R(cid:8) De Mori(cid:3) G(cid:8) Flammia(cid:3) and R(cid:8) Kompe(cid:3) (cid:29)Global

pp(cid:8) (cid:18)(cid:6)(cid:15)(cid:31)(cid:18)(cid:6)(cid:16)(cid:3) The MIT Press(cid:3) Cambridge(cid:8)

optimization of a neural network(cid:2)hidden Markov model hy(cid:2)

(cid:26)(cid:20)(cid:7)(cid:27) G(cid:8) Cybenko(cid:3) (cid:29)Approximation by superpositions of sigmoidal

brid(cid:3)(cid:30) IEEE Transactions on Neural Networks(cid:3) vol(cid:8) (cid:18)(cid:3) no(cid:8) (cid:15)(cid:3)

functions(cid:3)(cid:30) Mathematics of Control(cid:16) Signals(cid:16) and Systems(cid:3) vol(cid:8)

pp(cid:8) (cid:15)(cid:20)(cid:15)(cid:31)(cid:15)(cid:20)(cid:17)(cid:3) (cid:5)(cid:17)(cid:17)(cid:15)(cid:8)

(cid:15)(cid:3) no(cid:8) (cid:21)(cid:3) pp(cid:8) (cid:18)(cid:6)(cid:18)(cid:31)(cid:18)(cid:5)(cid:21)(cid:3) (cid:5)(cid:17)(cid:16)(cid:17)(cid:8)

(cid:26)(cid:7)(cid:16)(cid:27) P(cid:8) Ha(cid:10)ner and A(cid:8) H(cid:8) Waibel(cid:3) (cid:29)Multi(cid:2)state time(cid:2)delay neural

(cid:26)(cid:20)(cid:16)(cid:27) L(cid:8) Bottou and V(cid:8) N(cid:8) Vapnik(cid:3) (cid:29)Local learning algorithms(cid:3)(cid:30) Neu(cid:8)

networks for continuous speech recognition(cid:3)(cid:30) in Advances in

ral Computation(cid:3) vol(cid:8) (cid:21)(cid:3) no(cid:8) (cid:14)(cid:3) pp(cid:8) (cid:16)(cid:16)(cid:16)(cid:31)(cid:17)(cid:6)(cid:6)(cid:3) (cid:5)(cid:17)(cid:17)(cid:15)(cid:8)

Neural Information Processing Systems(cid:8) (cid:5)(cid:17)(cid:17)(cid:15)(cid:3) vol(cid:8) (cid:21)(cid:3) pp(cid:8) (cid:20)(cid:7)(cid:17)(cid:31)

(cid:26)(cid:20)(cid:17)(cid:27) R(cid:8) E(cid:8) Schapire(cid:3) (cid:29)The strength of weak learnability(cid:3)(cid:30) Machine

(cid:20)(cid:16)(cid:16)(cid:3) Morgan Kaufmann(cid:3) San Mateo(cid:8)

Learning(cid:3) vol(cid:8) (cid:20)(cid:3) no(cid:8) (cid:15)(cid:3) pp(cid:8) (cid:5)(cid:17)(cid:7)(cid:31)(cid:15)(cid:15)(cid:7)(cid:3) (cid:5)(cid:17)(cid:17)(cid:6)(cid:8)

(cid:26)(cid:7)(cid:17)(cid:27) Y(cid:8) Bengio(cid:3) (cid:3) P(cid:8) Simard(cid:3) and P(cid:8) Frasconi(cid:3) (cid:29)Learning long(cid:2)term

(cid:26)(cid:14)(cid:6)(cid:27) H(cid:8) Drucker(cid:3) R(cid:8) Schapire(cid:3) and P(cid:8) Simard(cid:3) (cid:29)Improving perfor(cid:2)

dependencies with gradient descent is di!cult(cid:3)(cid:30) IEEE Trans(cid:8)

mance in neural networks using a boosting algorithm(cid:3)(cid:30) in Ad(cid:8)

actions on Neural Networks(cid:3) vol(cid:8) (cid:20)(cid:3) no(cid:8) (cid:15)(cid:3) pp(cid:8) (cid:5)(cid:20)(cid:7)(cid:31)(cid:5)(cid:14)(cid:14)(cid:3) March

vances in Neural Information Processing Systems (cid:10)(cid:3) S(cid:8) J(cid:8) Han(cid:2)

(cid:5)(cid:17)(cid:17)(cid:21)(cid:3) Special Issue on Recurrent Neural Network(cid:8)

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:9)(cid:10)

(cid:26)(cid:16)(cid:6)(cid:27) T(cid:8) Kohonen(cid:3) G(cid:8) Barna(cid:3) and R(cid:8) Chrisley(cid:3) (cid:29)Statistical pattern

Lippmann(cid:3) Eds(cid:8)(cid:3) Denver(cid:3) CO(cid:3) (cid:5)(cid:17)(cid:17)(cid:15)(cid:3) pp(cid:8) (cid:5)(cid:7)(cid:20)(cid:31)(cid:5)(cid:16)(cid:15)(cid:3) Morgan Kauf(cid:2)

recognition with neural network(cid:9) Benchmarking studies(cid:3)(cid:30) in

mann(cid:8)

Proceedings of the IEEE Second International Conference on

(cid:26)(cid:5)(cid:6)(cid:6)(cid:27) F(cid:8) C(cid:8) N(cid:8) Pereira and M(cid:8) Riley(cid:3) (cid:29)Speech recognition by compo(cid:2)

Neural Networks(cid:3) San Diego(cid:3) (cid:5)(cid:17)(cid:16)(cid:16)(cid:3) vol(cid:8) (cid:5)(cid:3) pp(cid:8) (cid:14)(cid:5)(cid:31)(cid:14)(cid:16)(cid:8)

sition of weighted (cid:19)nite automata(cid:3)(cid:30) in Finite(cid:8)State Devices for

(cid:26)(cid:16)(cid:5)(cid:27) P(cid:8) Ha(cid:10)ner(cid:3) (cid:29)Connectionist speech recognition with a global

Natural Langue Processing(cid:3) Cambridge(cid:3) Massachusetts(cid:3) (cid:5)(cid:17)(cid:17)(cid:7)(cid:3)

MMI algorithm(cid:3)(cid:30) in EUROSPEECH(cid:12)(cid:15)(cid:19)(cid:16) (cid:19)rd European Confer(cid:8)

MIT Press(cid:8)

ence on Speech Communication and Technology(cid:3) Berlin(cid:3) Sept(cid:8)

(cid:26)(cid:5)(cid:6)(cid:5)(cid:27) M(cid:8) Mohri(cid:3) (cid:29)Finite(cid:2)state transducers in language and speech

(cid:5)(cid:17)(cid:17)(cid:18)(cid:8)

processing(cid:3)(cid:30) Computational Linguistics(cid:3) vol(cid:8) (cid:15)(cid:18)(cid:3) no(cid:8) (cid:15)(cid:3) pp(cid:8) (cid:15)(cid:14)(cid:17)(cid:31)

(cid:26)(cid:16)(cid:15)(cid:27)

J(cid:8) S(cid:8) Denker and C(cid:8) J(cid:8) Burges(cid:3) (cid:29)Image segmentation and recog(cid:2)

(cid:18)(cid:5)(cid:5)(cid:3) (cid:5)(cid:17)(cid:17)(cid:7)(cid:8)

nition(cid:3)(cid:30) in The Mathematics of Induction(cid:8) (cid:5)(cid:17)(cid:17)(cid:20)(cid:3) Addison Wes(cid:2)

(cid:26)(cid:5)(cid:6)(cid:15)(cid:27) I(cid:8) Guyon(cid:3) M(cid:8) Schenkel(cid:3) and J(cid:8) Denker(cid:3) (cid:29)Overview and syn(cid:2)

ley(cid:8)

thesis of on(cid:2)line cursive handwriting recognition techniques(cid:3)(cid:30)

(cid:26)(cid:16)(cid:18)(cid:27) L(cid:8) Bottou(cid:3) Une Approche th(cid:20)eorique de l(cid:12)Apprentissage Connex(cid:8)

in Handbook on Optical Character Recognition and Document

ionniste(cid:4) Applications (cid:11)a la Reconnaissance de la Parole(cid:3) Ph(cid:8)D(cid:8)

Image Analysis(cid:3) P(cid:8) S(cid:8) P(cid:8) Wang and Bunke H(cid:8)(cid:3) Eds(cid:8) (cid:5)(cid:17)(cid:17)(cid:14)(cid:3) World

thesis(cid:3) Universit(cid:12)e de Paris XI(cid:3) (cid:17)(cid:5)(cid:21)(cid:6)(cid:20) Orsay cedex(cid:3) France(cid:3) (cid:5)(cid:17)(cid:17)(cid:5)(cid:8)

Scienti(cid:19)c(cid:8)

(cid:26)(cid:16)(cid:21)(cid:27) M(cid:8) Rahim(cid:3) Y(cid:8) Bengio(cid:3) and Y(cid:8) LeCun(cid:3) (cid:29)Discriminative feature

(cid:26)(cid:5)(cid:6)(cid:18)(cid:27) M(cid:8) Mohri and M(cid:8) Riley(cid:3) (cid:29)Weighted determinization and min(cid:2)

and model design for automatic speech recognition(cid:3)(cid:30) in Proc(cid:6)

imization for large vocabulary recognition(cid:3)(cid:30) in Proceedings of

of Eurospeech(cid:3) Rhodes(cid:3) Greece(cid:3) (cid:5)(cid:17)(cid:17)(cid:7)(cid:8)

Eurospeech (cid:12)(cid:15)(cid:21)(cid:3) Rhodes(cid:3) Greece(cid:3) September (cid:5)(cid:17)(cid:17)(cid:7)(cid:3) pp(cid:8) (cid:5)(cid:18)(cid:5)(cid:31)(cid:5)(cid:18)(cid:21)(cid:8)

(cid:26)(cid:16)(cid:20)(cid:27) U(cid:8) Bodenhausen(cid:3) S(cid:8) Manke(cid:3) and A(cid:8) Waibel(cid:3) (cid:29)Connectionist ar(cid:2)

(cid:26)(cid:5)(cid:6)(cid:21)(cid:27) Y(cid:8) Bengio and P(cid:8) Frasconi(cid:3) (cid:29)An input(cid:28)output HMM architec(cid:2)

chitectural learning for high performance character and speech

ture(cid:3)(cid:30) in Advances in Neural Information Processing Systems(cid:3)

recognition(cid:3)(cid:30) in International Conference on Acoustics(cid:16) Speech(cid:16)

G(cid:8) Tesauro(cid:3) D Touretzky(cid:3) and T(cid:8) Leen(cid:3) Eds(cid:8) (cid:5)(cid:17)(cid:17)(cid:14)(cid:3) vol(cid:8) (cid:7)(cid:3) pp(cid:8)

and Signal Processing(cid:3) Minneapolis(cid:3) (cid:5)(cid:17)(cid:17)(cid:18)(cid:3) vol(cid:8) (cid:5)(cid:3) pp(cid:8) (cid:14)(cid:15)(cid:20)(cid:31)(cid:14)(cid:15)(cid:16)(cid:8)

(cid:21)(cid:15)(cid:7)(cid:31)(cid:21)(cid:18)(cid:21)(cid:3) MIT Press(cid:3) Cambridge(cid:3) MA(cid:8)

(cid:26)(cid:16)(cid:14)(cid:27) F(cid:8) Pereira(cid:3) M(cid:8) Riley(cid:3) and R(cid:8) Sproat(cid:3) (cid:29)Weighted rational trans(cid:2)

(cid:26)(cid:5)(cid:6)(cid:20)(cid:27) Y(cid:8) Bengio and P(cid:8) Frasconi(cid:3) (cid:29)Input(cid:28)Output HMMs for sequence

ductions and their application to human language processing(cid:3)(cid:30)

processing(cid:3)(cid:30) IEEE Transactions on Neural Networks(cid:3) vol(cid:8) (cid:7)(cid:3)

in ARPA Natural Language Processing workshop(cid:3) (cid:5)(cid:17)(cid:17)(cid:21)(cid:8)

no(cid:8) (cid:20)(cid:3) pp(cid:8) (cid:5)(cid:15)(cid:18)(cid:5)(cid:31)(cid:5)(cid:15)(cid:21)(cid:17)(cid:3) (cid:5)(cid:17)(cid:17)(cid:14)(cid:8)

(cid:26)(cid:16)(cid:7)(cid:27) M(cid:8) Lades(cid:3) J(cid:8) C(cid:8) Vorbr uggen(cid:3) J(cid:8) Buhmann(cid:3) and C(cid:8) von der Mals(cid:2)

(cid:26)(cid:5)(cid:6)(cid:14)(cid:27) M(cid:8) Mohri(cid:3) F(cid:8) C(cid:8) N(cid:8) Pereira(cid:3) and M(cid:8) Riley(cid:3) A rational design

burg(cid:3) (cid:29)Distortion invariant ob ject recognition in the dynamic

for a weighted (cid:2)nite(cid:8)state transducer library(cid:3) Lecture Notes in

link architecture(cid:3)(cid:30) IEEE Trans(cid:6) Comp(cid:6)(cid:3) vol(cid:8) (cid:21)(cid:15)(cid:3) no(cid:8) (cid:18)(cid:3) pp(cid:8)

Computer Science(cid:8) Springer Verlag(cid:3) (cid:5)(cid:17)(cid:17)(cid:7)(cid:8)

(cid:18)(cid:6)(cid:6)(cid:31)(cid:18)(cid:5)(cid:5)(cid:3) (cid:5)(cid:17)(cid:17)(cid:18)(cid:8)

(cid:26)(cid:5)(cid:6)(cid:7)(cid:27) M(cid:8) Rahim(cid:3) C(cid:8) H(cid:8) Lee(cid:3) and B(cid:8) H(cid:8) Juang(cid:3) (cid:29)Discriminative ut(cid:2)

(cid:26)(cid:16)(cid:16)(cid:27) B(cid:8) Boser(cid:3) E(cid:8) S ackinger(cid:3) J(cid:8) Bromley(cid:3) Y(cid:8) LeCun(cid:3) and L(cid:8) Jackel(cid:3)

terance veri(cid:19)cation for connected digits recognition(cid:3)(cid:30) IEEE

(cid:29)An analog neural network processor with programmable topol(cid:2)

Trans(cid:6) on Speech (cid:22) Audio Proc(cid:6)(cid:3) vol(cid:8) (cid:20)(cid:3) pp(cid:8) (cid:15)(cid:14)(cid:14)(cid:31)(cid:15)(cid:7)(cid:7)(cid:3) (cid:5)(cid:17)(cid:17)(cid:7)(cid:8)

ogy(cid:3)(cid:30) IEEE Journal of Solid(cid:8)State Circuits(cid:3) vol(cid:8) (cid:15)(cid:14)(cid:3) no(cid:8) (cid:5)(cid:15)(cid:3) pp(cid:8)

(cid:26)(cid:5)(cid:6)(cid:16)(cid:27) M(cid:8) Rahim(cid:3) Y(cid:8) Bengio(cid:3) and Y(cid:8) LeCun(cid:3) (cid:29)Discriminative feature

(cid:15)(cid:6)(cid:5)(cid:7)(cid:31)(cid:15)(cid:6)(cid:15)(cid:20)(cid:3) December (cid:5)(cid:17)(cid:17)(cid:5)(cid:8)

and model design for automatic speech recognition(cid:3)(cid:30) in Eu(cid:8)

(cid:26)(cid:16)(cid:17)(cid:27) M(cid:8) Schenkel(cid:3) H(cid:8) Weissman(cid:3) I(cid:8) Guyon(cid:3) C(cid:8) Nohl(cid:3) and D(cid:8) Hender(cid:2)

rospeech (cid:12)(cid:15)(cid:21)(cid:3) Rhodes(cid:3) Greece(cid:3) (cid:5)(cid:17)(cid:17)(cid:7)(cid:3) pp(cid:8) (cid:7)(cid:20)(cid:31)(cid:7)(cid:16)(cid:8)

son(cid:3) (cid:29)Recognition(cid:2)based segmentation of on(cid:2)line hand(cid:2)printed

(cid:26)(cid:5)(cid:6)(cid:17)(cid:27) S(cid:8) Bengio and Y(cid:8) Bengio(cid:3) (cid:29)An EM algorithm for asynchronous

words(cid:3)(cid:30) in Advances in Neural Information Processing Systems

input(cid:28)output hidden Markov models(cid:3)(cid:30) in International Con(cid:8)

(cid:10)(cid:3) S(cid:8) J(cid:8) Hanson(cid:3) J(cid:8) D(cid:8) Cowan(cid:3) and C(cid:8) L(cid:8) Giles(cid:3) Eds(cid:8)(cid:3) Denver(cid:3)

ference On Neural Information Processing(cid:3) L(cid:8) Xu(cid:3) Ed(cid:8)(cid:3) Hong(cid:2)

CO(cid:3) (cid:5)(cid:17)(cid:17)(cid:18)(cid:3) pp(cid:8) (cid:7)(cid:15)(cid:18)(cid:31)(cid:7)(cid:18)(cid:6)(cid:8)

Kong(cid:3) (cid:5)(cid:17)(cid:17)(cid:14)(cid:3) pp(cid:8) (cid:18)(cid:15)(cid:16)(cid:31)(cid:18)(cid:18)(cid:21)(cid:8)

(cid:26)(cid:17)(cid:6)(cid:27) C(cid:8) Dugast(cid:3) L(cid:8) Devillers(cid:3) and X(cid:8) Aubert(cid:3) (cid:29)Combining TDNN

(cid:26)(cid:5)(cid:5)(cid:6)(cid:27) C(cid:8) Tappert(cid:3) C(cid:8) Suen(cid:3) and T(cid:8) Wakahara(cid:3) (cid:29)The state of the

and HMM in a hybrid system for improved continuous(cid:2)speech

art in on(cid:2)line handwriting recognition(cid:3)(cid:30) IEEE Transactions on

recognition(cid:3)(cid:30) IEEE Transactions on Speech and Audio Pro(cid:8)

Pattern Analysis and Machine Intel ligence(cid:3) vol(cid:8) (cid:16)(cid:3) no(cid:8) (cid:5)(cid:15)(cid:3) pp(cid:8)

cessing(cid:3) vol(cid:8) (cid:15)(cid:3) no(cid:8) (cid:5)(cid:3) pp(cid:8) (cid:15)(cid:5)(cid:7)(cid:31)(cid:15)(cid:15)(cid:21)(cid:3) (cid:5)(cid:17)(cid:17)(cid:21)(cid:8)

(cid:7)(cid:16)(cid:7)(cid:31)(cid:16)(cid:6)(cid:16)(cid:3) (cid:5)(cid:17)(cid:17)(cid:6)(cid:8)

(cid:26)(cid:17)(cid:5)(cid:27) Ofer Matan(cid:3) Henry S(cid:8) Baird(cid:3) Jane Bromley(cid:3) Christopher J(cid:8) C(cid:8)

(cid:26)(cid:5)(cid:5)(cid:5)(cid:27) S(cid:8) Manke and U(cid:8) Bodenhausen(cid:3) (cid:29)A connectionist recognizer for

Burges(cid:3) John S(cid:8) Denker(cid:3) Lawrence D(cid:8) Jackel(cid:3) Yann Le Cun(cid:3) Ed(cid:2)

on(cid:2)line cursive handwriting recognition(cid:3)(cid:30) in International Con(cid:8)

win P(cid:8) D(cid:8) Pednault(cid:3) William D(cid:8) Satter(cid:19)eld(cid:3) Charles E(cid:8) Stenard(cid:3)

ference on Acoustics(cid:16) Speech(cid:16) and Signal Processing(cid:3) Adelaide(cid:3)

and Timothy J(cid:8) Thompson(cid:3) (cid:29)Reading handwritten digits(cid:9) A

(cid:5)(cid:17)(cid:17)(cid:21)(cid:3) vol(cid:8) (cid:15)(cid:3) pp(cid:8) (cid:14)(cid:18)(cid:18)(cid:31)(cid:14)(cid:18)(cid:14)(cid:8)

ZIP code recognition system(cid:3)(cid:30) Computer(cid:3) vol(cid:8) (cid:15)(cid:20)(cid:3) no(cid:8) (cid:7)(cid:3) pp(cid:8)

(cid:26)(cid:5)(cid:5)(cid:15)(cid:27) M(cid:8) Gilloux and M(cid:8) Leroux(cid:3)

(cid:29)Recognition of cursive script

(cid:20)(cid:17)(cid:31)(cid:14)(cid:15)(cid:3) July (cid:5)(cid:17)(cid:17)(cid:15)(cid:8)

amounts on postal checks(cid:3)(cid:30) in European Conference dedicated

(cid:26)(cid:17)(cid:15)(cid:27) Y(cid:8) Bengio and Y(cid:8) Le Cun(cid:3) (cid:29)Word normalization for on(cid:2)line

to Postal Technologies(cid:3) Nantes(cid:3) France(cid:3) June (cid:5)(cid:17)(cid:17)(cid:18)(cid:3) pp(cid:8) (cid:7)(cid:6)(cid:20)(cid:31)

handwritten word recognition(cid:3)(cid:30) in Proc(cid:6) of the International

(cid:7)(cid:5)(cid:15)(cid:8)

Conference on Pattern Recognition(cid:3) IAPR(cid:3) Ed(cid:8)(cid:3) Jerusalem(cid:3)

(cid:26)(cid:5)(cid:5)(cid:18)(cid:27) D(cid:8) Guillevic and C(cid:8) Y(cid:8) Suen(cid:3) (cid:29)Cursive script recognition applied

(cid:5)(cid:17)(cid:17)(cid:21)(cid:3) IEEE(cid:8)

to the processing of bank checks(cid:3)(cid:30) in Int(cid:6) Conf(cid:6) on Document

(cid:26)(cid:17)(cid:18)(cid:27) R(cid:8) Vaillant(cid:3) C(cid:8) Monrocq(cid:3) and Y(cid:8) LeCun(cid:3) (cid:29)Original approach

Analysis and Recognition(cid:3) Montreal(cid:3) Canada(cid:3) August (cid:5)(cid:17)(cid:17)(cid:20)(cid:3) pp(cid:8)

for the localization of ob jects in images(cid:3)(cid:30) IEE Proc on Vision(cid:16)

(cid:5)(cid:5)(cid:31)(cid:5)(cid:21)(cid:8)

Image(cid:16) and Signal Processing(cid:3) vol(cid:8) (cid:5)(cid:21)(cid:5)(cid:3) no(cid:8) (cid:21)(cid:3) pp(cid:8) (cid:15)(cid:21)(cid:20)(cid:31)(cid:15)(cid:20)(cid:6)(cid:3)

(cid:26)(cid:5)(cid:5)(cid:21)(cid:27) L(cid:8) Lam(cid:3) C(cid:8) Y(cid:8) Suen(cid:3) D(cid:8) Guillevic(cid:3) N(cid:8) W(cid:8) Strathy(cid:3) M(cid:8) Cheriet(cid:3)

August (cid:5)(cid:17)(cid:17)(cid:21)(cid:8)

K(cid:8) Liu(cid:3) and J(cid:8) N(cid:8) Said(cid:3) (cid:29)Automatic processing of information

(cid:26)(cid:17)(cid:21)(cid:27) R(cid:8) Wolf and J(cid:8) Platt(cid:3) (cid:29)Postal address block location using a

on checks(cid:3)(cid:30) in Int(cid:6) Conf(cid:6) on Systems(cid:16) Man (cid:22) Cybernetics(cid:3)

convolutional locator network(cid:3)(cid:30) in Advances in Neural Infor(cid:8)

Vancouver(cid:3) Canada(cid:3) October (cid:5)(cid:17)(cid:17)(cid:20)(cid:3) pp(cid:8) (cid:15)(cid:18)(cid:20)(cid:18)(cid:31)(cid:15)(cid:18)(cid:20)(cid:16)(cid:8)

mation Processing Systems (cid:3)(cid:3) J(cid:8) D(cid:8) Cowan(cid:3) G(cid:8) Tesauro(cid:3) and

(cid:26)(cid:5)(cid:5)(cid:20)(cid:27) C(cid:8) J(cid:8) C(cid:8) Burges(cid:3) J(cid:8) I(cid:8) Ben(cid:3) J(cid:8) S(cid:8) Denker(cid:3) Y(cid:8) LeCun(cid:3) and C(cid:8) R(cid:8)

J(cid:8) Alspector(cid:3) Eds(cid:8) (cid:5)(cid:17)(cid:17)(cid:21)(cid:3) pp(cid:8) (cid:7)(cid:21)(cid:20)(cid:31)(cid:7)(cid:20)(cid:15)(cid:3) Morgan Kaufmann Pub(cid:2)

Nohl(cid:3) (cid:29)O(cid:10) line recognition of handwritten postal words using

lishers(cid:3) San Mateo(cid:3) CA(cid:8)

neural networks(cid:3)(cid:30) Int(cid:6) Journal of Pattern Recognition and Ar(cid:8)

(cid:26)(cid:17)(cid:20)(cid:27) S(cid:8) Nowlan and J(cid:8) Platt(cid:3) (cid:29)A convolutional neural network hand

ti(cid:2)cial Intel ligence(cid:3) vol(cid:8) (cid:7)(cid:3) no(cid:8) (cid:21)(cid:3) pp(cid:8) (cid:14)(cid:16)(cid:17)(cid:3) (cid:5)(cid:17)(cid:17)(cid:18)(cid:3) Special Issue

tracker(cid:3)(cid:30) in Advances in Neural Information Processing Sys(cid:8)

on Applications of Neural Networks to Pattern Recognition (cid:22)I(cid:8)

tems (cid:21)(cid:3) G(cid:8) Tesauro(cid:3) D(cid:8) Touretzky(cid:3) and T(cid:8) Leen(cid:3) Eds(cid:8)(cid:3) San Ma(cid:2)

Guyon Ed(cid:8)(cid:23)(cid:8)

teo(cid:3) CA(cid:3) (cid:5)(cid:17)(cid:17)(cid:20)(cid:3) pp(cid:8) (cid:17)(cid:6)(cid:5)(cid:31)(cid:17)(cid:6)(cid:16)(cid:3) Morgan Kaufmann(cid:8)

(cid:26)(cid:5)(cid:5)(cid:14)(cid:27) Y(cid:8) LeCun(cid:3) Y(cid:8) Bengio(cid:3) D(cid:8) Henderson(cid:3) A(cid:8) Weisbuch(cid:3) H(cid:8) Weiss(cid:2)

(cid:26)(cid:17)(cid:14)(cid:27) H(cid:8) A(cid:8) Rowley(cid:3) S(cid:8) Baluja(cid:3) and T(cid:8) Kanade(cid:3) (cid:29)Neural network(cid:2)

man(cid:3) and Jackel(cid:8) L(cid:8)(cid:3) (cid:29)On(cid:2)line handwriting recognition with

based face detection(cid:3)(cid:30) in Proceedings of CVPR(cid:12)(cid:15)(cid:3)(cid:8) (cid:5)(cid:17)(cid:17)(cid:14)(cid:3) pp(cid:8)

neural networks(cid:9) spatial representation versus temporal repre(cid:2)

(cid:15)(cid:6)(cid:18)(cid:31)(cid:15)(cid:6)(cid:16)(cid:3) IEEE Computer Society Press(cid:8)

sentation(cid:8)(cid:3)(cid:30) in Proc(cid:6) International Conference on handwriting

(cid:26)(cid:17)(cid:7)(cid:27) E(cid:8) Osuna(cid:3) R(cid:8) Freund(cid:3) and F(cid:8) Girosi(cid:3) (cid:29)Training support vector

and drawing(cid:6) (cid:5)(cid:17)(cid:17)(cid:18)(cid:3) Ecole Nationale Superieure des Telecommu(cid:2)

machines(cid:9) an application to face detection(cid:3)(cid:30) in Proceedings of

nications(cid:8)

CVPR(cid:12)(cid:15)(cid:3)(cid:8) (cid:5)(cid:17)(cid:17)(cid:7)(cid:3) pp(cid:8) (cid:5)(cid:18)(cid:6)(cid:31)(cid:5)(cid:18)(cid:14)(cid:3) IEEE Computer Society Press(cid:8)

(cid:26)(cid:5)(cid:5)(cid:7)(cid:27) U(cid:8) M uller(cid:3) A(cid:8) Gunzinger(cid:3) and W(cid:8) Guggenb uhl(cid:3) (cid:29)Fast neural

(cid:26)(cid:17)(cid:16)(cid:27) H(cid:8) Bourlard and C(cid:8) J(cid:8) Wellekens(cid:3) (cid:29)Links between Markov mod(cid:2)

net simulation with a DSP processor array(cid:3)(cid:30) IEEE Trans(cid:6) on

els and multilayer perceptrons(cid:3)(cid:30) in Advances in Neural Infor(cid:8)

Neural Networks(cid:3) vol(cid:8) (cid:14)(cid:3) no(cid:8) (cid:5)(cid:3) pp(cid:8) (cid:15)(cid:6)(cid:18)(cid:31)(cid:15)(cid:5)(cid:18)(cid:3) (cid:5)(cid:17)(cid:17)(cid:20)(cid:8)

mation Processing Systems(cid:3) D(cid:8) Touretzky(cid:3) Ed(cid:8)(cid:3) Denver(cid:3) (cid:5)(cid:17)(cid:16)(cid:17)(cid:3)

(cid:26)(cid:5)(cid:5)(cid:16)(cid:27) R(cid:8) Battiti(cid:3) (cid:29)First(cid:2) and second(cid:2)order methods for learning(cid:9) Be(cid:2)

vol(cid:8) (cid:5)(cid:3) pp(cid:8) (cid:5)(cid:16)(cid:14)(cid:31)(cid:5)(cid:16)(cid:7)(cid:3) Morgan(cid:2)Kaufmann(cid:8)

tween steepest descent and newton(cid:13)s method(cid:8)(cid:3)(cid:30) Neural Com(cid:8)

(cid:26)(cid:17)(cid:17)(cid:27) Y(cid:8) Bengio(cid:3) R(cid:8) De Mori(cid:3) G(cid:8) Flammia(cid:3) and R(cid:8) Kompe(cid:3) (cid:29)Neu(cid:2)

putation(cid:3) vol(cid:8) (cid:21)(cid:3) no(cid:8) (cid:15)(cid:3) pp(cid:8) (cid:5)(cid:21)(cid:5)(cid:31)(cid:5)(cid:14)(cid:14)(cid:3) (cid:5)(cid:17)(cid:17)(cid:15)(cid:8)

ral network (cid:2) gaussian mixture hybrid for speech recognition

(cid:26)(cid:5)(cid:5)(cid:17)(cid:27) A(cid:8) H(cid:8) Kramer and A(cid:8) Sangiovanni(cid:2)Vincentelli(cid:3) (cid:29)E!cient par(cid:2)

or density estimation(cid:3)(cid:30) in Advances in Neural Information

allel learning algorithms for neural networks(cid:3)(cid:30) in Advances in

Processing Systems (cid:5)(cid:3) J(cid:8) E(cid:8) Moody(cid:3) S(cid:8) J(cid:8) Hanson(cid:3) and R(cid:8) P(cid:8)

Neural Information Processing Systems(cid:3) D(cid:8)S(cid:8) Touretzky(cid:3) Ed(cid:8)(cid:3)

PROC(cid:2) OF THE IEEE(cid:3) NOVEMBER (cid:4)(cid:5)(cid:5)(cid:6)

(cid:9)(cid:12)

Denver (cid:5)(cid:17)(cid:16)(cid:16)(cid:3) (cid:5)(cid:17)(cid:16)(cid:17)(cid:3) vol(cid:8) (cid:5)(cid:3) pp(cid:8) (cid:21)(cid:6)(cid:31)(cid:21)(cid:16)(cid:3) Morgan Kaufmann(cid:3) San

Yoshua Bengio Yoshua Bengio received his

Mateo(cid:8)

B(cid:8)Eng(cid:8)

in electrical engineering in (cid:5)(cid:17)(cid:16)(cid:14) from

(cid:26)(cid:5)(cid:15)(cid:6)(cid:27) M(cid:8) Moller(cid:3) E(cid:23)cient Training of Feed(cid:8)Forward Neural Net(cid:8)

works(cid:3) Ph(cid:8)D(cid:8) thesis(cid:3) Aarhus University(cid:3) Aarhus(cid:3) Denmark(cid:3)

(cid:5)(cid:17)(cid:17)(cid:18)(cid:8)

(cid:26)(cid:5)(cid:15)(cid:5)(cid:27) S(cid:8) Becker and Y(cid:8) LeCun(cid:3) (cid:29)Improving the convergence of back(cid:2)

propagation learning with second(cid:2)order methods(cid:3)(cid:30) Tech(cid:8) Rep(cid:8)

CRG(cid:2)TR(cid:2)(cid:16)(cid:16)(cid:2)(cid:20)(cid:3) University of Toronto Connectionist Research

Group(cid:3) September (cid:5)(cid:17)(cid:16)(cid:16)(cid:8)

McGill University(cid:8) He also received a M(cid:8)Sc(cid:8)

and a Ph(cid:8)D(cid:8) in computer science from McGill

University in (cid:5)(cid:17)(cid:16)(cid:16) and (cid:5)(cid:17)(cid:17)(cid:5) respectively(cid:8)

In

(cid:5)(cid:17)(cid:17)(cid:5)(cid:2)(cid:5)(cid:17)(cid:17)(cid:15) he was a post(cid:2)doctoral fellow at the

Massachusetts Institute of Technology(cid:8) In (cid:5)(cid:17)(cid:17)(cid:15)

he joined AT(cid:4)T Bell Laboratories(cid:3) which later

became AT(cid:4)T Labs(cid:2)Research(cid:8)

In (cid:5)(cid:17)(cid:17)(cid:18) he

joined the faculty of the computer science de(cid:2)

partment of the Universit(cid:12)e de Montr(cid:12)eal where

he is now an associate professor(cid:8) Since his (cid:19)rst work on neural net(cid:2)

works in (cid:5)(cid:17)(cid:16)(cid:14)(cid:3) his research interests have been centered around learn(cid:2)

ing algorithms especially for data with a sequential or spatial nature(cid:3)

such as speech(cid:3) handwriting(cid:3) and time(cid:2)series(cid:8)

Yann LeCun Yann LeCun received a

Dipl#ome d(cid:13)Ing(cid:12)enieur from the Ecole Sup(cid:12)erieure

d(cid:13)Ing(cid:12)enieur en Electrotechnique et Electron(cid:2)

ique(cid:3) Paris in (cid:5)(cid:17)(cid:16)(cid:18)(cid:3) and a PhD in Computer

Science from the Universit(cid:12)e Pierre et Marie

Curie(cid:3) Paris(cid:3) in (cid:5)(cid:17)(cid:16)(cid:7)(cid:3) during which he proposed

an early version of the back(cid:2)propagation learn(cid:2)

ing algorithm for neural networks(cid:8) He then

joined the Department of Computer Science at

the University of Toronto as a research asso(cid:2)

ciate(cid:8) In (cid:5)(cid:17)(cid:16)(cid:16)(cid:3) he joined the Adaptive Systems

Patrick Ha(cid:11)ner Patrick Ha(cid:10)ner graduated

from Ecole Polytechnique(cid:3) Paris(cid:3) France in

(cid:5)(cid:17)(cid:16)(cid:7) and from Ecole Nationale Sup(cid:12)erieure des

T(cid:12)el(cid:12)ecommunications (cid:22)ENST(cid:23)(cid:3) Paris(cid:3) France in

(cid:5)(cid:17)(cid:16)(cid:17)(cid:8) He received his Ph(cid:8)D in speech and sig(cid:2)

nal processing from ENST in (cid:5)(cid:17)(cid:17)(cid:21)(cid:8)

In (cid:5)(cid:17)(cid:16)(cid:16)

and (cid:5)(cid:17)(cid:17)(cid:6)(cid:3) he worked with Alex Waibel on the

design of the TDNN and the MS(cid:2)TDNN ar(cid:2)

chitectures at ATR (cid:22)Japan(cid:23) and Carnegie Mel(cid:2)

lon University(cid:8) From (cid:5)(cid:17)(cid:16)(cid:17) to (cid:5)(cid:17)(cid:17)(cid:20)(cid:3) as a re(cid:2)

search scientist for CNET(cid:28)France(cid:2)T(cid:12)el(cid:12)ecom in

Research Department at AT(cid:4)T Bell Laboratories in Holmdel(cid:3) NJ(cid:3)

where he worked among other thing on neural networks(cid:3) machine

learning(cid:3) and handwriting recognition(cid:8) Following AT(cid:4)T(cid:13)s second

breakup in (cid:5)(cid:17)(cid:17)(cid:14)(cid:3) he became head of the Image Processing Services

Research Department at AT(cid:4)T Labs(cid:2)Research(cid:8)

Lannion(cid:3) France(cid:3) he developed connectionist learning algorithms for

telephone speech recognition(cid:8) In (cid:5)(cid:17)(cid:17)(cid:20)(cid:3) he joined AT(cid:4)T Bell Labora(cid:2)

tories and worked on the application of Optical Character Recognition

and transducers to the processing of (cid:19)nancial documents(cid:8) In (cid:5)(cid:17)(cid:17)(cid:7)(cid:3) he

joined the Image Processing Services Research Department at AT(cid:4)T

Labs(cid:2)Research(cid:8) His research interests include statistical and connec(cid:2)

He is serving on the board of the Machine Learning Journal(cid:3) and

tionist models for sequence recognition(cid:3) machine learning(cid:3) speech and

has served as associate editor of the IEEE Trans(cid:8) on Neural Networks(cid:8)

image recognition(cid:3) and information theory(cid:8)

He is general chair of the (cid:30)Machines that Learn(cid:30) workshop held every

year since (cid:5)(cid:17)(cid:16)(cid:14) in Snowbird(cid:3) Utah(cid:8) He has served as program co(cid:2)chair

of IJCNN (cid:16)(cid:17)(cid:3) INNC (cid:17)(cid:6)(cid:3) NIPS (cid:17)(cid:6)(cid:3)(cid:17)(cid:21)(cid:3) and (cid:17)(cid:20)(cid:8) He is a member of the

IEEE Neural Network for Signal Processing Technical Committee(cid:8)

He has published over (cid:7)(cid:6) technical papers and book chapters on

neural networks(cid:3) machine learning(cid:3) pattern recognition(cid:3) handwriting

recognition(cid:3) document understanding(cid:3) image processing(cid:3) VLSI design(cid:3)

and information theory(cid:8) In addition to the above topics(cid:3) his current

interests include video(cid:2)based user interfaces(cid:3) image compression(cid:3) and

content(cid:2)based indexing of multimedia material(cid:8)

L(cid:10)eon Bottou L(cid:12)eon Bottou received a Dipl#ome

from Ecole Polytechnique(cid:3) Paris in (cid:5)(cid:17)(cid:16)(cid:7)(cid:3) a

Magist$ere en Math(cid:12)ematiques Fondamentales et

Appliqu(cid:12)ees et Informatiques from Ecole Nor(cid:2)

male Sup(cid:12)erieure(cid:3) Paris in (cid:5)(cid:17)(cid:16)(cid:16)(cid:3) and a PhD

in Computer Science from Universit(cid:12)e de Paris(cid:2)

Sud in (cid:5)(cid:17)(cid:17)(cid:5)(cid:3) during which he worked on speech

recognition and proposed a framework for

stochastic gradient learning and global train(cid:2)

ing(cid:8) He then joined the Adaptive Systems Re(cid:2)

search Department at AT(cid:4)T Bell Laboratories

where he worked on neural network(cid:3) statistical learning theory and

local learning algorithms(cid:8) He returned to France in (cid:5)(cid:17)(cid:17)(cid:15) as a research

engineer at ONERA(cid:8) He then became chairman of Neuristique S(cid:8)A(cid:8)(cid:3)

a company making neural network simulators and tra!c forecast(cid:2)

ing software(cid:8) He eventually came back to AT(cid:4)T Bell Laboratories

in (cid:5)(cid:17)(cid:17)(cid:20) where he worked on graph transformer networks for optical

character recognition(cid:8) He is now a member of the Image Process(cid:2)

ing Services Research Department at AT(cid:4)T Labs(cid:2)Research(cid:8) Besides

learning algorithms(cid:3) his current interests include arithmetic coding(cid:3)

image compression and indexing(cid:8)


