5202
raM
4
]TS.htam[
4v62360.1042:viXra
Optimal linear prediction with
functional observations: Why you can
use a simple post-dimension reduction
estimator
Won-Ki Seo∗
Universityof Sydney
Abstract: We study the optimal linear prediction of a random function
that takes values in an infinite dimensional Hilbert space. We begin by
characterizingthemeansquarepredictionerror(MSPE)associatedwitha
linearpredictoranddiscussingtheminimalachievableMSPE.Thisanalysis
reveals that, in general, there are multiple non-unique linear predictors
thatminimizetheMSPE,andevenifauniquesolutionexists,consistently
estimating it from finite samples is generally impossible. Nevertheless, we
candefineasymptoticallyoptimallinearoperatorswhoseempiricalMSPEs
approachtheminimalachievablelevelasthesamplesizeincreases.Weshow
that, interestingly, standard post-dimension reduction estimators, which
havebeenwidelyusedintheliterature,attainsuchasymptotic optimality
underminimalconditions.
MSC2020 subject classifications:Primary60G25;secondary62J99.
Keywords and phrases: linear prediction, functional data, functional
linearmodels,regularization.
Contents
1 Introduction. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
2 Optimal linear prediction in Hilbert space . . . . . . . . . . . . . . . . 3
2.1 Preliminaries . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
2.2 Linear prediction in . . . . . . . . . . . . . . . . . . . . . . . 4
H
3 Estimation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
4 Discussions and extensions. . . . . . . . . . . . . . . . . . . . . . . . . 9
4.1 A more general result . . . . . . . . . . . . . . . . . . . . . . . . 9
4.2 Misspecified functional linear models and OLPO . . . . . . . . . 9
4.3 Requirement of sufficient dimension reduction . . . . . . . . . . . 10
4.4 Dimension reduction of the target variable . . . . . . . . . . . . . 10
5 Simulation. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
6 Concluding remarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
A Proofs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
A.1 Useful lemmas . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
A.2 Proofs of the theoretical results . . . . . . . . . . . . . . . . . . . 15
∗Won-KiSeoisthecorrespondingauthor.
1

Seo/Optimal linear prediction withfunctional observations 2
B Additional simulation results for alternative estimators . . . . . . . . . 20
References . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
1. Introduction
We study the optimal linear prediction in an arbitrary Hilbert space , and
H
how to estimate the optimal linear predictor. Given recent developments in
functionaldataanalysis,studiesonthissubjectholdsignificantimportanceand
relevance to many empirical applications; see e.g., [1], [2], [17], [18] and [23] to
name only a few recent papers. The reader is also referred to [8], [9] and [22]
containing an earlier mathematical exploration of this subject.
Let Y t t≥1 and X t t≥1 be stationary sequences of mean-zero random ele-
{ } { }
ments, both taking values in . Suppose that Y
t
=AoX
t
, where Ao is a contin-
H
uous linear operator,and it satisfies the followieng: for any arbitrarycontinuous
linear operator B,
E Y
t
Y
t
2 =E Y
t
AoX
t
2 E Y
t
BX
t
2, (1.1)
k − k k − k ≤ k − k
e
where is the norm defined on . We refer to Y
t
=AoX
t
as an optimal lin-
k·k H
ear predictor (OLP) and Ao as an optimal linear perediction operator (OLPO).
Given the observations Y ,X T , practitioners are often interested in con-
{
t t }t=1
structingapredictorY thatconverges(inprobability)toanOLPasT increases.
t
This is straightforwarbd when =R, and Y
t
and X
t
are real-valued mean-zero
H
random variables with positive variances. In this case, it is well known that
the unique solution to(1.1)is achieved by Ao = E[Y
t
X
t
]/E[X
t
2] and the min-
imal achievable mean squared prediction error (MSPE) is E Y
t
AoX
t
2 =
E[Y2] (E[Y X ])2/E[X2]. The conventionalplug-in-type estim k ato − r (orOL k S es-
t − t t t
t t h im e a w to ea r) k o la f w A o o f m la a r y ge b n e u d m e b fi e n r e s d h b o y lds A bfo = r { T Y − t 2 1 } P t≥1 T t= , 1 { Y X t t X 2 } t t / ≥ T 1 − a 1 n P d { T t= X 1 t X Y t t 2 } . t≥ W 1, h w en e
find that the following two results hold: (i) A
→ p
Ao =E[X
t
Y
t
]/E[X
t
2] and (ii)
T−1 T (Y AX )2 E[Y2] (E[X Y ])b2/E[X2], where and hereafter
Pt=1 t − t → p t − t t t → p
denotes the convbergence in probability with respect to the norm of . That is,
H
a consistentestimator of the OLPO can be constructed from the given observa-
tions.Furthermore,the empiricalMSPEobtainedfromthe estimatorconverges
to the minimal achievable MSPE.
However,inamoregeneralsituationwhereY andX takevaluesinapossibly
t t
infinite dimensional , it is generally impossible to obtain parallel results. To
H
see this with a simple example, suppose that Y t ,X t t≥1 satisfies the following:
{ }
for t 1,
≥
Y
t
=AoX
t
+ε
t
, Ao =
X
a
j
f
j ⊗
f
j
, a
≤|
a
j |≤
a, a,a>0, (1.2)
j≥1
where ε t is independent of X t and f j j≥1 is an orthonormal basis of . Ao
{ } H
specifiedin(1.2)is obviouslythe OLPO,but Ao is notconsistently estimable in

Seo/Optimal linear prediction withfunctional observations 3
general. As will be detailed in Example 1, this is because, it is not possible to
estimate a for j >T from T observations unless (i) a simplifying condition on
j
A, such as a =a for all j m for some finite m, holds and (ii) researchersare
j
≥
aware of this condition and use it appropriately.Even in the simple case where
Y t =A¯ oX t +ε t withA¯ o =aI (I denotesthe identity map)fora R,consistent
estimation of A¯ o is impossible for the same reason if we do no ∈ t know such a
simple structure of A¯ o and hence allow a more general case given in(1.2)(note
that, A¯ o =aI is a special case of Ao in(1.2)with a j =a).
Does this mean that it is impossible to statistically solve the optimal linear
prediction problem in this general setting? The answer is no. We will demon-
strate that, under mild conditions, there exists the minimum MSPE achievable
by a linear predictor and it is feasible to construct a possibly inconsistent esti-
mator A such that the empirical MSPE, computed as T−1 T Y AX 2,
Pt=1k t − t k
convergbes to the minimum MSPE. Particularly, we show that a standarbd post
dimension-reduction estimator, obtained by (i) reducing the dimensionality of
the predictive variable X using the principal directions of its sample covari-
t
ance and then (ii) applying the least squares method to estimate the linear
relationshipbetween the resulting lower dimensionalpredictive variable and Y ,
t
is effective for linear prediction. This post dimension-reduction estimator has
been widely used due to its simplicity. Its statistical properties have been stud-
ied under technical assumptions, which are challenging to verify, such as those
concerning the eigenstructure of the covariance of X ; the reader is referred to,
t
e.g., [13] and [26], where the assumptions of [12] are adopted for function-on-
function regression models. We show that, without such assumptions, a naive
use of this simple post dimension-reduction estimator can be justified as a way
toobtainasolutionwhichasymptoticallyminimizestheMSPE.Wealsoextend
this finding to show that similar estimators, which involve further dimension
reduction of Y , also possess this desirable property under mild conditions.
t
The paper proceeds as follows: Section 2 characterizes the minimal achiev-
able MSPE by a linear predictor, followed by a discussion on the estimation
of an asymptotically optimal predictor in Section 3. Further discussions and
extensions are givenin Section 4. Section 5 provides simulation evidence of our
theoretical findings, and Section 6 contains concluding remarks. The appendix
includesmathematicalproofsofthetheoreticalresultsandsomeadditionalsim-
ulation results.
2. Optimal linear prediction in Hilbert space
2.1. Preliminaries
For the subsequent discussion, we introduce notation. Let be a separable
H
Hilbert space with inner product , and norm . For V , let V⊥ be
h· ·i k·k ⊂ H
the orthogonal complement to V. We let be the set of continuous linear
∞
L
operators, and let , ∗, ran , and ker denote the operator norm, ad-
∞
kTk T T T
joint, range, and kernel of , respectively. is self-adjoint if = ∗. is
T T T T T

Seo/Optimal linear prediction withfunctional observations 4
callednonnegativeif x,x 0for anyx ,andpositive ifalso x,x =0
hT i≥ ∈H hT i6
for any x 0 . For x,y , we let x y be the operator given by
∈ H \ { } ∈ H ⊗
z x,z y for z . is compact if = a v w for some
7→ h i ∈ H T ∈ L ∞ T Pj≥1 j j ⊗ j
orthonormal bases v j j≥1 and w j j≥1 and a sequence of nonnegative num-
{ } { }
bers a j j≥1 tending to zero; if is also self-adjoint and nonnegative (see [7],
p. 35 { ), w } e may assume that v T = w . For any compact and p N,
j j ∞
let kTk Sp be defined by kTk p Sp = P ∞ j=1 ap j and let S p b T e t ∈ he L set of com ∈ pact
operators
T
with
kTk
Sp <
∞
;
S
p is called the Schatten p-class.
S
1 (resp.
S
2)
is also referred to the trace (resp. Hilbert-Schmidt) class. It is known that the
followinghold: and 2 = ∞ w 2 foranyor-
kTk ∞ ≤kTk S2 ≤kTk S1 kTkS2 Pj=1kT j k
thonormalbasis w
j
j≥1.For any -valued mean-zerorandomelements Z and
{ } H
Z with E
k
Z
k
2 <
∞
and E
k
Z
k
2 <
∞
, their cross-covariance C ZZ˜ = E[Z
⊗
Z˜]
ies a Schatten 1-class operatoer; if Z = Z˜, it reduces to the covariance of Z and
E Z 2 = C holds.
k k k
ZZ
k
S1
2.2. Linear prediction in H
Consider a weakly stationary sequence Y t ,X t t≥1 with nonzero covariances
C = E[Y Y ] and C = E[X { X ], a } long with the cross-covariance
YY t t XX t t
C =E[X ⊗ Y ] (or equivalently C∗ ⊗ ). We hereafterwrite C and C as
XY t ⊗ t YX YY XX
their spectral representations as follows, if necessary:
C = κ u u , C = λ v v , (2.1)
YY X j j ⊗ j XX X j j ⊗ j
j≥1 j≥1
where κ1 κ2 ... 0, λ1 λ2 ... 0, and u j j≥1 and v j j≥1
≥ ≥ ≥ ≥ ≥ ≥ { } { }
are orthonormal sets of . Unless otherwise stated, we assume that C and
YY
H
C are not finite rank operators and thus there are infinitely many nonzero
XX
eigenvaluesin(2.1),whichisasusuallyassumedforcovariancesofHilbert-valued
random elements in the literature on functional data analysis.
Note first that, for any B , E Y BX 2 = E[(Y BX ) (Y
∞ t t t t t
∈ L k − k k − ⊗ −
BX )] and hence the MSPE associated with B can be written as follows:
t
k
S1
E Y BX 2 = C C B∗ BC∗ +BC B∗ . (2.2)
k t − t k k YY − XY − XY XX k S1
Next,weprovidethemainresultofthissection,whichnotonlygivesusauseful
characterization of the MSPE in(2.2), but also provides essential preliminary
results for the subsequent discussion.
Proposition 2.1. For any B , there exists a unique element R
∞ XY ∞
1/2 1/ ∈2 L ∈L
such that C =C R C and
XY YY XY XX
E Y BX 2 = C C 1/2 R R∗ C 1/2 + BC 1/2 C 1/2 R 2 . (2.3)
k t − t k k YY − YY XY XY YYk S1 k XX− YY XY kS2
Proposition 2.1 shows that the MSPE associated with B is the sum
∞
∈ L
of the Schatten 1- and 2-norms of specific operators dependent on C , C ,
YY XX

Seo/Optimal linear prediction withfunctional observations 5
R and B; notably, only the latter term ( BC 1/2 C 1/2 R 2 ) in(2.3)de-
XY k XX− YY XY kS2
pendsonB.Thus,the formerterm( C C 1/2 R R∗ C 1/2 )represents
k YY − YY XY XY YYk S1
the minimal achievable MSPE by a linear predictor, while the latter can be un-
derstood as a measure of the inadequacy of B as a linear predictor. If w j j≥1
{ }
is an orthonormalbasis of , this inadequacy becomes zero if and only if
H
(BC 1/2 C 1/2 R )w 2 =0 for all j 1. (2.4)
k XX − YY XY j k ≥
Based on these findings, we obtain the following two characterizations of an
OLPO in Corollary 2.1: the first is a direct consequence of(2.4)and the Hahn-
Banach extension theorem (see, e.g., Theorem 1.9.1 of [21]), which, in turn,
implies the second due to the fact that C = C
1/2
R C
1/2
as observed in
XY YY XY XX
Proposition 2.1.
Corollary 2.1. A is an OLPO if and only if any of the following equivalent
conditions holds: (a) AC
1/2
=C
1/2
R and (b) AC =C .
XX YY XY XX XY
Condition (b) follows directly from condition (a) and Proposition 2.1, and it
is notably alignwith the characterizationof an OLPO providedby [8]; Remark
2.1 outlines the distinctions between our findings and the existing result in
more detail. From Corollary 2.1, we find that the minimum MSPE is attained
by A satisfying AC = C (or AC
1/2
= C
1/2
R ). However, such
∈ L ∞ XX XY XX YY XY
an operator A is not uniquely determined; particularly, the equation does not
specifyhowAactsonkerC ,allowingAtoagreewithanyarbitraryelementin
XX
onkerC (seeRemark2.2).WhenC isnotinjective,therearemultiple
∞ XX XX
L
choices of A that achieve the minimum MSPE. In infinite dimensional settings,
the injectivity of C is a stringent assumption, and verifying this condition
XX
from finite observations is impractical. Thus, pursuing prediction under the
existence of the unique OLPO, as in the standard univariate or multivariate
prediction, is restrictive. Even if a different setup is considered with a different
purpose,similar concernsabout the requirementsfor unique identificationwere
recently raised by [3], and the enthusiastic reader is referred to their paper for
more detailed discussion on the topic.
Remark 2.1. Condition (b) in Proposition 2.1 was earlier obtained as the
requirementforA tobeanOLPObyPropositions2.2-2.3of[8].Compared
∞
∈L
with this earlier result, Proposition 2.1 not only provides more general results,
suchastheexplicitexpressionofthegapbetweentheminimalattainableMSPE
and the MSPE associated with any B , but it also employs a distinct
∞
∈ L
approach. The result of [8] relies on the notion of a linearly closed subspace
and an extension of the standard projection theorem, while Proposition 2.1 is
establishedbyanalgebraicproofbasedontherepresentationofcross-covariance
operators in [4].
Remark2.2. ByinvokingtheHahn-Banachextensiontheorem(Theorem1.9.1
of[21]),wemayassumethatAsatisfyingAC =C is auniquecontinuous
XX XY
linearmapdefinedontheclosureofranC ,whichisnotequalto ifC is
XX XX
H
not injective. Thus, if there exists another continuous linear operator A which
e

Seo/Optimal linear prediction withfunctional observations 6
agrees with A on the closure of ranC but not on [ranC ]⊥, then A also
XX XX
satisfies that E Y AX 2 = C C 1/2 R R∗ C 1/2 . e
k t − t k k YY − YY XY XY YYk S1
e
The resultsgivenin Proposition2.1andRemark2.2 imply that, particularly
when the predictive variable is function-valued, there may be multiple OLPOs
that satisfy (1.1). Furthermore, even if a unique OLPO exists, it may not be
consistently estimable; a more detailed discussion is given in Example 1 below.
This means that we are in a somewhat different situation from the previous
simple univariate case discussed in Section 1, where we can estimate the OLP
by consistently estimating the OLPO.
Example 1. Suppose that Y t ,X t t≥1 satisfies(1.2), X t has a positive covari-
{ }
ance C , and ε is serially independent and also independent of X for any
XX t s
s. In this case, AoC
XX
= C
XY
, making Ao an OLPO. Since C
XX
is injective,
anycontinuouslinearoperatorAagreeswithAo onthe closureofranC
XX
also
agreeswithAo on (seeRemarek2.2andnotethattheclosureofranC
XX
is
H H
in this case). However, consistently estimating Ao without any further assump-
tions is impossible. To illustrate this, we may consider the case where f j j≥1
{ }
in(1.2)are known for simplicity. Even in this simplified scenario, there are in-
finitelymanyunknownparameters a
j
j≥1tobeestimatedfromonlyT samples,
{ }
necessitating additional assumptions on a j j≥1 for consistent estimation.
{ }
3. Estimation
We observedthat in a general Hilbert space setting, there can be not only mul-
tiple OLPOsbutalsoinstanceswhere,evenifaunique OLPOexists,consistent
estimationof it is impossible. Nevertheless,under mild conditions, we may con-
struct a predictor using a standard post dimension-reduction estimator in such
awaythatthe associatedempiricalMSPEconvergestothe minimumMPSEas
in the simple univariate case. To propose such a predictor, let
1
T kT kT
C = X Y , C = λˆ vˆ vˆ , C−1 = λˆ−1vˆ vˆ ,
XY T X t ⊗ t XX,kT X j j ⊗ j XX,kT X j j ⊗ j
b t=1 b j=1 b j=1
(3.1)
wherek
T
isanintegersatisfyingthefollowingassumption:below,weleta1 a2 =
min a1,a2 fora1,a2 R andassumethatmax j≥1 j :E j =1 ifthe con ∧ dition
{ } ∈ { }
E is not satisfied for all j 1.
j
≥
Assumption 1 (Elbow-like rule). k in(3.1)is given by
T
k
T
=m
j≥
a
1
x
n
j :λˆ
j ≥
λˆ j+1+τ
To∧
υ
T
−1,
where τ and υ are user-specific choices of positive constants decaying to 0 as
T T
T
→ ∞
, and both τ
T
−1 and υ
T
−1 are bounded above by γ0Tγ1 for some γ0 > 0
and γ1 (0,1/2).
∈

|       | Seo/Optimal                        |     | linear prediction | withfunctional |     | observations         |     |     | 7   |
| ----- | ---------------------------------- | --- | ----------------- | -------------- | --- | -------------------- | --- | --- | --- |
| C−1   | in(3.1)isunderstoodastheinverseofC |     |                   |                |     | viewedasamapactingon |     |     |     |
| XX,kT |                                    |     |                   |                |     | XX                   |     |     |     |
thebrestrictedsubspacespan vˆ kT andk inAsbsumption1isarandominteger
|     |     |     | j }j=1 | T   |     |     |     |     |     |
| --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- |
{
by its construction (see Remark 3.1 below). By including υ T in Assumption 1,
we ensure that k becomes o (T1/2), which facilitates our theoretical analysis.
|     | T   |     | p   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Even if the choice of k depends on various contexts requiring a regularized
T
inverse of C XX , it is commonly set to a much smaller number than T, and
thus this cobndition does not impose any practical restrictions. A practically
λˆ
| more meaningful | decision |     | is made | by the | first component, |     | max | j≥1 j | : j |
| --------------- | -------- | --- | ------- | ------ | ---------------- | --- | --- | ----- | --- |
|                 |          |     |         |        |                  |     |     | {     | ≥   |
λˆ + τ in Assumption 1. Firstly, given that λˆ > 0, this condition
| j+1             | T              |     |     |     |              | kT +1 |     |     |     |
| --------------- | -------------- | --- | --- | --- | ------------ | ----- | --- | --- | --- |
| impliesthatλˆ−1 | } <τ−1,andthus |     | C−1 |     | τ−1,whereτ−1 |       |     |     |     |
divergesslowly
|     | kT  | T   | k   | XX,kTk | ∞ ≤ T |     | T   |     |     |
| --- | --- | --- | --- | ------ | ----- | --- | --- | --- | --- |
compared to T; clearly, this is onebof the essential requirements for k (as the
T
rank of a regularized inverse of C ) to satisfy in the literature employing
XX
similar regularized inverses. Seconbdly, k is determined near the point where
T
| λˆ  | λˆ  |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
the gap j j+1 is no longer smaller than a specified threshold τ T for the last
−
time. This approach is, in fact, analogous to the standard elbow rule used to
determinethenumberofprincipalcomponentsinmultivariateanalysisbasedon
the scree plot. Thus, even if Assumption 1 details some specific mathematical
requirements necessary for our asymptotic analysis, these seem to closely align
with existing practical rules for selecting k T , commonly employed in current
| practice; | for example, | see | [10]. |     |     |     |     |     |     |
| --------- | ------------ | --- | ----- | --- | --- | --- | --- | --- | --- |
C−1
| Using | the regularized | inverse |     | ,   | the proposed | predictor |     | is constructed |     |
| ----- | --------------- | ------- | --- | --- | ------------ | --------- | --- | -------------- | --- |
XX,kT
| as follows: |     |     | b   |     |     |     |     |     |     |
| ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
T kT
|       |         |       | C−1  |       | 1   |        | λˆ−1 |         |       |
| ----- | ------- | ----- | ---- | ----- | --- | ------ | ---- | ------- | ----- |
| Y =AX | , where | A()=C |      |       | ()= |        | vˆ   | , vˆ ,X | Y .   |
| t     | t       |       | · XY | XX,kT | · T | XX     | j h  | j ·ih j | t i t |
| b b   |         | b     | b b  |       |     | t=1j=1 |      |         |       |
(3.2)
The above predictor is standard in the literature on functional data analysis
as the least squares predictor of Y given the projection of X onto the space
|     |     |     | t   |     |     |     | t   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
spanned by the eigenvectorscorrespondingto the first k T largesteigenvalues;a
similarestimatorwasearlierconsideredby [23].The predictordescribedin(3.2)
andits modifications(suchasthosethat willbe consideredinSection4.4)have
been widely studied in the literature and adapted to various contexts; see e.g.,
| [1], [2], [7], | [17], [20], | [27] | and [28]. |     |     |     |     |     |     |
| -------------- | ----------- | ---- | --------- | --- | --- | --- | --- | --- | --- |
Remark 3.1. A significant difference in A compared to most of the existing
estimators,liesinthechoiceofk .Inmanybearlierarticles,k isdirectlychosen
|     |     |     | T   |     |     |     | T   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
by researchers and thus regarded as deterministic. However, as pointed out by
[26], even in this case,it is generally not recommended to choose k arbitrarily
T
λˆ
without taking the eigenvalues j into account. Therefore, it is natural to view
| k as random | from | a practical | point | of view. |     |     |     |     |     |
| ----------- | ---- | ----------- | ----- | -------- | --- | --- | --- | --- | --- |
T
ToestablishconsistencyofAin(3.2),certainassumptionshavebeenemployed
in the aforementioned literatubre, particularly concerning the eigenstructure of
C and the unique identification of the target estimator A. However, as illus-
XX
trated by Example 1 and Remark 2.2, these assumptions are not guaranteedto

Seo/Optimal linear prediction withfunctional observations 8
hold even in cases where there exists a well defined OLPO.Thus, in this paper,
wedo notmakesuchassumptionsforconsistency,but allowAto potentially be
inconsistent. Our main result in this section is that, despitebnot relying on the
typical assumptions for consistency, the predictor Y asymptotically minimizes
t
theMSPE,whichonlyrequiresmildconditionsonthbesample(cross-)covariance
operators.
For the subsequent discussion, we define the following: for any B ,
∞
∈L
T
1
Σ(B)= (Y BX ) (Y BX ).
T X t − t ⊗ t − t
t=1
Notethat Σ(B) isequivalenttotheempiricalMSPE,givenbyT−1 T Y
BX 2. W k e then k S d 1 efine an asymptotically OLPO as a random bound P ed t= li 1 n k ea t r −
t
k
operatorproducingapredictorthatasymptoticallyminimizes the MSPEinthe
following sense:
Definition 1. Any random bounded linear operator B is called an asymptoti-
cally OLPO if b
k
Σ(B)
k
S1
→
p
Σmin as T
→∞
,
where Σmin =
k
C
YY −
C
Y
1/
Y
b2 R
XY
R
X
∗
Y
C
Y
1/
Y
2
k S1
, which is the minimum MSPE
that can be achieved by a linear predictor, as defined in Proposition 2.1.
We will employ the following assumption:
Assumption 2 (Standard rate of convergence). C C , C
YY YY ∞ XX
k − k k −
C , and C C are O (T−1/2). b b
XX ∞ XY XY ∞ p
k k − k
b
We observe that Y
t
Y
t
C
YY
t≥1, X
t
X
t
C
XX
t≥1, and X
t
{ ⊗ − } { ⊗ − } { ⊗
Y t C XY t≥1 are stationary sequences of Schatten 2-class operators. These
− }
operator-valued sequences may also be understood as stationary sequences in
a separable Hilbert space ([7], p. 34). Then, Assumption 2 is satisfied under
some non-restrictive regularity conditions; see e.g., Theorems 2.16-2.18 of [7]
concerningthecentrallimittheoremsforHilbert-valuedrandomelements.Given
thatwearedealingwithaweaklystationarysequence Y
t
,X
t
t≥1,Assumption2
{ }
appearstobestandard.Furthermore,itcanbe relaxedtoaweakerrequirement
by imposing stricter conditions on τ and υ in Assumption 1; this will be
T T
detailed in Section 4.1.
We now present the main result of this paper.
Theorem 3.1. Under Assumptions 1-2, A is an asymptotically OLPO, i.e.,
k
Σ(A)
k
S1
→
p Σmin. b
b
As stated, an appropriate growth rate of k , detailed in Assumption 1, and
T
thestandardrateofconvergenceofthesample(cross-)covarianceoperators(As-
sumption 2) are all that we need in order to demonstrate that the proposed
predictor in (3.2) is an asymptotically OLPO. As discussed earlier, since the
choice rule in Assumption 1 is practically similar to existing rules for select-
ing k , a naive use of the simple post-dimension reduction estimator A in(3.2)
T
b

|     |     | Seo/Optimal | linear | prediction | withfunctional |     | observations |     | 9   |
| --- | --- | ----------- | ------ | ---------- | -------------- | --- | ------------ | --- | --- |
without the usual assumptions for the unique identification and/or consistency,
which are widely employed but challenging to verity, can still be justified as a
way to asymptotically minimize the MSPE (see Section 4.2). Some discussions
andextensionsonthe abovetheoremaregiveninthe nextsection.We alsocon-
sider the case where A is replaced by another estimator employing a different
| regularizationschemeb(Sections |      |         |            | 4.3-4.4). |     |     |     |     |     |
| ------------------------------ | ---- | ------- | ---------- | --------- | --- | --- | --- | --- | --- |
| 4. Discussions                 |      | and     | extensions |           |     |     |     |     |     |
| 4.1. A                         | more | general | result     |           |     |     |     |     |     |
In Assumption 2, we assumed that the sample (cross-)covariances converge to
the population counterparts with √T-rate. However, the exact √T-rate is not
mandatory for the desired result and can be relaxed if we make appropriate
| adjustments | on    | τ and   | υ in           | Assumption |            | 1 as follows: |             |         |         |
| ----------- | ----- | ------- | -------------- | ---------- | ---------- | ------------- | ----------- | ------- | ------- |
|             |       | T       | T              |            |            |               |             |         |         |
| Corollary   | 4.1.  |         |                |            |            |               |             |         |         |
|             |       | Suppose | that,          | for        | β (0,1/2], |               | C YY        | C YY ∞  | , C XX  |
|             |       |         |                |            | ∈          |               | k           | − k     | k −     |
| C           | , and | C       | C              | are        | O (T−β).   | If            | Assbumption | 1 holds | fobr γ1 |
| XX ∞        |       | XY      | XY             | ∞          | p          |               |             |         |         |
| k           |       | k       | −              | k          |            |               |             |         | ∈       |
| (0,β), then | A     | isban   | asymptotically |            | OLPO.      |               |             |         |         |
b
| That | is, A | is an | asymptotically |     | OLPO | under | weaker | assumptions | on the |
| ---- | ----- | ----- | -------------- | --- | ---- | ----- | ------ | ----------- | ------ |
sample(crossb-)covarianceoperatorsifweimposestricterconditionsonthedecay
rates of τ and υ . τ and υ are user-specific choices dependent on T, so
|     | T   | T   | T   | T   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
researcherscan easily manipulate their decay rates. Corollary4.1, thus, tells us
that we can make A an asymptotically OLPO under more general scenarios by
| simply            | reducing | thebdecay | rates      | of     | τ T and | υ T . |      |     |     |
| ----------------- | -------- | --------- | ---------- | ------ | ------- | ----- | ---- | --- | --- |
| 4.2. Misspecified |          |           | functional | linear | models  | and   | OLPO |     |     |
| Consider          | the      | standard  | functional | linear | model   |       |      |     |     |
E[X
|     |     |     | Y =AX | +ε  | ,   | ε   | ]=0, |     | (4.1) |
| --- | --- | --- | ----- | --- | --- | --- | ---- | --- | ----- |
|     |     |     | t     | t   | t   | t ⊗ | t    |     |       |
where A is typically assumed to satisfy the following two conditions: (i) A is
Hilbert-Schmidt (i.e., A 2) and (ii) A is uniquely identified (in 2). A com-
|     |     |     | ∈ S |     |     |     |     | S   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
monassumptionemployedfortheuniqueidentificationisthatranC isdense
XX
in (see Remark 2.2). Under additional technical assumptions on the eigen-
H
structure of C XX , such as those on the spectral gap (λ j λ j−1) as in [12], the
−
proposed estimator A in(3.2)turns out to be consistent if k grows appropri-
T
ately.Ofcourse,somebalternativeestimators,suchastheleastsquaresestimator
with Tikhonov regularization(see, e.g., [6]), do not require any assumptions on
the spectral gap, which is a well known advantage of such methods. However,
theystillrequirecondition(i),theHilbert-SchmidtpropertyofA,withsomead-
ditionalregularityassumptionsonA,andalsoimposeassumptionsforcondition
| (ii) when | discussing |     | the consistency |     | of those | estimators. |     |     |     |
| --------- | ---------- | --- | --------------- | --- | -------- | ----------- | --- | --- | --- |

Seo/Optimal linear prediction withfunctional observations 10
Itiscrucialtohaveconditions(i)and(ii)togetherwiththemodel(4.1),since
a violation of either of these can easily lead to inconsistency (see Example 1).
While these requirements are standard for estimation, they exclude many nat-
ural data generating mechanisms in (see Examples 1-2). Consequently, the
H
model may suffer from misspecificationissues. However,even in such cases,our
results show that A attains the minimum MSPE asymptotically if k grows
T
at an appropriate rbate, and thus affirm the potential use of the standard post-
dimensionreductionestimator in practice without a carefulexaminationof var-
ious technical conditions.
Example 2. Similar to the example given in Section 5 of [5], suppose that
X t =Y t−1 andY t satisfies the functional AR(1) law ofmotion: Y t =AY t−1+ε t
∞
with A = Pj=1 a j w j ⊗ w j for some orthonormal basis { w j } j≥1 and iid se-
quence ε t t∈Z (seealso[2]).ThisleadstothefollowingpointwiseAR(1)model:
{ }
Y
t
,w
j
= a
j
Y t−1,w
j
+ ε
t
,w
j
. This time series is guaranteed to be station-
h i h i h i
ary if sup a < 1 (this can be demonstrated with only a slight and obvious
j| j |
modification of Theorem 3.1 of [7] or Proposition 3.2 of [25]). On the other
hand, for A to be a Hilbert-Schmidt operator,a much more stringentcondition
P
∞
j=1|
a
j |
2 <
∞
is required,and,asaconsequence,
h
Y
t
,w
j i
and
h
Y t−1,w
j i
need
to be nearly uncorrelated for large j while they can be arbitrarily correlated
under the aforementioned condition for weak stationarity.
4.3. Requirement of sufficient dimension reduction
In our proof of Theorem 3.1 (resp. Corollary 4.1), it is crucial to have a reg-
ularized inverse of C , denoted as C−1 , whose rank k grows at a suffi-
XX XX,kT T
ciently slower rate tbhan T; see e.g.,(Ab.10)and(A.11)in Appendix A. Due to
this requirement,ourargumentsforprovingthe mainresults(Theorem3.1and
Corollary 4.1) are not straightforwardly extended to other popular estimators
without dimension reduction, such as least squares-type estimators with ridge
or Tikhonov regularization (see, e.g., [6]). Of course, this does not mean that
these estimators cannot achieve prediction results similar to those in Theorem
3.1 and Corollary 4.1 without requiring the standard assumptions of Hilbert-
Schmidtness andthe unique identificationofA.Rather,it simply suggeststhat,
for estimators computed without dimension reduction, alternative approaches
may be needed under different sets of assumptions. This could be a potential
direction for future research.
4.4. Dimension reduction of the target variable
TheproposedestimatorAiscommonlyusedasastandardestimatorofthefunc-
tionallinearmodel(4.1).bNotethat,inourconstructionofA,thetargetvariable
Y
t
is used as is, without any dimension reduction. In the lbiterature, estimators
similar to A in(3.2), but where Y is replacedwith its versionobtained through
t
dimension breduction, also appear to be popular and are used in practice (see

Seo/Optimal linear prediction withfunctional observations 11
e.g.,[27]).Asnotedinrecentarticles,dimensionreductionofthetargetvariable
not only is generally non-essentialfor establishing certain key asymptotic prop-
erties (suchas consistency)ofthe estimator but mayalso leadto a less optimal
estimator (see Remark 1 of [13]).
Consider the following predictor and estimator, constructed using a version
of Y with reduced dimension:
t
Y =AX ,
t t
e e
where
1
T kT
A()=Π C C−1 ()= λˆ−1 vˆ , vˆ ,X Π Y
· Y,ℓT XY XX,kT · T XX j h j ·ih j t i Y,ℓT t
e b b b t=1j=1 b
andΠ Y,ℓT = P ℓ j T =1 wˆ j ⊗ wˆ j forsomeorthonormalbasis { wˆ j } j≥1 andℓ T growing
asTbincreases;often, wˆ issettothe eigenvectorofC orC corresponding
j YY XX
to the j-th largest eigenvalue, but our subsequent abnalysisbis not restricted to
these specific cases. Even if the additional dimension reduction applied to Y
t
introduces some complications in our theoretical analysis, it can also be shown
that A is an asymptotically OLPO under an additional mild condition.
e
Corollary 4.2. Suppose that Assumptions 1-2 hold and there exists a sequence
m tending to infinity as T such that m Π C
1/2
C
1/2
0 and
T →∞ T k Y,ℓT YY − YYk ∞ → p
m ℓ eventually. Then, A is an asymptoticalbly OLPO.
T T
≤
e
Given that Π is the orthogonal projection with a growing rank, the con-
Y,ℓT
dition given inbCorollary 4.2 becomes easier to be satisfied if ℓ grows more
T
rapidly. Viewed in this light, the scenario in Theorem 3.1 can be seen as the
limiting case where Π =I, indicating no dimension reduction applied to Y .
Y,ℓT t
Corollary 4.2 tells ubs that estimators obtained by reducing the dimensionality
of Y tend to be asymptotically OLPOs under non-restrictiveconditions. Given
t
that C
1/2
and C share the same eigenvectors, one may conjecture that sat-
YY YY
isfying the requirement in Corollary 4.2 could be easier if w is an eigenvector
j
ofC
YY
. Theoreticaljustification ofthis conjecture may requbirefurther assump-
tionbsontheeigenstructureofC .Giventhefocusonoptimallinearprediction
YY
under minimal conditions, we do not pursue this direction and leave it for
H
future study.
5. Simulation
We provide simulation evidence for our theoretical findings, focusing on cases
that have not been sufficiently explored in the literature and where consistent
estimationoftheOLPOisimpossible.Inallsimulationexperiments,thenumber
of replications is 1000.
Let f j j≥1 be the Fourier basis of L2[0,1], the Hilbert space of square-
{ }
integrable functions on [0,1], i.e., for x [0,1], f1(x) = 1, and for j 2,
∈ ≥

Seo/Optimal linear prediction withfunctional observations 12
f =√2sin(2πjx) if j is even, and f =√2cos(2πjx) if j is odd. We define X
j j t
and Y as follows: for some real numbers a 101,
t
{
j }j=1
101
Y
t
=AX
t
+ε
t
, X
t
=
X
a
j h
X t−1,f
j i
f
j
+e
t
,
j=1
where e t t≥1 and ε t t≥1 areassumedtobemutuallyandseriallyindependent
{ } { }
sequencesofrandomelements.Inthissimulationsetup,theminimalMSPEthat
can be achieved by a linear operator equals the Schatten 1-norm (trace norm)
of the covariance operator C of ε .
ε t
We conducted experiments in three different cases.In the first case (referred
to as Case BB), e and ε are set to independent realizations of the standard
t t
Brownian bridge, and in the second case (referred to as Case CBM), they are
set to independent realizations of the centered Brownian motion. In these first
two cases, the j-th largest eigenvalue of C is given by π−2j−2; see [14] (p. 86)
ε
and[16] (p.1465).Froma wellknownresultonthe Riemannzeta function (see
e.g., [15]), we find that these eigenvalues add up to 1/6, which is the minimal
achievableMSPE.Inthe lastcase(referredtoasCaseBM),e andε aresetas
t t
realizations of the standard Brownian motion multiplied by a constant, which
is properly chosen to ensure that the minimal MSPE in this scenario matches
that in the previous two cases.
We will subsequently consider the following four models, depending on the
specification of a and A for generating X and Y : for all j 1,
j t t
≥
Model1 (M1): a j iid U( 0.1,0.25)and Af j =b0f j ,
∼ −
Model2 (M2): a j iid U( 0.1,0.25)and Af j =b j f j 1 j 100 +f j 1 j >100 ,
∼ − { ≤ } { }
Model3 (M3): a j iid U( 0.1,0.75)and Af j =b0f j ,
∼ −
Model4 (M4): a j iid U( 0.1,0.75)and Af j =b j f j 1 j 100 +f j 1 j >100 ,
∼ − { ≤ } { }
whereb0 U[ 2.5,2.5]andb j iid U[ 2.5,2.5]forj 1.Note that the param-
∼ − ∼ − ≥
eters a
j
, b0 and b
j
are generated differently in each simulation run; this allows
us to assess the average performance of the proposed predictor across various
parameter choices. In many empirical examples involving dependent sequences
offunctionsX ,itisoftenexpectedthat X ,v foranyv exhibitsapositive
t t
h i ∈H
lag-oneautocorrelation,soweleta tendtotakepositivevaluesmorefrequently
j
inoursimulationsettings.Inanyoftheabovecases,Aisnon-compactandhence
cannot be consistently estimated without prior knowledge on the structure of
Af
j
j≥1, which is as in the operator considered in Example 1.
{ }
We set τ = 0.01 C Tγ and υ = 0.5Tγ for some γ > 0, where note
T
k
XX
k
S1 T
that τ
T
is designed tobreflect the scale of X
t
, as proposed by [26] in a similar
context.We then computed the empiricalMSPE associatedwith A,introduced
in(3.2). Table 1 reports the excess MSPE (the empirical MSPE mbinus 1/6) for
eachofthe consideredcases.As expectedfromourmaintheoreticalresults,the
excess MSPE associatedwith A approacheszero as the sample size T increases,
even if A is not consistent. Notbably, in Case BM, the excess MSPE tends to be
b

|     | Seo/Optimal |          | linear prediction | withfunctional |           | observations |         | 13  |
| --- | ----------- | -------- | ----------------- | -------------- | --------- | ------------ | ------- | --- |
|     |             |          |                   | Table 1        |           |              |         |     |
|     |             | Excess   | MSPE of           | the proposed   | predictor |              |         |     |
|     |             |          | (a)               | Case BB        |           |              |         |     |
|     |             | γ =0.475 |                   |                |           |              | γ =0.45 |     |
| T   | 50 100      | 200      | 400               | 800            | 50        | 100          | 200 400 | 800 |
M1 0.043 0.035 0.023 0.018 0.012 0.072 0.051 0.031 0.022 0.014
M2 0.041 0.033 0.022 0.017 0.011 0.068 0.049 0.029 0.021 0.013
M3 0.056 0.046 0.031 0.023 0.016 0.094 0.066 0.040 0.028 0.019
M4 0.054 0.044 0.029 0.022 0.015 0.089 0.063 0.038 0.027 0.018
|     |        |          | (b) | Case CBM |     |     |         |     |
| --- | ------ | -------- | --- | -------- | --- | --- | ------- | --- |
|     |        | γ =0.475 |     |          |     |     | γ =0.45 |     |
| T   | 50 100 | 200      | 400 | 800      | 50  | 100 | 200 400 | 800 |
M1 0.045 0.036 0.024 0.018 0.012 0.074 0.052 0.031 0.022 0.014
M2 0.041 0.034 0.022 0.017 0.011 0.069 0.049 0.029 0.020 0.013
M3 0.059 0.048 0.032 0.024 0.016 0.098 0.068 0.041 0.029 0.019
M4 0.055 0.045 0.030 0.022 0.015 0.091 0.064 0.038 0.027 0.018
|     |        |          | (c) | Case BM |     |     |         |     |
| --- | ------ | -------- | --- | ------- | --- | --- | ------- | --- |
|     |        | γ =0.475 |     |         |     |     | γ =0.45 |     |
| T   | 50 100 | 200      | 400 | 800     | 50  | 100 | 200 400 | 800 |
M1 0.011 0.010 0.006 0.004 0.004 0.026 0.017 0.009 0.006 0.004
M2 0.012 0.010 0.006 0.004 0.004 0.027 0.017 0.009 0.006 0.004
M3 0.018 0.014 0.009 0.006 0.005 0.037 0.024 0.013 0.008 0.006
M4 0.018 0.015 0.009 0.006 0.005 0.038 0.024 0.013 0.008 0.006
Notes:TheexcessMSPEiscalculatedastheempiricalMSPEminus1/6,where1/6represents
|                                               |     |     |     |     |         |        | Tγ    | =0.5Tγ |
| --------------------------------------------- | --- | --- | --- | --- | ------- | ------ | ----- | ------ |
| theminimalachievableMSPEbyalinearpredictor.τT |     |     |     |     | =0.01kC | bXXkS1 | andυT |        |
forγ∈{0.45,0.475}.ThereportedMSPEsareapproximatelycomputedby(i)generatingXt
andYt on afine gridof [0,1] with200 equally spaced gridpoints, and then (ii)representing
thosewith100cubicB-splinefunctions.Theresultsexhibitlittlechangewithvaryingnumbers
ofgridpointsandB-splinefunctions.
significantlysmallerthanintheothertwocases(CaseBBandCaseCMB);even
with a moderately large number of observations, the empirical MSPE in Case
BMtendstobeclosetotheminimalMSPE.Thissuggeststhattheperformance
of the proposed predictor significantly depends on the specification of ε . Over-
t
all,the simulationresultsreportedin Table 1 supportour theoreticalfinding in
Section 3. We also experimented with an alternative estimator A which is in-
troduced in Section 4.4 and obtained qualitatively similar supporeting evidence;
| some of       | the simulation | results | are reported |     | in Appendix |     | B of the appendix. |     |
| ------------- | -------------- | ------- | ------------ | --- | ----------- | --- | ------------------ | --- |
| 6. Concluding | remarks        |         |              |     |             |     |                    |     |
This paper studies linear prediction in a Hilbert space, demonstrating that,
under mild conditions, the empirical MSPEs associated with standard post-

Seo/Optimal linear prediction withfunctional observations 14
dimensionreductionestimatorsapproachthe minimal achievableMSPE.There
isampleroomforfutureresearch;forexample,itwouldbeintriguingtoexplore
whether similar prediction results can be obtained from various alternatives or
modifications of the simple post-dimension reduction estimators considered in
the literature.

Seo/Optimal linear prediction withfunctional observations 15
Appendix A: Proofs
A.1. Useful lemmas
LemmaA.1. LetΓbeanonnegativeself-adjoint Schatten1-class operator.For
any D , the following hold.
∞
∈L
(i) Γ1/2DD∗Γ1/2 andDΓD∗ areSchatten1-classoperatorsandtheirSchatten
1-norms are bounded above by D 2 Γ .
(ii) Γ1/2D and D∗Γ1/2 are Schatte k n 2 k - ∞ cla k ss k o S p 1 erators.
∞
Proof. LetΓ=
Pj=1
c
j
w
j ⊗
w
j
,wherec1
≥
c2
≥
...
≥
0andc
j
maybezero.By
allocatingapropervectortoeachzeroeigenvalue,wemayassumethat w j j≥1
isanorthonormalbasisof .Toshow(i),wenotethat Γ1/2DD∗Γ1/2w { ,w } =
j j
c w ,DD∗w c D 2 H , from which Γ1/2DD∗Γ1/2h D 2 Γ is i es-
j h j j i ≤ j k k∞ k k S1 ≤ k k∞k k S1
tablished. The desiredresultfor DΓD∗ is alreadywell known,see e.g., [11]
(p.267).Toshow(ii),weobservet k hat Γ1k / S 2 1 D 2 = D∗Γ1/2 2 D 2 ∞ Γ1/2w 2 =
D 2 ∞ c < , which establishe
k
s the d
k
es
S
i
2
red
k
result.
kS2 ≤k k∞ Pj=1k j k
k k∞ Pj=1 j ∞
Lemma A.2. Let Γ j j≥1 be a sequence of random Schatten 1-class operators
{ }
and let Γ be a self-adjoint Schatten 1-class operator. Then, for any orthonormal
basis w j j≥1 of and m T 0,
{ } H ≥
∞
 
Γ Γ =O ( Γ Γ )+O m Γ Γ + Γw ,w .
k j − k S1 p k j k S1−k k S1 p T k j − k ∞ X h j j i
 j=mT +1 
(A.1)
Moreover, Γ Γ 0 if Γ Γ 0 and m Γ Γ 0
k
j
− k
S1
→
p
k
j
k
S1
−k k
S1
→
p T
k
j
− k
∞
→
p
as m and T .
T
→∞ →∞
Proof. Equation(A.1)directly follows from Lemma 2 (and also Theorem 2) of
[19]. Moreover, if Γ Γ 0 and m Γ Γ 0 as m
k
j
k
S1
−k k
S1
→
p T
∞k
j
− k
∞
→
p T
→ ∞
and T , we find that Γ Γ = O ( Γw ,w ). Since Γ is a
→ ∞ k j − k S1 p Pj=
∞
mT +1h j j i
self-adjoint Schatten 1-class operator, we have Γw ,w Γ < ,
∞
Pj=1h j j i → p k k S1 ∞
from which we conclude that Γw ,w 0 as m .
Pj=mT +1h j j i→ p T →∞
A.2. Proofs of the theoretical results
Proof of Proposition 2.1. From Theorem 1 of [4], we find that C allows the
XY
following representation: for a unique bounded linear operator R satisfying
XY
R 1,
XY ∞
k k ≤
C =C
1/2
R C
1/2
.
XY YY XY XX
Let Cmin = C
YY −
C
Y
1/
Y
2 R
XY
R
X
∗
Y
C
Y
1/
Y
2 , which is clearly self-adjoint. Moreover,
Cmin is a nonnegative Schatten 1-class operator. To see this, note that for any
w ,
∈H
h
Cminw,w
i
=
h
(C
YY −
C
Y
1/
Y
2 R
XY
R
X
∗
Y
C
Y
1/
Y
2 )w,w
i

Seo/Optimal linear prediction withfunctional observations 16
= C 1/2 w 2 R∗ C 1/2 w 2 0 (A.2)
k YY k −k XY YY k ≥
(i.e.,Cminisnonnegative),whichisbecause
k
R
X
∗
Y
C
Y
1/
Y
2 w
k
2
≤k
R
X
∗
Yk
2
∞k
C
Y
1/
Y
2 w
k
2
≤
C 1/2 w 2. Moreover, from(A.2)and Lemma A.1(ii), we know that, for any or-
k YY k
thonormal basis
{
w
j }
j≥1,
k
Cmin
k S1
=
P
∞
j=1
(
k
C
Y
1/
Y
2 w
j k
2
−k
R
X
∗
Y
C
Y
1/
Y
2 w
j k
2) <
.
∞
We then find the following holds for any B and w :
∞
∈L ∈H
(C C B∗ BC∗ +BC B∗)w,w
h YY − XY − XY XX i
= C 1/2 w 2+ C 1/2 B∗w 2 2 C 1/2 B∗w,R∗ C 1/2 w
k YY k k XX k − h XX XY YY i
= C 1/2 B∗w R∗ C 1/2 w 2+ C 1/2 w 2 R∗ C 1/2 w 2
k XX − XY YY k k YY k −k XY YY k
=
k
C
X
1/
X
2 B∗w
−
R
X
∗
Y
C
Y
1/
Y
2 w
k
2+
h
Cminw,w
i
. (A.3)
We know from(A.3)that
∞
E Y BX 2 = (C C B∗ BC∗ +BC B∗)w ,w
k t − t k Xh YY − XY − XY XX j j i
j=1
∞
=
Xk
C
X
1/
X
2 B∗w
j −
R
X
∗
Y
C
Y
1/
Y
2 w
j k
2+
k
Cmin
k S1
, (A.4)
j=1
where { w j } j≥1 isanyorthonormalbasisof H .NotethatC X 1/ X 2 B∗ − R X ∗ Y C Y 1/ Y 2 isa
Schatten2-classoperator(LemmaA.1(ii))andthuswefindthat ∞ C 1/2 B∗w
Pj=1k XX j −
R∗ C 1/2 w 2 = C 1/2 B∗ R∗ C 1/2 2 .Fromthisresultcombinedwith(A.4),
XY YY j k k XX − XY YYkS2
the desired result immediately follows.
Proofs of Theorem 3.1 and Corollary 4.1. Toaccommodatemoregeneralcases,
which are considered in Corollary 4.1, we hereafter assume that τ−1 =O (Tγ1)
T p
andυ
T
−1 =O
p
(Tγ1)forsomeγ1
∈
(0,β),and
k
C
XX −
C
XX k ∞
,
k
C
YY −
C
YY k ∞
and C C areallO (Tβ)forsomeβ b (0,1/2].OurprobofofTheorem
XY XY ∞ p
k − k ∈
3.1 corbresponds to the particular case with β =1/2.
Note that sup λˆ λ C C = O (T−β) (Lemma 4.2 of
j≥1
|
j
−
j
| ≤ k
XX
−
XX
k
∞ p
[7]). Under Assumption 1, we havebλˆ kT
−
λˆ kT +1
≥
τ T and thus
Tβ(λˆ kT
−
λˆ kT +1)=Tβ(λˆ kT
−
λ kT +λ kT +1
−
λˆ kT +1)+Tβ(λ kT
−
λ kT +1)
≥
Tβτ T .
(A.5)
If λ
kT
= λ
kT
+1, (A.5) reduces to Tβ(λˆ
kT
−
λˆ
kT
+1)
≥
Tβτ
T
. Moreover, since
Tβ(λˆ kT
−
λˆ kT +1) = O p (1) and Tβτ T
→
p
∞
, P
{
λˆ kT
−
λˆ kT +1
≥
τ T
|
λ kT =
λ kT +1
}→
0asT
→∞
.UsingtheBayes’ruleandthefactsthatP
{
λˆ kT− λˆ kT +1
≥
τ ti T o } n = 1, 1 w a e n fi d n P d { t λ h k a T t = P { λ λ k k T T +1 = } λ = kT P + { 1 λ } kT → = 0. λ k T T o + e 1 s | λ t ˆ a k b T l − ish λˆ k t T h + e 1 d ≥ esi τ r T ed } c b o y n A si s s s t u en m c p y - ,
we thus may subsequently assume that λ
kT
>λ
kT
+1.

Seo/Optimal linear prediction withfunctional observations 17
Let Cmin = C
YY −
C
Y
1/
Y
2 R
XY
R
X
∗
Y
C
Y
1/
Y
2 . Since there exists A
∈ L ∞
such
that C
XY
= AC
XX
, Cmin = C
YY
AC
XX
A∗ holds (Corollary 2.1). From the
−
triangular inequality applied to the 1-norm, we then find that
S
k
Σ(A)
−
Cmin
k
S1
≤k
C
YY
−
C
YY
k
S1
+
k
AC
XX,kT
A∗
−
AC
XX,kT
A∗
k
S1
b +bAC A∗ ACbb A∗ b. (A.6)
k
XX,kT
−
XX
k
S1
It suffices to show that each summand in the RHS of(A.6)is o (1).
p
Wewillfirstconsiderthe firstterminthe RHSof(A.6).Letm beanydiver-
T
gentsequence(depending onT)butsatisfyT−βm 0.Notethat C
T
→ k
YY
k
S1−
C = mT (µˆ µ ) + ∞ (µˆ µ ) m C Cb +
k YY k S1 Pj=1 j − j Pj=mT +1 j − j ≤ T k YY − YY k ∞
∞ (µˆ µ ). For every δ >0, let E = ∞ (µˆb µ ) > δ and
Pj=mT +1 j − j δ {|Pj=mT +1 j − j | 2}
F = m C C > δ .Since P m C C + ∞ (µˆ
µ δ )> { δ T kPb Y E Y − + Y P Y k F ∞ , w 2 e } find that { T k b YY − YY k ∞ Pj=mT +1 j −
j δ δ
}≤ { } { }
P C C >δ P E +P F .
{k
YY
k
S1
−k
YY
k
S1
}≤ {
δ
} {
δ
}
b
NotethatC andC areSchatten1-classoperators(almostsurely)andalso
YY YY
m increasbes without bound. Moreover, m C C = O (m T−β) =
T T YY YY ∞ p T
o
p
(1)underourassumptions.Theseresultsim k pblyth − atP E k
δ
0andP F
δ
{ }→ { }→
0asT ,andthusweconcludethat C C 0.Combining
→∞ k
YY
k
S1−k YY
k
S1
→
p
this result with Lemma A.2 and the fact bthat m C C = o (1), we
T YY YY ∞ p
k − k
find that C C 0 as desired. b
k
YY
−
YY
k
S1
→
p
We nextbconsider the third term in the RHS of(A.6). We know from Lemma
A.1that AC A∗ AC A∗ A 2 C C .SinceC is
k XX,kT − XX k S1 ≤k k∞k XX,kT− XX k S1 XX
aSchatten1-classoperatorandk growswithoutbound, C C =
∞ λ 0 and thus AC T A∗ AC A∗ k = X o X ( , 1 k ) T a − s d X es X ir k e S d 1 .
Pj=kT +1 j → p k XX,kT − XX k S1 p
We lastly consider the second term in the RHS of(A.6). Let ε = Y AX .
t t t
−
Since C =AC +C and AC A∗ =C C−1 C∗ , we have
XY XX εX XX,kT XY XX,kT XY
b b b bb b b b b
AC A∗ =(AC +C )C−1 (AC +C )∗
XX,kT XX εX XX,kT XX εX
bb b =AC b A b ∗+A b Π C∗ b +C b Π A∗+C C−1 C∗ ,
XX,kT X,kT εX εX X,kT εX XX,kT εX
b b b b b b b b
where Π = kT vˆ vˆ . Therefore, we have
X,kT Pj=1 j ⊗ j
b
AC A∗ AC A∗
k
XX,kT
−
XX,kT
k
S1
bb AC b A∗ AC A∗
≤k
XX,kT
−
XX,kT
k
S1
+ A b Π C∗ +C Π A∗+C C−1 C∗ . (A.7)
k X,kT εX εX X,kT εX XX,kT εXk S1
b b b b b b b
It will be proved later that the first term in the RHS of(A.7)is o (1), i.e.,
p
AC A∗ AC A∗ =o (1). (A.8)
k
XX,kT
−
XX,kT
k
S1 p
b
We deduce from Lemma A.1(i)that the second term in the RHS of(A.7), below
denoted simply as , satisfies
D
O(1) Π C∗ + C C−1 C∗ 2 . (A.9)
D ≤ k kT εXk S1 k εX XX,kT εXkS1
b b b b b

Seo/Optimal linear prediction withfunctional observations 18
Since C = T−1 T X Y T−1 T X AX = C AC +
O
p
(T−βb X )= ε O
p
(T−βP ), t w = e 1 fin t d ⊗ tha t t − Pt=1 t ⊗ t YX − XX
max Π C∗ , C Π C Π =O (T−β)k =o (1)
nk kT εXk S1 k εX kTk S1o≤k εX k ∞ k kTk S1 p T p
b b b b b b
(A.10)
and also
∞ kT
C C−1 C∗ = λˆ−1 C vˆ ,vˆ 2
k εX XX,kT εXk S1 XX j h εX j ℓ i
b b b ℓ=1j=1 b
kT kT
λˆ−1 C vˆ 2 C 2 λˆ−1 =o (1), (A.11)
≤X j k εX j k ≤k εX k∞X j p
j=1 b b j=1
where the last equality follows from the fact that C 2 = O (T−2β) and
k εX k∞ p
kT λˆ−1 τ−1k = o(T2β) hold under Assumptionbs 1 and 2. As shown by
Pj=1 j ≤ T T
(A.9)-(A.11),the secondtermintheRHSof(A.7)is o (1).Combiningthis result
p
with(A.8), we find that AC A∗ AC A∗ =o (1) as desired.
k
XX,kT
−
XX,kT
k
S1 p
Itremainstoverify(A.8b)btocompbletetheproof.FromLemmaA.1(i),weknow
that
AC A∗ AC A∗ A 2 C C , (A.12)
k XX,kT − XX,kT k S1 ≤k k∞k XX,kT − XX,kTk S1
b b
andthusitsufficesto showthattheRHSof(A.12)iso (1).Tothis end,wefirst
p
note that
kT
C C = (λˆ λ ) k C C 0.
k XX,kTk S1 −k XX,kTk S1 X j − j ≤ T k XX − XX k ∞ → p
b j=1 b
(A.13)
We then obtain an upper bound of C C as follows:
k
XX,kT
−
XX,kTk ∞
b
k C b XX,kT − C XX,kTk ∞ ≤ (cid:13) (cid:13) (cid:13) (cid:13) (cid:13) Λ b kT + X j k = T 1 (λˆ j − λ j )f j ⊗ f j (cid:13) (cid:13) (cid:13) (cid:13) (cid:13)∞ ≤ (cid:13) (cid:13) (cid:13) Λ b kT(cid:13) (cid:13) (cid:13) ∞ +O p (T−β),
(cid:13) (cid:13) (A.14)
where Λ = kT λˆ fˆ fˆ kT λˆ f f . From similar algebra used in
kT Pj=1 j j ⊗ j −Pj=1 j j ⊗ j
the probof of Lemma 3.1 of [24] and the fact that
k·k
∞
≤k·k
S2
, we find that
2 kT ∞ kT ∞ kT kT
Λ λˆ2 fˆ,f 2+ λˆ2 fˆ,f 2+ (λˆ λˆ )2 fˆ,f 2.
(cid:13) (cid:13) (cid:13)b kT(cid:13) (cid:13) (cid:13) ∞ ≤X j=1 j ℓ= X kT +1 h j ℓ i X ℓ=1 ℓ j= X kT +1 h j ℓ i X j=1 X ℓ=1 j − ℓ h j ℓ i
(A.15)
Observe that, for every ℓ=1,...,k ,
T
∞ ∞
(λˆ
ℓ −
λˆ ℓ+1)2
X h
fˆ
j
,f
ℓ i
2
≤ X
(λˆ
ℓ −
λˆ
j
)2
h
fˆ
j
,f
ℓ i
2
j=kT +1 j=kT +1

Seo/Optimal linear prediction withfunctional observations 19
∞
= ( λˆ fˆ,f C fˆ,f )2
X h ℓ j ℓ i−h XX j ℓ i
j=kT +1 b
∞
= ( fˆ,(λˆ λ )f + fˆ,(C C )f )2
X h j ℓ − ℓ ℓ i h j XX − XX ℓ i
j=kT +1 b
(λˆ λ )f +(C C )f 2. (A.16)
ℓ ℓ ℓ XX XX ℓ
≤k − − k
b
Since sup λˆ λ C C =O (T−β), we find that the RHS of
ℓ≥1
|
ℓ
−
ℓ
|≤k
XX
−
XX
k
∞ p
(A.16)isO
p
(T−2β).Usingtbhefactthatτ
T
−1
≥
(λˆ
ℓ −
λˆ ℓ+1)−1 forallℓ=1,...,k
T
,
the following is deduced: for some δ >0,
kT ∞ kT O (T−2β)λˆ2 kT
λˆ2 fˆ,f 2 = p ℓ O (τ−2T−2β) λˆ2 =O (T−δ),
X
ℓ=1
ℓ
j=
X
kT +1
h j ℓ i X
ℓ=1
(λˆ
ℓ −
λˆ ℓ+1)2 ≤ p T X
ℓ=1
ℓ p
(A.17)
where the last equality is deduced from the facts that kT λˆ2 = O (1) and
τ
T
−1 = O
p
(Tγ1) for some γ1
∈
(0,β) under the emplo P ye ℓ d =1 co ℓ ndition p s. From
nearly identical arguments, we also find that
kT ∞
λˆ2 fˆ,f 2 =O (T−δ). (A.18)
X j X h j ℓ i p
j=1 ℓ=kT +1
Moreover,from similar algebra used in(A.16)we find that
kT kT
(λˆ λˆ )2 fˆ,f 2
XX j − ℓ h j ℓ i
j=1ℓ=1
kT kT
= ( fˆ,C f fˆ,(λˆ λ )f + fˆ,C f )2
XX h j XX ℓ i−h j ℓ − ℓ ℓ i h j XX ℓ i
j=1ℓ=1 b
kT
(λˆ λ )f +(C C )f 2 =O (k T−2β). (A.19)
≤Xk ℓ − ℓ ℓ XX − XX ℓ k p T
j=1 b
From(A.14)-(A.19), we know that there is a divergent sequence m such that
T
m C C 0. Combining this result with(A.13)and Lemma
T
k
XX,kT
−
XX,kTk ∞
→
p
A.2, wbe find that the RHS of(A.12)is o
p
(1) (and thus(A.8)holds).
Proof of Corollary 2.1. (a) directly follows from the facts that (i) (BC
1/2
k XX −
C 1/2 R )w 2 = 0 for all j 1 if and only if BC 1/2 C 1/2 R 2 = 0
YY XY j k ≥ k XX − YY XY kS2
and (ii) the Hahn-Banach extension theorem (see e.g., Theorem 1.9.1 of [21]).
This result in turn implies (b) due to the fact that C = C
1/2
R C
1/2
as
XY YY XY XX
observed in Proposition 2.1.
Proof of Corollary 4.2. Note that (A.6) holds when A is replaced by A , and
C C = o (1) and AC A∗ AC b A∗ = o (1)ecan be
k
YY
−
YY
k
S1 p
k
XX,kT
−
XX
k
S1 p
b

Seo/Optimal linear prediction withfunctional observations 20
shown as in our proof of Theorem 3.1. We thus have
k
Σ(A)
−
Cmin
k
S1
≤
o
p
(1)+
k
AC
XX,kT
A∗
−
AC
XX,kT
A∗
k
S1
,
e eb e
andhenceitsufficestoshowthat AC A∗ AC A∗ =o (1).Note
k
XX,kT
−
XX,kT
k
S1 p
that eb e
AC A∗ =Π (AC A∗+AΠ C∗ +C A∗+C C−1 C∗ )Π .
XX,kT Y,ℓT XX,kT X,kT εX εX εX XX,kT εX Y,ℓT
eb e b b b b b b b b b
Using similar arguments used in our proof of Theorem 3.1, Lemma A.1 and
the facts that Π 1 for any arbitrary ℓ 1 and AC A∗
k
Y,ℓTk ∞
≤
T
≥ k
XX,kT
−
AC A∗ =ob(1),wefindthat Π (AC A∗ AC A∗)Π =
XX
k
S1 p
k
Y,ℓT XX,kT
−
XX,kT Y,ℓTk S1
o (1), Π (AC A∗ AC Ab∗)Π b =o (1),and Π (AbΠ C∗ +
p k Y,ℓT XX,kT − XX Y,ℓTk S1 p k Y,ℓT X,kT εX
C A∗+b C C−1 C∗ )Π =bo (1). Therefore, b b b
εX εX XX,kT εX Y,ℓTk S1 p
b b b b b
AC A∗ Π AC A∗Π 0.
k
XX,kT
−
Y,ℓT XX Y,ℓTk S1
→
p
eb e b b
If Π AC A∗Π AC A∗ 0, the desired result is established.
k
Y,ℓT XX Y,ℓT
−
XX
k
S1
→
p
LetbΥ = C 1/2 R band Υ = Π C 1/2 R . We then have AC A∗ = ΥΥ∗
YY XY Y,ℓT YY XY XX
(see Proposition2.1) andbΠ bAC A∗Π =ΥΥ∗. Observe that
Y,ℓT XX Y,ℓT
b b bb
ΥΥ∗ ΥΥ∗ = Υ(Υ∗ Υ∗)+(Υ Υ)Υ∗ O (1) Υ Υ
∞ ∞ p ∞
k − k k − − k ≤ k − k
bb b b b
O (1) Π
b
C
1/2
C
1/2
.
≤ p k Y,ℓT YY − YYk ∞
b
This implies that ΥΥ∗ ΥΥ∗ 0 if there exists a divergentsequence m
∞ p T
k − k →
such that m Π
bCb1/2
C
1/2
0 and m ℓ for large T. Moreover,
T k Y,ℓT YY − YYk ∞ → p T ≤ T
we find the follbowing from the fact that ΥΥ∗ is nonnegative self-adjoint and
wˆ ∞ is an orthonormalbasis of :
{
j }j=1
H
∞ ∞
ΥΥ∗ ΥΥ = ΥΥ∗wˆ ,wˆ ΥΥ∗wˆ ,wˆ ,
k k S1 −k k S1 X h j j i≤ X h j j i
bb j=ℓT +1 j=mT +1
which is o (1) since ΥΥ∗ = C 1/2 R R∗ C 1/2 is a Schatten 1-class (Lemma
p YY XY XY YY
A.1(i)).We thusdeduce fromLemma A.2that ΥΥ∗ ΥΥ∗ 0asdesired.
k − k
S1
→
p
bb
Appendix B: Additional simulation results for alternative
estimators
In this section, we experiment with an alternative estimator A which is intro-
duced in Section 4.4. We replicate the same simulation experimeents conducted
inSection5usingthealternativeestimatorAintroducedinSection4.4.Wecon-
struct Π
Y,ℓT
using the eigenvectors
{
uˆ
j
}
j≥1eof C
YY
with ℓ
T
=
⌊
T1/2
⌋
. Overall,
wefounbdthatthe computedexcessMSPEstendbtobesimilartothoseobtained

Seo/Optimal linear prediction withfunctional observations 21
with A, providing supporting evidence for our finding in Corollary 4.2. The
simulabtion results are reported in Table 2. We also experimented with Π
Y,ℓT
constructedfromthe eigenvectors vˆ j j≥1 ofC XX ,andthe resultsarerepborted
{ }
in Table 3. Even if the empirical MSPEs tendbto be larger than in the previous
case, we obtained overall similar simulation results.
Table 2
Excess MSPE of the alternative predictor, Π bY,ℓT =Pℓ j T =1uˆj⊗uˆj
(a) Case BB
γ =0.475 γ =0.45
T 50 100 200 400 800 50 100 200 400 800
M1 0.043 0.035 0.023 0.017 0.011 0.071 0.049 0.029 0.021 0.013
M2 0.045 0.035 0.023 0.017 0.012 0.071 0.051 0.030 0.021 0.014
M3 0.056 0.045 0.030 0.022 0.015 0.091 0.063 0.038 0.027 0.017
M4 0.057 0.046 0.031 0.023 0.015 0.092 0.065 0.039 0.028 0.018
(b) Case CBM
γ =0.475 γ =0.45
T 50 100 200 400 800 50 100 200 400 800
M1 0.043 0.035 0.023 0.017 0.011 0.071 0.050 0.029 0.021 0.013
M2 0.044 0.036 0.023 0.017 0.012 0.072 0.051 0.030 0.021 0.014
M3 0.057 0.046 0.030 0.022 0.015 0.093 0.065 0.039 0.027 0.018
M4 0.057 0.047 0.031 0.023 0.015 0.094 0.066 0.039 0.028 0.018
(c) Case BM
γ =0.475 γ =0.45
T 50 100 200 400 800 50 100 200 400 800
M1 0.012 0.010 0.006 0.004 0.004 0.026 0.017 0.009 0.006 0.004
M2 0.012 0.010 0.006 0.004 0.004 0.027 0.018 0.009 0.006 0.004
M3 0.018 0.014 0.009 0.006 0.005 0.037 0.024 0.013 0.008 0.006
M4 0.018 0.015 0.009 0.006 0.005 0.038 0.025 0.013 0.008 0.006
Notes:TheexcessMSPEsarecalculatedasinTable1.

|     | Seo/Optimal | linear prediction | withfunctional | observations | 22  |
| --- | ----------- | ----------------- | -------------- | ------------ | --- |
Table 3
=Pℓ T
|     | Excess MSPE | of the alternative | predictor, Π bY,ℓT | j =1vˆj⊗vˆj |     |
| --- | ----------- | ------------------ | ------------------ | ----------- | --- |
(a) Case BB
|     |        | γ =0.475 |        | γ =0.45     |     |
| --- | ------ | -------- | ------ | ----------- | --- |
| T   | 50 100 | 200 400  | 800 50 | 100 200 400 | 800 |
M1 0.043 0.035 0.023 0.017 0.011 0.071 0.049 0.029 0.021 0.013
M2 0.066 0.049 0.032 0.023 0.015 0.089 0.063 0.039 0.026 0.017
M3 0.056 0.045 0.030 0.022 0.015 0.091 0.063 0.038 0.027 0.017
M4 0.078 0.057 0.036 0.025 0.017 0.109 0.075 0.044 0.030 0.019
(b) Case CBM
|     |        | γ =0.475 |        | γ =0.45     |     |
| --- | ------ | -------- | ------ | ----------- | --- |
| T   | 50 100 | 200 400  | 800 50 | 100 200 400 | 800 |
M1 0.044 0.035 0.023 0.017 0.011 0.071 0.050 0.029 0.021 0.013
M2 0.075 0.055 0.036 0.025 0.016 0.099 0.069 0.043 0.029 0.018
M3 0.057 0.046 0.030 0.022 0.015 0.093 0.065 0.039 0.027 0.018
M4 0.096 0.071 0.046 0.032 0.021 0.127 0.089 0.055 0.037 0.023
(c) Case BM
|     |        | γ =0.475 |        | γ =0.45     |     |
| --- | ------ | -------- | ------ | ----------- | --- |
| T   | 50 100 | 200 400  | 800 50 | 100 200 400 | 800 |
M1 0.012 0.010 0.006 0.004 0.004 0.026 0.017 0.009 0.006 0.004
M2 0.032 0.022 0.014 0.009 0.007 0.046 0.030 0.017 0.011 0.008
M3 0.018 0.015 0.009 0.006 0.005 0.037 0.024 0.013 0.008 0.006
M4 0.040 0.028 0.017 0.011 0.008 0.058 0.037 0.021 0.013 0.009
Notes:TheexcessMSPEsarecalculatedasinTable1.

|     | Seo/Optimal |     | linear | prediction | withfunctional |     | observations |     |     | 23  |
| --- | ----------- | --- | ------ | ---------- | -------------- | --- | ------------ | --- | --- | --- |
References
[1] Aue,A.andKlepsch,J.(2017). Estimatingfunctionaltimeseriesbymoving
| average | model | fitting, | arxiv:1701.00770[stat.ME]. |     |     |     |     |     |     |     |
| ------- | ----- | -------- | -------------------------- | --- | --- | --- | --- | --- | --- | --- |
[2] Aue, A., Norinho, D. D., and Ho¨rmann, S. (2015). On the prediction of
| stationary | functional |     | time | series. | Journal | of the | American | Statistical | Associ- |     |
| ---------- | ---------- | --- | ---- | ------- | ------- | ------ | -------- | ----------- | ------- | --- |
ation, 110(509):378–392.
[3] Babii, A. and Florens, J.-P. (2025). Is completeness necessary? estimation
| in nonidentified |     | linear | models. | Econometric |     | Theory, | page | 1–38. |     |     |
| ---------------- | --- | ------ | ------- | ----------- | --- | ------- | ---- | ----- | --- | --- |
[4] Baker, C. R. (1973). Joint measures and cross-covarianceoperators. Trans-
| actions | of the | American | Mathematical |     | Society, |     | 186:273–289. |     |     |     |
| ------- | ------ | -------- | ------------ | --- | -------- | --- | ------------ | --- | --- | --- |
[5] Beare, B. K., Seo, J., and Seo, W.-K. (2017). Cointegrated linear processes
| in Hilbert | space. | Journal | of  | Time | Series | Analysis, | 38(6):1010–1027. |     |     |     |
| ---------- | ------ | ------- | --- | ---- | ------ | --------- | ---------------- | --- | --- | --- |
[6] Benatia, D., Carrasco, M., and Florens, J.-P. (2017). Functional linear re-
| gression | with | functional | response. |     | Journal | of Econometrics, |     | 201(2):269–291. |     |     |
| -------- | ---- | ---------- | --------- | --- | ------- | ---------------- | --- | --------------- | --- | --- |
[7] Bosq,D. (2000). Linear Processes in Function Spaces. Springer-VerlagNew
York.
[8] Bosq, D. (2007). General linear processes in Hilbert spaces and prediction.
| Journal | of Statistical |     | Planning | and | Inference, | 137(3):879–894. |     |     |     |     |
| ------- | -------------- | --- | -------- | --- | ---------- | --------------- | --- | --- | --- | --- |
[9] Bosq, D. (2014). Computing the best linear predictor in a Hilbert space.
| Applications |     | togeneralARMAH |     | processes. |     | Journal | of  | Multivariate | Analysis, |     |
| ------------ | --- | -------------- | --- | ---------- | --- | ------- | --- | ------------ | --------- | --- |
124:436–450.
[10] Chang, Y., Choi, Y., Kim, S., and Park, J. (2021). Stock market return
| predictability |             | dormant    | in option |        | panels.       | Mimeo, | Department | of        | Economics, |     |
| -------------- | ----------- | ---------- | --------- | ------ | ------------- | ------ | ---------- | --------- | ---------- | --- |
| Indiana        | Univeristy. |            |           |        |               |        |            |           |            |     |
| [11] Conway,   | J.          | B. (1994). | A         | Course | in Functional |        | Analysis.  | Springer. |            |     |
[12] Hall, P. and Horowitz, J. L. (2007). Methodology and convergence rates
| for functional |     | linear | regression. | Annals | of  | Statistics, | 35(1):70 | –   | 91. |     |
| -------------- | --- | ------ | ----------- | ------ | --- | ----------- | -------- | --- | --- | --- |
[13] Imaizumi, M. and Kato, K. (2018). PCA-based estimation for functional
| linear regressionwith |     |     | functional | responses. |     | Journal | of  | Multivariate | Analysis, |     |
| --------------------- | --- | --- | ---------- | ---------- | --- | ------- | --- | ------------ | --------- | --- |
163:15–36.
[14] Jaimez, R. G. and Bonnet, M. J. V. (1987). On the Karhunen-Loeve ex-
| pansion | for transformed |     | processes. |     | Trabajos | de  | Estadistica, | 2:81–90. |     |     |
| ------- | --------------- | --- | ---------- | --- | -------- | --- | ------------ | -------- | --- | --- |
[15] Kalman, D. and McKinzie, M. (2012). Another way to sum a series: gen-
| erating  | functions,    | euler, | and | the dilog | function. |     | The American | Mathematical |     |     |
| -------- | ------------- | ------ | --- | --------- | --------- | --- | ------------ | ------------ | --- | --- |
| Monthly, | 119(1):42–51. |        |     |           |           |     |              |              |     |     |
[16] Karol,A., Nazarov,A., and Nikitin, Y. (2008). Small ball probabilities for
| Gaussian | random       | fields | and          | tensor | products | of                | compact | operators. | Transac- |     |
| -------- | ------------ | ------ | ------------ | ------ | -------- | ----------------- | ------- | ---------- | -------- | --- |
| tions of | the American |        | Mathematical |        | Society, | 360(3):1443–1474. |         |            |          |     |
[17] Klepsch, J., Klu¨ppelberg, C., and Wei, T. (2017). Prediction of functional
| ARMAprocesseswithanapplicationtotrafficdata. |     |     |     |     |     |     | EconometricsandStatis- |     |     |     |
| -------------------------------------------- | --- | --- | --- | --- | --- | --- | ---------------------- | --- | --- | --- |
tics, 1:128–149.
[18] Klepsch, J. and Klu¨ppelberg, C. (2017). An innovations algorithm for the
| prediction | of  | functional | linear | processes. |     | Journal | of  | Multivariate | Analysis, |     |
| ---------- | --- | ---------- | ------ | ---------- | --- | ------- | --- | ------------ | --------- | --- |
155:252–271.
[19] Kubrusly, C. (1985). On convergence of nuclear and correlation operators

|            | Seo/Optimal      | linear | prediction | withfunctional |     | observations  |     |             | 24  |
| ---------- | ---------------- | ------ | ---------- | -------------- | --- | ------------- | --- | ----------- | --- |
| in Hilbert | space. Technical |        | report,    | Laboratorio    |     | de Computacao |     | Cientifica. |     |
[20] Luo,R.andQi,X.(2017). Function-on-functionlinearregressionbysignal
| compression. | Journal | of the | American | Statistical |     | Association, |     | 112(518):690– |     |
| ------------ | ------- | ------ | -------- | ----------- | --- | ------------ | --- | ------------- | --- |
705.
[21] Megginson, R. E. (2012). Introduction to Banach Space Theory. Springer
New York.
[22] Mollenhauer, M., Mu¨cke, N., and Sullivan, T. J. (2023). Learning linear
| operators: | Infinite-dimensional |     | regression |     | as a | well-behaved | non-compact |     | in- |
| ---------- | -------------------- | --- | ---------- | --- | ---- | ------------ | ----------- | --- | --- |
verse problem.
[23] Park, J. Y. and Qian, J. (2012). Functional regression of continuous state
| distributions. | Journal | of Econometrics, |     | 167(2):397–412. |     |     |     |     |     |
| -------------- | ------- | ---------------- | --- | --------------- | --- | --- | --- | --- | --- |
[24] Reimherr, M. (2015). Functional regression with repeated eigenvalues.
| Statistics | & Probability | Letters, | 107:62–70. |     |     |     |     |     |     |
| ---------- | ------------- | -------- | ---------- | --- | --- | --- | --- | --- | --- |
[25] Seo, W.-K. (2023). Cointegration and representation of cointegrated au-
| toregressive | processes | in Banch | spaces. | Econometric |     | Theory, | 39(4):737–788. |     |     |
| ------------ | --------- | -------- | ------- | ----------- | --- | ------- | -------------- | --- | --- |
[26] Seong, D. and Seo, W.-K. (2021). Functional instrumental variable regres-
| sion with | an application | to  | estimating | the | impact | of immigration |     | on  | native |
| --------- | -------------- | --- | ---------- | --- | ------ | -------------- | --- | --- | ------ |
wages, arXiv:2110.12722[econ.EM].
[27] Yao,F.,Mu¨ller,H.-G.,andWang,J.-L.(2005).Functionallinearregression
| analysis | for longitudinal | data. | The | Annals | of Statistics, |     | 33(6):2873–2903. |     |     |
| -------- | ---------------- | ----- | --- | ------ | -------------- | --- | ---------------- | --- | --- |
[28] Zhang, X., Chiou, J.-M., and Ma, Y. (2018). Functional prediction
| throughaveragingestimated |     |     | functional | linearregressionmodels. |     |     |     | Biometrika, |     |
| ------------------------- | --- | --- | ---------- | ----------------------- | --- | --- | --- | ----------- | --- |
105(4):945–962.