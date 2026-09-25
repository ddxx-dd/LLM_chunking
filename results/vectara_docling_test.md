## Optimal linear prediction with functional observations: Why you can use a simple post-dimension reduction estimator

## Won-Ki Seo ∗

University of Sydney

Abstract: We study the optimal linear prediction of a random function that takes values in an infinite dimensional Hilbert space. We begin by characterizing the mean square prediction error (MSPE) associated with a linear predictor and discussing the minimal achievable MSPE. This analysis reveals that, in general, there are multiple non-unique linear predictors that minimize the MSPE, and even if a unique solution exists, consistently estimating it from finite samples is generally impossible. Nevertheless, we can define asymptotically optimal linear operators whose empirical MSPEs approach the minimal achievable level as the sample size increases. We show that, interestingly, standard post-dimension reduction estimators, which have been widely used in the literature, attain such asymptotic optimality under minimal conditions.

MSC2020 subject classifications: Primary 60G25; secondary 62J99. Keywords and phrases: linear prediction, functional data, functional linear models, regularization.

## Contents

| 1   | Introduction . . .                                                                | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2                     |
|-----|-----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| 2   | Optimal linear prediction in Hilbert space . . . . . . . . . . .                  | . . . . . 3                                                                       |
|     | 2.1                                                                               | Preliminaries . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3         |
|     | 2.2                                                                               | Linear prediction in H . . . . . . . . . . . . . . . . . . . . . . . 4            |
| 3   | Estimation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .      | 6                                                                                 |
| 4   | Discussions and extensions . . . . . . . . . . . . . . . . . . . . . . . . .      | 9                                                                                 |
|     | 4.1                                                                               | A more general result . . . . . . . . . . . . . . . . . . . . . . . . 9           |
|     | 4.2                                                                               | Misspecified functional linear models and OLPO . . . . . . . . . 9                |
|     | 4.3                                                                               | Requirement of sufficient dimension reduction . . . . . . . . . . . 10            |
|     | 4.4                                                                               | Dimension reduction of the target variable . . . . . . . . . . . . . 10           |
| 5   | Simulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11 | Simulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11 |
| 6   | Concluding remarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13     | Concluding remarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13     |
| A   | Proofs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15 | Proofs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15 |
|     | A.1                                                                               | Useful lemmas . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15          |
|     | A.2                                                                               | Proofs of the theoretical results . . . . . . . . . . . . . . . . . . . 15        |

∗ Won-Ki Seo is the corresponding author.

| B          | Additional simulation results for alternative estimators . . . . . . . . . 20   |
|------------|---------------------------------------------------------------------------------|
| References | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23      |

## 1. Introduction

We study the optimal linear prediction in an arbitrary Hilbert space H , and how to estimate the optimal linear predictor. Given recent developments in functional data analysis, studies on this subject hold significant importance and relevance to many empirical applications; see e.g., [1], [2], [17], [18] and [23] to name only a few recent papers. The reader is also referred to [8], [9] and [22] containing an earlier mathematical exploration of this subject.

Let { Y t } t ≥ 1 and { X t } t ≥ 1 be stationary sequences of mean-zero random elements, both taking values in H . Suppose that ˜ Y t = A o X t , where A o is a continuous linear operator, and it satisfies the following: for any arbitrary continuous linear operator B ,

<!-- formula-not-decoded -->

where ‖ · ‖ is the norm defined on H . We refer to ˜ Y t = A o X t as an optimal linear predictor (OLP) and A o as an optimal linear prediction operator (OLPO). Given the observations { Y t , X t } T t =1 , practitioners are often interested in constructing a predictor ̂ Y t that converges (in probability) to an OLP as T increases. This is straightforward when H = R , and Y t and X t are real-valued mean-zero random variables with positive variances. In this case, it is well known that the unique solution to (1.1) is achieved by A o = E [ Y t X t ] / E [ X 2 t ] and the minimal achievable mean squared prediction error (MSPE) is E ‖ Y t - A o X t ‖ 2 = E [ Y 2 t ] - ( E [ Y t X t ]) 2 / E [ X 2 t ]. The conventional plug-in-type estimator (or OLS estimator) of A o may be defined by ̂ A = T - 1 ∑ T t =1 Y t X t /T - 1 ∑ T t =1 X 2 t . When the weak law of large numbers holds for { Y 2 t } t ≥ 1 , { X 2 t } t ≥ 1 and { X t Y t } t ≥ 1 , we find that the following two results hold: (i) ̂ A → p A o = E [ X t Y t ] / E [ X 2 t ] and (ii) T - 1 ∑ T t =1 ( Y t - ̂ AX t ) 2 → p E [ Y 2 t ] - ( E [ X t Y t ]) 2 / E [ X 2 t ], where and hereafter → p denotes the convergence in probability with respect to the norm of H . That is, a consistent estimator of the OLPO can be constructed from the given observations. Furthermore, the empirical MSPE obtained from the estimator converges to the minimal achievable MSPE.

However, in a more general situation where Y t and X t take values in a possibly infinite dimensional H , it is generally impossible to obtain parallel results. To see this with a simple example, suppose that { Y t , X t } t ≥ 1 satisfies the following: for t ≥ 1,

<!-- formula-not-decoded -->

where ε t is independent of X t and { f j } j ≥ 1 is an orthonormal basis of H . A o specified in (1.2) is obviously the OLPO, but A o is not consistently estimable in general. As will be detailed in Example 1, this is because, it is not possible to estimate a j for j &gt; T from T observations unless (i) a simplifying condition on A , such as a j = a for all j ≥ m for some finite m , holds and (ii) researchers are aware of this condition and use it appropriately. Even in the simple case where Y t = ¯ A o X t + ε t with ¯ A o = aI ( I denotes the identity map) for a ∈ R , consistent estimation of ¯ A o is impossible for the same reason if we do not know such a simple structure of ¯ A o and hence allow a more general case given in (1.2) (note that, ¯ A o = aI is a special case of A o in (1.2) with a j = a ).

Does this mean that it is impossible to statistically solve the optimal linear prediction problem in this general setting? The answer is no. We will demonstrate that, under mild conditions, there exists the minimum MSPE achievable by a linear predictor and it is feasible to construct a possibly inconsistent estimator ̂ A such that the empirical MSPE, computed as T - 1 ∑ T t =1 ‖ Y t - ̂ AX t ‖ 2 , converges to the minimum MSPE. Particularly, we show that a standard post dimension-reduction estimator, obtained by (i) reducing the dimensionality of the predictive variable X t using the principal directions of its sample covariance and then (ii) applying the least squares method to estimate the linear relationship between the resulting lower dimensional predictive variable and Y t , is effective for linear prediction. This post dimension-reduction estimator has been widely used due to its simplicity. Its statistical properties have been studied under technical assumptions, which are challenging to verify, such as those concerning the eigenstructure of the covariance of X t ; the reader is referred to, e.g., [13] and [26], where the assumptions of [12] are adopted for function-onfunction regression models. We show that, without such assumptions, a naive use of this simple post dimension-reduction estimator can be justified as a way to obtain a solution which asymptotically minimizes the MSPE. We also extend this finding to show that similar estimators, which involve further dimension reduction of Y t , also possess this desirable property under mild conditions.

The paper proceeds as follows: Section 2 characterizes the minimal achievable MSPE by a linear predictor, followed by a discussion on the estimation of an asymptotically optimal predictor in Section 3. Further discussions and extensions are given in Section 4. Section 5 provides simulation evidence of our theoretical findings, and Section 6 contains concluding remarks. The appendix includes mathematical proofs of the theoretical results and some additional simulation results.

## 2. Optimal linear prediction in Hilbert space

## 2.1. Preliminaries

For the subsequent discussion, we introduce notation. Let H be a separable Hilbert space with inner product 〈· , ·〉 and norm ‖ · ‖ . For V ⊂ H , let V ⊥ be the orthogonal complement to V . We let L ∞ be the set of continuous linear operators, and let ‖T ‖ ∞ , T ∗ , ran T , and ker T denote the operator norm, adjoint, range, and kernel of T , respectively. T is self-adjoint if T = T ∗ . T is called nonnegative if 〈T x, x 〉 ≥ 0 for any x ∈ H , and positive if also 〈T x, x 〉 ̸ = 0 for any x ∈ H \ { 0 } . For x, y ∈ H , we let x ⊗ y be the operator given by z ↦→ 〈 x, z 〉 y for z ∈ H . T ∈ L ∞ is compact if T = ∑ j ≥ 1 a j v j ⊗ w j for some orthonormal bases { v j } j ≥ 1 and { w j } j ≥ 1 and a sequence of nonnegative numbers { a j } j ≥ 1 tending to zero; if T is also self-adjoint and nonnegative (see [7], p. 35), we may assume that v j = w j . For any compact T ∈ L ∞ and p ∈ N , let ‖T ‖ S p be defined by ‖T ‖ p S p = ∑ ∞ j =1 a p j and let S p be the set of compact operators T with ‖T ‖ S p &lt; ∞ ; S p is called the Schatten p -class. S 1 (resp. S 2 ) is also referred to the trace (resp. Hilbert-Schmidt) class. It is known that the following hold: ‖T ‖ ∞ ≤ ‖T ‖ S 2 ≤ ‖T ‖ S 1 and ‖T ‖ 2 S 2 = ∑ ∞ j =1 ‖T w j ‖ 2 for any orthonormal basis { w j } j ≥ 1 . For any H -valued mean-zero random elements Z and ˜ Z with E ‖ Z ‖ 2 &lt; ∞ and E ‖ ˜ Z ‖ 2 &lt; ∞ , their cross-covariance C Z ˜ Z = E [ Z ⊗ ˜ Z ] is a Schatten 1-class operator; if Z = ˜ Z , it reduces to the covariance of Z and E ‖ Z ‖ 2 = ‖ C ZZ ‖ S 1 holds.

## 2.2. Linear prediction in H

Consider a weakly stationary sequence { Y t , X t } t ≥ 1 with nonzero covariances C Y Y = E [ Y t ⊗ Y t ] and C XX = E [ X t ⊗ X t ], along with the cross-covariance C XY = E [ X t ⊗ Y t ] (or equivalently C ∗ Y X ). We hereafter write C Y Y and C XX as their spectral representations as follows, if necessary:

<!-- formula-not-decoded -->

where κ 1 ≥ κ 2 ≥ . . . ≥ 0, λ 1 ≥ λ 2 ≥ . . . ≥ 0, and { u j } j ≥ 1 and { v j } j ≥ 1 are orthonormal sets of H . Unless otherwise stated, we assume that C Y Y and C XX are not finite rank operators and thus there are infinitely many nonzero eigenvalues in(2.1), which is as usually assumed for covariances of Hilbert-valued random elements in the literature on functional data analysis.

Note first that, for any B ∈ L ∞ , E ‖ Y t - BX t ‖ 2 = ‖ E [( Y t - BX t ) ⊗ ( Y t BX t )] ‖ S 1 and hence the MSPE associated with B can be written as follows:

-

<!-- formula-not-decoded -->

Next, we provide the main result of this section, which not only gives us a useful characterization of the MSPE in (2.2), but also provides essential preliminary results for the subsequent discussion.

Proposition 2.1. For any B ∈ L ∞ , there exists a unique element R XY ∈ L ∞ such that C XY = C 1 / 2 Y Y R XY C 1 / 2 XX and

<!-- formula-not-decoded -->

Proposition 2.1 shows that the MSPE associated with B ∈ L ∞ is the sum of the Schatten 1- and 2-norms of specific operators dependent on C Y Y , C XX , R XY and B ; notably, only the latter term ( ‖ BC 1 / 2 XX - C 1 / 2 Y Y R XY ‖ 2 S 2 ) in (2.3) depends on B . Thus, the former term ( ‖ C Y Y - C 1 / 2 Y Y R XY R ∗ XY C 1 / 2 Y Y ‖ S 1 ) represents the minimal achievable MSPE by a linear predictor, while the latter can be understood as a measure of the inadequacy of B as a linear predictor. If { w j } j ≥ 1 is an orthonormal basis of H , this inadequacy becomes zero if and only if

<!-- formula-not-decoded -->

Based on these findings, we obtain the following two characterizations of an OLPO in Corollary 2.1: the first is a direct consequence of (2.4) and the HahnBanach extension theorem (see, e.g., Theorem 1.9.1 of [21]), which, in turn, implies the second due to the fact that C XY = C 1 / 2 Y Y R XY C 1 / 2 XX as observed in Proposition 2.1.

Corollary 2.1. A is an OLPO if and only if any of the following equivalent conditions holds: (a) AC 1 / 2 XX = C 1 / 2 Y Y R XY and (b) AC XX = C XY .

Condition ( b ) follows directly from condition ( a ) and Proposition 2.1, and it is notably align with the characterization of an OLPO provided by [8]; Remark 2.1 outlines the distinctions between our findings and the existing result in more detail. From Corollary 2.1, we find that the minimum MSPE is attained by A ∈ L ∞ satisfying AC XX = C XY (or AC 1 / 2 XX = C 1 / 2 Y Y R XY ). However, such an operator A is not uniquely determined; particularly, the equation does not specify how A acts on ker C XX , allowing A to agree with any arbitrary element in L ∞ on ker C XX (see Remark 2.2). When C XX is not injective, there are multiple choices of A that achieve the minimum MSPE. In infinite dimensional settings, the injectivity of C XX is a stringent assumption, and verifying this condition from finite observations is impractical. Thus, pursuing prediction under the existence of the unique OLPO, as in the standard univariate or multivariate prediction, is restrictive. Even if a different setup is considered with a different purpose, similar concerns about the requirements for unique identification were recently raised by [3], and the enthusiastic reader is referred to their paper for more detailed discussion on the topic.

Remark 2.1. Condition ( b ) in Proposition 2.1 was earlier obtained as the requirement for A ∈ L ∞ to be an OLPO by Propositions 2.2-2.3 of [8]. Compared with this earlier result, Proposition 2.1 not only provides more general results, such as the explicit expression of the gap between the minimal attainable MSPE and the MSPE associated with any B ∈ L ∞ , but it also employs a distinct approach. The result of [8] relies on the notion of a linearly closed subspace and an extension of the standard projection theorem, while Proposition 2.1 is established by an algebraic proof based on the representation of cross-covariance operators in [4].

Remark 2.2. By invoking the Hahn-Banach extension theorem (Theorem 1.9.1 of [21]), we may assume that A satisfying AC XX = C XY is a unique continuous linear map defined on the closure of ran C XX , which is not equal to H if C XX is not injective. Thus, if there exists another continuous linear operator ˜ A which agrees with A on the closure of ran C XX but not on [ran C XX ] ⊥ , then ˜ A also satisfies that E ‖ Y t - ˜ AX t ‖ 2 = ‖ C Y Y - C 1 / 2 Y Y R XY R ∗ XY C 1 / 2 Y Y ‖ S 1 .

The results given in Proposition 2.1 and Remark 2.2 imply that, particularly when the predictive variable is function-valued, there may be multiple OLPOs that satisfy (1.1). Furthermore, even if a unique OLPO exists, it may not be consistently estimable; a more detailed discussion is given in Example 1 below. This means that we are in a somewhat different situation from the previous simple univariate case discussed in Section 1, where we can estimate the OLP by consistently estimating the OLPO.

Example 1. Suppose that { Y t , X t } t ≥ 1 satisfies (1.2), X t has a positive covariance C XX , and ε t is serially independent and also independent of X s for any s . In this case, A o C XX = C XY , making A o an OLPO. Since C XX is injective, any continuous linear operator ˜ A agrees with A o on the closure of ran C XX also agrees with A o on H (see Remark 2.2 and note that the closure of ran C XX is H in this case). However, consistently estimating A o without any further assumptions is impossible. To illustrate this, we may consider the case where { f j } j ≥ 1 in (1.2) are known for simplicity. Even in this simplified scenario, there are infinitely many unknown parameters { a j } j ≥ 1 to be estimated from only T samples, necessitating additional assumptions on { a j } j ≥ 1 for consistent estimation.

## 3. Estimation

We observed that in a general Hilbert space setting, there can be not only multiple OLPOs but also instances where, even if a unique OLPO exists, consistent estimation of it is impossible. Nevertheless, under mild conditions, we may construct a predictor using a standard post dimension-reduction estimator in such a way that the associated empirical MSPE converges to the minimum MPSE as in the simple univariate case. To propose such a predictor, let

<!-- formula-not-decoded -->

where k T is an integer satisfying the following assumption: below, we let a 1 ∧ a 2 = min { a 1 , a 2 } for a 1 , a 2 ∈ R and assume that max j ≥ 1 { j : E j } = 1 if the condition E j is not satisfied for all j ≥ 1.

Assumption 1 (Elbow-like rule) . k T in (3.1) is given by

<!-- formula-not-decoded -->

where τ T and υ T are user-specific choices of positive constants decaying to 0 as T → ∞ , and both τ - 1 T and υ - 1 T are bounded above by γ 0 T γ 1 for some γ 0 &gt; 0 and γ 1 ∈ (0 , 1 / 2) .

̂ C - 1 XX,k T in(3.1)is understood as the inverse of ̂ C XX viewed as a map acting on the restricted subspace span { ˆ v j } k T j =1 and k T in Assumption 1 is a random integer by its construction (see Remark 3.1 below). By including υ T in Assumption 1, we ensure that k T becomes o p ( T 1 / 2 ), which facilitates our theoretical analysis. Even if the choice of k T depends on various contexts requiring a regularized inverse of ̂ C XX , it is commonly set to a much smaller number than T , and thus this condition does not impose any practical restrictions. A practically more meaningful decision is made by the first component, max j ≥ 1 { j : ˆ λ j ≥ ˆ λ j +1 + τ T } in Assumption 1. Firstly, given that ˆ λ k T +1 &gt; 0, this condition implies that ˆ λ - 1 k T &lt; τ - 1 T , and thus ‖ ̂ C - 1 XX,k T ‖ ∞ ≤ τ - 1 T , where τ - 1 T diverges slowly compared to T ; clearly, this is one of the essential requirements for k T (as the rank of a regularized inverse of ̂ C XX ) to satisfy in the literature employing similar regularized inverses. Secondly, k T is determined near the point where the gap ˆ λ j - ˆ λ j +1 is no longer smaller than a specified threshold τ T for the last time. This approach is, in fact, analogous to the standard elbow rule used to determine the number of principal components in multivariate analysis based on the scree plot. Thus, even if Assumption 1 details some specific mathematical requirements necessary for our asymptotic analysis, these seem to closely align with existing practical rules for selecting k T , commonly employed in current practice; for example, see [10].

Using the regularized inverse ̂ C - 1 XX,k T , the proposed predictor is constructed as follows:

<!-- formula-not-decoded -->

The above predictor is standard in the literature on functional data analysis as the least squares predictor of Y t given the projection of X t onto the space spanned by the eigenvectors corresponding to the first k T largest eigenvalues; a similar estimator was earlier considered by [23]. The predictor described in (3.2) and its modifications (such as those that will be considered in Section 4.4) have been widely studied in the literature and adapted to various contexts; see e.g., [1], [2], [7], [17], [20], [27] and [28].

Remark 3.1. A significant difference in ̂ A compared to most of the existing estimators, lies in the choice of k T . In many earlier articles, k T is directly chosen by researchers and thus regarded as deterministic. However, as pointed out by [26], even in this case, it is generally not recommended to choose k T arbitrarily without taking the eigenvalues ˆ λ j into account. Therefore, it is natural to view k T as random from a practical point of view.

To establish consistency of ̂ A in(3.2), certain assumptions have been employed in the aforementioned literature, particularly concerning the eigenstructure of C XX and the unique identification of the target estimator A . However, as illustrated by Example 1 and Remark 2.2, these assumptions are not guaranteed to hold even in cases where there exists a well defined OLPO. Thus, in this paper, we do not make such assumptions for consistency, but allow ̂ A to potentially be inconsistent. Our main result in this section is that, despite not relying on the typical assumptions for consistency, the predictor ̂ Y t asymptotically minimizes the MSPE, which only requires mild conditions on the sample (cross-)covariance operators.

For the subsequent discussion, we define the following: for any B ∈ L ∞ ,

<!-- formula-not-decoded -->

-

Note that ‖ Σ( B ) ‖ S 1 is equivalent to the empirical MSPE, given by T - 1 ∑ T t =1 ‖ Y t BX t ‖ 2 . We then define an asymptotically OLPO as a random bounded linear operator producing a predictor that asymptotically minimizes the MSPE in the following sense:

Definition 1. Any random bounded linear operator ̂ B is called an asymptotically OLPO if

where Σ min = ‖ C Y Y - C 1 / 2 Y Y R XY R ∗ XY C 1 / 2 Y Y ‖ S 1 , which is the minimum MSPE that can be achieved by a linear predictor, as defined in Proposition 2.1.

<!-- formula-not-decoded -->

We will employ the following assumption:

Assumption 2 (Standard rate of convergence) . ‖ ̂ C Y Y - C Y Y ‖ ∞ , ‖ ̂ C XX C XX ‖ ∞ , and ‖ ̂ C XY - C XY ‖ ∞ are O p ( T - 1 / 2 ) .

-

We observe that { Y t ⊗ Y t - C Y Y } t ≥ 1 , { X t ⊗ X t - C XX } t ≥ 1 , and { X t ⊗ Y t - C XY } t ≥ 1 are stationary sequences of Schatten 2-class operators. These operator-valued sequences may also be understood as stationary sequences in a separable Hilbert space ([7], p. 34). Then, Assumption 2 is satisfied under some non-restrictive regularity conditions; see e.g., Theorems 2.16-2.18 of [7] concerning the central limit theorems for Hilbert-valued random elements. Given that we are dealing with a weakly stationary sequence { Y t , X t } t ≥ 1 , Assumption 2 appears to be standard. Furthermore, it can be relaxed to a weaker requirement by imposing stricter conditions on τ T and υ T in Assumption 1; this will be detailed in Section 4.1.

We now present the main result of this paper.

Theorem 3.1. Under Assumptions 1-2, ̂ A is an asymptotically OLPO, i.e., ‖ Σ( ̂ A ) ‖ S 1 → p Σ min .

As stated, an appropriate growth rate of k T , detailed in Assumption 1, and the standard rate of convergence of the sample (cross-)covariance operators (Assumption 2) are all that we need in order to demonstrate that the proposed predictor in (3.2) is an asymptotically OLPO. As discussed earlier, since the choice rule in Assumption 1 is practically similar to existing rules for selecting k T , a naive use of the simple post-dimension reduction estimator ̂ A in (3.2)

without the usual assumptions for the unique identification and/or consistency, which are widely employed but challenging to verity, can still be justified as a way to asymptotically minimize the MSPE (see Section 4.2). Some discussions and extensions on the above theorem are given in the next section. We also consider the case where ̂ A is replaced by another estimator employing a different regularization scheme (Sections 4.3-4.4).

## 4. Discussions and extensions

## 4.1. A more general result

In Assumption 2, we assumed that the sample (cross-)covariances converge to the population counterparts with √ T -rate. However, the exact √ T -rate is not mandatory for the desired result and can be relaxed if we make appropriate adjustments on τ T and υ T in Assumption 1 as follows:

Corollary 4.1. Suppose that, for β ∈ (0 , 1 / 2] , ‖ ̂ C Y Y - C Y Y ‖ ∞ , ‖ ̂ C XX - C XX ‖ ∞ , and ‖ ̂ C XY - C XY ‖ ∞ are O p ( T - β ) . If Assumption 1 holds for γ 1 ∈ (0 , β ) , then ̂ A is an asymptotically OLPO.

That is, ̂ A is an asymptotically OLPO under weaker assumptions on the sample (cross-)covariance operators if we impose stricter conditions on the decay rates of τ T and υ T . τ T and υ T are user-specific choices dependent on T , so researchers can easily manipulate their decay rates. Corollary 4.1, thus, tells us that we can make ̂ A an asymptotically OLPO under more general scenarios by simply reducing the decay rates of τ T and υ T .

## 4.2. Misspecified functional linear models and OLPO

Consider the standard functional linear model

<!-- formula-not-decoded -->

where A is typically assumed to satisfy the following two conditions: (i) A is Hilbert-Schmidt (i.e., A ∈ S 2 ) and (ii) A is uniquely identified (in S 2 ). A common assumption employed for the unique identification is that ran C XX is dense in H (see Remark 2.2). Under additional technical assumptions on the eigenstructure of C XX , such as those on the spectral gap ( λ j - λ j - 1 ) as in [12], the proposed estimator ̂ A in (3.2) turns out to be consistent if k T grows appropriately. Of course, some alternative estimators, such as the least squares estimator with Tikhonov regularization (see, e.g., [6]), do not require any assumptions on the spectral gap, which is a well known advantage of such methods. However, they still require condition (i), the Hilbert-Schmidt property of A , with some additional regularity assumptions on A , and also impose assumptions for condition (ii) when discussing the consistency of those estimators.

It is crucial to have conditions (i) and (ii) together with the model (4.1), since a violation of either of these can easily lead to inconsistency (see Example 1). While these requirements are standard for estimation, they exclude many natural data generating mechanisms in H (see Examples 1-2). Consequently, the model may suffer from misspecification issues. However, even in such cases, our results show that ̂ A attains the minimum MSPE asymptotically if k T grows at an appropriate rate, and thus affirm the potential use of the standard postdimension reduction estimator in practice without a careful examination of various technical conditions.

Example 2. Similar to the example given in Section 5 of [5], suppose that X t = Y t - 1 and Y t satisfies the functional AR(1) law of motion: Y t = AY t - 1 + ε t with A = ∑ ∞ j =1 a j w j ⊗ w j for some orthonormal basis { w j } j ≥ 1 and iid sequence { ε t } t ∈ Z (see also [2]). This leads to the following pointwise AR(1) model: 〈 Y t , w j 〉 = a j 〈 Y t - 1 , w j 〉 + 〈 ε t , w j 〉 . This time series is guaranteed to be stationary if sup j | a j | &lt; 1 (this can be demonstrated with only a slight and obvious modification of Theorem 3.1 of [7] or Proposition 3.2 of [25]). On the other hand, for A to be a Hilbert-Schmidt operator, a much more stringent condition ∑ ∞ j =1 | a j | 2 &lt; ∞ is required, and, as a consequence, 〈 Y t , w j 〉 and 〈 Y t - 1 , w j 〉 need to be nearly uncorrelated for large j while they can be arbitrarily correlated under the aforementioned condition for weak stationarity.

## 4.3. Requirement of sufficient dimension reduction

In our proof of Theorem 3.1 (resp. Corollary 4.1), it is crucial to have a regularized inverse of ̂ C XX , denoted as ̂ C - 1 XX,k T , whose rank k T grows at a sufficiently slower rate than T ; see e.g., (A.10) and (A.11) in Appendix A. Due to this requirement, our arguments for proving the main results (Theorem 3.1 and Corollary 4.1) are not straightforwardly extended to other popular estimators without dimension reduction, such as least squares-type estimators with ridge or Tikhonov regularization (see, e.g., [6]). Of course, this does not mean that these estimators cannot achieve prediction results similar to those in Theorem 3.1 and Corollary 4.1 without requiring the standard assumptions of HilbertSchmidtness and the unique identification of A . Rather, it simply suggests that, for estimators computed without dimension reduction, alternative approaches may be needed under different sets of assumptions. This could be a potential direction for future research.

## 4.4. Dimension reduction of the target variable

The proposed estimator ̂ A is commonly used as a standard estimator of the functional linear model (4.1). Note that, in our construction of ̂ A , the target variable Y t is used as is, without any dimension reduction. In the literature, estimators similar to ̂ A in (3.2), but where Y t is replaced with its version obtained through dimension reduction, also appear to be popular and are used in practice (see e.g., [27]). As noted in recent articles, dimension reduction of the target variable not only is generally non-essential for establishing certain key asymptotic properties (such as consistency) of the estimator but may also lead to a less optimal estimator (see Remark 1 of [13]).

Consider the following predictor and estimator, constructed using a version of Y t with reduced dimension:

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and ̂ Π Y,ℓ T = ∑ ℓ T j =1 ˆ w j ⊗ ˆ w j for some orthonormal basis { ˆ w j } j ≥ 1 and ℓ T growing as T increases; often, ˆ w j is set to the eigenvector of ̂ C Y Y or ̂ C XX corresponding to the j -th largest eigenvalue, but our subsequent analysis is not restricted to these specific cases. Even if the additional dimension reduction applied to Y t introduces some complications in our theoretical analysis, it can also be shown that ˜ A is an asymptotically OLPO under an additional mild condition.

Given that ̂ Π Y,ℓ T is the orthogonal projection with a growing rank, the condition given in Corollary 4.2 becomes easier to be satisfied if ℓ T grows more rapidly. Viewed in this light, the scenario in Theorem 3.1 can be seen as the limiting case where ̂ Π Y,ℓ T = I , indicating no dimension reduction applied to Y t . Corollary 4.2 tells us that estimators obtained by reducing the dimensionality of Y t tend to be asymptotically OLPOs under non-restrictive conditions. Given that C 1 / 2 Y Y and C Y Y share the same eigenvectors, one may conjecture that satisfying the requirement in Corollary 4.2 could be easier if ̂ w j is an eigenvector of ̂ C Y Y . Theoretical justification of this conjecture may require further assumptions on the eigenstructure of C Y Y . Given the focus on optimal linear prediction H under minimal conditions, we do not pursue this direction and leave it for future study.

Corollary 4.2. Suppose that Assumptions 1-2 hold and there exists a sequence m T tending to infinity as T →∞ such that m T ‖ ̂ Π Y,ℓ T C 1 / 2 Y Y - C 1 / 2 Y Y ‖ ∞ → p 0 and m T ≤ ℓ T eventually. Then, ˜ A is an asymptotically OLPO.

## 5. Simulation

We provide simulation evidence for our theoretical findings, focusing on cases that have not been sufficiently explored in the literature and where consistent estimation of the OLPO is impossible. In all simulation experiments, the number of replications is 1000.

Let { f j } j ≥ 1 be the Fourier basis of L 2 [0 , 1], the Hilbert space of squareintegrable functions on [0 , 1], i.e., for x ∈ [0 , 1], f 1 ( x ) = 1, and for j ≥ 2, f j = √ 2 sin(2 πjx ) if j is even, and f j = √ 2cos(2 πjx ) if j is odd. We define X t and Y t as follows: for some real numbers { a j } 101 j =1 ,

<!-- formula-not-decoded -->

where { e t } t ≥ 1 and { ε t } t ≥ 1 are assumed to be mutually and serially independent sequences of random elements. In this simulation setup, the minimal MSPE that can be achieved by a linear operator equals the Schatten 1-norm (trace norm) of the covariance operator C ε of ε t .

We conducted experiments in three different cases. In the first case (referred to as Case BB), e t and ε t are set to independent realizations of the standard Brownian bridge, and in the second case (referred to as Case CBM), they are set to independent realizations of the centered Brownian motion. In these first two cases, the j -th largest eigenvalue of C ε is given by π - 2 j - 2 ; see [14] (p. 86) and [16] (p. 1465). From a well known result on the Riemann zeta function (see e.g., [15]), we find that these eigenvalues add up to 1 / 6, which is the minimal achievable MSPE. In the last case (referred to as Case BM), e t and ε t are set as realizations of the standard Brownian motion multiplied by a constant, which is properly chosen to ensure that the minimal MSPE in this scenario matches that in the previous two cases.

We will subsequently consider the following four models, depending on the specification of a j and A for generating X t and Y t : for all j ≥ 1,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where b 0 ∼ U [ - 2 . 5 , 2 . 5] and b j ∼ iid U [ - 2 . 5 , 2 . 5] for j ≥ 1. Note that the parameters a j , b 0 and b j are generated differently in each simulation run; this allows us to assess the average performance of the proposed predictor across various parameter choices. In many empirical examples involving dependent sequences of functions X t , it is often expected that 〈 X t , v 〉 for any v ∈ H exhibits a positive lag-one autocorrelation, so we let a j tend to take positive values more frequently in our simulation settings. In any of the above cases, A is non-compact and hence cannot be consistently estimated without prior knowledge on the structure of { Af j } j ≥ 1 , which is as in the operator considered in Example 1.

We set τ T = 0 . 01 ‖ ̂ C XX ‖ S 1 T γ and υ T = 0 . 5 T γ for some γ &gt; 0, where note that τ T is designed to reflect the scale of X t , as proposed by [26] in a similar context. We then computed the empirical MSPE associated with ̂ A , introduced in (3.2). Table 1 reports the excess MSPE (the empirical MSPE minus 1 / 6) for each of the considered cases. As expected from our main theoretical results, the excess MSPE associated with ̂ A approaches zero as the sample size T increases, even if ̂ A is not consistent. Notably, in Case BM, the excess MSPE tends to be Notes: The excess MSPE is calculated as the empirical MSPE minus 1 / 6, where 1 / 6 represents the minimal achievable MSPE by a linear predictor. τ T = 0 . 01 ‖ ̂ C XX ‖ S 1 T γ and υ T = 0 . 5 T γ for γ ∈ { 0 . 45 , 0 . 475 } . The reported MSPEs are approximately computed by (i) generating X t and Y t on a fine grid of [0 , 1] with 200 equally spaced grid points, and then (ii) representing those with 100 cubic B-spline functions. The results exhibit little change with varying numbers of grid points and B-spline functions.

Table 1 Excess MSPE of the proposed predictor

(a) Case BB

| T   |   γ = 0 . 475 - 50 |   γ = 0 . 475 - 100 |   γ = 0 . 475 - 200 |   γ = 0 . 475 - 400 |   γ = 0 . 475 - 800 |   γ = 0 . 45 - 50 |   γ = 0 . 45 - 100 |   γ = 0 . 45 - 200 |   γ = 0 . 45 - 400 |   γ = 0 . 45 - 800 |
|-----|--------------------|---------------------|---------------------|---------------------|---------------------|-------------------|--------------------|--------------------|--------------------|--------------------|
| M1  |              0.043 |               0.035 |               0.023 |               0.018 |               0.012 |             0.072 |              0.051 |              0.031 |              0.022 |              0.014 |
| M2  |              0.041 |               0.033 |               0.022 |               0.017 |               0.011 |             0.068 |              0.049 |              0.029 |              0.021 |              0.013 |
| M3  |              0.056 |               0.046 |               0.031 |               0.023 |               0.016 |             0.094 |              0.066 |              0.040 |              0.028 |              0.019 |
| M4  |              0.054 |               0.044 |               0.029 |               0.022 |               0.015 |             0.089 |              0.063 |              0.038 |              0.027 |              0.018 |

(b) Case CBM

| T   |   γ = 0 . 475 - 50 |   γ = 0 . 475 - 100 |   γ = 0 . 475 - 200 |   γ = 0 . 475 - 400 |   γ = 0 . 475 - 800 |   γ = 0 . 45 - 50 |   γ = 0 . 45 - 100 |   γ = 0 . 45 - 200 |   γ = 0 . 45 - 400 |   γ = 0 . 45 - 800 |
|-----|--------------------|---------------------|---------------------|---------------------|---------------------|-------------------|--------------------|--------------------|--------------------|--------------------|
| M1  |              0.045 |               0.036 |               0.024 |               0.018 |               0.012 |             0.074 |              0.052 |              0.031 |              0.022 |              0.014 |
| M2  |              0.041 |               0.034 |               0.022 |               0.017 |               0.011 |             0.069 |              0.049 |              0.029 |              0.020 |              0.013 |
| M3  |              0.059 |               0.048 |               0.032 |               0.024 |               0.016 |             0.098 |              0.068 |              0.041 |              0.029 |              0.019 |
| M4  |              0.055 |               0.045 |               0.030 |               0.022 |               0.015 |             0.091 |              0.064 |              0.038 |              0.027 |              0.018 |

(c) Case BM

| T   |   γ = 0 . 475 - 50 |   γ = 0 . 475 - 100 |   γ = 0 . 475 - 200 |   γ = 0 . 475 - 400 |   γ = 0 . 475 - 800 |   γ = 0 . 45 - 50 |   γ = 0 . 45 - 100 |   γ = 0 . 45 - 200 |   γ = 0 . 45 - 400 |   γ = 0 . 45 - 800 |
|-----|--------------------|---------------------|---------------------|---------------------|---------------------|-------------------|--------------------|--------------------|--------------------|--------------------|
| M1  |              0.011 |               0.010 |               0.006 |               0.004 |               0.004 |             0.026 |              0.017 |              0.009 |              0.006 |              0.004 |
| M2  |              0.012 |               0.010 |               0.006 |               0.004 |               0.004 |             0.027 |              0.017 |              0.009 |              0.006 |              0.004 |
| M3  |              0.018 |               0.014 |               0.009 |               0.006 |               0.005 |             0.037 |              0.024 |              0.013 |              0.008 |              0.006 |
| M4  |              0.018 |               0.015 |               0.009 |               0.006 |               0.005 |             0.038 |              0.024 |              0.013 |              0.008 |              0.006 |

significantly smaller than in the other two cases (Case BB and Case CMB); even with a moderately large number of observations, the empirical MSPE in Case BMtends to be close to the minimal MSPE. This suggests that the performance of the proposed predictor significantly depends on the specification of ε t . Overall, the simulation results reported in Table 1 support our theoretical finding in Section 3. We also experimented with an alternative estimator ˜ A which is introduced in Section 4.4 and obtained qualitatively similar supporting evidence; some of the simulation results are reported in Appendix B of the appendix.

## 6. Concluding remarks

This paper studies linear prediction in a Hilbert space, demonstrating that, under mild conditions, the empirical MSPEs associated with standard postdimension reduction estimators approach the minimal achievable MSPE. There is ample room for future research; for example, it would be intriguing to explore whether similar prediction results can be obtained from various alternatives or modifications of the simple post-dimension reduction estimators considered in the literature.

## Appendix A: Proofs

## A.1. Useful lemmas

Lemma A.1. Let Γ be a nonnegative self-adjoint Schatten 1-class operator. For any D ∈ L ∞ , the following hold.

- (ii) Γ 1 / 2 D and D ∗ Γ 1 / 2 are Schatten 2-class operators.
- (i) Γ 1 / 2 DD ∗ Γ 1 / 2 and D Γ D ∗ are Schatten 1-class operators and their Schatten 1-norms are bounded above by ‖ D ‖ 2 ∞ ‖ Γ ‖ S 1 .

Proof. Let Γ = ∑ ∞ j =1 c j w j ⊗ w j , where c 1 ≥ c 2 ≥ . . . ≥ 0 and c j may be zero. By allocating a proper vector to each zero eigenvalue, we may assume that { w j } j ≥ 1 is an orthonormal basis of H . To show (i), we note that 〈 Γ 1 / 2 DD ∗ Γ 1 / 2 w j , w j 〉 = c j 〈 w j , DD ∗ w j 〉 ≤ c j ‖ D ‖ 2 ∞ , from which ‖ Γ 1 / 2 DD ∗ Γ 1 / 2 ‖ S 1 ≤ ‖ D ‖ 2 ∞ ‖ Γ ‖ S 1 is established. The desired result for ‖ D Γ D ∗ ‖ S 1 is already well known, see e.g., [11] (p. 267). To show (ii), we observe that ‖ Γ 1 / 2 D ‖ 2 S 2 = ‖ D ∗ Γ 1 / 2 ‖ 2 S 2 ≤ ‖ D ‖ 2 ∞ ∑ ∞ j =1 ‖ ‖ D ‖ 2 ∞ ∑ ∞ j =1 c j &lt; ∞ , which establishes the desired result.

Lemma A.2. Let { Γ j } j ≥ 1 be a sequence of random Schatten 1-class operators and let Γ be a self-adjoint Schatten 1-class operator. Then, for any orthonormal basis { w j } j ≥ 1 of H and m T ≥ 0 ,

<!-- formula-not-decoded -->

Moreover, ‖ Γ j - Γ ‖ S 1 → p 0 if ‖ Γ j ‖ S 1 -‖ Γ ‖ S 1 → p 0 and m T ‖ Γ j - Γ ‖ ∞ → p 0 as m T →∞ and T →∞ .

Proof. Equation (A.1) directly follows from Lemma 2 (and also Theorem 2) of [19]. Moreover, if ‖ Γ j ‖ S 1 -‖ Γ ‖ S 1 → p 0 and m T ‖ Γ j - Γ ‖ ∞ → p 0 as m T → ∞ and T → ∞ , we find that ‖ Γ j - Γ ‖ S 1 = O p ( ∑ ∞ j = m T +1 〈 Γ w j , w j 〉 ). Since Γ is a self-adjoint Schatten 1-class operator, we have ∑ ∞ j =1 〈 Γ w j , w j 〉 → p ‖ Γ ‖ S 1 &lt; ∞ , from which we conclude that ∑ ∞ j = m T +1 〈 Γ w j , w j 〉 → p 0 as m T →∞ .

## A.2. Proofs of the theoretical results

Proof of Proposition 2.1. From Theorem 1 of [4], we find that C XY allows the following representation: for a unique bounded linear operator R XY satisfying ‖ R XY ‖ ∞ ≤ 1,

<!-- formula-not-decoded -->

Let C min = C Y Y - C 1 / 2 Y Y R XY R ∗ XY C 1 / 2 Y Y , which is clearly self-adjoint. Moreover, C min is a nonnegative Schatten 1-class operator. To see this, note that for any w ∈ H ,

<!-- formula-not-decoded -->

Γ

1

/

2

w

j

‖

2

=

<!-- formula-not-decoded -->

(i.e., C min is nonnegative), which is because ‖ R ∗ XY C 1 / 2 Y Y w ‖ 2 ≤ ‖ R ∗ XY ‖ 2 ∞ ‖ C 1 / 2 Y Y w ‖ 2 ≤ ‖ C 1 / 2 Y Y w ‖ 2 . Moreover, from (A.2) and Lemma A.1(ii), we know that, for any orthonormal basis { w j } j ≥ 1 , ‖ C min ‖ S 1 = ∑ ∞ j =1 ( ‖ C 1 / 2 Y Y w j ‖ 2 - ‖ R ∗ XY C 1 / 2 Y Y w j ‖ 2 ) &lt; ∞ .

We then find the following holds for any B ∈ L ∞ and w ∈ H :

<!-- formula-not-decoded -->

We know from (A.3) that

<!-- formula-not-decoded -->

where { w j } j ≥ 1 is any orthonormal basis of H . Note that C 1 / 2 XX B ∗ - R ∗ XY C 1 / 2 Y Y is a Schatten 2-class operator (Lemma A.1(ii)) and thus we find that ∑ ∞ j =1 ‖ C 1 / 2 XX B ∗ w j R ∗ XY C 1 / 2 Y Y w j ‖ 2 = ‖ C 1 / 2 XX B ∗ - R ∗ XY C 1 / 2 Y Y ‖ 2 S 2 . From this result combined with(A.4), the desired result immediately follows.

Proofs of Theorem 3.1 and Corollary 4.1. To accommodate more general cases, which are considered in Corollary 4.1, we hereafter assume that τ - 1 T = O p ( T γ 1 ) and υ - 1 T = O p ( T γ 1 ) for some γ 1 ∈ (0 , β ), and ‖ ̂ C XX - C XX ‖ ∞ , ‖ ̂ C Y Y - C Y Y ‖ ∞ and ‖ ̂ C XY - C XY ‖ ∞ are all O p ( T β ) for some β ∈ (0 , 1 / 2]. Our proof of Theorem 3.1 corresponds to the particular case with β = 1 / 2.

Note that sup j ≥ 1 | ˆ λ j - λ j | ≤ ‖ ̂ C XX - C XX ‖ ∞ = O p ( T - β ) (Lemma 4.2 of [7]). Under Assumption 1, we have ˆ λ k T - ˆ λ k T +1 ≥ τ T and thus

<!-- formula-not-decoded -->

If λ k T = λ k T +1 , (A.5) reduces to T β ( ˆ λ k T - ˆ λ k T +1 ) ≥ T β τ T . Moreover, since T β ( ˆ λ k T - ˆ λ k T +1 ) = O p (1) and T β τ T → p ∞ , P { ˆ λ k T - ˆ λ k T +1 ≥ τ T | λ k T = λ k T +1 } → 0 as T →∞ . Using the Bayes' rule and the facts that P { ˆ λ k T - ˆ λ k T +1 ≥ τ T } = 1 and P { λ k T = λ k T +1 } = P { λ k T = λ k T +1 | ˆ λ k T - ˆ λ k T +1 ≥ τ T } by Assumption 1, we find that P { λ k T = λ k T +1 } → 0. To establish the desired consistency, we thus may subsequently assume that λ k T &gt; λ k T +1 .

-

Let C min = C Y Y - C 1 / 2 Y Y R XY R ∗ XY C 1 / 2 Y Y . Since there exists A ∈ L ∞ such that C XY = AC XX , C min = C Y Y - AC XX A ∗ holds (Corollary 2.1). From the triangular inequality applied to the S 1 -norm, we then find that

<!-- formula-not-decoded -->

It suffices to show that each summand in the RHS of (A.6) is o p (1).

We will first consider the first term in the RHS of (A.6). Let m T be any divergent sequence (depending on T ) but satisfy T - β m T → 0. Note that ‖ ̂ C Y Y ‖ S 1 - ‖ C Y Y ‖ S 1 = ∑ m T j =1 (ˆ µ j - µ j ) + ∑ ∞ j = m T +1 (ˆ µ j - µ j ) ≤ m T ‖ ̂ C Y Y - C Y Y ‖ ∞ + ∑ ∞ j = m T +1 (ˆ µ j - µ j ). For every δ &gt; 0, let E δ = {| ∑ ∞ j = m T +1 (ˆ µ j - µ j ) | &gt; δ 2 } and F δ = { m T ‖ ̂ C Y Y - C Y Y ‖ ∞ &gt; δ 2 } . Since P { m T ‖ ̂ C Y Y - C Y Y ‖ ∞ + ∑ ∞ j = m T +1 (ˆ µ j - µ j ) &gt; δ } ≤ P { E δ } + P { F δ } , we find that

<!-- formula-not-decoded -->

We next consider the third term in the RHS of (A.6). We know from Lemma A.1 that ‖ AC XX,k T A ∗ - AC XX A ∗ ‖ S 1 ≤ ‖ A ‖ 2 ∞ ‖ C XX,k T - C XX ‖ S 1 . Since C XX is a Schatten 1-class operator and k T grows without bound, ‖ C XX,k T - C XX ‖ S 1 = ∑ ∞ j = k T +1 λ j → p 0 and thus ‖ AC XX,k T A ∗ - AC XX A ∗ ‖ S 1 = o p (1) as desired. We lastly consider the second term in the RHS of (A.6). Let ε t = Y t - AX t .

Note that ̂ C Y Y and C Y Y are Schatten 1-class operators (almost surely) and also m T increases without bound. Moreover, m T ‖ ̂ C Y Y - C Y Y ‖ ∞ = O p ( m T T - β ) = o p (1) under our assumptions. These results imply that P { E δ } → 0 and P { F δ } → 0 as T →∞ , and thus we conclude that ‖ ̂ C Y Y ‖ S 1 -‖ C Y Y ‖ S 1 → p 0. Combining this result with Lemma A.2 and the fact that m T ‖ ̂ C Y Y - C Y Y ‖ ∞ = o p (1), we find that ‖ ̂ C Y Y - C Y Y ‖ S 1 → p 0 as desired.

Since ̂ C XY = A ̂ C XX + ̂ C εX and ̂ A ̂ C XX,k T ̂ A ∗ = ̂ C XY ̂ C - 1 XX,k T ̂ C ∗ XY , we have

where ̂ Π X,k T = ∑ k T j =1 ˆ v j ⊗ ˆ v j . Therefore, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It will be proved later that the first term in the RHS of (A.7) is o p (1), i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We deduce from Lemma A.1(i) that the second term in the RHS of (A.7), below denoted simply as D , satisfies Since ̂ C Xε = T - 1 ∑ T t =1 X t ⊗ Y t - T - 1 ∑ T t =1 X t ⊗ AX t = C Y X - AC XX + O p ( T - β ) = O p ( T - β ), we find that

<!-- formula-not-decoded -->

and also

<!-- formula-not-decoded -->

where the last equality follows from the fact that ‖ ̂ C εX ‖ 2 ∞ = O p ( T - 2 β ) and ∑ k T j =1 ˆ λ - 1 j ≤ τ - 1 T k T = o ( T 2 β ) hold under Assumptions 1 and 2. As shown by (A.9)-(A.11), the second term in the RHS of (A.7) is o p (1). Combining this result with (A.8), we find that ‖ ̂ A ̂ C XX,k T ̂ A ∗ - AC XX,k T A ∗ ‖ S 1 = o p (1) as desired.

It remains to verify (A.8) to complete the proof. From Lemma A.1(i), we know that

<!-- formula-not-decoded -->

and thus it suffices to show that the RHS of (A.12) is o p (1). To this end, we first note that

<!-- formula-not-decoded -->

We then obtain an upper bound of ‖ ̂ C XX,k T - C XX,k T ‖ ∞ as follows:

<!-- formula-not-decoded -->

where ̂ Λ k T = ∑ k T j =1 ˆ λ j ˆ f j ⊗ ˆ f j - ∑ k T j =1 ˆ λ j f j ⊗ f j . From similar algebra used in the proof of Lemma 3.1 of [24] and the fact that ‖ · ‖ ∞ ≤ ‖ · ‖ S 2 , we find that

<!-- formula-not-decoded -->

Observe that, for every ℓ = 1 , . . . , k T ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since sup ℓ ≥ 1 | ˆ λ ℓ - λ ℓ | ≤ ‖ ̂ C XX - C XX ‖ ∞ = O p ( T - β ), we find that the RHS of (A.16)is O p ( T - 2 β ). Using the fact that τ - 1 T ≥ ( ˆ λ ℓ - ˆ λ ℓ +1 ) - 1 for all ℓ = 1 , . . . , k T , the following is deduced: for some δ &gt; 0,

<!-- formula-not-decoded -->

where the last equality is deduced from the facts that ∑ k T ℓ =1 ˆ λ 2 ℓ = O p (1) and τ - 1 T = O p ( T γ 1 ) for some γ 1 ∈ (0 , β ) under the employed conditions. From nearly identical arguments, we also find that

<!-- formula-not-decoded -->

Moreover, from similar algebra used in (A.16) we find that

<!-- formula-not-decoded -->

From (A.14)-(A.19), we know that there is a divergent sequence m T such that m T ‖ ̂ C XX,k T - C XX,k T ‖ ∞ → p 0. Combining this result with (A.13) and Lemma A.2, we find that the RHS of (A.12) is o p (1) (and thus (A.8) holds). □

Proof of Corollary 2.1. (a) directly follows from the facts that (i) ‖ ( BC 1 / 2 XX - C 1 / 2 Y Y R XY ) w j ‖ 2 = 0 for all j ≥ 1 if and only if ‖ BC 1 / 2 XX - C 1 / 2 Y Y R XY ‖ 2 S 2 = 0 and (ii) the Hahn-Banach extension theorem (see e.g., Theorem 1.9.1 of [21]). This result in turn implies (b) due to the fact that C XY = C 1 / 2 Y Y R XY C 1 / 2 XX as observed in Proposition 2.1. □

Proof of Corollary 4.2. Note that (A.6) holds when ̂ A is replaced by ˜ A , and ‖ ̂ C Y Y - C Y Y ‖ S 1 = o p (1) and ‖ AC XX,k T A ∗ - AC XX A ∗ ‖ S 1 = o p (1) can be shown as in our proof of Theorem 3.1. We thus have

<!-- formula-not-decoded -->

and hence it suffices to show that ‖ ˜ A ̂ C XX,k T ˜ A ∗ - AC XX,k T A ∗ ‖ S 1 = o p (1). Note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using similar arguments used in our proof of Theorem 3.1, Lemma A.1 and the facts that ‖ ̂ Π Y,ℓ T ‖ ∞ ≤ 1 for any arbitrary ℓ T ≥ 1 and ‖ AC XX,k T A ∗ - AC XX A ∗ ‖ S 1 = o p (1), we find that ‖ ̂ Π Y,ℓ T ( A ̂ C XX,k T A ∗ - AC XX,k T A ∗ ) ̂ Π Y,ℓ T ‖ S 1 = o p (1), ‖ ̂ Π Y,ℓ T ( AC XX,k T A ∗ - AC XX A ∗ ) ̂ Π Y,ℓ T ‖ S 1 = o p (1), and ‖ ̂ Π Y,ℓ T ( A ̂ Π X,k T ̂ C ∗ εX + ̂ C εX A ∗ + ̂ C εX ̂ C - 1 XX,k T ̂ C ∗ εX ) ̂ Π Y,ℓ T ‖ S 1 = o p (1). Therefore,

If ‖ ̂ Π Y,ℓ T AC XX A ∗ ̂ Π Y,ℓ T - AC XX A ∗ ‖ S 1 → p 0, the desired result is established. Let Υ = C 1 / 2 Y Y R XY and ̂ Υ = ̂ Π Y,ℓ T C 1 / 2 Y Y R XY . We then have AC XX A ∗ = ΥΥ ∗ (see Proposition 2.1) and ̂ Π Y,ℓ T AC XX A ∗ ̂ Π Y,ℓ T = ̂ Υ ̂ Υ ∗ . Observe that

This implies that ‖ ̂ Υ ̂ Υ ∗ - ΥΥ ∗ ‖ ∞ → p 0 if there exists a divergent sequence m T such that m T ‖ ̂ Π Y,ℓ T C 1 / 2 Y Y - C 1 / 2 Y Y ‖ ∞ → p 0 and m T ≤ ℓ T for large T . Moreover, we find the following from the fact that ΥΥ ∗ is nonnegative self-adjoint and { ˆ w j } ∞ j =1 is an orthonormal basis of H :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which is o p (1) since ΥΥ ∗ = C 1 / 2 Y Y R XY R ∗ XY C 1 / 2 Y Y is a Schatten 1-class (Lemma A.1(i)). We thus deduce from Lemma A.2 that ‖ ̂ Υ ̂ Υ ∗ - ΥΥ ∗ ‖ S 1 → p 0 as desired.

## Appendix B: Additional simulation results for alternative estimators

In this section, we experiment with an alternative estimator ˜ A which is introduced in Section 4.4. We replicate the same simulation experiments conducted in Section 5 using the alternative estimator ˜ A introduced in Section 4.4. We construct ̂ Π Y,ℓ T using the eigenvectors { ˆ u j } j ≥ 1 of ̂ C Y Y with ℓ T = ⌊ T 1 / 2 ⌋ . Overall, we found that the computed excess MSPEs tend to be similar to those obtained with ̂ A , providing supporting evidence for our finding in Corollary 4.2. The simulation results are reported in Table 2. We also experimented with ̂ Π Y,ℓ T constructed from the eigenvectors { ˆ v j } j ≥ 1 of ̂ C XX , and the results are reported in Table 3. Even if the empirical MSPEs tend to be larger than in the previous case, we obtained overall similar simulation results.

Table 2 Excess MSPE of the alternative predictor, ̂ Π Y,ℓ T = ∑ ℓ T j =1 ˆ u j ⊗ ˆ u j

(a) Case BB

| T   |   γ = 0 . 475 - 50 |   γ = 0 . 475 - 100 |   γ = 0 . 475 - 200 |   γ = 0 . 475 - 400 |   γ = 0 . 475 - 800 |   γ = 0 . 45 - 50 |   γ = 0 . 45 - 100 |   γ = 0 . 45 - 200 |   γ = 0 . 45 - 400 |   γ = 0 . 45 - 800 |
|-----|--------------------|---------------------|---------------------|---------------------|---------------------|-------------------|--------------------|--------------------|--------------------|--------------------|
| M1  |              0.043 |               0.035 |               0.023 |               0.017 |               0.011 |             0.071 |              0.049 |              0.029 |              0.021 |              0.013 |
| M2  |              0.045 |               0.035 |               0.023 |               0.017 |               0.012 |             0.071 |              0.051 |              0.030 |              0.021 |              0.014 |
| M3  |              0.056 |               0.045 |               0.030 |               0.022 |               0.015 |             0.091 |              0.063 |              0.038 |              0.027 |              0.017 |
| M4  |              0.057 |               0.046 |               0.031 |               0.023 |               0.015 |             0.092 |              0.065 |              0.039 |              0.028 |              0.018 |

(b) Case CBM

| T   |   γ = 0 . 475 - 50 |   γ = 0 . 475 - 100 |   γ = 0 . 475 - 200 |   γ = 0 . 475 - 400 |   γ = 0 . 475 - 800 |   γ = 0 . 45 - 50 |   γ = 0 . 45 - 100 |   γ = 0 . 45 - 200 |   γ = 0 . 45 - 400 |   γ = 0 . 45 - 800 |
|-----|--------------------|---------------------|---------------------|---------------------|---------------------|-------------------|--------------------|--------------------|--------------------|--------------------|
| M1  |              0.043 |               0.035 |               0.023 |               0.017 |               0.011 |             0.071 |              0.050 |              0.029 |              0.021 |              0.013 |
| M2  |              0.044 |               0.036 |               0.023 |               0.017 |               0.012 |             0.072 |              0.051 |              0.030 |              0.021 |              0.014 |
| M3  |              0.057 |               0.046 |               0.030 |               0.022 |               0.015 |             0.093 |              0.065 |              0.039 |              0.027 |              0.018 |
| M4  |              0.057 |               0.047 |               0.031 |               0.023 |               0.015 |             0.094 |              0.066 |              0.039 |              0.028 |              0.018 |

(c) Case BM

| T   |   γ = 0 . 475 - 50 |   γ = 0 . 475 - 100 |   γ = 0 . 475 - 200 |   γ = 0 . 475 - 400 |   γ = 0 . 475 - 800 |   γ = 0 . 45 - 50 |   γ = 0 . 45 - 100 |   γ = 0 . 45 - 200 |   γ = 0 . 45 - 400 |   γ = 0 . 45 - 800 |
|-----|--------------------|---------------------|---------------------|---------------------|---------------------|-------------------|--------------------|--------------------|--------------------|--------------------|
| M1  |              0.012 |               0.010 |               0.006 |               0.004 |               0.004 |             0.026 |              0.017 |              0.009 |              0.006 |              0.004 |
| M2  |              0.012 |               0.010 |               0.006 |               0.004 |               0.004 |             0.027 |              0.018 |              0.009 |              0.006 |              0.004 |
| M3  |              0.018 |               0.014 |               0.009 |               0.006 |               0.005 |             0.037 |              0.024 |              0.013 |              0.008 |              0.006 |
| M4  |              0.018 |               0.015 |               0.009 |               0.006 |               0.005 |             0.038 |              0.025 |              0.013 |              0.008 |              0.006 |

Notes: The excess MSPEs are calculated as in Table 1.

Table 3 Excess MSPE of the alternative predictor, ̂ Π Y,ℓ T = ∑ ℓ T j =1 ˆ v j ⊗ ˆ v j (a) Case BB

| T   |   γ = 0 . 475 - 50 |   γ = 0 . 475 - 100 |   γ = 0 . 475 - 200 |   γ = 0 . 475 - 400 |   γ = 0 . 475 - 800 |   γ = 0 . 45 - 50 |   γ = 0 . 45 - 100 |   γ = 0 . 45 - 200 |   γ = 0 . 45 - 400 |   γ = 0 . 45 - 800 |
|-----|--------------------|---------------------|---------------------|---------------------|---------------------|-------------------|--------------------|--------------------|--------------------|--------------------|
| M1  |              0.043 |               0.035 |               0.023 |               0.017 |               0.011 |             0.071 |              0.049 |              0.029 |              0.021 |              0.013 |
| M2  |              0.066 |               0.049 |               0.032 |               0.023 |               0.015 |             0.089 |              0.063 |              0.039 |              0.026 |              0.017 |
| M3  |              0.056 |               0.045 |               0.030 |               0.022 |               0.015 |             0.091 |              0.063 |              0.038 |              0.027 |              0.017 |
| M4  |              0.078 |               0.057 |               0.036 |               0.025 |               0.017 |             0.109 |              0.075 |              0.044 |              0.030 |              0.019 |

(b) Case CBM

| T   |   γ = 0 . 475 - 50 |   γ = 0 . 475 - 100 |   γ = 0 . 475 - 200 |   γ = 0 . 475 - 400 |   γ = 0 . 475 - 800 |   γ = 0 . 45 - 50 |   γ = 0 . 45 - 100 |   γ = 0 . 45 - 200 |   γ = 0 . 45 - 400 |   γ = 0 . 45 - 800 |
|-----|--------------------|---------------------|---------------------|---------------------|---------------------|-------------------|--------------------|--------------------|--------------------|--------------------|
| M1  |              0.044 |               0.035 |               0.023 |               0.017 |               0.011 |             0.071 |              0.050 |              0.029 |              0.021 |              0.013 |
| M2  |              0.075 |               0.055 |               0.036 |               0.025 |               0.016 |             0.099 |              0.069 |              0.043 |              0.029 |              0.018 |
| M3  |              0.057 |               0.046 |               0.030 |               0.022 |               0.015 |             0.093 |              0.065 |              0.039 |              0.027 |              0.018 |
| M4  |              0.096 |               0.071 |               0.046 |               0.032 |               0.021 |             0.127 |              0.089 |              0.055 |              0.037 |              0.023 |

(c) Case BM

| T   |   γ = 0 . 475 - 50 |   γ = 0 . 475 - 100 |   γ = 0 . 475 - 200 |   γ = 0 . 475 - 400 |   γ = 0 . 475 - 800 |   γ = 0 . 45 - 50 |   γ = 0 . 45 - 100 |   γ = 0 . 45 - 200 |   γ = 0 . 45 - 400 |   γ = 0 . 45 - 800 |
|-----|--------------------|---------------------|---------------------|---------------------|---------------------|-------------------|--------------------|--------------------|--------------------|--------------------|
| M1  |              0.012 |               0.010 |               0.006 |               0.004 |               0.004 |             0.026 |              0.017 |              0.009 |              0.006 |              0.004 |
| M2  |              0.032 |               0.022 |               0.014 |               0.009 |               0.007 |             0.046 |              0.030 |              0.017 |              0.011 |              0.008 |
| M3  |              0.018 |               0.015 |               0.009 |               0.006 |               0.005 |             0.037 |              0.024 |              0.013 |              0.008 |              0.006 |
| M4  |              0.040 |               0.028 |               0.017 |               0.011 |               0.008 |             0.058 |              0.037 |              0.021 |              0.013 |              0.009 |

Notes: The excess MSPEs are calculated as in Table 1.

## References

- [1] Aue, A. and Klepsch, J. (2017). Estimating functional time series by moving average model fitting, arxiv:1701.00770 [stat.ME].
- [2] Aue, A., Norinho, D. D., and H¨ ormann, S. (2015). On the prediction of stationary functional time series. Journal of the American Statistical Association , 110(509):378-392.
- [3] Babii, A. and Florens, J.-P. (2025). Is completeness necessary? estimation in nonidentified linear models. Econometric Theory , page 1-38.
- [4] Baker, C. R. (1973). Joint measures and cross-covariance operators. Transactions of the American Mathematical Society , 186:273-289.
- [5] Beare, B. K., Seo, J., and Seo, W.-K. (2017). Cointegrated linear processes in Hilbert space. Journal of Time Series Analysis , 38(6):1010-1027.
- [6] Benatia, D., Carrasco, M., and Florens, J.-P. (2017). Functional linear regression with functional response. Journal of Econometrics , 201(2):269-291.
- [7] Bosq, D. (2000). Linear Processes in Function Spaces . Springer-Verlag New York.
- [8] Bosq, D. (2007). General linear processes in Hilbert spaces and prediction. Journal of Statistical Planning and Inference , 137(3):879-894.
- [9] Bosq, D. (2014). Computing the best linear predictor in a Hilbert space. Applications to general ARMAH processes. Journal of Multivariate Analysis , 124:436-450.
- [10] Chang, Y., Choi, Y., Kim, S., and Park, J. (2021). Stock market return predictability dormant in option panels. Mimeo, Department of Economics, Indiana Univeristy.
- [11] Conway, J. B. (1994). A Course in Functional Analysis . Springer.
- [12] Hall, P. and Horowitz, J. L. (2007). Methodology and convergence rates for functional linear regression. Annals of Statistics , 35(1):70 - 91.
- [13] Imaizumi, M. and Kato, K. (2018). PCA-based estimation for functional linear regression with functional responses. Journal of Multivariate Analysis , 163:15-36.
- [14] Jaimez, R. G. and Bonnet, M. J. V. (1987). On the Karhunen-Loeve expansion for transformed processes. Trabajos de Estadistica , 2:81-90.
- [15] Kalman, D. and McKinzie, M. (2012). Another way to sum a series: generating functions, euler, and the dilog function. The American Mathematical Monthly , 119(1):42-51.
- [16] Karol, A., Nazarov, A., and Nikitin, Y. (2008). Small ball probabilities for Gaussian random fields and tensor products of compact operators. Transactions of the American Mathematical Society , 360(3):1443-1474.
- [17] Klepsch, J., Kl¨ uppelberg, C., and Wei, T. (2017). Prediction of functional ARMA processes with an application to traffic data. Econometrics and Statistics , 1:128-149.
- [18] Klepsch, J. and Kl¨ uppelberg, C. (2017). An innovations algorithm for the prediction of functional linear processes. Journal of Multivariate Analysis , 155:252-271.
- [19] Kubrusly, C. (1985). On convergence of nuclear and correlation operators

- in Hilbert space. Technical report, Laboratorio de Computacao Cientifica.
- [20] Luo, R. and Qi, X. (2017). Function-on-function linear regression by signal compression. Journal of the American Statistical Association , 112(518):690705.
- [21] Megginson, R. E. (2012). Introduction to Banach Space Theory . Springer New York.
- [22] Mollenhauer, M., M¨ ucke, N., and Sullivan, T. J. (2023). Learning linear operators: Infinite-dimensional regression as a well-behaved non-compact inverse problem.
- [23] Park, J. Y. and Qian, J. (2012). Functional regression of continuous state distributions. Journal of Econometrics , 167(2):397-412.
- [24] Reimherr, M. (2015). Functional regression with repeated eigenvalues. Statistics &amp; Probability Letters , 107:62-70.
- [25] Seo, W.-K. (2023). Cointegration and representation of cointegrated autoregressive processes in Banch spaces. Econometric Theory , 39(4):737-788.
- [26] Seong, D. and Seo, W.-K. (2021). Functional instrumental variable regression with an application to estimating the impact of immigration on native wages, arXiv:2110.12722 [econ.EM].
- [27] Yao, F., M¨ uller, H.-G., and Wang, J.-L. (2005). Functional linear regression analysis for longitudinal data. The Annals of Statistics , 33(6):2873-2903.
- [28] Zhang, X., Chiou, J.-M., and Ma, Y. (2018). Functional prediction through averaging estimated functional linear regression models. Biometrika , 105(4):945-962.