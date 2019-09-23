# LearnigModels
Some estimators for various problem, clustering, classification, regression, and so on.

Table of Contents
=================

* [gp_classifier](#gp_classifier)
* [gp_regression](#gp_regression)
* [hsmm](#hsmm)
* [lib](#lib)
  * [learning](##learning)
  * [util](##util)
* [lrmm](#lrmm)

# gp_classifier
Estimators for classification problem by using Gaussian Process prior.

# gp_regression
Estimators for regression problem by using Gaussian Process prior.
The gaussian process prior is like this:
```math
\begin{equation}
p(f|X) \propto \exp(-f(x)^T K(x,x) f(x)),
f(x) \in \mathbb{R}^{dim(X)}.
\end{equation}
```
Distribution for regression problem exists not only normal distribution, but also others like hyperbolic secant distribution, Von-Mises distribution, and so on. In this folder, various kinds of distribution with gaussian process prior are used.

# hsmm
Hyperbolic Secant Mixture Model(HSMM).
Hyperbolic secant distribution is like this:  
```math
\begin{equation}
 p(x|b,s) = \frac{\sqrt{s}}{2 \pi} 1/\cosh(\frac{\sqrt{s}}{2}(x - b).
\end{equation}
```
Since mixture model can be used for clustering, the HSMM also used for clustering.
Performance for the model are evaluated in this folder.

# lib
General Library in this repository
## learning
  General library especially used in the fit part.
## util
  General library used in the all parts.

# lrmm
Logistic Regression Mixture Model.
Let y be binary random variable,
the logistic regression is like this:
```math
\begin{equation}
p(y = 1|x,w) = \sigma(x \cdot w).
\end{equation}
```
Mixture model gives richer regression, the LRMM used to enhance model representation.
