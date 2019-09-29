# Library for hyperbolic secant distribution
This library is for hyperbolic secant distribution. The distribution is as follows:  
<img src="https://latex.codecogs.com/gif.latex?p(x|w)&space;=&space;\frac{\sqrt{s}}{2\pi}sech&space;\bigl(&space;\frac{s}{2}x-b&space;\bigr)" title="p(x|w) = \frac{\sqrt{s}}{2\pi}sech \bigl( \frac{\sqrt{s}}{2}(x-b) \bigr)" />,

where <img src="https://latex.codecogs.com/gif.latex?x,&space;b&space;\in&space;\mathbb{R}^M" title="x, b \in \mathbb{R}"/> and <img src="https://latex.codecogs.com/gif.latex?s&space;\in&space;\mathbb{R}_&plus;" title="s \in \mathbb{R}_+" />.

To estimate the distribution, local variational approximation is used.
Through the estimation, there are various usage:

1. Probability density function itself
  + When we have data x, we may want to know the mean and scale. In this case, the distribution is used in the case that the data is following to the family of the distribution.
2. Regression
  + <img src="https://latex.codecogs.com/gif.latex?p(y|x,w)&space;=&space;\frac{s}{2\pi}&space;sech(y&space;-&space;f(x,b))," title="p(y|x,w) = \frac{\sqrt{s}}{2\pi} sech(\frac{\sqrt{s}}{2}y - f(x,b))" /> is regression model.  
  Here, there are various possibility for the form of f(x,b). Simple case is <img src="https://latex.codecogs.com/gif.latex?f(x,b)&space;=&space;x&space;\cdot&space;b,&space;x,b&space;\in&space;\mathbb{R}^M" title="f(x,b) = x \cdot b, x,b \in \mathbb{R}^M" />.
  + This library does not consider the above case, but also non-parameteric case, i.e. the prior distribution is gaussian process.
3. Clustering
  + When we want to separate our data into some groups, we need to label each data. Considering probability density function for each data given label, the distribution can be applicable.

# Sample Code for use
  + Each notebook is sample programs:
    1. sample_pdf_est.ipynb
      + Sample for probability density function.
    2. sample_regression.ipynb
      + Sample for regression problem.
    3. sample_clustering.ipynb
      + Sample for clustering problem.
