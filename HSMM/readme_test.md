# Library for hyperbolic secant distribution
This library is for hyperbolic secant distribution. The distribution is as follows:
$ p(x|w) = \frac{s}{2\pi} sech(\frac{s}{2}(x - b))$,  
where $x, b \in \mathbb{R}, s \in \mathbb{R}_+$.

To estimate the distribution, local variational approximation is used.
Through the estimation, there are various usage.
1. Regression
  + <img src="https://latex.codecogs.com/gif.latex?p(y|x,w)&space;=&space;\frac{s}{2\pi}&space;sech(y&space;-&space;f(x,b))," title="p(y|x,w) = \frac{s}{2\pi} sech(y - f(x,b))," /> is regression model.  
  Here, there are various possibility for the form of f(x,b). Simple case is <img src="https://latex.codecogs.com/gif.latex?f(x,b)&space;=&space;x&space;\cdot&space;b,&space;x,b&space;\in&space;\mathbb{R}^M" title="f(x,b) = x \cdot b, x,b \in \mathbb{R}^M" />.
  + This library does not consider the above case, but also non-parameteric case, i.e. the prior distribution is gaussian process.

2. 

# Sample Code for use
See example for more detail.

