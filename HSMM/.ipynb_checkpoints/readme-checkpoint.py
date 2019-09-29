# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Library for hyperbolic secant distribution
# This library is for hyperbolic secant distribution. The distribution is as follows:
# $$ p(x|w) = \frac{s}{2\pi} sech(\frac{s}{2}(x - b))  $$,
#
# where <img src="https://latex.codecogs.com/gif.latex?x,&space;b&space;\in&space;\mathbb{R}^M" title="x, b \in \mathbb{R}"/> and <img src="https://latex.codecogs.com/gif.latex?s&space;\in&space;\mathbb{R}_&plus;" title="s \in \mathbb{R}_+" />.
#
# To estimate the distribution, local variational approximation is used.
# Through the estimation, there are various usage.
# 1. Regression
#   + <img src="https://latex.codecogs.com/gif.latex?p(y|x,w)&space;=&space;\frac{s}{2\pi}&space;sech(y&space;-&space;f(x,b))," title="p(y|x,w) = \frac{s}{2\pi} sech(y - f(x,b))," /> is regression model.  
#   Here, there are various possibility for the form of f(x,b). Simple case is <img src="https://latex.codecogs.com/gif.latex?f(x,b)&space;=&space;x&space;\cdot&space;b,&space;x,b&space;\in&space;\mathbb{R}^M" title="f(x,b) = x \cdot b, x,b \in \mathbb{R}^M" />.
#   + This library does not consider the above case, but also non-parameteric case, i.e. the prior distribution is gaussian process.
#
# 2. 
#
# # Sample Code for use
# See example for more detail.
#

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


def multi_hyperbolic_secant(x:np.ndarray, s:float):
    return 1/np.cosh(np.sqrt(s*np.abs(x[:,0])*np.abs(x[:,1]))/2) * np.sqrt(s)/(2*np.pi)**2


# +
m = 2 #dimension
s = 2

N = 100
x1 = np.linspace(-10, 10, N)
x2 = np.linspace(-10, 10, N)

X1, X2 = np.meshgrid(x1, x2)
X = np.c_[np.ravel(X1), np.ravel(X2)]

Y_plot = multi_hyperbolic_secant(x=X, s=s)
Y_plot = Y_plot.reshape(X1.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X1, X2, Y_plot, cmap='bwr', linewidth=0)
fig.colorbar(surf)
ax.set_title("Surface Plot")
fig.show()

# -

Y_plot.sum()


