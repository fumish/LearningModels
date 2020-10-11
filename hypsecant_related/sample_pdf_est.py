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

# # Sample program for pdf estimation
# + Model
#     $p(x|w) = \prod_{j=1}^M \frac{\sqrt{s}}{2\pi} \frac{1}{\cosh(\frac{\sqrt{s_j}}{2\pi} (x_j - b_j))}$
#     + In many cases, we have just data x, and want to know $b_j and s_j$ to summarize data form.
#     + Note: this sample case focuses on $M=2$. However, in many cases, $M$ is very large, thus summarize without visualization is necessary,
#     and we consider the summarization.
#     
# + In this notebook, the following two things are demonstrated:
#     1. Plot data following to the above distribution
#     1. Estimation of $b$ and $s$.

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import sys
sys.path.append("../lib")

# +
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hypsecant, norm
from mpl_toolkits.mplot3d import Axes3D

from HyperbolicSecantMixtureModelVB import HyperbolicSecantMixtureVB
from learning import GaussianMixtureModelVB

# +
### data settings
n = 100 ### The number of training data
M = 2 ### dimension of the data
data_seed = 20190930

### true b and s
true_b = np.array([0, 0])
true_s = np.array([2, 0.5])
# -

# # Plot data following to the above distribution
# + In this sample case,
#     $$(x_1, x_2) \sim \frac{\sqrt{2}}{2\pi}\frac{1}{\cosh(\frac{\sqrt{2}}{2}x_1)}\frac{\sqrt{0.5}}{2\pi}\frac{1}{\cosh(\frac{\sqrt{0.5}}{2}x_2)}$$
# + Here, the data is plotted.

train_X = np.array([hypsecant.rvs(loc = true_b[j], scale = 2/(np.sqrt(true_s[j])), size = n) for j in range(M)]).T

plt.scatter(train_X[:,0], train_X[:,1])
plt.show()

train_X.mean(axis = 0)

train_X.std(axis = 0)

# # Estimation of b and s
# + When we have the above data, the following command enables us to summarize its mean and scale.
# + Note: When you prepare another data whose data type is np.ndarray with $n \times M$, $n$ is the number of the data, $M$ is dimension of the data,  
#     the following calculation can be applied.

hsm_obj = HyperbolicSecantMixtureVB(K = 1, step = 1)
hsm_obj.fit(train_X)

# ## Estimated true_b = $(0,0)$ is this one.

hsm_obj.result_["mean"]

# ## Estimated true_s = $(2, 0.5)$ is this one.

hsm_obj.result_["precision"]

# ## Compare p(x|true_w) with p(x|est_w)

# +
N = 1000
x1 = np.linspace(-5, 5, N)
x2 = np.linspace(-5, 5, N)

X1, X2 = np.meshgrid(x1, x2)
X = np.c_[np.ravel(X1), np.ravel(X2)]

True_Y = hypsecant.pdf(x=X, loc=true_b, scale=2/np.sqrt(true_s)).prod(axis=1)
True_Y = True_Y.reshape(X1.shape)

Est_Y = hypsecant.pdf(x=X, loc=hsm_obj.result_["mean"], scale=2/np.sqrt(hsm_obj.result_["precision"])).prod(axis=1)
Est_Y = Est_Y.reshape(X1.shape)

# +
plt.subplots_adjust(wspace=0.6, hspace=0.6)
plt.subplots(figsize=(8,4))

ax = plt.subplot(1, 2, 1)
cont10 = ax.contourf(X1, X2, True_Y, levels = 10)
ax.set_title("True contour")
plt.colorbar(cont10)

ax = plt.subplot(1, 2, 2)
cont10 = ax.contourf(X1, X2, Est_Y, levels = 10)
ax.set_title("Estimated contour")
plt.colorbar(cont10)

plt.show()
# -

# ## For reference, compare estimation based on Gaussian model with this model.

gm_obj = GaussianMixtureModelVB(K = 1, step = 1)
gm_obj.fit(train_X)

# +
N = 1000
x1 = np.linspace(-5, 5, N)
x2 = np.linspace(-5, 5, N)

X1, X2 = np.meshgrid(x1, x2)
X = np.c_[np.ravel(X1), np.ravel(X2)]

True_Y = hypsecant.pdf(x=X, loc=true_b, scale=2/np.sqrt(true_s)).prod(axis=1)
True_Y = True_Y.reshape(X1.shape)

norm_Est_Y = norm.pdf(x=X, loc=gm_obj.result_["mean"], scale=1/np.sqrt(gm_obj.result_["precision"])).prod(axis=1)
norm_Est_Y = norm_Est_Y.reshape(X1.shape)

hypsecant_Est_Y = hypsecant.pdf(x=X, loc=hsm_obj.result_["mean"], scale=2/np.sqrt(hsm_obj.result_["precision"])).prod(axis=1)
hypsecant_Est_Y = hypsecant_Est_Y.reshape(X1.shape)

# +
plt.subplots_adjust(wspace=2, hspace=1)
plt.subplots(figsize=(8,4))

ax = plt.subplot(1, 3, 1)
cont10 = ax.contourf(X1, X2, True_Y, levels = 10)
ax.set_title("True contour")
plt.colorbar(cont10)

ax = plt.subplot(1, 3, 2)
cont10 = ax.contourf(X1, X2, norm_Est_Y, levels = 10)
ax.set_title("Estimated contour \n by Gaussian")
plt.colorbar(cont10)

ax = plt.subplot(1, 3, 3)
cont10 = ax.contourf(X1, X2, hypsecant_Est_Y, levels = 10)
ax.set_title("Estimated contour \n by Hyperbolic secant")
plt.colorbar(cont10)

plt.show()
# -

# + From above result, Estimation based on Gaussian overestimate the variance.


