# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Sparse Gaussian Process by Varitional Inference
# + $p(u,f|y,X)$ are approximated by $q(u,f|y,X)$, where
# $$ q(u,f|y,X) = \argmin_{\tilde{q}} \int \tilde{q}(u,f|y,X) \log \frac{\tilde{q}(u,f|y,X)}{p(y|f,X) p(u|f,X) p(u)} df du , \equiv \argmin_{\tilde{q}} \mathcal{F}(\tilde{q}) $$
# $$ \tilde{q}(u,f|y,X) = \tilde{q}(u) \tilde{q}(f|u) = \tilde{q}(u) \prod_{i=1}^n \tilde{q}(f_i|u)$$
# + When we minimize $q(f|u)$, then it is a same form with $p(f|u)$. Thus, let $q(u) = N(u|\hat{u}, \hat{\Sigma}_u)$, then eventually,
# \begin{equation} \mathcal{F} = \int \tilde{q}(u,f|y,X) \log \frac{1}{p(y|f,X)} df du + KL(q(u) || p(u)) \\
#  = \sum_{i=1}^n \Bigl\{ \frac{1}{2\sigma^2} (y_i - a_i^T \hat{u})^2 + \frac{1}{2\sigma^2}\hat{\sigma}_i^2 + \frac{1}{2\sigma^2} a_i^T \hat{\Sigma}_u a_i \Bigr\} + \frac{1}{2} \log \frac{|K_{M,M}|}{|\hat{\Sigma}_u|} -\frac{M}{2} + \frac{1}{2} tr(K_{M,M}^{-1} \hat{\Sigma}_u) + \frac{1}{2} \hat{u}^T \hat{\Sigma}_u \hat{u}, \end{equation}
#  where $a_i = K_{M,M}^{-1}K_{M,i}, \hat{\sigma}_i = K_{i,i} - K_{i,M} K_{M,M}^{-1} K_{M,i}$
# + Hence, to obtain the optimal $q(\cdot)$, we need to search an optimal $\hat{u}$ and $\hat{\Sigma}_u$ for $\mathcal{F}$:
#     + By using gradient for both $\hat{u}$ and $\hat{\Sigma}_u$, we can obtain a local minimum. The gradients are
#     \begin{equation} \frac{\partial}{\partial \hat{u}} \mathcal{F} = (K_{M,M}^{-1} + \frac{1}{\sigma^2} a^T a)\hat{u} - \frac{1}{\sigma^2}ay, \end{equation}
#     \begin{equation} \frac{\partial}{\partial \hat{\Sigma}_u} \mathcal{F} = -\frac{1}{2}\hat{\Sigma}_u + \frac{1}{2}K_{M,M}^{-1} + \frac{1}{2\sigma^2} a^T a, \end{equation}
#     where $a = (a_1, a_2, \ldots a_n)^T \in \mathrm{R}^{n \times M}$.

# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

# + {"toc-hr-collapsed": false, "cell_type": "markdown"}
# # Preparation for learning
# -

# ## Setting for data

# +
n = 100
M = 1
data_seed = 20190727
N = 100

x_domain = (-5, 5)
true_func = lambda x:(np.sin(4*x)/x).sum(axis = 1)
noise_y = 1
# -

# ## Generating data

# +
np.random.seed(data_seed)
train_X = np.random.uniform(low = x_domain[0], high = x_domain[1], size = (n,M))
train_Y = true_func(train_X) + np.random.normal(scale = noise_y, size = n)

test_X = np.random.uniform(low = x_domain[0], high = x_domain[1], size = (N,M))
test_Y = true_func(test_X) + np.random.normal(scale = noise_y, size = N)
# -

plt.scatter(train_X[:,0], train_Y)
plt.scatter(train_X[:,0], true_func(train_X))
plt.show()

# ## Setting for learning




