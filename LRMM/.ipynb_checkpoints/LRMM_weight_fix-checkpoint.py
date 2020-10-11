# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 混合ロジスティックで混合比を一定にした場合の推定
# + $p(y=1|x,w) = \sum_{k=1}^K \frac{1}{K} r(b_k^T x)$の推定を行う
#     + 特に$K$を大きくしたとき、どのようになるか調べる

# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

### problem setting
n = 1000
N = 1000
M = 2
X_domain = (-10, 10)
data_seed = 20191123
true_func = lambda x: (x * np.sin(x)).sum(axis = 1)

test = np.random.normal(size = (2,3))

# +
### data generation
np.random.seed(data_seed)
def data_generation(n:int):
    ret_X = np.zeros((n, M))
    base_X = np.random.uniform(low = X_domain[0], high = X_domain[1], size = n)
    for j in range(M):
        ret_X[:,j] = base_X**j
    ret_func = true_func(ret_X)
    ret_prob = expit(ret_func)
    ret_Y = np.random.binomial(n = 1, p = ret_prob, size = n)

    return (ret_X, ret_Y, ret_func, ret_prob)
    
(train_X, train_Y, train_func, train_prob) = data_generation(n)
(test_X, test_Y, test_func, test_prob) = data_generation(N)
# -

### learning setting
learning_seed = 20181123
iteration = 1000
K = 10
pri_beta = 0.0001

# +
### initial learning
np.random.seed(learning_seed)
est_u_xi = np.random.dirichlet(alpha = np.ones(K), size = n)
est_g_eta = np.abs(np.random.normal(size = (n,K)))
est_v_eta = -est_u_xi*np.tanh(np.sqrt(est_g_eta)/2)/(4*np.sqrt(est_g_eta))

in_out_matrix = np.repeat((train_Y - 0.5),M).reshape(n,M) * train_X
# -

### iteration
for ite in range(iteration):
    ### update param posterior
    est_beta = np.repeat(pri_beta * np.eye(M), K).reshape(M,M,K)
    for i in range(M):
        for j in range(M):
            est_beta[i,j,:] += train_X[:,i] * train_X[:,j] @ (-2*est_v_eta)
    est_inv_beta = np.array([np.linalg.inv(est_beta[:,:, k]) for k in range(K)]).transpose((1,2,0))
    est_b = np.zeros((M,K))
    for j in range(M):
        est_b[j,:] = (est_inv_beta[j,:,:] * (in_out_matrix.T @ est_u_xi)).sum(axis = 0)
    
    ### update g_eta
    est_g_eta = np.zeros((n,K))
    for i in range(M):
        for j in range(M):
            est_g_eta += np.repeat(train_X[:,i] * train_X[:,j], K).reshape((n,K)) * np.repeat(est_b[i,:] * est_b[j,:] + est_inv_beta[i,j,:], n).reshape((K,n)).T
            
    
    ### update h_xi
    break
    pass



debug_g_eta = np.zeros((n,K))
for k in range(K):
    debug_g_eta[:,k] = np.diag(train_X @ (est_b[:,0].reshape((M,1)) @ est_b[:,0].reshape((1,M)) + est_inv_beta[:,:,k]) @ train_X.T)
    pass

np.allclose(est_g_eta, debug_g_eta)

est_g_eta

debug_g_eta

est_inv_beta.transpose((1,2,0)).shape

est_inv_beta.shape
