# -*- coding: utf-8 -*-
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

# # スパースガウス過程でロジスティック回帰をminibatchでやるための試作

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import sys
sys.path.append("../lib")

# +
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.spatial import distance_matrix
from scipy.stats import wishart

from util import logcosh, ratio_tanh_x
# -

# ## 学習設定

# +
data_seed = 20190908
n = 10000
sub_n = 400
M = 1
domain_X = (-10, 10)

true_func = lambda x: (x * np.sin(x)).sum(axis = 1)
# -

# ## データ生成

# +
np.random.seed(data_seed)

train_X = np.random.uniform(low = domain_X[0], high = domain_X[1], size = (n, M))
train_func = true_func(train_X)
train_prob = expit(train_func)
train_Y = np.random.binomial(n = 1, p = train_prob, size = n)
# -

# # バッチ処理による学習開始

# ## 学習設定

# +
learning_seed = 20190909

iteration = 1000
theta1 = 1; theta2 = 1; theta3 = 0.0001
kronecker_delta = lambda x,y: np.exp(-distance_matrix(x,y)**2/(0.00001))
gauss_kernel = lambda x,y: theta1 * np.exp(-distance_matrix(x,y)**2/(theta2)) + theta3 * kronecker_delta(x,y)
gauss_kernel_diag = lambda x: theta1 * np.ones(len(x))  + theta3

used_kernel = gauss_kernel
used_kernel_diag = gauss_kernel_diag

step = 0.5
tol = 1e-7
# -

# ## 部分データの選択

np.random.seed(learning_seed)
sub_train_X = np.random.uniform(low = domain_X[0], high = domain_X[1], size = (sub_n, M))
# sub_ind = np.random.permutation(n)[:sub_n]
# sub_train_X = train_X[sub_ind,:]

# ## 学習前の事前計算

# +
## 学習で用いるカーネルの計算
sub_sub_kernel = used_kernel(sub_train_X, sub_train_X)
inv_sub_sub_kernel = np.linalg.inv(sub_sub_kernel)
train_sub_kernel = used_kernel(train_X, sub_train_X)

## 事後分布の形状パラメータの初期化
est_u = np.random.normal(size = sub_n)
est_Sigma = wishart.rvs(df = sub_n + 2, scale = np.eye(sub_n), size = 1)
# est_g_xi = np.random.normal(size = n)
est_h_xi = np.abs(np.random.normal(size = n))
nu1 = np.linalg.solve(est_Sigma, est_u)
nu2 = -np.linalg.inv(est_Sigma)/2
# -

# ## メインの学習部分

a = train_sub_kernel @ inv_sub_sub_kernel
est_var_fu = used_kernel_diag(train_X) - ((train_sub_kernel @ inv_sub_sub_kernel) * train_sub_kernel).sum(axis = 1)

current_F = np.inf
for ite in range(iteration):
    ## 補助変数のを計算用に変換する
    est_sq_h_xi = np.sqrt(est_h_xi)
    est_v_xi = -ratio_tanh_x(est_sq_h_xi/2)/8
    
    ## 事後分布の形状パラメータの計算
    b = np.sqrt(-est_v_xi.repeat(sub_n).reshape((n, sub_n)))*a
    dFdnu1 = -(train_Y - 0.5) @ a + nu1
    dFdnu2 = b.T @ b + inv_sub_sub_kernel/2 + nu2
    nu1 += -step * dFdnu1
    nu2 += -step * dFdnu2
    
    ## 補助変数の計算 -> もともとのものにlogを付けた変数の最適化を行っている
    est_Sigma = -np.linalg.inv(nu2)/2
    est_u = -np.linalg.solve(nu2, nu1)/2
    m2_u = est_Sigma + est_u.reshape((sub_n, 1)) @ est_u.reshape((1, sub_n))
    est_h_xi = est_var_fu + ((a @ m2_u) * a).sum(axis = 1)
    ### エネルギーの計算
    est_sq_h_xi = np.sqrt(est_h_xi)
    est_v_xi = -ratio_tanh_x(est_sq_h_xi/2)/8
    phi_h = (-logcosh(est_sq_h_xi/2)-np.log(2)).sum()
    F = 0
    F += -phi_h + est_v_xi @ est_h_xi
    F += -(train_Y - 0.5) @ a @ est_u - est_v_xi @ (est_var_fu + ((a @ m2_u)*a).sum(axis=1))
    F += (np.trace(inv_sub_sub_kernel @ est_Sigma) + est_u @ inv_sub_sub_kernel @ est_u - sub_n + np.linalg.slogdet(sub_sub_kernel)[1] -  np.linalg.slogdet(est_Sigma)[1])/2
    
    print(F, np.sqrt((dFdnu1**2).mean()), np.sqrt((dFdnu2**2).mean()))
    
    if np.abs(F - current_F) < tol:
        break
    current_F = F
    
    pass

plt.scatter(sub_train_X[:,0], true_func(sub_train_X))
plt.scatter(sub_train_X[:,0], est_u)
plt.show()

plt.scatter(sub_train_X[:,0], expit(true_func(sub_train_X)))
plt.scatter(sub_train_X[:,0], expit(est_u))
plt.show()



# # ミニバッチ処理による学習開始

# ## 学習設定

# +
learning_seed = 20190909

iteration = 1000
theta1 = 1; theta2 = 1; theta3 = 0.0001
kronecker_delta = lambda x,y: np.exp(-distance_matrix(x,y)**2/(0.00001))
gauss_kernel = lambda x,y: theta1 * np.exp(-distance_matrix(x,y)**2/(theta2)) + theta3 * kronecker_delta(x,y)
gauss_kernel_diag = lambda x: theta1 * np.ones(len(x))  + theta3

used_kernel = gauss_kernel
used_kernel_diag = gauss_kernel_diag

alpha = 0.5
tol = 1e-7

max_minibatch_size = 1000
epoch_num = 50
# -

# ## 部分データの選択

np.random.seed(learning_seed)
sub_train_X = np.random.uniform(low = domain_X[0], high = domain_X[1], size = (sub_n, M))
# sub_ind = np.random.permutation(n)[:sub_n]
# sub_train_X = train_X[sub_ind,:]

# ## 学習前の事前計算

# +
## 学習で用いるカーネルの計算
sub_sub_kernel = used_kernel(sub_train_X, sub_train_X)
inv_sub_sub_kernel = np.linalg.inv(sub_sub_kernel)

## 事後分布の形状パラメータの初期化
est_u = np.random.normal(size = sub_n)
est_Sigma = wishart.rvs(df = sub_n + 2, scale = np.eye(sub_n), size = 1)
# est_g_xi = np.random.normal(size = n)
est_h_xi = np.abs(np.random.normal(size = n))
nu1 = np.linalg.solve(est_Sigma, est_u)
nu2 = -np.linalg.inv(est_Sigma)/2
# -

# ## メインの学習部分

minibatch_num = int(np.ceil(n / max_minibatch_size))

total_ite = 0
for ite_epoch in range(epoch_num):
    current_perm = np.random.permutation(n)
    for minibatch_ite in range(minibatch_num):
        picked_ind = current_perm[(minibatch_ite*max_minibatch_size):((minibatch_ite+1)*max_minibatch_size)] if minibatch_ite < minibatch_num-1 else current_perm[(minibatch_ite*max_minibatch_size):]
        current_train_X = train_X[picked_ind,:]
        current_train_Y = train_Y[picked_ind]
        minibatch_size = len(picked_ind)
        
        train_sub_kernel = used_kernel(current_train_X, sub_train_X)
        a = train_sub_kernel @ inv_sub_sub_kernel
        est_var_fu = used_kernel_diag(current_train_X) - ((train_sub_kernel @ inv_sub_sub_kernel) * train_sub_kernel).sum(axis = 1)
        
        ## 補助変数の計算 -> もともとのものにlogを付けた変数の最適化を行っている
        est_Sigma = -np.linalg.inv(nu2)/2
        est_u = -np.linalg.solve(nu2, nu1)/2
        m2_u = est_Sigma + est_u.reshape((sub_n, 1)) @ est_u.reshape((1, sub_n))
        current_h_xi = est_var_fu + ((a @ m2_u) * a).sum(axis = 1)
        est_h_xi[picked_ind] = current_h_xi                
        
        ## 補助変数を計算用に変換する
        current_h_xi = est_h_xi[picked_ind]
        est_sq_h_xi = np.sqrt(current_h_xi)
        est_v_xi = -ratio_tanh_x(est_sq_h_xi/2)/8

        ## 事後分布の形状パラメータの計算
        b = np.sqrt(-est_v_xi.repeat(sub_n).reshape((minibatch_size, sub_n)))*a
        dFdnu1 = -(current_train_Y - 0.5) @ a + nu1
        dFdnu2 = b.T @ b + inv_sub_sub_kernel/2 + nu2
        nu1 += -alpha/(total_ite+1) * dFdnu1
        nu2 += -alpha/(total_ite+1) * dFdnu2
#         nu1 += -alpha * dFdnu1
#         nu2 += -alpha * dFdnu2
        
        ### エネルギーの計算
        est_sq_h_xi = np.sqrt(current_h_xi)
        est_v_xi = -ratio_tanh_x(est_sq_h_xi/2)/8
        phi_h = (-logcosh(est_sq_h_xi/2)-np.log(2)).sum()
        F = 0
        F += -phi_h + est_v_xi @ current_h_xi
        F += -(current_train_Y - 0.5) @ a @ est_u - est_v_xi @ (est_var_fu + ((a @ m2_u)*a).sum(axis=1))
        F += (np.trace(inv_sub_sub_kernel @ est_Sigma) + est_u @ inv_sub_sub_kernel @ est_u - sub_n + np.linalg.slogdet(sub_sub_kernel)[1] -  np.linalg.slogdet(est_Sigma)[1])/2

        pass
    print(F, np.sqrt((dFdnu1**2).mean()), np.sqrt((dFdnu2**2).mean()))
    pass

plt.scatter(sub_train_X[:,0], true_func(sub_train_X))
plt.scatter(sub_train_X[:,0], est_u)
plt.show()

plt.scatter(sub_train_X[:,0], expit(true_func(sub_train_X)))
plt.scatter(sub_train_X[:,0], expit(est_u))
plt.show()


