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

# # スパースガウス過程でロジスティック回帰をminibatchでやるための試作

# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.spatial import distance_matrix
from scipy.stats import wishart

# ## 学習設定

# +
data_seed = 20190908
n = 1000
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

# # 学習開始

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
est_g_xi = np.random.normal(size = n)

nu1 = np.linalg.solve(est_Sigma, est_u)
nu2 = -np.linalg.inv(est_Sigma)/2
# -

# ## メインの学習部分

a = train_sub_kernel @ inv_sub_sub_kernel
est_var_fu = used_kernel_diag(train_X) - ((train_sub_kernel @ inv_sub_sub_kernel) * train_sub_kernel).sum(axis = 1)


def logcosh(x):
    return np.abs(x) + np.log((1 + np.exp(-2*np.abs(x)))/2)


zero_condition = 0.000001
def ratio_tanh_x(x):
    """
    tanh(x/2)/xの計算
    原点がoverflowの原因になるが、原点の値は1/2
    """
    ratio_val = np.zeros(len(sq_h_xi))
    zero_ind = np.abs(x) < zero_condition
    ratio_val[zero_ind] = 1/2
    ratio_val[~zero_ind] = np.tanh(ratio_val[~zero_ind]/2)/(ratio_val[~zero_ind])
    return ratio_val


# +
# zero_condition = 0.000001
# def calc_dphidh(sq_h_xi):
#     v_xi = np.zeros(len(sq_h_xi))
#     zero_ind = np.abs(sq_h_xi) < zero_condition
#     v_xi[zero_ind] = -1/8
#     v_xi[~zero_ind] = -np.tanh(sq_h_xi[~zero_ind]/2)/(4*sq_h_xi[~zero_ind])
#     return v_xi
# -

est_h_xi = np.exp(est_g_xi)
est_sq_h_xi = np.sqrt(est_h_xi)
est_v_xi = calc_dphidh(est_sq_h_xi)


dFdnu1 = -(train_Y - 0.5) @ a + nu1
dFdnu2 = np.einsum("i, ij, ik -> jk", -est_v_xi, a, a) + inv_sub_sub_kernel/2 + nu2
nu1 += -step * dFdnu1
nu2 += -step * dFdnu2


dvdg = -((1/np.cosh(est_sq_h_xi/2)**2/4) - np.tanh(est_sq_h_xi/2)/est_sq_h_xi/2)/4
dFdg = -dvdg*(est_var_fu + ((a @ nu2) * a).sum(axis = 1) - est_h_xi)
est_g_xi += -step * dFdg


for ite in range(iteration):
    ## 補助変数のを計算用に変換する
    est_h_xi = np.exp(est_g_xi)
    est_sq_h_xi = np.sqrt(est_h_xi)
    est_v_xi = calc_dphidh(est_sq_h_xi)
    
    ## 事後分布の形状パラメータの計算
    dFdnu1 = -(train_Y - 0.5) @ a + nu1
    dFdnu2 = np.einsum("i, ij, ik -> jk", -est_v_xi, a, a) + inv_sub_sub_kernel/2 + nu2
    nu1 += -step * dFdnu1
    nu2 += -step * dFdnu2
    
    ## 補助変数の計算 -> もともとのものにlogを付けた変数の最適化を行っている
    dvdg = -((1/np.cosh(est_sq_h_xi/2)**2/4) - np.tanh(est_sq_h_xi/2)/est_sq_h_xi/2)/4
    dFdg = -dvdg*(est_var_fu + ((a @ nu2) * a).sum(axis = 1) - est_h_xi)
    est_g_xi += -step * dFdg
    
    ### エネルギーの計算
    est_h_xi = np.exp(est_g_xi)
    est_sq_h_xi = np.sqrt(est_h_xi)
    est_v_xi = calc_dphidh(est_sq_h_xi)    
    phi_h = (-logcosh(est_g_xi)-np.log(2)).sum()
    est_Sigma = nu2 - nu1.reshape((sub_n,1)) @ nu1.reshape((1,sub_n))
    F = 0
    F += -phi_h + est_v_xi @ est_h_xi
    F += -(train_Y - 0.5) @ a @ nu1 - est_v_xi @ (est_var_fu + ((a @ nu2)*a).sum(axis=1))
    F += (np.trace(inv_sub_sub_kernel @ est_Sigma) + nu1 @ inv_sub_sub_kernel @ nu1 - sub_n + np.linalg.slogdet(sub_sub_kernel)[1] -  np.linalg.slogdet(est_Sigma)[1])/2
    
    print(F, np.sqrt((dFdg**2).mean()), np.sqrt((dFdnu1**2).mean()), np.sqrt((dFdnu2**2).mean()))
    break
    pass

est_g_xi.min()


