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

# # クロネッカー積のテスト
# + ガウスカーネル行列のクロネッカー積で格子点のカーネル行列を表現できるかということ
# + クロネッカー積の固有値が掛けられた2つの行列の固有値の積であるかということ
# + クロネッカー積の固有ベクトルが2つの行列の固有ベクトルのクロネッカー積になるといったことを確認する

# %matplotlib inline

# +
import itertools

import numpy as np
from scipy.spatial import distance_matrix
from sklearn.utils.extmath import cartesian
# -

# # 2軸の格子点の定義域を設定する

n1 = 10
n2 = 15
X1_domain = (-10, 5)
X2_domain = (-8, -3)
X1_candidate = np.linspace(start = X1_domain[0], stop = X1_domain[1], num = n1).reshape(n1, 1)
X2_candidate = np.linspace(start = X2_domain[0], stop = X2_domain[1], num = n2).reshape(n2, 1)

# ## 格子点の各座標を求める
# + X1とX2の直積を求める

X_candidate = cartesian([X1_candidate.squeeze(), X2_candidate.squeeze()])

# ## カーネル行列を定義する
# + 今回は、ガウスカーネルを考える

gauss_kernel = lambda x,y: np.exp(-distance_matrix(x,y, p=2)**2/2)

quasi_gauss_kernel = lambda x,y: np.exp(-distance_matrix(x,y, p=2)/2)

# # クロネッカー積が正しく動くか検証する

K1 = gauss_kernel(X1_candidate, X1_candidate)
K2 = gauss_kernel(X2_candidate, X2_candidate)
K = gauss_kernel(X_candidate, X_candidate)
K_kron = np.kron(K1, K2)
np.allclose(K, K_kron)

K1 = quasi_gauss_kernel(X1_candidate, X1_candidate)
K2 = quasi_gauss_kernel(X2_candidate, X2_candidate)
K = quasi_gauss_kernel(X_candidate, X_candidate)
K_kron = np.kron(K1, K2)
np.allclose(K, K_kron)


