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

# # クロネッカー積の固有値が2つの行列の固有値の積かどうか検証する

(eig1_val, eig1_vec) = np.linalg.eigh(K1)
(eig2_val, eig2_vec) = np.linalg.eigh(K2)
(eig_val, eig_vec) = np.linalg.eigh(K)
kron_eig_val = np.kron(eig1_val, eig2_val)
np.allclose(np.sort(kron_eig_val), eig_val)

# # クロネッカー積の固有ベクトルが2つの行列の固有ベクトルのクロネッカー積かどうか検証する
# + 固有ベクトル同士を完全に同じにするのは難しい(長さの自由度がある)ので以下が成り立つかどうかを考えた:
# $$
# K = P V P^T.
# $$
# $V$は固有値のクロネッカー積を対角成分に持つ行列(大きい順にソート), Pは固有ベクトルのクロネッカー積をVに対応するように並べ替えたもの

kron_eig_vec = np.kron(eig1_vec, eig2_vec)
kron_perm_ind = np.argsort(kron_eig_val)
np.allclose(K, kron_eig_vec[:,kron_perm_ind] @ np.diag(np.sort(kron_eig_val))  @ kron_eig_vec[:,kron_perm_ind].T)

# # 結論
# + 以下のことが言えた:
#
#
# 1. カーネル行列が独立(要素ごとの積)で表せる場合は、クロネッカー積でカーネル行列を表現できる
# 2. クロネッカー積でカーネル行列を表すことができると、固有値、固有ベクトルもクロネッカー積で与えることができる
# 3. 固有値、固有ベクトルが求まれば、格子状に与えられた補助変数のカーネル行列の逆行列や行列式が求まるのでうれしい


