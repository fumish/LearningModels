# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# # HSMMの性能比較の数値実験

import sys
sys.path.append("../lib")

# +
import math

from IPython.core.display import display, Markdown, Latex
import numpy as np
from scipy.special import gammaln, psi
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t, cauchy, laplace, gumbel_r, gamma, skewnorm, pareto, multivariate_normal
from typing import Callable
from sklearn.mixture import BayesianGaussianMixture

from learning.MixtureModel import HyperbolicSecantMixtureVB
from learning.MixtureModel import GaussianMixtureModelVB
from util.elementary_function import GaussianMixtureModel, HyperbolicSecantMixtureModel
# -

# # 問題設定

# ## 真の分布の設定
# + データ生成分布は変更しますが、混合比, 中心, scaleは同じものを流用

true_ratio = np.array([0.33, 0.33, 0.34])
true_delta = 0
true_s = np.array([[0.1, 0.1], [0.5, 0.5], [1, 1]])
true_b = np.array([[2, 4], [-4, -2], [0, 0]])
true_param = dict()
true_param["ratio"] = true_ratio
true_param["mean"] = true_b
true_param["precision"] = true_s
true_param["scale"] = np.array([np.diag(1/np.sqrt(true_s[k,:])) for k in range(len(true_ratio))])
K0 = len(true_ratio)
M = true_b.shape[1]

# ## Learning setting:

# +
### 学習データの数
n = 400

### テストデータの数
N = 10000

### データの出方の個数
ndataset = 30

### 事前分布のハイパーパラメータ
pri_params = {
    "pri_alpha": 0.1,
    "pri_beta": 0.001,
    "pri_gamma": M+2,
    "pri_delta": 1
}

### データ生成の回数
data_seed_start = 201907
data_seeds = np.arange(start = data_seed_start, stop = data_seed_start + ndataset, step = 1)

### 学習モデルの初期値の乱数 -> データseedにoffsetを加えたものを使う
learning_num = 10
learning_seed_offset = 100

### 繰り返しアルゴリズムの繰り返し回数
learning_iteration = 1000

### 学習モデルのコンポーネントの数
K = np.array([3, 5])
# -

# # 性能評価
# + 1連の流れ
#     1. データ生成する
#     1. 学習を行う
#     1. 精度評価を行う
#     1. 1に戻って再度計算

# # コンポーネントの分布が正規分布の場合

# +
gerror_gmm = np.zeros(len(data_seeds))
cklerror_gmm = np.zeros(len(data_seeds))
c01error_gmm = np.zeros(len(data_seeds))

gerror_hsmm = np.zeros(len(data_seeds))
cklerror_hsmm = np.zeros(len(data_seeds))
c01error_hsmm = np.zeros(len(data_seeds))

for i, data_seed in enumerate(data_seeds):
    ### データを生成する
    (train_X, train_label, train_label_arg) = GaussianMixtureModel().rvs(true_ratio, true_b, true_s, size = n, data_seed = data_seed)
    (test_X, test_label, test_label_arg) = GaussianMixtureModel().rvs(true_ratio, true_b, true_s, size = N)
    
    gmm_diag_obj = GaussianMixtureModelVB(K = K[0],
                                     pri_alpha = pri_params["pri_alpha"], pri_beta = pri_params["pri_beta"], pri_gamma = pri_params["pri_gamma"], pri_delta = pri_params["pri_delta"], 
                                     iteration = 1000, restart_num=learning_num, learning_seed=data_seed + learning_seed_offset, method = "diag")
    gmm_diag_obj.fit(train_X)
    
    hsmm_obj = HyperbolicSecantMixtureVB(K = K[0],                                     
                                         pri_alpha = pri_params["pri_alpha"], pri_beta = pri_params["pri_beta"], pri_gamma = pri_params["pri_gamma"], pri_delta = pri_params["pri_delta"], 
                                         iteration = 1000, restart_num=learning_num, learning_seed=data_seed + learning_seed_offset)
    hsmm_obj.fit(train_X)
    true_latent_kl, _ = GaussianMixtureModel().score_latent_kl(train_X, true_ratio, true_b, true_s)
    cklerror_gmm[i] = (-true_latent_kl + gmm_diag_obj.score_latent_kl())/len(train_X)
    cklerror_hsmm[i] = (-true_latent_kl + hsmm_obj.score_latent_kl())/len(train_X)
    
    c01error_gmm[i] = gmm_diag_obj.score_clustering(train_label_arg)[0]/len(train_X)
    c01error_hsmm[i] = hsmm_obj.score_clustering(train_label_arg)[0]/len(train_X)
    
    true_empirical_entropy = -GaussianMixtureModel().logpdf(test_X, true_ratio, true_b, true_s)
    gerror_gmm[i] = (-true_empirical_entropy - gmm_diag_obj.predict_logproba(test_X))/len(test_X)
    gerror_hsmm[i] = (-true_empirical_entropy - hsmm_obj.predict_logproba(test_X))/len(test_X)
# -


print(f"""
gerror_gmm: {gerror_gmm.mean()},
gerror_hsmm: {gerror_hsmm.mean()},
cklerror_gmm: {cklerror_gmm.mean()},
cklerror_hsmm: {cklerror_hsmm.mean()},
c01error_gmm: {c01error_gmm.mean()},
c01error_hsmm: {c01error_hsmm.mean()}
""")

# # コンポーネントの分布が双曲線正割分布の場合

# +
gerror_gmm = np.zeros(len(data_seeds))
cklerror_gmm = np.zeros(len(data_seeds))
c01error_gmm = np.zeros(len(data_seeds))
norm_energy_gmm = np.zeros(len(data_seeds))

gerror_hsmm = np.zeros(len(data_seeds))
cklerror_hsmm = np.zeros(len(data_seeds))
c01error_hsmm = np.zeros(len(data_seeds))
norm_energy_hsmm = np.zeros(len(data_seeds))

for i, data_seed in enumerate(data_seeds):
    ### データを生成する
    (train_X, train_label, train_label_arg) = HyperbolicSecantMixtureModel().rvs(true_ratio, true_b, true_s, size = n, data_seed = data_seed)
    (test_X, test_label, test_label_arg) = HyperbolicSecantMixtureModel().rvs(true_ratio, true_b, true_s, size = N)
    
    gmm_diag_obj = GaussianMixtureModelVB(K = K[0],
                                     pri_alpha = pri_params["pri_alpha"], pri_beta = pri_params["pri_beta"], pri_gamma = pri_params["pri_gamma"], pri_delta = pri_params["pri_delta"], 
                                     iteration = 1000, method = "diag", 
                                     restart_num=learning_num, learning_seed=data_seed + learning_seed_offset)
    gmm_diag_obj.fit(train_X)
    
    hsmm_obj = HyperbolicSecantMixtureVB(K = K[0],                                     
                                         pri_alpha = pri_params["pri_alpha"], pri_beta = pri_params["pri_beta"], pri_gamma = pri_params["pri_gamma"], pri_delta = pri_params["pri_delta"], 
                                         iteration = 1000, restart_num=learning_num, learning_seed=data_seed + learning_seed_offset)
    hsmm_obj.fit(train_X)
    
    true_latent_kl, _ = HyperbolicSecantMixtureModel().score_latent_kl(train_X, true_ratio, true_b, true_s)
    cklerror_gmm[i] = (-true_latent_kl + gmm_diag_obj.score_latent_kl())/len(train_X)
    cklerror_hsmm[i] = (-true_latent_kl + hsmm_obj.score_latent_kl())/len(train_X)
    
    c01error_gmm[i] = gmm_diag_obj.score_clustering(train_label_arg)[0]/len(train_X)
    c01error_hsmm[i] = hsmm_obj.score_clustering(train_label_arg)[0]/len(train_X)
    
    true_empirical_entropy = -HyperbolicSecantMixtureModel().logpdf(test_X, true_ratio, true_b, true_s)
    gerror_gmm[i] = (-true_empirical_entropy - gmm_diag_obj.predict_logproba(test_X))/len(test_X)
    gerror_hsmm[i] = (-true_empirical_entropy - hsmm_obj.predict_logproba(test_X))/len(test_X)
    
# -

print(f"""
gerror_gmm: {gerror_gmm.mean()},
gerror_hsmm: {gerror_hsmm.mean()},
cklerror_gmm: {cklerror_gmm.mean()},
cklerror_hsmm: {cklerror_hsmm.mean()},
c01error_gmm: {c01error_gmm.mean()},
c01error_hsmm: {c01error_hsmm.mean()}
""")







# ### Label setting for each data
# + Remark: Label is fixed through each cluster distribution.

true_train_label = np.random.multinomial(n = 1, pvals = true_ratio, size = n)
true_train_label_arg = np.argmax(true_train_label, axis = 1)
true_test_label = np.random.multinomial(n = 1, pvals = true_ratio, size = test_data_num)
true_test_label_arg = np.argmax(true_test_label, axis = 1)



# +
np.random.seed(data_seed)
train_x = np.zeros((n, M))
for i in range(n):
    for j in range(M):
        train_x[i, j] = t.rvs(df = 3, loc=true_b[true_train_label_arg[i],j], scale=1/true_s[true_train_label_arg[i],j], size=1)

noise_data_num = math.ceil(n*true_delta)
if noise_data_num > 0:
    train_x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
    
np.random.seed(test_seed)
test_x = np.zeros((test_data_num, M))
for i in range(test_data_num):
    for j in range(M):
        test_x[i, j] = t.rvs(df = 1.5, loc=true_b[true_test_label_arg[i],j], scale=1/true_s[true_test_label_arg[i],j], size=1)

noise_data_num = math.ceil(test_data_num*true_delta)
if noise_data_num > 0:
    test_x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)

# +
from sklearn.mixture import BayesianGaussianMixture
printmd("### 1. Data distribution:")
plot_scatter_with_label(train_x, true_train_label_arg,  K0, noise_data_num)

printmd("### 2. Learning by sklearn.mixture.BayesianGaussianMixture:")
sklearn_gmm_obj = BayesianGaussianMixture(n_components=K,covariance_type="full", max_iter=1000, weight_concentration_prior_type="dirichlet_process", weight_concentration_prior=5)
sklearn_gmm_obj.fit(train_x)
print("mean plug-in parameters \n {0}".format({
    "est_ratio": sklearn_gmm_obj.weights_,
    "est_mean": sklearn_gmm_obj.means_,
    "est_precision": sklearn_gmm_obj.covariances_
}))
(correct_num_skgmm, perm_skgmm, label_arg_skgmm) = evaluate_correct_cluster_number(None, noise_data_num, true_train_label_arg, K, predict_label = sklearn_gmm_obj.predict(train_x))    

printmd("### 3. Learning by GMM:")
gmm_result = fit_lva_gmm(train_x, K, pri_alpha = pri_alpha, pri_beta = pri_beta, pri_gamma = pri_gamma, pri_delta = pri_delta, learning_seeds = learning_seeds)
print("mean plug-in parameters: \n {0}".format({
    "est_ratio": gmm_result["alpha"] / sum(gmm_result["alpha"]),
    "est_mean": gmm_result["mu"],
    "est_precision": gmm_result["gamma"] / gmm_result["delta"]
}))
(correct_num_gmm, perm_gmm, label_arg_gmm) = evaluate_correct_cluster_number(gmm_result, noise_data_num, true_train_label_arg, K)

printmd("### 4. Learning by HSMM:")
hsmm_result = fit_lva_hsmm(train_x, K, pri_alpha = pri_alpha, pri_beta = pri_beta, pri_gamma = pri_gamma, pri_delta = pri_delta, learning_seeds=learning_seeds)
print("mean plug-in parameters: \n {0}".format({
    "est_ratio": hsmm_result["alpha"] / sum(hsmm_result["alpha"]),
    "est_mean": hsmm_result["mu"],
    "est_precision": hsmm_result["gamma"] / hsmm_result["delta"]
}))
(correct_num_hsmm, perm_hsmm, label_arg_hsmm) = evaluate_correct_cluster_number(hsmm_result, noise_data_num, true_train_label_arg, K)


# -

ussebba = 50

# +
label_arg = true_train_label_arg


for i in range(K):
    plt.scatter(train_x[np.where((label_arg == i) & (np.abs(train_x[:,0]) < ussebba) & (np.abs(train_x[:,1]) < ussebba))[0],0],
                train_x[np.where((label_arg == i) & (np.abs(train_x[:,0]) < ussebba) & (np.abs(train_x[:,1]) < ussebba))[0],1])        
plt.show()

# +
label_arg = label_arg_skgmm 

for i in range(K):
    plt.scatter(train_x[np.where((label_arg == i) & (np.abs(train_x[:,0]) < ussebba) & (np.abs(train_x[:,1]) < ussebba))[0],0],
                train_x[np.where((label_arg == i) & (np.abs(train_x[:,0]) < ussebba) & (np.abs(train_x[:,1]) < ussebba))[0],1])        
plt.show()

# +
label_arg = label_arg_gmm

for i in range(K):
    plt.scatter(train_x[np.where((label_arg == i) & (np.abs(train_x[:,0]) < ussebba) & (np.abs(train_x[:,1]) < ussebba))[0],0],
                train_x[np.where((label_arg == i) & (np.abs(train_x[:,0]) < ussebba) & (np.abs(train_x[:,1]) < ussebba))[0],1])        
plt.show()

# +
label_arg = label_arg_hsmm

for i in range(K):
    plt.scatter(train_x[np.where((label_arg == i) & (np.abs(train_x[:,0]) < ussebba) & (np.abs(train_x[:,1]) < ussebba))[0],0],
                train_x[np.where((label_arg == i) & (np.abs(train_x[:,0]) < ussebba) & (np.abs(train_x[:,1]) < ussebba))[0],1])        
plt.show()
# -

plt.scatter(train_x[focus_ind1,0], train_x[focus_ind1,1])

# +
printmd("### 5. Correct number of labeling of GMM:")
printmd("+ {0}/{1}".format(correct_num_skgmm, len(label_arg_hsmm)))

printmd("### 5. Correct number of labeling of GMM:")
printmd("+ {0}/{1}".format(correct_num_gmm, len(label_arg_hsmm)))

printmd("### 6. Correct number of labeling of HSMM:")
printmd("+ {0}/{1}".format(correct_num_hsmm, len(label_arg_hsmm)))

printmd("### 7. Generalization error of GMM:")
printmd("+ {0}".format(evaluate_log_loss(gmm_result, true_param, noise_data_num, test_x, true_logpdf, pred_logpdf_gmm)))

printmd("### 8. Generalization error of HSMM:")
printmd("+ {0}".format(evaluate_log_loss(hsmm_result, true_param, noise_data_num, test_x, true_logpdf, pred_logpdf_hsmm)))

printmd("### 9. Data distribution labeled by GMM:")
plot_scatter_with_label(train_x, label_arg_gmm,  K, noise_data_num)

printmd("### 10. Data distribution labeled by HSMM:")
plot_scatter_with_label(train_x, label_arg_hsmm,  K, noise_data_num)
# -









# ## 1. Cluster distribution is Gaussian distribution

true_logpdf = lambda x, param: logpdf_mixture_dist(x, param, logpdf_multivariate_normal)

# +
np.random.seed(data_seed)
train_x = np.zeros((n, M))
for i in range(n):
    for j in range(M):
        train_x[i, j] = norm.rvs(loc=true_b[true_train_label_arg[i],j], scale=1/np.sqrt(true_s[true_train_label_arg[i],j]), size = 1)

noise_data_num = math.ceil(n*true_delta)
if noise_data_num > 0:
    train_x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
    
np.random.seed(test_seed)
test_x = np.zeros((test_data_num, M))
for i in range(test_data_num):
    for j in range(M):
        test_x[i, j] = norm.rvs(loc=true_b[true_test_label_arg[i],j], scale=1/np.sqrt(true_s[true_test_label_arg[i],j]), size = 1)

noise_data_num = math.ceil(test_data_num*true_delta)
if noise_data_num > 0:
    test_x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
# -

learning_and_labeling()

# ## 2. Cluster distribution is Hyperbolic secant distribution

true_logpdf = lambda x, param: logpdf_mixture_dist(x, param, logpdf_hypsecant)

# +
np.random.seed(data_seed)
train_x = np.zeros((n, M))
for i in range(n):
    for j in range(M):
        train_x[i, j] = random_hsm(n = 1, loc=true_b[true_train_label_arg[i],j], scale=true_s[true_train_label_arg[i],j])

noise_data_num = math.ceil(n*true_delta)
if noise_data_num > 0:
    train_x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
    
np.random.seed(test_seed)
test_x = np.zeros((test_data_num, M))
for i in range(test_data_num):
    for j in range(M):
        test_x[i, j] = random_hsm(n = 1, loc=true_b[true_test_label_arg[i],j], scale=true_s[true_test_label_arg[i],j])

noise_data_num = math.ceil(test_data_num*true_delta)
if noise_data_num > 0:
    test_x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
# -

learning_and_labeling()

# ## 3. Cluster distribution is Laplace distribution

logpdf_laplace = lambda x, mean, precision: laplace.logpdf(test_x, loc=mean, scale=1/np.diag(precision)).sum(axis=1)
true_logpdf = lambda x, param: logpdf_mixture_dist(x, param, logpdf_laplace)

# +
np.random.seed(data_seed)
train_x = np.zeros((n, M))
for i in range(n):
    for j in range(M):
        train_x[i, j] = laplace.rvs(loc=true_b[true_train_label_arg[i],j], scale=1/true_s[true_train_label_arg[i],j], size=1)

noise_data_num = math.ceil(n*true_delta)
if noise_data_num > 0:
    train_x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
    
np.random.seed(test_seed)
test_x = np.zeros((test_data_num, M))
for i in range(test_data_num):
    for j in range(M):
        test_x[i, j] = laplace.rvs(loc=true_b[true_test_label_arg[i],j], scale=1/true_s[true_test_label_arg[i],j], size=1)

noise_data_num = math.ceil(test_data_num*true_delta)
if noise_data_num > 0:
    test_x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
# -

learning_and_labeling()

# ## 4. Cluster distribution is Gumbel distribution

logpdf_gumbel = lambda x, mean, precision: gumbel_r.logpdf(test_x, loc=mean, scale=1/np.diag(precision)).sum(axis=1)
true_logpdf = lambda x, param: logpdf_mixture_dist(x, param, logpdf_gumbel)

# +
np.random.seed(data_seed)
train_x = np.zeros((n, M))
for i in range(n):
    for j in range(M):
        train_x[i, j] = gumbel_r.rvs(loc=true_b[true_train_label_arg[i],j], scale=1/true_s[true_train_label_arg[i],j], size=1)

noise_data_num = math.ceil(n*true_delta)
if noise_data_num > 0:
    train_x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
    
np.random.seed(test_seed)
test_x = np.zeros((test_data_num, M))
for i in range(test_data_num):
    for j in range(M):
        test_x[i, j] = gumbel_r.rvs(loc=true_b[true_test_label_arg[i],j], scale=1/true_s[true_test_label_arg[i],j], size=1)

noise_data_num = math.ceil(test_data_num*true_delta)
if noise_data_num > 0:
    test_x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
# -

learning_and_labeling()

# ## 5. Cluster distribution is student distribution

logpdf_t = lambda x, mean, precision: t.logpdf(test_x, df=1.5, loc=mean, scale=1/np.diag(precision)).sum(axis=1)
true_logpdf = lambda x, param: logpdf_mixture_dist(x, param, logpdf_t)

# +
np.random.seed(data_seed)
train_x = np.zeros((n, M))
for i in range(n):
    for j in range(M):
        train_x[i, j] = t.rvs(df = 2, loc=true_b[true_train_label_arg[i],j], scale=1/true_s[true_train_label_arg[i],j], size=1)

noise_data_num = math.ceil(n*true_delta)
if noise_data_num > 0:
    train_x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
    
np.random.seed(test_seed)
test_x = np.zeros((test_data_num, M))
for i in range(test_data_num):
    for j in range(M):
        test_x[i, j] = t.rvs(df = 1.5, loc=true_b[true_test_label_arg[i],j], scale=1/true_s[true_test_label_arg[i],j], size=1)

noise_data_num = math.ceil(test_data_num*true_delta)
if noise_data_num > 0:
    test_x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
# -

learning_and_labeling()

# ## 6. Cluster distribution is Cauchy distribution

logpdf_cauchy = lambda x, mean, precision: cauchy.logpdf(test_x, loc=mean, scale=1/np.diag(precision)).sum(axis=1)
true_logpdf = lambda x, param: logpdf_mixture_dist(x, param, logpdf_cauchy)

# +
np.random.seed(data_seed)
train_x = np.zeros((n, M))
for i in range(n):
    for j in range(M):
        train_x[i, j] = cauchy.rvs(loc=true_b[true_train_label_arg[i],j], scale=1/true_s[true_train_label_arg[i],j], size=1)

noise_data_num = math.ceil(n*true_delta)
if noise_data_num > 0:
    train_x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
    
np.random.seed(test_seed)
test_x = np.zeros((test_data_num, M))
for i in range(test_data_num):
    for j in range(M):
        test_x[i, j] = cauchy.rvs(loc=true_b[true_test_label_arg[i],j], scale=1/true_s[true_test_label_arg[i],j], size=1)

noise_data_num = math.ceil(test_data_num*true_delta)
if noise_data_num > 0:
    test_x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
# -

learning_and_labeling()

# ## 7. Cluster distribution is Gamma distribution
# + Remark: Actually support of gamma distribution is not whole real line, but scipy can generate data with loc on real value.

logpdf_gamma = lambda x, mean, precision: gamma.logpdf(test_x, a=1, loc=mean, scale=1/np.diag(precision)).sum(axis=1)
true_logpdf = lambda x, param: logpdf_mixture_dist(x, param, logpdf_gamma)

# +
np.random.seed(data_seed)
train_x = np.zeros((n, M))
for i in range(n):
    for j in range(M):
        train_x[i, j] = gamma.rvs(a = 1, loc=true_b[true_train_label_arg[i],j], scale=1/true_s[true_train_label_arg[i],j], size=1)

noise_data_num = math.ceil(n*true_delta)
if noise_data_num > 0:
    train_x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
    
np.random.seed(test_seed)
test_x = np.zeros((test_data_num, M))
for i in range(test_data_num):
    for j in range(M):
        test_x[i, j] = gamma.rvs(a = 1, loc=true_b[true_test_label_arg[i],j], scale=1/true_s[true_test_label_arg[i],j], size=1)

noise_data_num = math.ceil(test_data_num*true_delta)
if noise_data_num > 0:
    test_x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
# -

learning_and_labeling()

# ## 8. Cluster distribution is Skew Normal distribution

logpdf_skewnormal = lambda x, mean, precision: skewnorm.logpdf(test_x, a=2, loc=mean, scale=1/np.diag(precision)).sum(axis=1)
true_logpdf = lambda x, param: logpdf_mixture_dist(x, param, logpdf_skewnormal)

# +
np.random.seed(data_seed)
train_x = np.zeros((n, M))
for i in range(n):
    for j in range(M):
        train_x[i, j] = skewnorm.rvs(a = 2, loc=true_b[true_train_label_arg[i],j], scale=1/true_s[true_train_label_arg[i],j], size=1)

noise_data_num = math.ceil(n*true_delta)
if noise_data_num > 0:
    train_x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
    
np.random.seed(test_seed)
test_x = np.zeros((test_data_num, M))
for i in range(test_data_num):
    for j in range(M):
        test_x[i, j] = skewnorm.rvs(a = 2, loc=true_b[true_test_label_arg[i],j], scale=1/true_s[true_test_label_arg[i],j], size=1)

noise_data_num = math.ceil(test_data_num*true_delta)
if noise_data_num > 0:
    test_x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
# -

learning_and_labeling()

# ## 9. Cluster distribution is Parato distribution
# + Parato distribution has inifite variance if $shape \leq 2$.

logpdf_pareto = lambda x, mean, precision: pareto.logpdf(test_x, b=1.5, loc=mean, scale=1/np.diag(precision)).sum(axis=1)
true_logpdf = lambda x, param: logpdf_mixture_dist(x, param, logpdf_pareto)

# +
np.random.seed(data_seed)
train_x = np.zeros((n, M))
for i in range(n):
    for j in range(M):
        train_x[i, j] = pareto.rvs(b = 1.5, loc=true_b[true_train_label_arg[i],j], scale=1/true_s[true_train_label_arg[i],j], size=1)

noise_data_num = math.ceil(n*true_delta)
if noise_data_num > 0:
    train_x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
    
np.random.seed(test_seed)
test_x = np.zeros((test_data_num, M))
for i in range(test_data_num):
    for j in range(M):
        test_x[i, j] = pareto.rvs(b = 1.5, loc=true_b[true_test_label_arg[i],j], scale=1/true_s[true_test_label_arg[i],j], size=1)

noise_data_num = math.ceil(test_data_num*true_delta)
if noise_data_num > 0:
    test_x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
# -

learning_and_labeling()


