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

from IPython.core.display import display, Markdown, Latex
import math
import numpy as np
from scipy.special import gammaln, psi
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t, cauchy, laplace, gumbel_r, gamma, skewnorm, pareto, multivariate_normal
from typing import Callable

from sklearn.mixture import BayesianGaussianMixture

import sys
sys.path.append("../lib")
from learning.MixtureModel import HyperbolicSecantMixtureVB
from learning.MixtureModel import GaussianMixtureModelVB
from util.elementary_function import GaussianMixtureModel

# # 問題設定

# ## 真の分布の設定
# + データ生成分布は変更しますが、混合比, 中心, scaleは同じものを流用

true_ratio = np.array([0.33, 0.33, 0.34])
true_delta = 0
true_s = np.array([[1.5, 1.5], [0.5, 0.5], [1, 1]])
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
ndataset = 10

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

gerror_gmm.mean()

gerror_hsmm.mean()



gerror_gmm = np.zeros(len(data_seeds))
gerror_hsmm = np.zeros(len(data_seeds))
for i, data_seed in enumerate(data_seeds):
    ### データを生成する
    (train_X, train_label, train_label_arg) = GaussianMixtureModel().rvs(true_ratio, true_b, true_s, size = n, data_seed = data_seed)
    (test_X, test_label, test_label_arg) = GaussianMixtureModel().rvs(true_ratio, true_b, true_s, size = N)
    
    gmm_obj = GaussianMixtureModelVB(K = K[0],
                                     pri_alpha = pri_params["pri_alpha"], pri_beta = pri_params["pri_beta"], pri_gamma = pri_params["pri_gamma"], pri_delta = pri_params["pri_delta"], 
                                     iteration = 1000, restart_num=learning_num, learning_seed=data_seed + learning_seed_offset)
    gmm_obj.fit(train_X)
    
    hsmm_obj = HyperbolicSecantMixtureVB(K = K[0],                                     
                                         pri_alpha = pri_params["pri_alpha"], pri_beta = pri_params["pri_beta"], pri_gamma = pri_params["pri_gamma"], pri_delta = pri_params["pri_delta"], 
                                         iteration = 1000, restart_num=learning_num, learning_seed=data_seed + learning_seed_offset)
    hsmm_obj.fit(train_X)
    
    true_empirical_entropy = GaussianMixtureModel().logpdf(test_X, true_ratio, true_b, true_s)
    gerror_gmm[i] = (true_empirical_entropy - gmm_obj.predict_logproba(test_X))/len(test_X)
    gerror_hsmm[i] = (true_empirical_entropy - hsmm_obj.predict_logproba(test_X))/len(test_X)


# +


gmm_obj.fit(train_X)
# -

(GaussianMixtureModel().logpdf(test_X, true_ratio, true_b, true_s) - gmm_obj.predict_logproba(test_X))/N

# +
hsmm_obj = HyperbolicSecantMixtureVB(iteration = 1000, step=2)

hsmm_obj.fit(train_X)
# -

hsmm_obj._result

(GaussianMixtureModel().logpdf(test_X, true_ratio, true_b, true_s) - hsmm_obj.predict_logproba(test_X))/N

a = 0
if a == 0: print("aa")

true_train_label = np.random.multinomial(n = 1, pvals = true_ratio, size = n)
true_train_label_arg = np.argmax(true_train_label, axis = 1)
# true_test_label = np.random.multinomial(n = 1, pvals = true_ratio, size = test_data_num)
# true_test_label_arg = np.argmax(true_test_label, axis = 1)



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
# -













# # HSMMの数値実験
# 1. 比較対象:
#     + HSMM, GMM(diag), GMM(cov)
# 2. 評価方法:
#     + データの出方を変えて、平均汎化誤差, クラスタ一致の平均0-1損失, クラスタの平均KL,
#         + クラスタKL $KL(q(y^n|x^n) || p(y^n|x^n)) = \sum_{(y^n)^{(l)} \in Cluster} \log \frac{q((y^n)^{(l)},|(x^n)^{(l)})}{p((y^n)^{(l)},|(x^n)^{(l)})},$
#         + 学習に用いる乱数の種はそれぞれの推定で共有 -> 推定の初期値は同じで計算する
#         + 真の分布の変更に対して同じラベルを用いて推定と評価を行う
#         + 真の分布について
#             + データの事件は2, コンポーネントの数は3, 混合比は等分割, 平均は(4,4), (-4,-4), (0,0), scaleは(1.5, 1.5), (0.5, 0.5), (1,1)
#         + **その他の実験設定は後程記述**
# 3. 真の分布:混合分布で以下の分布の混合
#     1. 正規分布
#     2. 双曲正割分布
#     3. ラプラス分布
#     4. t-分布
#     5. 他にうまく行く分布
# 4. 実験の詳細設定
#     1. 学習データの個数: 400個
#     2 データの出方を変える回数: 10回
#     3. 1回の学習の繰り返し回数: 1000回
#     4. 1回の学習の初期値を変える回数: 10回
#     5. 学習モデルのコンポーネントの数: trueと同じ場合, trueより2大きい場合
#     6. ハイパーパラメータ$ \eta = (\alpha, \beta, \nu, \sigma)$: $\eta = (\alpha, \beta, \nu, \sigma) = (0.1, 0.001, M+2, I_M)$
#     7. 汎化誤差の近似に用いるテストデータの数: 10000個

# ### Suplementary material: Local Variational Approximation algorithm for HSMM
# + Model:
#     + $p(x|w) = \sum_{k=1}^K a_k \prod_{j = 1}^M \frac{\sqrt{s_{kj}}}{2\pi} \frac{1}{\cosh(\frac{s_{kj}}{2}(x_j - b_{kj}))}$
#     + $x, b_k \in \mathbb{R}^M, s_k \in \mathbb{R}_+^M$
# + Prior distribution:
#     + $\varphi(w) = Dir(a|\{ \alpha_k \}_{k=1}^K) \prod_{k=1}^K N(b_k|0, (s_k \beta_k)^{-1} ) Gam(s_k|\gamma_k, \delta_k)$
# + Algorithm
#    1. Initializing the following values:
#        + $g_{ikj}(\eta), v_{ikj}(\eta), h_{ik}(\xi), u_{ik}(\xi)$
#    2. Update the following values
#        + $\hat{\alpha}_k = \alpha_k + \sum_{i=1}^n u_{ik}(\xi)$
#        + $\hat{\beta}_{kj} = \beta_k + \sum_{i=1}^n -2v_{ikj}(\eta)$
#        + $\hat{m}_{kj} = \frac{1}{\hat{\beta}_k} \sum_{i=1}^n -2v_{ik}(\eta)x_i$
#        + $\hat{\gamma}_{kj} = \gamma_k + \frac{1}{2}\sum_{i=1}^n u_{ik}(\xi)$
#        + $\hat{\delta}_{kj} = \delta_k + \sum_{i=1}^n -v_{ikj}(\eta)x_{ij}^2 - \frac{\hat{\beta}_{kj}}{2}\hat{m}_{kj}^2$
#    3. Update the following values
#        + $g_{ikj}(\eta) = \frac{\hat{\gamma}_{kj}}{\hat{\delta}_{kj}} (x_{ij} - \hat{m}_{kj})^2 + \frac{1}{\hat{\beta}_{kj}}$
#        + $v_{ikj}(\eta) = -u_{ik}(\xi)\frac{ \tanh(\sqrt{g_{ikj}(\eta)}/2) }{4\sqrt{g_{ikj}(\eta)}}$
#    4. Update the following values
#        + $h_{ik}(\xi) = \psi(\hat{\alpha}_k) - \psi(\sum_{l=1}^K \hat{\alpha}_l) + \frac{1}{2} \sum_{j=1}^M (\psi(\hat{\gamma}_{kj}) - \log(\hat{\delta}_{kj})) - \sum_{j=1}^M \log(\cosh(\sqrt{g_{ikj}(\eta)}/2)) $
#        + $u_{ik}(\xi) = \frac{ \exp(h_{ik}(\xi)) }{ \sum_{l=1}^K \exp(h_{il}(\xi)) }$
#        + where,$\psi(x) = \frac{d}{dx}\log \Gamma(x)$
#    5. Return back to 2.
#    
# + Evaluation function $\overline{F}_{\xi, \eta}(x^n)$:
#     + $\overline{F}_{\xi, \eta}(x^n) = - \phi(h(\xi)) - \psi(g(\eta)) + u(\xi) \cdot h(\xi) + v(\eta) \cdot g(\eta) $  
#         $+ nM \log 2 \pi + \log \Gamma(\sum_{l = 1}^K \hat{\alpha}_l) - \log \Gamma({\sum_{l = 1}^K\alpha}_l) + \sum_{k=1}^K \log \frac{\Gamma(\alpha_k)}{\Gamma(\hat{\alpha}_k)}$  
#         $+ \sum_{k=1}^K \sum_{j=1}^M \bigl\{ \frac{1}{2} \log \frac{\hat{\beta}_{kj}}{\beta_{kj}} + \hat{\gamma}_{kj} \log \hat{\delta}_{kj} - \gamma_{kj} \log \delta_{kj} - \log \Gamma(\hat{\gamma}_{kj}) + \log \Gamma(\gamma_{kj}) \bigr\}$

# ## Used funtions

def printmd(x):
    display(Markdown(x))


def random_hsm(n, loc = 0, scale = 1):
    """
    Generate data following hyperbolic secant distribution.
    Let $Y \sim standard_cauchy(x)$,  
    random variable $X = \frac{2}{\sqrt{s}}\sinh^{-1}(Y) + b$ follows to  
    $X \sim p(x) = \frac{\sqrt{s}}{2\pi}\frac{1}{\cosh(s(x-b)/2)}$.
    """
    Y = np.random.standard_cauchy(size=n)
    X = 2/np.sqrt(scale)*np.arcsinh(Y) + loc    
    return X


# +
def logpdf_hypsecant(x:np.ndarray, mean:np.ndarray, precision:np.ndarray):
    """
    Calculate \log p(x|w) = \sum_{j=1}^M \log(\frac{\sqrt{s_j}}{2\pi} 1/cosh(\sqrt{s_j}/2(x_j - b_j)))
    Input:
     + x: n*M
     + mean: M
     + precision :M*M
    Output:
     + n*M
    """
    (n, M) = x.shape
    expand_precision = np.repeat(np.diag(precision), n).reshape(M,n).T
    y = np.sqrt(expand_precision)*(x - np.repeat(mean, n).reshape(M,n).T)/2
    return(np.log(expand_precision)/2 - np.log(2*np.pi) - logcosh(y)).sum(axis = 1)

def logpdf_multivariate_normal(x:np.ndarray, mean:np.ndarray, cov:np.ndarray):
    """
    Calculate \log p(x|w) = \sum_{j=1}^M \log(\frac{\sqrt{s_j}}{2\pi} 1/cosh(\sqrt{s_j}/2(x_j - b_j)))
    Input:
     + x: n*M
     + mean: M
     + cov :M * M
    Output:
     + n*M
    """
    return(multivariate_normal.logpdf(x, mean = mean, cov = cov))


# +
def logcosh(x:np.ndarray):
    """
    対数双曲線関数の計算
    + 普通にライブラリを呼ぶと、オーバーフローするのでその対策
    """
    return np.abs(x) + np.log((1 + np.exp(-2 * np.abs(x)))/2)
    
#     ret_val = -x + np.log((1 + np.exp(2*x))/2)
#     (row, col) = np.where(x > 0)
#     ret_val[row, col] = x[row, col] + np.log((1 + np.exp(-2*x[row,col]))/2)
#     return ret_val

def logpdf_mixture_dist(x:np.ndarray, param:dict, component_log_dist:Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]):
    """
    対数尤度の計算
    確率分布が混合分布の時を想定:
    $\log p(x|w) = \sum_{i = 1}^n \log p(x_i|w) = \sum_{i = 1}^n \log \exp(L_{ik}(w)) =  \sum_{i = 1}^n \{\hat{L}_{i} + \log \exp(L_{ik}(w) - \hat{L}(i)) \}$,
    where L_{ik} = \log a_k + \log p(x_i|w, y_{ik} = 1)\hat{L}(i) = \max_{k} L_{ik}, p(x_i|w, y_{ik}=1):i番目のサンプルのk番目のクラスタの確率分布    
    
    + 入力:
        1. x:入力データ(n*M)
        2. param: 確率分布のパラメータ(ratio: 混合比, mean: 各クラスタの平均値 K*M, scale: 各クラスタのscale(正規分布における標準偏差) K*M)
        3. component_log_dist: 各クラスタの対数確率密度の値 logp(x|w,y)    
    """
    n = x.shape[0]
    K = len(param["ratio"])
    loglik = np.zeros((n,K))
    for k in range(K):
        if param["scale"].ndim == 2:
            loglik[:,k] = np.log(param["ratio"][k]) + component_log_dist(test_x, param["mean"][k,:],  param["scale"][k,:])
        elif param["scale"].ndim == 3:
            loglik[:,k] = np.log(param["ratio"][k]) + component_log_dist(test_x, param["mean"][k,:],  param["scale"][k,:,:])
        else:
            raise ValueError("Error precision, dimension of precision must be 2 or 3!")
    max_loglik = loglik.max(axis = 1)
    norm_loglik = loglik - np.repeat(max_loglik,K).reshape(n,K)
    return (np.log(np.exp(norm_loglik).sum(axis = 1)) + max_loglik)


# -

from sklearn.mixture import BayesianGaussianMixture
def fit_gmm_sklearn(train_X:np.ndarray, K:int,
                    pri_alpha = 0.1, pri_beta = 0.001, pri_gamma = 2, pri_delta = 2,
                    iteration = 1000, restart_num:int = 5, learning_seeds:list = None):
    """
    Estimating GMM by sklearn library.
    
    + Input:
        + train_X: input data
        + pri_alpha: hyperparameter for prior distribution of symmetric Dirichlet distribution.
        + pri_beta: hyperparameter for prior distribution of Normal distribution for inverse variance.
        + pri_gamma: hyperparameter for prior distribution of Gamma distribution for shape parameter.
        + pri_delta: hyperparameter for prior distribution of Gamma distribution for rate parameter.
        + iteration: Number of iteration.
        + restart_num: Number of restart of inital values.
        + learning_seeds: Seeds for initial values.
        
    + Output:
        + Dictionary of the best estimated result:
            1. alpha: parameter for posterior distribution of Dirichlet distribution.
            2. mu: parameter for posterior distribution of Normal distribution for mean parameter.
            3. beta: parameter for posterior distribution of Normal distribution for inverse variance parameter.
            4. gamma: parameter for posterior distribution of Gamma distribution for shape parameter.
            5. delta: parameter for posterior distribution of Gamma distribution for rate parameter.
            6. h_xi: Value of E_w[log p(x_i, z_i = k|w)], where z_i is latent variable. This parameters form posterior latent distribution.
            7. u_xi: Value of p(z_i = k). This parameters represent posterior probability of latent variable
            8. energy: Value of the best evaluation function.
            9. seed: Value of the best learning seed.
    """
    M = train_X.shape[1]
    sklearn_gmm_obj = BayesianGaussianMixture(n_components=K,
                                              covariance_type="full",
                                              max_iter=iteration,
                                              mean_precision_prior = pri_beta,
                                              degrees_of_freedom_prior = M*pri_gamma,
                                              covariance_prior = pri_delta * np.eye(M),
                                              weight_concentration_prior_type="dirichlet_distribution",
                                              weight_concentration_prior=pri_alpha,
                                              n_init=5)
    sklearn_gmm_obj.fit(train_X)
        
    result = dict()
    result["ratio"] = sklearn_gmm_obj.weights_
    result["mean"] = sklearn_gmm_obj.means_
    result["precision"] = sklearn_gmm_obj.precisions_
    result["scale"] = sklearn_gmm_obj.covariances_
    result["u_xi"] = sklearn_gmm_obj.predict_proba(train_X)
    
    return result


def fit_lva_gmm(train_X:np.ndarray, K:int,
                 pri_alpha = 0.1, pri_beta = 0.001, pri_gamma = 2, pri_delta = 2,
                 iteration = 1000, restart_num:int = 5, learning_seeds:list = None):
    """
    LVA for GMM.
    This is same with Variational Bayes inference for GMM.
    Since the algorithm fails to local minima, the best estimator are chosen in several initial values.
    
    + Input:
        + train_X: input data
        + pri_alpha: hyperparameter for prior distribution of symmetric Dirichlet distribution.
        + pri_beta: hyperparameter for prior distribution of Normal distribution for inverse variance.
        + pri_gamma: hyperparameter for prior distribution of Gamma distribution for shape parameter.
        + pri_delta: hyperparameter for prior distribution of Gamma distribution for rate parameter.
        + iteration: Number of iteration.
        + restart_num: Number of restart of inital values.
        + learning_seeds: Seeds for initial values.
        
    + Output:
        + Dictionary of the best estimated result:
            1. alpha: parameter for posterior distribution of Dirichlet distribution.
            2. mu: parameter for posterior distribution of Normal distribution for mean parameter.
            3. beta: parameter for posterior distribution of Normal distribution for inverse variance parameter.
            4. gamma: parameter for posterior distribution of Gamma distribution for shape parameter.
            5. delta: parameter for posterior distribution of Gamma distribution for rate parameter.
            6. h_xi: Value of E_w[log p(x_i, z_i = k|w)], where z_i is latent variable. This parameters form posterior latent distribution.
            7. u_xi: Value of p(z_i = k). This parameters represent posterior probability of latent variable
            8. energy: Value of the best evaluation function.
            9. seed: Value of the best learning seed.
    """
    
    (n, M) = train_X.shape
    ### Setting for static variable in the algorithm.
    expand_x = np.repeat(train_X, K).reshape(n, M, K).transpose((0, 2, 1)) ### n * K * M data with the same matrix among 2nd dimension

    min_energy = np.inf
    result = dict()
    
    for restart in range(restart_num):
        ### Set learning seed if learning_seeds is specified.
        if learning_seeds is not None and len(learning_seeds) >= restart:
            np.random.seed(learning_seeds[restart])

        energy = np.zeros(iteration)
        ### Setting for initial value
        est_u_xi = np.random.dirichlet(alpha = np.ones(K), size=n)
    
        ### Start learning.
        for ite in range(iteration):
            ### Update posterior distribution of parameter.
            est_alpha = pri_alpha + est_u_xi.sum(axis = 0)
            est_beta = np.repeat(pri_beta + est_u_xi.sum(axis = 0), M).reshape(K,M)
            est_m = est_u_xi.T @ train_X / est_beta
            est_gamma = np.repeat(pri_gamma + est_u_xi.sum(axis = 0)/2, M).reshape(K,M)
            est_delta = pri_delta + est_u_xi.T @ (train_X**2) /2 - est_beta / 2 * est_m**2
            
            ### Update posterior distribution of latent variable
            est_g_eta = np.repeat(est_gamma / est_delta, n).reshape(K,M,n).transpose((2,0,1)) * (expand_x - np.repeat(est_m,n).reshape(K,M,n).transpose((2,0,1)))**2 + 1/np.repeat(est_beta, n).reshape(K,M,n).transpose((2,0,1))
            est_h_xi = -M/2*np.log(2*np.pi) + np.repeat(psi(est_alpha) - psi(est_alpha.sum()) + (psi(est_gamma) - np.log(est_delta)).sum(axis = 1)/2, n).reshape(K,n).T - est_g_eta.sum(axis = 2)/2
            max_h_xi = est_h_xi.max(axis = 1)
            norm_h_xi = est_h_xi - np.repeat(max_h_xi,K).reshape(n,K)
            est_u_xi = np.exp(norm_h_xi) / np.repeat(np.exp(norm_h_xi).sum(axis = 1), K).reshape(n,K)

            ### Calculate evaluation function
            energy[ite] = - (np.log(np.exp(norm_h_xi).sum(axis = 1)) + max_h_xi).sum() + (est_u_xi * est_h_xi).sum()
            energy[ite] += gammaln(est_alpha.sum()) - gammaln(K*pri_alpha) + (-gammaln(est_alpha) + gammaln(pri_alpha)).sum()
            energy[ite] += (np.log(est_beta/pri_beta)/2 + est_gamma * np.log(est_delta) - pri_gamma * np.log(pri_delta) - gammaln(est_gamma) + gammaln(pri_gamma)).sum()
        
        print(energy[-1])        
        if energy[-1] < min_energy:
            min_energy = energy[-1]
            result["ratio"] = est_alpha / est_alpha.sum()
            result["mean"] = est_m
            result["precision"] = est_gamma / est_delta
            result["scale"] = np.array([np.diag(est_delta[k,:] / est_gamma[k,:]) for k in range(K)])
            result["alpha"] = est_alpha
            result["mu"] = est_m
            result["beta"] = est_beta
            result["gamma"] = est_gamma
            result["delta"] = est_delta
            result["h_xi"] = est_h_xi
            result["u_xi"] = est_u_xi
            result["energy"] = energy
            if learning_seeds is not None and len(learning_seeds) >= restart:
                result["seed"] = learning_seeds[restart]
    return result


def fit_lva_hsmm(train_X:np.ndarray, K:int,
                 pri_alpha = 0.1, pri_beta = 0.001, pri_gamma = 2, pri_delta = 2,
                 iteration = 1000, restart_num:int = 5, learning_seeds:list = None):
    """
    LVA for HSMM.
    The algorithm is described in the above cell.
    Since the algorithm fails to local minima, the best estimator are chosen in several initial values.
    
    + Input:
        + train_X: input data
        + pri_alpha: hyperparameter for prior distribution of symmetric Dirichlet distribution.
        + pri_beta: hyperparameter for prior distribution of Normal distribution for inverse variance.
        + pri_gamma: hyperparameter for prior distribution of Gamma distribution for shape parameter.
        + pri_delta: hyperparameter for prior distribution of Gamma distribution for rate parameter.
        + iteration: Number of iteration.
        + restart_num: Number of restart of inital values.
        + learning_seeds: Seeds for initial values.
        
    + Output:
        + Dictionary of the best estimated result:
            1. alpha: parameter for posterior distribution of Dirichlet distribution.
            2. mu: parameter for posterior distribution of Normal distribution for mean parameter.
            3. beta: parameter for posterior distribution of Normal distribution for inverse variance parameter.
            4. gamma: parameter for posterior distribution of Gamma distribution for shape parameter.
            5. delta: parameter for posterior distribution of Gamma distribution for rate parameter.
            6. h_xi: Value of E_w[log p(x_i, z_i = k|w)], where z_i is latent variable. This parameters form posterior latent distribution.
            7. u_xi: Value of p(z_i = k). This parameters represent posterior probability of latent variable.
            8. g_eta: Value of auxiliary variable, which represents g(\eta) in the algorithm.
            9. v_eta: Value of auxiliary variable, which represents v(\eta) in the algorithm.
            10. energy: Value of the best evaluation function.
            11. seed: Value of the best learning seed.
    """    
    
    (n, M) = train_X.shape
    ### Setting for static variable in the algorithm.
    expand_x = np.repeat(train_X, K).reshape(n, M, K).transpose((0, 2, 1)) ### n * K * M data with the same matrix among 2nd dimension

    min_energy = np.inf
    result = dict()
    
    for restart in range(restart_num):
        ### Set learning seed if learning_seeds is specified.
        if learning_seeds is not None and len(learning_seeds) >= restart:
            np.random.seed(learning_seeds[restart])

        energy = np.zeros(iteration)
        ### Setting for initial value
        est_u_xi = np.random.dirichlet(alpha = np.ones(K), size=n)

        est_g_eta = np.abs(np.random.normal(size=(n,K,M)))
        est_v_eta = - np.repeat(est_u_xi, M).reshape(n, K, M) * np.tanh(np.sqrt(est_g_eta)/2)/(4*np.sqrt(est_g_eta))

        ### Start learning.
        for ite in range(iteration):
            ### Update posterior distribution of parameter
            est_alpha = pri_alpha + est_u_xi.sum(axis = 0)
            est_beta = pri_beta + (-2*est_v_eta.sum(axis = 0))
            est_m = -2 * (expand_x * est_v_eta).sum(axis = 0) / est_beta
            est_gamma = np.repeat(pri_gamma + est_u_xi.sum(axis = 0)/2, M).reshape(K,M)
            est_delta = pri_delta - (expand_x**2 * est_v_eta).sum(axis = 0) - est_beta / 2 * est_m**2
            
            ### Update auxiliary variables
            est_g_eta = np.repeat(est_gamma / est_delta, n).reshape(K,M,n).transpose((2,0,1)) * (expand_x - np.repeat(est_m,n).reshape(K,M,n).transpose((2,0,1)))**2 + 1/np.repeat(est_beta, n).reshape(K,M,n).transpose((2,0,1))
            est_v_eta = - np.repeat(est_u_xi, M).reshape(n, K, M) * np.tanh(np.sqrt(est_g_eta)/2)/(4*np.sqrt(est_g_eta))

            ### Update posterior distribution of latent variable
            sqrt_g_eta = np.sqrt(est_g_eta)
            est_h_xi = np.repeat(psi(est_alpha) - psi(est_alpha.sum()) + (psi(est_gamma) - np.log(est_delta)).sum(axis = 1)/2, n).reshape(K,n).T - (sqrt_g_eta/2 +  np.log( (1 + np.exp(-2*sqrt_g_eta/2 ))/2)).sum(axis = 2)
            max_h_xi = est_h_xi.max(axis = 1)
            norm_h_xi = est_h_xi - np.repeat(max_h_xi,K).reshape(n,K)
            est_u_xi = np.exp(norm_h_xi) / np.repeat(np.exp(norm_h_xi).sum(axis = 1), K).reshape(n,K)

            ### Calculate evaluation function
            energy[ite] = (np.repeat(est_u_xi, M).reshape(n, K, M) * (sqrt_g_eta/2 +  np.log( (1 + np.exp(-2*sqrt_g_eta/2 ))/2)) ).sum() - (np.log(np.exp(norm_h_xi).sum(axis = 1)) + max_h_xi).sum() + (est_u_xi * est_h_xi).sum() + (est_v_eta * est_g_eta).sum()
            energy[ite] += gammaln(est_alpha.sum()) - gammaln(K*pri_alpha) + (-gammaln(est_alpha) + gammaln(pri_alpha)).sum()
            energy[ite] += (np.log(est_beta/pri_beta)/2 + est_gamma * np.log(est_delta) - pri_gamma * np.log(pri_delta) - gammaln(est_gamma) + gammaln(pri_gamma)).sum()
        
        print(energy[-1])        
        if energy[-1] < min_energy:
            min_energy = energy[-1]
            result["ratio"] = est_alpha / est_alpha.sum()
            result["mean"] = est_m
            result["precision"] = est_gamma / est_delta
            result["scale"] = np.array([np.diag(est_delta[k,:] / est_gamma[k,:]) for k in range(K)])
            result["alpha"] = est_alpha
            result["beta"] = est_beta
            result["mu"] = est_m
            result["gamma"] = est_gamma
            result["delta"] = est_delta
            result["h_xi"] = est_h_xi
            result["u_xi"] = est_u_xi
            result["g_eta"] = est_g_eta
            result["v_eta"] = est_v_eta            
            result["energy"] = energy
            if learning_seeds is not None and len(learning_seeds) >= restart:
                result["seed"] = learning_seeds[restart]
    return result


import itertools
def evaluate_correct_cluster_number(result:dict, noise_data_num:int, true_label_arg, K:int, predict_label = None) -> (float, np.ndarray, np.ndarray):
    """
    0-1損失によるクラスタ分布の評価
    1つのデータセットに対する評価で、ラベルの一致数の最大値の平均値を計算する
    + 入力:
        1. result: dict, u_xiをキーに持つ必要がある
        2. noise_data_num: 外れ値データの数
        3. true_label_arg: 真のラベル番号
        4. predict_label: 予測されたラベル番号
    + 出力:
        1. max_correct_num: ラベルの最大一致数
        2. max_perm: 最大一致数を与える置換
        3. max_est_label_arg: 最大の一致数を与えるラベルの番号
    
    """
    
    if predict_label is not None:
        est_label_arg = predict_label
    else:        
        est_label_prob = result["u_xi"]
        target_label_arg = true_label_arg
        est_label_arg = np.argmax(est_label_prob, axis = 1)

    if noise_data_num > 0:
        target_label_arg = true_label_arg[:-noise_data_num]
        est_label_arg = est_label_arg[:-noise_data_num]
    else:
        target_label_arg = true_label_arg
        
    max_correct_num = 0
    for perm in list(itertools.permutations(range(K), K)):
        permed_est_label_arg = est_label_arg.copy()
        for i in range(len(perm)):
            permed_est_label_arg[est_label_arg == i] = perm[i]
        correct_num = (permed_est_label_arg == target_label_arg).sum()
        if correct_num > max_correct_num:
            max_correct_num = correct_num
            max_perm = perm
            max_est_label_arg = permed_est_label_arg
    return (max_correct_num, max_perm, max_est_label_arg)


def evaluate_log_loss(fit_result:dict, true_param:dict, noise_data_num:int, test_x:np.ndarray,
                      true_logpdf:Callable[[np.ndarray, dict],np.ndarray], pred_logpdf:Callable[[np.ndarray, dict], np.ndarray]):
    """
    汎化誤差の計算を行う
    HSMMについては予測分布の計算が面倒なので、とりあえず平均プラグインでそれぞれの分布の性能比較を行う。
    + 汎化誤差G(x^n,y^n):
        + G(x^n) = \int q(x) \log q(x) / p(x|x^n) dx \arrox \sum_{l = 1}^N \log q(x^l) / p(x^l|x^n), x_l ~ q(x_l),
        + where p(x|x^n) = \int p(x|w) p(w|x^n) dw \approx p(x|E[w])
    + 入力
        1. fit_result: dict, ratio, mean, scaleをキーに持つ必要がある
        2. true_param: dict: ratio, mean, scaleをキーに持つ必要がある
        3. noise_data_num: データに含まれる外れ値の数
        4. test_x: x_jのデータ N*M行列
        5. true_logpdf: 真の分布の対数尤度関数
        6. pred_logpdf: 予測分布の対数尤度関数
    + 出力 汎化誤差の値
    """
    
    if noise_data_num > 0:
        return (true_logpdf(test_x, true_param) - pred_logpdf(test_x, fit_result))[:-noise_data_num].mean()
    else:
        return (true_logpdf(test_x, true_param) - pred_logpdf(test_x, fit_result)).mean()


from sklearn.mixture import BayesianGaussianMixture
def learning_and_labeling():
    """
    テストに用いたメイン関数
    やっていること:
    1. データ生成分布のプロット
    2. それぞれの分布による推定
    3. 0-1損失をそれぞれ計算
    4. 汎化誤差の計算
    5. それぞれの推定によって得られたラベルを用いて色付けした結果
    """
    
    printmd("### 1. Data distribution:")
    plot_scatter_with_label(train_x, true_train_label_arg,  K0, noise_data_num)
    
    printmd("### 2. Learning by sklearn.mixture.BayesianGaussianMixture:")
    sklearn_gmm_result = fit_gmm_sklearn(train_x, K, pri_alpha = pri_alpha, pri_beta = pri_beta, pri_gamma = pri_gamma, pri_delta = pri_delta, learning_seeds = learning_seeds)
    print("mean plug-in parameters \n {0}".format({
        "est_ratio": sklearn_gmm_result["ratio"],
        "est_mean": sklearn_gmm_result["mean"],
        "est_precision": sklearn_gmm_result["precision"]
    }))
    (correct_num_skgmm, perm_skgmm, label_arg_skgmm) = evaluate_correct_cluster_number(sklearn_gmm_result, noise_data_num, true_train_label_arg, K)    
    
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

    printmd("### 5. Correct number of labeling of GMM by sklearn:")
    printmd("+ {0}/{1}".format(correct_num_skgmm, len(label_arg_hsmm)))
        
    printmd("### 5. Correct number of labeling of GMM:")
    printmd("+ {0}/{1}".format(correct_num_gmm, len(label_arg_hsmm)))

    printmd("### 6. Correct number of labeling of HSMM:")
    printmd("+ {0}/{1}".format(correct_num_hsmm, len(label_arg_hsmm)))

    printmd("### 7. Generalization error of GMM by sklearn:")
    printmd("+ {0}".format(evaluate_log_loss(sklearn_gmm_result, true_param, noise_data_num, test_x, true_logpdf, pred_logpdf_gmm)))
    
    printmd("### 8. Generalization error of GMM:")
    printmd("+ {0}".format(evaluate_log_loss(gmm_result, true_param, noise_data_num, test_x, true_logpdf, pred_logpdf_gmm)))
    
    printmd("### 9. Generalization error of HSMM:")
    printmd("+ {0}".format(evaluate_log_loss(hsmm_result, true_param, noise_data_num, test_x, true_logpdf, pred_logpdf_hsmm)))

    printmd("### 10. Data distribution labeled by GMM by sklearn:")
    plot_scatter_with_label(train_x, label_arg_skgmm,  K, noise_data_num)
    
    printmd("### 11. Data distribution labeled by GMM:")
    plot_scatter_with_label(train_x, label_arg_gmm,  K, noise_data_num)

    printmd("### 12. Data distribution labeled by HSMM:")
    plot_scatter_with_label(train_x, label_arg_hsmm,  K, noise_data_num)


def plot_scatter_with_label(x:np.ndarray, label_arg:np.ndarray,  K:int, noise_data_num):
    """
    Scatter plot for data x
    """
    for i in range(K):
        if noise_data_num > 0:
            plt.scatter(x[np.where(label_arg[:-noise_data_num] == i)[0],0], x[np.where(label_arg[:-noise_data_num] == i)[0],1])
        else:
            plt.scatter(x[np.where(label_arg == i)[0],0], x[np.where(label_arg == i)[0],1])        
    plt.show()


# ## Problem setting:

true_ratio = np.array([0.33, 0.33, 0.34])
true_delta = 0
true_s = np.array([[1.5, 1.5], [0.5, 0.5], [1, 1]])
true_b = np.array([[4, 4], [-4, -4], [0, 0]])
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
ndataset = 10

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
learning_seed_offset = np.arange(learning_num)+1

### 繰り返しアルゴリズムの繰り返し回数
learning_iteration = 1000

### 学習モデルのコンポーネントの数
K = np.array([3, 5])

### 予測分布の対数確率密度巻子の値を生成する関数
pred_logpdf_gmm = lambda x, param: logpdf_mixture_dist(x, param, logpdf_multivariate_normal)
pred_logpdf_hsmm = lambda x, param: logpdf_mixture_dist(x, param, logpdf_hypsecant)
# -

# ## 実験設定で変更するパラメータを設定する
# + このパラメータをgrid計算して学習と評価を繰り返す

### 実験設定で変更するパラメータを設定する
experiment_params = {
    "learning_ncomponent": K,
    "learning_seed_offset": learning_seed_offset,
    "pri_params": [pri_params],
    "ndataset": [ndataset],
    "learning_iteration": [learning_iteration]
}

# ## 学習部分

from sklearn.model_selection import ParameterGrid

experiment_grids = ParameterGrid(experiment_params)

for data_seed in data_seeds:
    ### データ生成
    np.random.seed(data_seed)
    true_train_label = np.random.multinomial(n = 1, pvals = true_ratio, size = n)
    true_train_label_arg = np.argmax(true_train_label, axis = 1)
    true_test_label = np.random.multinomial(n = 1, pvals = true_ratio, size = N)
    true_test_label_arg = np.argmax(true_test_label, axis = 1)
    
    for experiment_grid in experiment_grids:
        
        break

for a in ParameterGrid(experiment_params):
    print(a)
    break

for params, value in a["pri_params"].items():
    setattr(hsmm_obj, params, value)

a["pri_params"]

hsmm_obj = HyperbolicSecantMixtureVB()
hsmm_obj.set_params(params = a["pri_params"])

hsmm_obj







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


