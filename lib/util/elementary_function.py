
"""
This module is a set of useful function, while famous library does not seem to contain.
"""
## standard libraries

## 3rd party libraries
import numpy as np
from scipy.stats import multivariate_normal

## local libraries

def logcosh(x:np.ndarray):
    """
    Calculating a log cosh(x).
    When absolute value of x is very large, this function are overflow,
    so we avoid it.
    """
    return np.abs(x) + np.log((1 + np.exp(-2 * np.abs(x)))/2)

class GaussianMixtureModel(object):
    """
    This is class of Gaussian mixture model
    Like random variable of scipy.stats, this class has rvs, pdf, logpdf, and so on.
    """

    def __init__(self):
        pass

    def rvs(self, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray, size:int=1, data_seed:int=-1):
        if data_seed > 0: np.random.seed(data_seed)
        data_label = np.random.multinomial(n = 1, pvals = ratio, size = size)
        data_label_arg = np.argmax(data_label, axis = 1)
        X = np.array([multivariate_normal.rvs(mean=mean[data_label_arg[i],:], cov=np.diag(1/precision[data_label_arg[i],:]), size=1) for i in range(size)])
        return (X, data_label, data_label_arg)

    def logpdf(self, X:np.ndarray, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray):
        return np.exp(self.logpdf(X, ratio, mean, precision))

    def logpdf(self, X:np.ndarray, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray):
        n = X.shape[0]
        K = len(ratio)

        loglik = np.zeros((n,K))
        for k in range(K):
            if precision.ndim == 2:
                loglik[:,k] = np.log(ratio[k]) + multivariate_normal.logpdf(X, mean[k,:], np.diag(1/precision[k,:]))
            elif precision.ndim == 3:
                loglik[:,k] = np.log(ratio[k]) + multivariate_normal.logpdf(X, mean[k,:],  1/precision[k,:,:])
            else:
                raise ValueError("Error precision, dimension of precision must be 2 or 3!")
        max_loglik = loglik.max(axis = 1)
        norm_loglik = loglik - np.repeat(max_loglik,K).reshape(n,K)
        return (np.log(np.exp(norm_loglik).sum(axis = 1)) + max_loglik).sum()

def rgmm(ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray, size:int=1, data_seed:int = -1):
    """
    Generate data following to mixture of a Gaussian distribution with ratio, mean, and precision.
    Assigning the data size and seed is admissible for this function.

    + Input:
        + ratio: K dimensional ratio vector, i.e. (K-1) dimensional simplex.
        + mean: K times M dimensional vector, where K is the number of component.
        + precision: K dimensional R_+ vector representing a precision for each distribution.
        + size: (optional) the number of data, default is 1.
        + data_seed: (optional) seed to gerante data.
    """
    if data_seed > 0: np.random.seed(data_seed)
    data_label = np.random.multinomial(n = 1, pvals = ratio, size = size)
    data_label_arg = np.argmax(data_label, axis = 1)
    M = mean.shape[1]
    X = np.array([np.random.normal(loc=mean[data_label_arg[i],:], scale=1/precision[data_label_arg[i],:]) for i in range(size)])
    return (X, data_label, data_label_arg)

# def logpdf_mixture_dist(x:np.ndarray, param:dict, component_log_dist:Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]):
#     """
#     対数尤度の計算
#     確率分布が混合分布の時を想定:
#     $\log p(x|w) = \sum_{i = 1}^n \log p(x_i|w) = \sum_{i = 1}^n \log \exp(L_{ik}(w)) =  \sum_{i = 1}^n \{\hat{L}_{i} + \log \exp(L_{ik}(w) - \hat{L}(i)) \}$,
#     where L_{ik} = \log a_k + \log p(x_i|w, y_{ik} = 1)\hat{L}(i) = \max_{k} L_{ik}, p(x_i|w, y_{ik}=1):i番目のサンプルのk番目のクラスタの確率分布
#
#     + 入力:
#         1. x:入力データ(n*M)
#         2. param: 確率分布のパラメータ(ratio: 混合比, mean: 各クラスタの平均値 K*M, scale: 各クラスタのscale(正規分布における標準偏差) K*M)
#         3. component_log_dist: 各クラスタの対数確率密度の値 logp(x|w,y)
#     """
#     n = x.shape[0]
#     K = len(param["ratio"])
#     loglik = np.zeros((n,K))
#     for k in range(K):
#         if param["scale"].ndim == 2:
#             loglik[:,k] = np.log(param["ratio"][k]) + component_log_dist(test_x, param["mean"][k,:],  param["scale"][k,:])
#         elif param["scale"].ndim == 3:
#             loglik[:,k] = np.log(param["ratio"][k]) + component_log_dist(test_x, param["mean"][k,:],  param["scale"][k,:,:])
#         else:
#             raise ValueError("Error precision, dimension of precision must be 2 or 3!")
#     max_loglik = loglik.max(axis = 1)
#     norm_loglik = loglik - np.repeat(max_loglik,K).reshape(n,K)
#     return (np.log(np.exp(norm_loglik).sum(axis = 1)) + max_loglik)
