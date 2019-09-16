
"""
This module is a set of useful function, while famous library does not seem to contain.
"""
## standard libraries

## 3rd party libraries
import numpy as np
from scipy.stats import multivariate_normal

## local libraries

def ratio_tanh_x(x):
    """
    Calculating f(x)=tanh(x)/x.
    While lim_{x -> 0} f(x) = 1, overflow is occured at zero point.
    Thus, this function is conditioned by zero point and other points.
    """
    zero_condition = 1e-20
    ret_val = np.zeros(x.shape)
    zero_ind = np.abs(x) < zero_condition
    ret_val[~zero_ind] = np.tanh(x[~zero_ind])/x[~zero_ind]
    ret_val[zero_ind] = 1
    return ret_val

def logcosh(x:np.ndarray):
    """
    Calculating a log cosh(x).
    When absolute value of x is very large, this function are overflow,
    so we avoid it.
    """
    return np.abs(x) + np.log((1 + np.exp(-2 * np.abs(x)))/2)

class HyperbolicSecantMixtureModel(object):
    """
    This is class of Gaussian mixture model
    Like random variable of scipy.stats, this class has rvs, pdf, logpdf, and so on.
    """

    @classmethod
    def logpdf_hypsecant(clf, x:np.ndarray, mean:np.ndarray, precision:np.ndarray):
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

    @classmethod
    def logpdf(self, X:np.ndarray, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray):
        n = X.shape[0]
        K = len(ratio)

        loglik = np.zeros((n,K))
        for k in range(K):
            if precision.ndim == 2:
                loglik[:,k] = np.log(ratio[k]) + HyperbolicSecantMixtureModel.logpdf_hypsecant(X, mean[k,:], np.diag(1/precision[k,:]))
            elif precision.ndim == 3:
                loglik[:,k] = np.log(ratio[k]) + HyperbolicSecantMixtureModel.logpdf_hypsecant(X, mean[k,:],  1/precision[k,:,:])
            else:
                raise ValueError("Error precision, dimension of precision must be 2 or 3!")
        max_loglik = loglik.max(axis = 1)
        norm_loglik = loglik - np.repeat(max_loglik,K).reshape(n,K)
        return (np.log(np.exp(norm_loglik).sum(axis = 1)) + max_loglik).sum()

    @classmethod
    def random_hsm(clf, n, loc = 0, scale = 1):
        """
        Generate data following hyperbolic secant distribution.
        Let $Y \sim standard_cauchy(x)$,
        random variable $X = \frac{2}{\sqrt{s}}\sinh^{-1}(Y) + b$ follows to
        $X \sim p(x) = \frac{\sqrt{s}}{2\pi}\frac{1}{\cosh(s(x-b)/2)}$.
        """
        Y = np.random.standard_cauchy(size=n)
        X = 2/np.sqrt(scale)*np.arcsinh(Y) + loc
        return X

    @classmethod
    def rvs(clf, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray, size:int=1, data_seed:int=-1):
        if data_seed > 0: np.random.seed(data_seed)
        data_label = np.random.multinomial(n = 1, pvals = ratio, size = size)
        data_label_arg = np.argmax(data_label, axis = 1)
        X = np.array([[HyperbolicSecantMixtureModel.random_hsm(n = 1, loc=mean[data_label_arg[i],j], scale=1/precision[data_label_arg[i],j]) for j in range(mean.shape[1])] for i in range(size)]).squeeze()
        return (X, data_label, data_label_arg)

    @classmethod
    def score_latent_kl(clf, x:np.ndarray, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray):
        K = len(ratio)
        (n, M) = x.shape
        expand_x = np.repeat(x, K).reshape(n, M, K).transpose((0, 2, 1))

        y = np.repeat(np.sqrt(precision)/2, n).reshape(K,M,n).transpose((2,0,1)) * (expand_x - np.repeat(mean,n).reshape(K,M,n).transpose((2,0,1)))
        log_complente_likelihood = np.repeat(np.log(ratio) + np.log(precision).sum(axis = 1)/2 - M*np.log(2*np.pi), n).reshape(K,n).T - logcosh(y).sum(axis = 2)
        max_log_complente_likelihood = log_complente_likelihood.max(axis = 1)
        norm_log_complente_likelihood = log_complente_likelihood - np.repeat(max_log_complente_likelihood, K).reshape(n, K)
        posterior_prob = np.exp(norm_log_complente_likelihood) / np.repeat(np.exp(norm_log_complente_likelihood).sum(axis = 1), K).reshape(n, K)
        return (-np.log(posterior_prob).sum(), posterior_prob)

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

    # def logpdf(self, X:np.ndarray, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray):
    #     return np.exp(self.logpdf(X, ratio, mean, precision))

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

    @classmethod
    def score_latent_kl(clf, x:np.ndarray, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray):
        K = len(ratio)
        (n, M) = x.shape
        expand_x = np.repeat(x, K).reshape(n, M, K).transpose((0, 2, 1))

        y = np.repeat(precision, n).reshape(K,M,n).transpose((2,0,1)) * (expand_x - np.repeat(mean,n).reshape(K,M,n).transpose((2,0,1)))**2
        log_complente_likelihood = np.repeat(np.log(ratio) + np.log(precision).sum(axis = 1)/2 - M/2*np.log(2*np.pi), n).reshape(K,n).T - y.sum(axis = 2)/2
        max_log_complente_likelihood = log_complente_likelihood.max(axis = 1)
        norm_log_complente_likelihood = log_complente_likelihood - np.repeat(max_log_complente_likelihood, K).reshape(n, K)
        posterior_prob = np.exp(norm_log_complente_likelihood) / np.repeat(np.exp(norm_log_complente_likelihood).sum(axis = 1), K).reshape(n, K)
        return (-norm_log_complente_likelihood.sum() + np.log(np.exp(norm_log_complente_likelihood).sum(axis=1)).sum(), posterior_prob)

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
