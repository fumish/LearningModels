"""
This library is for probability distribution
"""

## standard libraries
from abc import ABCMeta, abstractmethod

## 3rd party libraries
import numpy as np
from scipy.stats import multivariate_normal, t, laplace, gumbel_r, hypsecant
from util.elementary_function import logcosh

class AbstractMixtureModel(metaclass = ABCMeta):
    """
    This class is abstract class for mixture models to generate random variables, calculate probability density function,
    and calculate posterior latent distribution.
    """
    # def _rvs_component(cls, loc:float=0, scale:float=1, size:int =1, **kwargs):
    @classmethod
    @abstractmethod
    def _rvs_component(cls, loc:float=0, scale:float=1, size:int =1, **kwargs):
        """
        Generate random variable for each component distribution.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def _logpdf_component(cls, x:np.ndarray, loc:float=0, scale:float=1, **kwargs):
        """
        Calculate log probability density function.
        """
        raise NotImplementedError()

    @classmethod
    def rvs(cls, ratio:np.ndarray, loc:np.ndarray, scale:np.ndarray, size:int=1, data_seed:int=-1, **kwargs):
        if data_seed > 0: np.random.seed(data_seed)
        data_label = np.random.multinomial(n = 1, pvals = ratio, size = size)
        data_label_arg = np.argmax(data_label, axis = 1)
        X = np.array([[cls._rvs_component(loc[data_label_arg[i],j], scale[data_label_arg[i],j], size=1, **kwargs) for j in range(loc.shape[1])] for i in range(size)]).squeeze()
        return (X, data_label, data_label_arg)

    @classmethod
    def logpdf(cls, X:np.ndarray, ratio:np.ndarray, loc:np.ndarray, scale:np.ndarray, **kwargs):
        n = X.shape[0]
        K = len(ratio)

        loglik = np.zeros((n,K))
        for k in range(K):
            if scale.ndim == 2:
                loglik[:,k] = np.log(ratio[k]) + cls._logpdf_component(X, loc[k,:], scale[k,:], **kwargs).sum(axis=1)
            elif scale.ndim == 3:
                loglik[:,k] = np.log(ratio[k]) + cls._logpdf_component(X, loc[k,:],  scale[k,:,:], **kwargs).sum(axis=1)
            else:
                raise ValueError("Error precision, dimension of precision must be 2 or 3!")
        max_loglik = loglik.max(axis = 1)
        norm_loglik = loglik - np.repeat(max_loglik,K).reshape(n,K)
        return (np.log(np.exp(norm_loglik).sum(axis = 1)) + max_loglik).sum()

    @classmethod
    def latent_posterior_logprob(cls, x:np.ndarray, ratio:np.ndarray, loc:np.ndarray, scale:np.ndarray):
        K = len(ratio)
        (n, M) = x.shape

        log_complete_likelihood = (np.repeat(np.log(ratio),n).reshape(K,n) + np.array([cls._logpdf_component(x, loc = loc[k,:], scale = scale[k,:], **kwargs).sum(axis=1) for k in range(K)])).T
        max_log_complete_likelihood = log_complete_likelihood.max(axis = 1)
        norm_log_complete_likelihood = log_complete_likelihood - np.repeat(max_log_complete_likelihood, K).reshape(n, K)
        log_posterior_p =  norm_log_complete_likelihood - np.log(np.repeat(np.exp(norm_log_complete_likelihood).sum(axis=1), K).reshape(n, K))
        return log_posterior_p

class GumbelMixtureModel(AbstractMixtureModel):
    """
    This is a class of mixture of gumbel distribution:
    p(x|w) propto sum_k a_k exp(-(x-b_k)/s_k + exp(-(x-b_k)/s_k)),
    w = (a_k, b_k, s_k)_k^K
    """
    @classmethod
    def _rvs_component(cls, loc:float=0, scale:float=1, size:int =1, **kwargs):
        """
        Generate random variable for each component distribution.
        """
        return gumbel_r.rvs(loc=loc, scale=scale, size=size)

    @classmethod
    def _logpdf_component(cls, x:np.ndarray, loc:float=0, scale:float=1, **kwargs):
        """
        Calculate log probability density function.
        """
        return gumbel_r.logpdf(x, loc, scale)

class HyperbolicSecantMixtureModel(AbstractMixtureModel):
    @classmethod
    def _rvs_component(cls, loc:float=0, scale:float=1, size:int =1, **kwargs):
        """
        Generate data following hyperbolic secant distribution.
        Let $Y \sim standard_cauchy(x)$,
        random variable $X = \frac{2}{\sqrt{s}}\sinh^{-1}(Y) + b$ follows to
        X \sim p(x) = \frac{\sqrt{s}}{2\pi}\frac{1}{\cosh(s(x-b)/2)}.
        """
        # Y = np.random.standard_cauchy(size=size)
        # X = 2/np.sqrt(scale)*np.arcsinh(Y) + loc
        return hypsecant.rvs(loc=loc, scale=scale, size=size)

    @classmethod
    def _logpdf_component(cls, x:np.ndarray, loc:float=0, scale:float=1, **kwargs):
        """
        Calculate log probability density function.
        """
        (n, M) = x.shape
        expand_scale = np.repeat(np.diag(scale), n).reshape(M,n).T
        y = np.sqrt(expand_scale)*(x - np.repeat(loc, n).reshape(M,n).T)/2
        return(np.log(expand_precision)/2 - np.log(2*np.pi) - logcosh(y))

class StudentMixtureModel(AbstractMixtureModel):
    """
    Mixture model for t-distribution.
    Basically, this class is implemented class of AbstractMixtureModel.
    """
    DEFAULT_DF = 5

    @classmethod
    def _rvs_component(cls, loc:float=0, scale:float=1, size:int =1, **kwargs):
        """
        Generate random variable for each component distribution.
        """
        df = kwargs["df"] if "df" in kwargs.keys() else StudentMixtureModel.DEFAULT_DF
        return t.rvs(df = df, loc=loc, scale=scale, size=size)

    @classmethod
    def _logpdf_component(cls, x:np.ndarray, loc:float=0, scale:float=1, **kwargs):
        """
        Calculate log probability density function.
        """
        df = kwargs["df"] if "df" in kwargs.keys() else StudentMixtureModel.DEFAULT_DF
        return t.logpdf(x, df = df, loc = loc, scale = scale)

class LaplaceMixtureModel(AbstractMixtureModel):
    """
    Mixture model for Laplace distribution.
    Basically, this class is implemented class of AbstractMixtureModel.
    """

    @classmethod
    def _rvs_component(cls, loc:float=0, scale:float=1, size:int =1, **kwargs):
        """
        Generate random variable for each component distribution.
        """
        return laplace.rvs(loc=loc, scale=scale, size=size)

    @classmethod
    def _logpdf_component(cls, x:np.ndarray, loc:float=0, scale:float=1, **kwargs):
        """
        Calculate log probability density function.
        """
        return -np.abs(x - loc)/scale - np.log(2 * scale)
    pass

# class GaussianMixtureModel(AbstractMixtureModel):
#     """
#     Mixture model for Laplace distribution.
#     Basically, this class is implemented class of AbstractMixtureModel.
#     """
#
#     @classmethod
#     def _rvs_component(cls, loc:float=0, scale:float=1, size:int =1, **kwargs):
#         """
#         Generate random variable for each component distribution.
#         """
#         cov = kwargs["cov"] if "cov" in kwargs.keys() else np.linalg.inv(scale)
#         return multivariate_normal.rvs(mean=loc, cov=cov, size=size)
#
#     @classmethod
#     def _logpdf_component(cls, x:np.ndarray, loc:float=0, scale:float=1, **kwargs):
#         """
#         Calculate log probability density function.
#         """
#         cov = kwargs["cov"] if "cov" in kwargs.keys() else np.linalg.inv(scale)
#         return multivariate_normal.logpdf(x, mean=loc, cov=cov)
#     pass

class _HyperbolicSecantMixtureModel(object):
    """
    This is class of Hyperbolic Secant mixture model
    Like random variable of scipy.stats, this class has rvs, pdf, logpdf, and so on.
    """

    @classmethod
    def logpdf_hypsecant(cls, x:np.ndarray, mean:np.ndarray, precision:np.ndarray):
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
    def logpdf(cls, X:np.ndarray, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray):
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
    def random_hsm(cls, n, loc = 0, scale = 1):
        """
        Generate data following hyperbolic secant distribution.
        Let $Y \sim standard_cauchy(x)$,
        random variable $X = \frac{2}{\sqrt{s}}\sinh^{-1}(Y) + b$ follows to
        X \sim p(x) = \frac{\sqrt{s}}{2\pi}\frac{1}{\cosh(s(x-b)/2)}.
        """
        Y = np.random.standard_cauchy(size=n)
        X = 2/np.sqrt(scale)*np.arcsinh(Y) + loc
        return X

    @classmethod
    def rvs(cls, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray, size:int=1, data_seed:int=-1):
        if data_seed > 0: np.random.seed(data_seed)
        data_label = np.random.multinomial(n = 1, pvals = ratio, size = size)
        data_label_arg = np.argmax(data_label, axis = 1)
        X = np.array([[HyperbolicSecantMixtureModel.random_hsm(n = 1, loc=mean[data_label_arg[i],j], scale=1/precision[data_label_arg[i],j]) for j in range(mean.shape[1])] for i in range(size)]).squeeze()
        return (X, data_label, data_label_arg)

    @classmethod
    def score_latent_kl(cls, x:np.ndarray, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray):
        K = len(ratio)
        (n, M) = x.shape
        expand_x = np.repeat(x, K).reshape(n, M, K).transpose((0, 2, 1))

        y = np.repeat(np.sqrt(precision)/2, n).reshape(K,M,n).transpose((2,0,1)) * (expand_x - np.repeat(mean,n).reshape(K,M,n).transpose((2,0,1)))
        log_complete_likelihood = np.repeat(np.log(ratio) + np.log(precision).sum(axis = 1)/2 - M*np.log(2*np.pi), n).reshape(K,n).T - logcosh(y).sum(axis = 2)
        max_log_complete_likelihood = log_complete_likelihood.max(axis = 1)
        norm_log_complete_likelihood = log_complete_likelihood - np.repeat(max_log_complete_likelihood, K).reshape(n, K)
        posterior_prob = np.exp(norm_log_complete_likelihood) / np.repeat(np.exp(norm_log_complete_likelihood).sum(axis = 1), K).reshape(n, K)
        return (-np.log(posterior_prob).sum(), posterior_prob)

class _StudentMixtureModel(object):
    """
    This is class of Student mixture model
    Like random variable of scipy.stats, this class has rvs, pdf, logpdf, and so on.
    """

    @classmethod
    def logpdf(cls, X:np.ndarray, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray, df:float=1.5):
        n = X.shape[0]
        K = len(ratio)

        loglik = np.zeros((n,K))
        for k in range(K):
            if precision.ndim == 2:
                loglik[:,k] = np.log(ratio[k]) + t.logpdf(X, df = df, loc = mean[k,:], scale = 1/precision[k,:]).sum(axis = 1)
            else:
                raise ValueError("Error precision, dimension of precision must be 2!")
        max_loglik = loglik.max(axis = 1)
        norm_loglik = loglik - np.repeat(max_loglik,K).reshape(n,K)
        return (np.log(np.exp(norm_loglik).sum(axis = 1)) + max_loglik).sum()

    @classmethod
    def rvs(cls, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray, size:int=1, data_seed:int=-1, df:float = 1.5):
        if data_seed > 0: np.random.seed(data_seed)
        data_label = np.random.multinomial(n = 1, pvals = ratio, size = size)
        data_label_arg = np.argmax(data_label, axis = 1)
        X = np.array([[t.rvs(df = df, loc=mean[data_label_arg[i],j], scale=1/precision[data_label_arg[i],j], size=1) for j in range(mean.shape[1])] for i in range(size)]).squeeze()
        return (X, data_label, data_label_arg)

    @classmethod
    def latent_posterior_logprob(cls, x:np.ndarray, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray, df:float=1.5):
        K = len(ratio)
        (n, M) = x.shape

        log_complete_likelihood = (np.repeat(np.log(ratio),n).reshape(K,n) + np.array([t.logpdf(x, df = df, loc = mean[k,:], scale = precision[k,:]).sum(axis=1) for k in range(K)])).T
        max_log_complete_likelihood = log_complete_likelihood.max(axis = 1)
        norm_log_complete_likelihood = log_complete_likelihood - np.repeat(max_log_complete_likelihood, K).reshape(n, K)
        log_posterior_p =  norm_log_complete_likelihood - np.log(np.repeat(np.exp(norm_log_complete_likelihood).sum(axis=1), K).reshape(n, K))
        return log_posterior_p

class _LaplaceMixtureModel(object):
    """
    This is class of Gaussian mixture model
    Like random variable of scipy.stats, this class has rvs, pdf, logpdf, and so on.
    """

    @classmethod
    def logpdf_laplace(cls, X:np.ndarray, loc:np.ndarray, scale:np.ndarray):
        """
        Since value of scipy.stats.laplace.logpdf goes to -np.inf, logpdf function is redefined here.
        Note: Comparing this function with scipy.stats.laplace.logpdf, these functions are same except for the value goes to -np.inf
        """
        return -np.abs(X - loc)/scale - np.log(2 * scale)

    @classmethod
    def logpdf(cls, X:np.ndarray, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray):
        n = X.shape[0]
        K = len(ratio)

        loglik = np.zeros((n,K))
        for k in range(K):
            if precision.ndim == 2:
                loglik[:,k] = np.log(ratio[k]) + LaplaceMixtureModel.logpdf_laplace(X, loc = mean[k,:], scale = 1/precision[k,:]).sum(axis = 1)
            else:
                raise ValueError("Error precision, dimension of precision must be 2 or 3!")
        max_loglik = loglik.max(axis = 1)
        norm_loglik = loglik - np.repeat(max_loglik,K).reshape(n,K)
        return (np.log(np.exp(norm_loglik).sum(axis = 1)) + max_loglik).sum()

    @classmethod
    def rvs(cls, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray, size:int=1, data_seed:int=-1):
        if data_seed > 0: np.random.seed(data_seed)
        data_label = np.random.multinomial(n = 1, pvals = ratio, size = size)
        data_label_arg = np.argmax(data_label, axis = 1)
        X = np.array([[laplace.rvs(loc=mean[data_label_arg[i],j], scale=1/precision[data_label_arg[i],j], size=1) for j in range(mean.shape[1])] for i in range(size)]).squeeze()
        return (X, data_label, data_label_arg)

    @classmethod
    def latent_posterior_logprob(cls, x:np.ndarray, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray):
        K = len(ratio)
        (n, M) = x.shape
        expand_x = np.repeat(x, K).reshape(n, M, K).transpose((0, 2, 1))

        log_complete_likelihood = (np.repeat(np.log(ratio),n).reshape(K,n) + np.array([LaplaceMixtureModel.logpdf_laplace(x, loc = mean[k,:], scale = precision[k,:]).sum(axis=1) for k in range(K)])).T
        max_log_complete_likelihood = log_complete_likelihood.max(axis = 1)
        norm_log_complete_likelihood = log_complete_likelihood - np.repeat(max_log_complete_likelihood, K).reshape(n, K)
        log_posterior_p =  norm_log_complete_likelihood - np.log(np.repeat(np.exp(norm_log_complete_likelihood).sum(axis=1), K).reshape(n, K))
        return log_posterior_p

class _HyperbolicSecantMixtureModel(object):
    """
    This is class of Gaussian mixture model
    Like random variable of scipy.stats, this class has rvs, pdf, logpdf, and so on.
    """

    @classmethod
    def logpdf_hypsecant(cls, x:np.ndarray, mean:np.ndarray, precision:np.ndarray):
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
    def logpdf(cls, X:np.ndarray, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray):
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
    def random_hsm(cls, n, loc = 0, scale = 1):
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
    def rvs(cls, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray, size:int=1, data_seed:int=-1):
        if data_seed > 0: np.random.seed(data_seed)
        data_label = np.random.multinomial(n = 1, pvals = ratio, size = size)
        data_label_arg = np.argmax(data_label, axis = 1)
        X = np.array([[HyperbolicSecantMixtureModel.random_hsm(n = 1, loc=mean[data_label_arg[i],j], scale=1/precision[data_label_arg[i],j]) for j in range(mean.shape[1])] for i in range(size)]).squeeze()
        return (X, data_label, data_label_arg)

    @classmethod
    def latent_posterior_logprob(cls, x:np.ndarray, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray):
        K = len(ratio)
        (n, M) = x.shape
        expand_x = np.repeat(x, K).reshape(n, M, K).transpose((0, 2, 1))

        y = np.repeat(np.sqrt(precision)/2, n).reshape(K,M,n).transpose((2,0,1)) * (expand_x - np.repeat(mean,n).reshape(K,M,n).transpose((2,0,1)))
        log_complete_likelihood = np.repeat(np.log(ratio) + np.log(precision).sum(axis = 1)/2 - M*np.log(2*np.pi), n).reshape(K,n).T - logcosh(y).sum(axis = 2)
        max_log_complete_likelihood = log_complete_likelihood.max(axis = 1)
        norm_log_complete_likelihood = log_complete_likelihood - np.repeat(max_log_complete_likelihood, K).reshape(n, K)
        log_posterior_p =  norm_log_complete_likelihood - np.log(np.repeat(np.exp(norm_log_complete_likelihood).sum(axis=1), K).reshape(n, K))
        return log_posterior_p

class GaussianMixtureModel(object):
    """
    This is class of Gaussian mixture model
    Like random variable of scipy.stats, this class has rvs, pdf, logpdf, and so on.
    """

    @classmethod
    def rvs(cls, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray, size:int=1, data_seed:int=-1):
        if data_seed > 0: np.random.seed(data_seed)
        data_label = np.random.multinomial(n = 1, pvals = ratio, size = size)
        data_label_arg = np.argmax(data_label, axis = 1)
        X = np.array([multivariate_normal.rvs(mean=mean[data_label_arg[i],:], cov=np.diag(1/precision[data_label_arg[i],:]), size=1) for i in range(size)])
        return (X, data_label, data_label_arg)

    # def logpdf(self, X:np.ndarray, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray):
    #     return np.exp(self.logpdf(X, ratio, mean, precision))

    @classmethod
    def logpdf(cls, X:np.ndarray, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray):
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
    def latent_posterior_logprob(cls, x:np.ndarray, ratio:np.ndarray, mean:np.ndarray, precision:np.ndarray):
        K = len(ratio)
        (n, M) = x.shape
        expand_x = np.repeat(x, K).reshape(n, M, K).transpose((0, 2, 1))

        y = np.repeat(precision, n).reshape(K,M,n).transpose((2,0,1)) * (expand_x - np.repeat(mean,n).reshape(K,M,n).transpose((2,0,1)))**2
        log_complete_likelihood = np.repeat(np.log(ratio) + np.log(precision).sum(axis = 1)/2 - M/2*np.log(2*np.pi), n).reshape(K,n).T - y.sum(axis = 2)/2
        max_log_complete_likelihood = log_complete_likelihood.max(axis = 1)
        norm_log_complete_likelihood = log_complete_likelihood - np.repeat(max_log_complete_likelihood, K).reshape(n, K)
        log_posterior_p =  norm_log_complete_likelihood - np.log(np.repeat(np.exp(norm_log_complete_likelihood).sum(axis=1), K).reshape(n, K))
        return log_posterior_p
