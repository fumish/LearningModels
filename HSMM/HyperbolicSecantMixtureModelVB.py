## standard libraries
import math
import itertools
from abc import ABCMeta, abstractmethod

## 3rd party libraries
import numpy as np
from scipy.special import gammaln, psi, multigammaln
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator
from sklearn.base import DensityMixin
from sklearn.utils.validation import check_is_fitted

## local libraries
from util import logcosh, ratio_tanh_x, multipsi
from learning import AbstractMixtureModel

class HyperbolicSecantMixtureVB(AbstractMixtureModel, DensityMixin, BaseEstimator):
    """
    This class is to infer a hyperbolic secant mixture model (HSMM) by Variational Bayesian approach.
    # Model p(x|w):
    p(x|w) = sum_{k=1}^K a_k prod_{j = 1}^M frac{sqrt{s_{kj}}}{2pi} frac{1}{cosh(frac{s_{kj}}{2}(x_j - b_{kj}))},
    where x, b_k in mathbb{R}^{M}, s_k in mathbb{R}_+^M and w = {a_k, b_k, s_k}_{k=1}^K.
    Note that the distribution of each component k is called hyperbolic secant distribution.

    # Prior distribution varphi(w):
    varphi(w) = Dir(a|{alpha_k}_{k=1}^K) prod_{k=1}^K N(b_k|0_M, beta I_M),
    where Dir(a|{alpha_k}_{k=1}^K) represents a Dirichlet distribution with parameter alpha_k
    and N(b_k|0_M, 1/beta I_M) a multivariate Gaussian distribution with mean 0 and covariance 1/beta I_M.
    I_M is M dimensional identity matrix.
    Note that although this prior distribution is not cojugate prior for the model,
    by approximating the model, the prior is conjugate prior for the approximated model.

    # Approximating the model p(x|w):
    Let p_{xi, eta}(x|w) be the approximated distribution, it is represented by
    p_{xi, eta}(x|w) = p(x|w) exp(-d_{phi}(h(w), h(xi)) -d_{psi}(g(w),g(eta)) ),
    where d_{phi}(h(w), h(xi)) and d_{psi}(g(w),g(eta)) are Bregman divergence
    with convex function phi and psi respectively.
    After doing a boring calculation, we obtain the following algorithm (I will append the detail if I have some motivation for it :( ):

   1. Initializing the following values:
       # g_{ikj}(eta), v_{ikj}(eta), h_{ik}(xi), u_{ik}(xi)
   2. Update the following values
       # hat{alpha}_k = alpha_k + sum_{i=1}^n u_{ik}(xi)
       # hat{beta}_{kj} = beta_k + sum_{i=1}^n -2v_{ikj}(eta)
       # hat{m}_{kj} = frac{1}{hat{beta}_k} sum_{i=1}^n -2v_{ik}(eta)x_i
       # hat{gamma}_{kj} = gamma_k + frac{1}{2}sum_{i=1}^n u_{ik}(xi)
       # hat{delta}_{kj} = delta_k + sum_{i=1}^n -v_{ikj}(eta)x_{ij}^2 - frac{hat{beta}_{kj}}{2}hat{m}_{kj}^2
   3. Update the following values
       # g_{ikj}(eta) = frac{hat{gamma}_{kj}}{hat{delta}_{kj}} (x_{ij} - hat{m}_{kj})^2 + frac{1}{hat{beta}_{kj}}
       # v_{ikj}(eta) = -u_{ik}(xi)frac{ tanh(sqrt{g_{ikj}(eta)}/2) }{4sqrt{g_{ikj}(eta)}}
   4. Update the following values
       # h_{ik}(xi) = psi(hat{alpha}_k) - psi(sum_{l=1}^K hat{alpha}_l) + frac{1}{2} sum_{j=1}^M (psi(hat{gamma}_{kj}) - log(hat{delta}_{kj})) - sum_{j=1}^M log(cosh(sqrt{g_{ikj}(eta)}/2))
       # u_{ik}(xi) = frac{ exp(h_{ik}(xi)) }{ sum_{l=1}^K exp(h_{il}(xi)) }
       # where,psi(x) = frac{d}{dx}log Gamma(x)
   5. Return back to 2.

   # Evaluating the performance:
   Fundamentally, this algorithm descrease the value of the objective function through the above algorithm,
   and small value is better for the algorithm.
   The value is described as follows:
   Evaluation function overline{F}_{xi, eta}(x^n):
   overline{F}_{xi, eta}(x^n) = - phi(h(xi)) - psi(g(eta)) + u(xi) cdot h(xi) + v(eta) cdot g(eta)
    + nM log 2 pi + log Gamma(sum_{l = 1}^K hat{alpha}_l) - log Gamma({sum_{l = 1}^Kalpha}_l) + sum_{k=1}^K log frac{Gamma(alpha_k)}{Gamma(hat{alpha}_k)}
    + sum_{k=1}^K sum_{j=1}^M bigl{ frac{1}{2} log frac{hat{beta}_{kj}}{beta_{kj}} + hat{gamma}_{kj} log hat{delta}_{kj} - gamma_{kj} log delta_{kj} - log Gamma(hat{gamma}_{kj}) + log Gamma(gamma_{kj}) bigr}
    """

    def __init__(self, K:int = 3,
                 pri_alpha:float = 0.1, pri_beta:float = 0.001, pri_gamma:float = 2, pri_delta:float = 2,
                 iteration:int = 1000, restart_num:int = 5, learning_seed:int = -1, tol:float = 1e-5, step:int = 20, is_trace:bool = False):
        """
        Initialize the following parameters:
        1. pri_alpha: hyperparameter for prior distribution of symmetric Dirichlet distribution.
        2. pri_beta: hyperparameter for prior distribution of Normal distribution for inverse variance.
        3. pri_gamma: hyperparameter for prior distribution of Gamma distribution for shape parameter.
        4. pri_delta: hyperparameter for prior distribution of Gamma distribution for rate parameter.
        5. iteration: Number of iteration.
        6. restart_num: Number of restart of inital values.
        7. learning_seed: Seed for initial values.
        8. tol: tolerance to stop the algorithm
        9. step: interval to calculate the objective function
            Note: Since evaluating the objective function is a little bit heavy (roughly it may have O(n*M*K)),
            so by this parameter we avoid watching the value every time.
        """
        self.K = K
        self.pri_alpha = pri_alpha
        self.pri_beta = pri_beta
        self.pri_gamma = pri_gamma
        self.pri_delta = pri_delta
        self.iteration = iteration
        self.restart_num = restart_num
        self.learning_seed = learning_seed
        self.tol = tol
        self.step = step
        self.is_trace = is_trace
        pass

    def fit(self, train_X:np.ndarray, y:np.ndarray=None):
        """
        LVA for HSMM.
        The algorithm is described in the above cell.
        Since the algorithm fails to local minima, the best estimator are chosen in several initial values.

        + Input:
            + train_X: array like, input data

        + Output:
            + The following parameters are restored in self instance:
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
        if self.learning_seed > 0:
            np.random.seed(self.learning_seed)

        ### Setting for static variable in the algorithm.
        expand_x = np.repeat(train_X, self.K).reshape(n, M, self.K).transpose((0, 2, 1))
        ### n * K * M data with the same matrix among 2nd dimension

        min_energy = np.inf
        result = dict()

        for restart in range(self.restart_num):
            energy = np.zeros(np.floor(self.iteration/self.step).astype(int))
            calc_ind = 0
            ### Setting for initial value
            # est_alpha = np.random.dirichlet(alpha = np.ones(self.K), size = 1)
            # est_m = np.random.normal(size = (self.K, M))
            # est_gamma = np.random.gamma(shape=1, size = (self.K, M))
            # est_delta = np.random.gamma(shape=1, size = (self.K, M))
            # est_beta = np.random.gamma(shape=1, size=(self.K, M))

            est_u_xi = np.random.dirichlet(alpha = np.ones(self.K), size=n)
            est_g_eta = np.abs(np.random.normal(size=(n,self.K,M)))
            est_v_eta = -np.repeat(est_u_xi, M).reshape(n, self.K, M) * ratio_tanh_x(np.sqrt(est_g_eta)/2)/8

            ### Start learning.
            for ite in range(self.iteration):
                ### Update posterior distribution of parameter
                est_alpha = self.pri_alpha + est_u_xi.sum(axis = 0)
                est_beta = self.pri_beta + (-2*est_v_eta.sum(axis = 0))
                est_m = -2 * (expand_x * est_v_eta).sum(axis = 0) / est_beta
                est_gamma = np.repeat(self.pri_gamma + est_u_xi.sum(axis = 0)/2, M).reshape(self.K,M)
                est_delta = self.pri_delta - (expand_x**2 * est_v_eta).sum(axis = 0) - est_beta / 2 * est_m**2

                ### Update auxiliary variables
                est_g_eta = np.repeat(est_gamma / est_delta, n).reshape(self.K,M,n).transpose((2,0,1)) * (expand_x - np.repeat(est_m,n).reshape(self.K,M,n).transpose((2,0,1)))**2 + 1/np.repeat(est_beta, n).reshape(self.K,M,n).transpose((2,0,1))
                est_v_eta = - np.repeat(est_u_xi, M).reshape(n, self.K, M) * ratio_tanh_x(np.sqrt(est_g_eta)/2)/8

                ### Update posterior distribution of latent variable
                sqrt_g_eta = np.sqrt(est_g_eta)
                est_h_xi = np.repeat(psi(est_alpha) - psi(est_alpha.sum()) + (psi(est_gamma) - np.log(est_delta)).sum(axis = 1)/2 - M*np.log(2*np.pi)/2, n).reshape(self.K,n).T - logcosh(sqrt_g_eta/2).sum(axis = 2)
                max_h_xi = est_h_xi.max(axis = 1)
                norm_h_xi = est_h_xi - np.repeat(max_h_xi, self.K).reshape(n,self.K)
                est_u_xi = np.exp(norm_h_xi) / np.repeat(np.exp(norm_h_xi).sum(axis = 1), self.K).reshape(n, self.K)

                if ite % self.step == 0:
                    ### Calculate evaluation function
                    energy[calc_ind] =  self._calc_obj_func(est_u_xi = est_u_xi, est_h_xi = est_h_xi,
                                                            est_v_eta = est_v_eta, est_g_eta = est_g_eta,
                                                            est_alpha = est_alpha, est_beta = est_beta, est_gamma = est_gamma, est_delta = est_delta)
                    if self.is_trace: print(energy[calc_ind])
                    if calc_ind > 0 and np.abs(energy[calc_ind] - energy[calc_ind-1]) < self.tol:
                        energy = energy[:calc_ind]
                        break
                    calc_ind += 1
                    pass
                pass
            energy[-1] = self._calc_obj_func(est_u_xi = est_u_xi, est_h_xi = est_h_xi, est_v_eta = est_v_eta, est_g_eta = est_g_eta,
                                             est_alpha = est_alpha, est_beta = est_beta, est_gamma = est_gamma, est_delta = est_delta)
            # energy = energy[:calc_ind]
            if self.is_trace: print(energy[-1])
            if energy[-1] < min_energy:
                min_energy = energy[-1]
                result["ratio"] = est_alpha / est_alpha.sum()
                result["mean"] = est_m
                result["precision"] = est_gamma / est_delta
                result["scale"] = np.array([np.diag(est_delta[k,:] / est_gamma[k,:]) for k in range(self.K)])
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
            pass
        self.result_ = result
        return self

    def _logpdf(self, x:np.ndarray, mean:np.ndarray, precision:np.ndarray):
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

    def _calc_obj_func(self, **kwargs) -> float:
        """
        -ELBO is calculated.
        + Necessary arguments are as follows:
            1. est_u_xi
            2. est_h_xi
            3. est_v_eta
            4. est_g_eta
            5. est_alpha
            6. est_beta
            7. est_gamma
            8. est_delta
        """
        est_u_xi = kwargs["est_u_xi"]
        est_h_xi = kwargs["est_h_xi"]
        est_v_eta = kwargs["est_v_eta"]
        est_g_eta = kwargs["est_g_eta"]
        est_alpha = kwargs["est_alpha"]
        est_beta = kwargs["est_beta"]
        est_gamma = kwargs["est_gamma"]
        est_delta = kwargs["est_delta"]

        n, _, M = est_v_eta.shape

        sqrt_g_eta = np.sqrt(est_g_eta)
        max_h_xi = est_h_xi.max(axis = 1)
        norm_h_xi = est_h_xi - np.repeat(max_h_xi, self.K).reshape(n,self.K)

        energy =  (np.repeat(est_u_xi, M).reshape(n, self.K, M) * logcosh(sqrt_g_eta/2)).sum() - (np.log(np.exp(norm_h_xi).sum(axis = 1)) + max_h_xi).sum() + (est_u_xi * est_h_xi).sum() + (est_v_eta * est_g_eta).sum()
        energy += gammaln(est_alpha.sum()) - gammaln(self.K*self.pri_alpha) + (-gammaln(est_alpha) + gammaln(self.pri_alpha)).sum()
        energy += (np.log(est_beta/self.pri_beta)/2 + est_gamma * np.log(est_delta) - self.pri_gamma * np.log(self.pri_delta) - gammaln(est_gamma) + gammaln(self.pri_gamma)).sum()
        energy += n*M*np.log(2*np.pi)
        return energy

    def get_params(self, deep = True):
        return{
               "K": self.K,
               "pri_alpha": self.pri_alpha,
               "pri_beta": self.pri_beta,
               "pri_gamma":self.pri_gamma,
               "pri_delta":self.pri_delta,
               "iteration":self.iteration,
               "restart_num":self.restart_num,
               "learning_seed":self.learning_seed,
               "tol":self.tol,
               "step":self.step
        }

    def set_params(self, **params):
        for params, value in params.items():
            setattr(self, params, value)
        return self
