""" Mixture Models """
# standard libraries
import math
import itertools
from abc import ABCMeta, abstractmethod

# 3rd party libraries
import numpy as np
from scipy.special import gammaln, psi, multigammaln
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator
from sklearn.base import DensityMixin
from sklearn.utils.validation import check_is_fitted

# local libraries
from util.elementary_function import logcosh, ratio_tanh_x, multipsi

"""
This is a library for probability distribution of mixture.
Mainly this library focuses on Bayesian method to estimate the models.
Hereafter, we use latex like notation for equations.
Back slash is often occurs syntax error, so we describe each equation without back slash.
"""


class AbstractMixtureModel(metaclass=ABCMeta):
    """
    Abstract class for mixture model.
    Since every mixture model has cluster, cluster relevant method are prepared in this method.

    + To inherit this class, the following variables are necessary to use methods include here:
        1. result_: dictionary of result for fitted values.
            + u_xi: posterior distribution of latent variable.
            + h_xi: corresponding to complete log likelihood.
            + alpha: mixture ratio.
            + mean: mean of each component
            + precision: precision of each component
    """

    @abstractmethod
    def fit(self, train_X: np.ndarray, train_Y: np.ndarray):
        """
        Learning for this instance.
        """
        raise NotImplementedError()

    @abstractmethod
    def _logpdf(self, x: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
        """
        Log value of probability density function.
        """
        raise NotImplementedError()

    @abstractmethod
    def _calc_obj_func(self, **kwargs) -> float:
        """
        Calculate objective function to evaluate the estimated parameters.
        Every inherited class will minimize the value of this function.
        Arguments depend on each class, so kwargs are adopted.
        """
        raise NotImplementedError()

    def predict_logproba(self, test_X: np.ndarray):
        """
        Calculate log value of predictive distribution.
        Difference among each implemented class is _logpdf, this method is commonized.
        """

        check_is_fitted(self, "result_")
        n = test_X.shape[0]
        loglik = np.zeros((n, self.K))
        for k in range(self.K):
            if self.result_["precision"].ndim == 2:
                loglik[:, k] = np.log(self.result_["ratio"][k]) + self._logpdf(
                    test_X, self.result_["mean"][k, :],  np.diag(self.result_["precision"][k, :]))
            elif self.result_["precision"].ndim == 3:
                loglik[:, k] = np.log(self.result_["ratio"][k]) + self._logpdf(
                    test_X, self.result_["mean"][k, :],  self.result_["precision"][:, :, k])
            else:
                raise ValueError(
                    "Error precision, dimension of precision must be 2 or 3!")
        max_loglik = loglik.max(axis=1)
        norm_loglik = loglik - np.repeat(max_loglik, self.K).reshape(n, self.K)
        return (np.log(np.exp(norm_loglik).sum(axis=1)) + max_loglik).sum()

    def score_clustering(self, true_label_arg: np.ndarray):
        """
        Score the clustering distribution by 0-1 loss.
        Since label has degree to change the value itself,
        this function chooses the most fitted permutation.

        + Input:
            1. true_label_arg: label of true distribution
        + Output:
            1. max_correct_num: the best fitted 0-1 loss.
            2. max_perm: permutation that gives the best fitted one.
            3. max_est_label_arg: label number that gives the best fitted one.
        """
        check_is_fitted(self, "result_")
        K = len(self.result_["ratio"])

        est_label_prob = self.result_["u_xi"]
        target_label_arg = true_label_arg
        est_label_arg = np.argmax(est_label_prob, axis=1)

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

    def score_latent_kl(self, true_logp: np.ndarray):
        """
        Calculate the value of negative log posterior distribution of latent varialble -log(p(z|x,w)).
        """
        check_is_fitted(self, "result_")
        log_complete_likelihood = self.result_["h_xi"]
        (n, K) = log_complete_likelihood.shape
        min_K = np.array([K, true_logp.shape[1]]).min()

        max_log_complete_likelihood = log_complete_likelihood.max(axis=1)
        norm_log_complete_likelihood = log_complete_likelihood - \
            np.repeat(max_log_complete_likelihood, K).reshape(n, K)
        log_pred_p = norm_log_complete_likelihood - \
            np.log(np.repeat(np.exp(norm_log_complete_likelihood).sum(
                axis=1), K).reshape(n, K))

        min_kl = np.inf
        min_perm = -1
        min_log_pred_p = -1
        for perm in list(itertools.permutations(range(K), K)):
            permed_log_pred_p = log_pred_p.copy()
            for i in range(len(perm)):
                permed_log_pred_p[:, perm[i]] = log_pred_p[:, i]
            cluster_kl = (np.exp(
                true_logp[:, :min_K]) * (true_logp[:, :min_K] - permed_log_pred_p[:, :min_K])).sum()
            if cluster_kl < min_kl:
                min_kl = cluster_kl
                min_perm = perm
                min_log_pred_p = permed_log_pred_p
        return (min_kl, min_perm, min_log_pred_p)

    pass


class GaussianMixtureModelVB(AbstractMixtureModel, DensityMixin, BaseEstimator):
    """
    Gaussian Mixture with Variational Bayes.
    This class is created to compare with the HSMM, whereas sklearn has already implemented the estimator.
    """

    def __init__(self, K: int = 3,
                 pri_alpha: float = 0.1, pri_beta: float = 0.001, pri_gamma: float = 2, pri_delta: float = 2,
                 iteration: int = 1000, restart_num: int = 5, learning_seed: int = -1, method="diag",
                 tol: float = 1e-5, step: int = 20, is_trace=False):
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
        self.method = method
        self.tol = tol
        self.step = step
        self.is_trace = is_trace
        pass

    def fit(self, train_X: np.ndarray, y: np.ndarray = None):
        if self.method == "diag":
            return self._fit_diag(train_X, y)
        elif self.method == "full":
            return self._fit_full(train_X, y)
        # elif method == "single":
        #     return self._fit_signle(train_X, y)
        else:
            raise ValueError("""
                            Method name is not supported. Supported method are as follows:
                            1. diag: This is used for diagonal variance.
                            2. full: This is used for Full covariance.
                             """)
        pass

    # def _fit_single(self, train_X:np.ndarray, y:np.ndarray = None):

    def _fit_full(self, train_X, y: np.ndarray):
        (n, M) = train_X.shape
        if self.learning_seed > 0:
            np.random.seed(self.learning_seed)

        # Setting for static variable in the algorithm.
        expand_x = np.repeat(train_X, self.K).reshape(n, M, self.K).transpose(
            (0, 2, 1))  # n * K * M data with the same matrix among 2nd dimension

        min_energy = np.inf
        result = dict()

        pri_Sigma = self.pri_delta * np.eye(M)
        pri_inv_Sigma = 1 / self.pri_delta * np.eye(M)

        for restart in range(self.restart_num):

            energy = np.zeros(np.floor(self.iteration / self.step).astype(int))
            calc_ind = 0
            # Setting for initial value
            est_u_xi = np.random.dirichlet(alpha=np.ones(self.K), size=n)

            # Start learning.
            for ite in range(self.iteration):
                # Update posterior distribution of parameter.
                est_alpha = self.pri_alpha + est_u_xi.sum(axis=0)
                est_beta = self.pri_beta + est_u_xi.sum(axis=0)
                est_m = est_u_xi.T @ train_X / \
                    np.repeat(est_beta, M).reshape(self.K, M)
                est_gamma = self.pri_gamma + est_u_xi.sum(axis=0)
                est_delta = np.array([(np.repeat(est_u_xi[:, k], M).reshape(n, M) * train_X).T @ train_X - est_m[k, :].reshape(
                    M, 1) @ est_m[k, :].reshape(1, M) * est_beta[k] + pri_inv_Sigma for k in range(self.K)]).reshape(self.K, M, M).transpose((1, 2, 0))
                # est_inv_delta = np.array([np.linalg.inv(est_delta[:,:,k]) for k in range(self.K)]).reshape(self.K, M,M).transpose((1,2,0))

                # Update posterior distribution of latent variable
                est_h_xi = np.zeros((n, self.K))
                for k in range(self.K):
                    est_g_eta = (train_X - est_m[k, :]) * np.linalg.solve(
                        est_delta[:, :, k] / est_gamma[k], (train_X - est_m[k, :]).T).T + 1 / est_beta[k]
                    est_h_xi[:, k] = -M / 2 * np.log(2 * np.pi) + psi(est_alpha[k]) - psi(est_alpha.sum()) - est_g_eta.sum(
                        axis=1) / 2 + multipsi(est_gamma[k] / 2, M) / 2 + M / 2 * np.log(2) - np.linalg.slogdet(est_delta[:, :, k])[1] / 2
                    pass
                max_h_xi = est_h_xi.max(axis=1)
                norm_h_xi = est_h_xi - \
                    np.repeat(max_h_xi, self.K).reshape(n, self.K)
                est_u_xi = np.exp(
                    norm_h_xi) / np.repeat(np.exp(norm_h_xi).sum(axis=1), self.K).reshape(n, self.K)

                # Calculate evaluation function
                if ite % self.step == 0:
                    # Calculate evaluation function
                    energy[calc_ind] = -(np.log(np.exp(norm_h_xi).sum(axis=1)) +
                                         max_h_xi).sum() + (est_u_xi * est_h_xi).sum()
                    energy[calc_ind] += gammaln(est_alpha.sum()) - gammaln(
                        self.K * self.pri_alpha) + (-gammaln(est_alpha) + gammaln(self.pri_alpha)).sum()
                    energy[calc_ind] += (np.log(M *
                                                est_beta / self.pri_beta) / 2).sum()
                    energy[calc_ind] += (M * (self.pri_gamma - est_gamma) / 2 * np.log(2) + self.pri_gamma / 2 * np.log(M * self.pri_delta) + est_gamma / 2 * np.array(
                        [np.linalg.slogdet(est_delta[:, :, k])[1] for k in range(self.K)]) + multigammaln(self.pri_gamma / 2, M) - multigammaln(est_gamma / 2, M)).sum()
                    energy[calc_ind] += n * M * np.log(2 * np.pi) / 2
                    if self.is_trace:
                        print(energy[calc_ind])
                    if calc_ind > 0 and np.abs(energy[calc_ind] - energy[calc_ind - 1]) < self.tol:
                        energy = energy[:calc_ind]
                        break
                    calc_ind += 1
                    pass
                pass
            if self.is_trace:
                print(f"{energy[-1]} \n")
            if energy[-1] < min_energy:
                min_energy = energy[-1]
                result["ratio"] = est_alpha / est_alpha.sum()
                result["mean"] = est_m
                result["scale"] = np.array([est_delta[:, :, k] / est_gamma[k]
                                            for k in range(self.K)]).reshape(self.K, M, M).transpose((1, 2, 0))
                result["precision"] = np.array([np.linalg.inv(result["scale"][:, :, k]) for k in range(
                    self.K)]).reshape(self.K, M, M).transpose((1, 2, 0))
                result["alpha"] = est_alpha
                result["beta"] = est_beta
                result["mu"] = est_m
                result["gamma"] = est_gamma
                result["delta"] = est_delta
                result["h_xi"] = est_h_xi
                result["u_xi"] = est_u_xi
                result["energy"] = energy
            pass
        self.result_ = result

    def _fit_diag(self, train_X: np.ndarray, y: np.ndarray = None):
        (n, M) = train_X.shape
        if self.learning_seed > 0:
            np.random.seed(self.learning_seed)

        # Setting for static variable in the algorithm.
        expand_x = np.repeat(train_X, self.K).reshape(n, M, self.K).transpose(
            (0, 2, 1))  # n * K * M data with the same matrix among 2nd dimension

        min_energy = np.inf
        result = dict()

        for restart in range(self.restart_num):

            energy = np.zeros(np.floor(self.iteration / self.step).astype(int))
            calc_ind = 0
            # Setting for initial value
            # est_alpha = np.random.dirichlet(alpha = np.ones(self.K), size = 1)
            # est_m = np.random.normal(size = (self.K, M))
            # est_gamma = np.random.gamma(shape=1, size = (self.K, M))
            # est_delta = np.random.gamma(shape=1, size = (self.K, M))
            # est_beta = np.random.gamma(shape=1, size=(self.K, M))
            est_u_xi = np.random.dirichlet(alpha=np.ones(self.K), size=n)

            # Start learning.
            for ite in range(self.iteration):
                # Update posterior distribution of parameter.
                est_alpha = self.pri_alpha + est_u_xi.sum(axis=0)
                est_beta = np.repeat(
                    self.pri_beta + est_u_xi.sum(axis=0), M).reshape(self.K, M)
                est_m = est_u_xi.T @ train_X / est_beta
                est_gamma = np.repeat(
                    self.pri_gamma + est_u_xi.sum(axis=0) / 2, M).reshape(self.K, M)
                est_delta = self.pri_delta + \
                    est_u_xi.T @ (train_X**2) / 2 - est_beta / 2 * est_m**2

                # Update posterior distribution of latent variable
                est_g_eta = np.repeat(est_gamma / est_delta, n).reshape(self.K, M, n).transpose((2, 0, 1)) * (expand_x - np.repeat(
                    est_m, n).reshape(self.K, M, n).transpose((2, 0, 1)))**2 + 1 / np.repeat(est_beta, n).reshape(self.K, M, n).transpose((2, 0, 1))
                est_h_xi = -M / 2 * np.log(2 * np.pi) + np.repeat(psi(est_alpha) - psi(est_alpha.sum()) + (psi(
                    est_gamma) - np.log(est_delta)).sum(axis=1) / 2, n).reshape(self.K, n).T - est_g_eta.sum(axis=2) / 2
                max_h_xi = est_h_xi.max(axis=1)
                norm_h_xi = est_h_xi - \
                    np.repeat(max_h_xi, self.K).reshape(n, self.K)
                est_u_xi = np.exp(
                    norm_h_xi) / np.repeat(np.exp(norm_h_xi).sum(axis=1), self.K).reshape(n, self.K)

                # Calculate evaluation function
                if ite % self.step == 0:
                    # Calculate evaluation function
                    energy[calc_ind] = self._calc_obj_func(
                        est_u_xi=est_u_xi, est_h_xi=est_h_xi, est_alpha=est_alpha, est_beta=est_beta, est_gamma=est_gamma, est_delta=est_delta)
                    if self.is_trace:
                        print(energy[calc_ind])
                    if calc_ind > 0 and np.abs(energy[calc_ind] - energy[calc_ind - 1]) < self.tol:
                        energy = energy[:calc_ind]
                        break
                    calc_ind += 1
                    pass
                pass
            energy[-1] = self._calc_obj_func(est_u_xi=est_u_xi, est_h_xi=est_h_xi, est_alpha=est_alpha,
                                             est_beta=est_beta, est_gamma=est_gamma, est_delta=est_delta)

            if self.is_trace:
                print(energy[-1])
            if energy[-1] < min_energy:
                min_energy = energy[-1]
                result["ratio"] = est_alpha / est_alpha.sum()
                result["mean"] = est_m
                result["precision"] = est_gamma / est_delta
                result["scale"] = np.array(
                    [np.diag(est_delta[k, :] / est_gamma[k, :]) for k in range(self.K)])
                result["alpha"] = est_alpha
                result["beta"] = est_beta
                result["mu"] = est_m
                result["gamma"] = est_gamma
                result["delta"] = est_delta
                result["h_xi"] = est_h_xi
                result["u_xi"] = est_u_xi
                result["energy"] = energy
            pass
        self.result_ = result
        return self

    def _logpdf(self, x: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
        return multivariate_normal.logpdf(x, mean, cov=np.linalg.inv(precision))

    def _calc_obj_func(self, **kwargs) -> float:
        """
        -ELBO is calculated.
        + Necessary arguments are as follows:
            1. est_u_xi
            2. est_h_xi
            3. est_alpha
            4. est_beta
            5. est_gamma
            6. est_delta
        """
        est_u_xi = kwargs["est_u_xi"]
        est_h_xi = kwargs["est_h_xi"]
        est_alpha = kwargs["est_alpha"]
        est_beta = kwargs["est_beta"]
        est_gamma = kwargs["est_gamma"]
        est_delta = kwargs["est_delta"]

        n = est_h_xi.shape[0]
        M = est_delta.shape[1]

        max_h_xi = est_h_xi.max(axis=1)
        norm_h_xi = est_h_xi - np.repeat(max_h_xi, self.K).reshape(n, self.K)
        energy = -(np.log(np.exp(norm_h_xi).sum(axis=1)) +
                   max_h_xi).sum() + (est_u_xi * est_h_xi).sum()
        energy += gammaln(est_alpha.sum()) - gammaln(self.K * self.pri_alpha) + \
            (-gammaln(est_alpha) + gammaln(self.pri_alpha)).sum()
        if self.method == "diag":
            # energy =  -(np.log(np.exp(norm_h_xi).sum(axis = 1)) + max_h_xi).sum() + (est_u_xi * est_h_xi).sum()
            # energy += gammaln(est_alpha.sum()) - gammaln(self.K*self.pri_alpha) + (-gammaln(est_alpha) + gammaln(self.pri_alpha)).sum()
            energy += (np.log(est_beta / self.pri_beta) / 2 + est_gamma * np.log(est_delta) -
                       self.pri_gamma * np.log(self.pri_delta) - gammaln(est_gamma) + gammaln(self.pri_gamma)).sum()
        elif self.method == "full":
            energy += (np.log(M * est_beta / self.pri_beta) / 2).sum()
            energy += (M * (self.pri_gamma - est_gamma) / 2 * np.log(2) + self.pri_gamma / 2 * np.log(M * self.pri_delta) + est_gamma / 2 *
                       np.array([np.linalg(est_delta[:, :, k])[1] for k in self.K]) + multigammaln(self.pri_gamma / 2) - multigammaln(est_gamma / 2)).sum()
        energy += n * M * np.log(2 * np.pi) / 2

        return energy

    def get_params(self, deep=True):
        return{
            "K": self.K,
            "pri_alpha": self.pri_alpha,
            "pri_beta": self.pri_beta,
            "pri_gamma": self.pri_gamma,
            "pri_delta": self.pri_delta,
            "iteration": self.iteration,
            "restart_num": self.restart_num,
            "learning_seed": self.learning_seed,
            "tol": self.tol,
            "step": self.step
        }

    def set_params(self, **params):
        for params, value in params.items():
            setattr(self, params, value)
        return self
