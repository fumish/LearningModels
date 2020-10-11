"""
Algorithm for linear regression based
on Variational Bayes is defined in this module.

Formulation:
At first, variational bayes method approximates
a Bayesian posterior distribution based on their objective fuction F:

F = int q(w) log frac{q(w)}{p(y|x,w) p(w)},
where q(w) is the variational bayes posterior distribution,
p(y|x,w) is likelihood, p(w) is prior distribution.

Note that many paper gives the objective function as the
Evidence Lower Bound(ELBO), and it's same as -F.
In order to correspond to KL minimization, we use the F as the objective.

When we constaraint the form of q(w) as a normal distribution with
mean mu and covariance sigma, the task of here is to find the minimum F
in terms of mu and sigma.

Note that although log p(w) has non-differential points,
in some cases, the problem can be solved by E_q[log p(w)].
When the prior distribution is the Laplace distribution,
E_q[log p(w)] eliminates the non-differential point.

Moreover, when we apply the natural gradient method in this task,
we can give an very efficient algorithm.


Component of this module:
1. class VBLaplace uses a Laplace distribution for p(w)
2. class VBNormal uses a normal distribution for p(w)
3. class VBApproxLaplace uses an approximated Laplace distribution for p(w)
by the normal distribution.

"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class VBLaplace(BaseEstimator, RegressorMixin):
    def __init__(
        self, pri_beta: float = 20, pri_opt_flag: bool = True,
        seed: int = -1, iteration: int = 1000,
        tol: float = 1e-8, step: float = 0.1,
        is_trace: bool = False, trace_step: int = 20
    ):
        """
        VB algorithm with the Laplace distribution as the prior distribution.
        Since F is not closed-form, natural gradient method is used.

        + Input:
            1. pri_beta: hyperparameter of the prior distributuion
                -> p(w) propto prod_{j=1}^M exp(-1/pri_beta |w_j|)

            2. pri_opt_flag: whether the pri_beta
            is optimized in terms of F or not.

            3. seed: whether the inialization is conducted by same seed or not.

            4. iteration: # of iteration for gradient method.

            5. tol: tolerance, i.e. if |F_t - F_{t+1}| < tol,
            then calculation is finished.

            6. step: frequency of print, Only valid when is_trace is True.

            7. is_trace: whether the result is printed or not.
                -> square of derivative for F and F itself is printed out.
        """
        self.pri_beta = pri_beta
        self.pri_opt_flag = pri_opt_flag
        self.seed = seed
        self.iteration = iteration
        self.tol = tol
        self.step = step
        self.is_trace = is_trace
        self.trace_step = trace_step
        pass

    def _initialization(self, M: int):
        seed = self.seed

        if seed > 0:
            np.random.seed(seed)

        mean = np.random.normal(size=M)
        sigma = invwishart.rvs(df=M + 2, scale=np.eye(M), size=1)
        pri_beta = np.random.gamma(
            shape=3, size=1) if self.pri_opt_flag else self.pri_beta

        self.mean_ = mean
        self.sigma_ = sigma
        self.pri_beta_ = pri_beta
        pass

    def _obj_func(self, X: np.ndarray, y: np.ndarray, pri_beta: float,
                  mean: np.ndarray, sigma: np.ndarray) -> float:
        """
        Calculate objective function.

        + Input:
            1. X: input matrix (n, M) matrix
            2. y: output vector (n, ) matrix
            3. mean: mean parameter of vb posterior
            4. sigma: covariance matrix of vb posterior

        + Output:
            value of the objective function.

        """

        n, M = X.shape

        sq_sigma_diag = np.sqrt(np.diag(sigma))
        log_2pi = np.log(2 * np.pi)

        F = 0
        # const values
        F += -M / 2 * log_2pi - M / 2 + M * log_2pi + \
            n * M / 2 * log_2pi + M * np.log(2 * pri_beta)

        F += ((y - X @ mean)**2).sum() / 2 - \
            np.linalg.slogdet(sigma)[1] / 2 + np.trace(X.T @ X @ sigma) / 2

        # term obtained from laplace prior
        F += ((mean + 2 * sq_sigma_diag * norm.pdf(-mean / sq_sigma_diag) -
               2 * mean * norm.cdf(-mean / sq_sigma_diag)) / pri_beta).sum()

        return F

    def fit(self, train_X: np.ndarray, train_Y: np.ndarray):
        pri_beta = self.pri_beta
        iteration = self.iteration
        step = self.step
        tol = self.tol

        is_trace = self.is_trace
        trace_step = self.trace_step

        M = train_X.shape[1]

        if not hasattr(self, "mean_"):
            self._initialization(M)

        est_mean = self.mean_
        est_sigma = self.sigma_
        est_pri_beta = self.pri_beta_

        # transformation to natural parameter
        theta1 = np.linalg.solve(est_sigma, est_mean)
        theta2 = -np.linalg.inv(est_sigma) / 2

        F = []

        cov_X = train_X.T @ train_X
        cov_YX = train_Y @ train_X
        for ite in range(iteration):
            sq_sigma_diag = np.sqrt(np.diag(est_sigma))

            # update mean and sigma by natural gradient
            mean_sigma_ratio = est_mean / sq_sigma_diag
            dFdnu1 = theta1 - cov_YX
            dFdnu1 += (1 - 2 * mean_sigma_ratio * norm.pdf(-mean_sigma_ratio) -
                       2 * norm.cdf(-mean_sigma_ratio)) / est_pri_beta
            dFdnu2 = theta2 + cov_X / 2
            dFdnu2[np.diag_indices(M)] += 1 / sq_sigma_diag * \
                norm.pdf(-mean_sigma_ratio) / est_pri_beta

            theta1 += -step * dFdnu1
            theta2 += -step * dFdnu2
            est_sigma = -np.linalg.inv(theta2) / 2
            est_mean = est_sigma @ theta1

            # update pri_beta by extreme value
            sq_sigma_diag = np.sqrt(np.diag(est_sigma))
            mean_sigma_ratio = est_mean / sq_sigma_diag
            est_pri_beta = ((est_mean +
                             2 * sq_sigma_diag * norm.pdf(-mean_sigma_ratio)
                             - 2 * est_mean * norm.cdf(-mean_sigma_ratio))).mean() if self.pri_opt_flag else pri_beta
            current_F = self._obj_func(
                train_X, train_Y, est_pri_beta, est_mean, est_sigma)
            if is_trace and ite % trace_step == 0:
                print(current_F, (dFdnu1**2).sum(), (dFdnu2**2).sum())

            if ite > 0 and np.abs(current_F - F[ite - 1]) < tol:
                if is_trace:
                    print(current_F, (dFdnu1**2).sum(), (dFdnu2**2).sum())
                break
            else:
                F.append(current_F)
            pass

        self.F_ = F
        self.mean_ = est_mean
        self.sigma_ = est_sigma
        self.pri_beta_ = est_pri_beta

        return self
        pass

    def predict(self, test_X: np.ndarray):
        if not hasattr(self, "mean_"):
            raise ValueError(
                "fit has not finished yet, should fit before predict.")
        return test_X @ self.mean_
        pass

    pass


class VBNormal(BaseEstimator, RegressorMixin):
    def __init__(
        self, pri_beta: float = 20, pri_opt_flag: bool = True,
        seed: int = -1, iteration: int = 1000,
        tol: float = 1e-8, step: float = 0.1,
        is_trace: bool = False, trace_step: int = 20
    ):
        """
        VB algorithm with the normal distribution as the prior distribution.
        Since F is closed-form, alternative optimization method is used
        when pri_beta is optmized.

        + Input:
            1. pri_beta: hyperparameter of the prior distributuion
                -> p(w) propto prod_{j=1}^M exp(-1/pri_beta |w_j|)

            2. pri_opt_flag: whether the pri_beta
            is optimized in terms of F or not.

            3. seed: whether the inialization is conducted by same seed or not.

            4. iteration: # of iteration for gradient method.

            5. tol: tolerance, i.e. if |F_t - F_{t+1}| < tol,
            then calculation is finished.

            6. step: frequency of print, Only valid when is_trace is True.

            7. is_trace: whether the result is printed or not.
                -> square of derivative for F and F itself is printed out.
        """
        self.pri_beta = pri_beta
        self.pri_opt_flag = pri_opt_flag
        self.seed = seed
        self.iteration = iteration
        self.tol = tol
        self.step = step
        self.is_trace = is_trace
        self.trace_step = trace_step
        pass

    def _initialization(self, M: int):
        seed = self.seed

        if seed > 0:
            np.random.seed(seed)

        mean = np.random.normal(size=M)
        sigma = invwishart.rvs(df=M + 2, scale=np.eye(M), size=1)
        pri_beta = np.random.gamma(
            shape=3, size=1) if self.pri_opt_flag else self.pri_beta

        self.mean_ = mean
        self.sigma_ = sigma
        self.pri_beta_ = pri_beta
        pass

    def _obj_func(self, X: np.ndarray, y: np.ndarray, pri_beta: float,
                  mean: np.ndarray, sigma: np.ndarray) -> float:
        """
        Calculate objective function.

        + Input:
            1. X: input matrix (n, M) matrix
            2. y: output vector (n, ) matrix
            3. mean: mean parameter of vb posterior
            4. sigma: covariance matrix of vb posterior

        + Output:
            value of the objective function.

        """

        n, M = X.shape

        log_2pi = np.log(2 * np.pi)

        F = 0
        # const values
        F += -M / 2 * log_2pi - M / 2 + M * log_2pi + \
            n * M / 2 * log_2pi + M * np.log(2 * pri_beta)

        F += ((y - X @ mean)**2).sum() / 2 - \
            np.linalg.slogdet(sigma)[1] / 2 + np.trace(X.T @ X @ sigma) / 2

        # term obtained from Normal prior
        F += pri_beta / 2 * (mean @ mean + np.trace(sigma)) - \
            M / 2 * np.log(pri_beta) + M / 2 * log_2pi

        return F

    def fit(self, train_X: np.ndarray, train_Y: np.ndarray):
        pri_beta = self.pri_beta
        iteration = self.iteration
        step = self.step
        tol = self.tol

        is_trace = self.is_trace
        trace_step = self.trace_step

        M = train_X.shape[1]

        if not hasattr(self, "mean_"):
            self._initialization(M)

        est_mean = self.mean_
        est_sigma = self.sigma_
        est_pri_beta = self.pri_beta_

        F = []
        XY_cov = train_Y @ train_X
        X_cov = train_X.T @ train_X

        for ite in range(iteration):
            sigma_inv = X_cov + est_pri_beta * np.eye(M)
            est_mean = np.linalg.solve(sigma_inv, XY_cov)
            est_sigma = np.linalg.inv(sigma_inv)

            # update pri_beta by extreme value
            est_pri_beta = M / \
                (est_mean @ est_mean + np.trace(est_sigma)
                 ) if self.pri_opt_flag else pri_beta
            current_F = self._obj_func(
                train_X, train_Y, est_pri_beta, est_mean, est_sigma)
            if is_trace and ite % trace_step == 0:
                print(current_F, (dFdnu1**2).sum(), (dFdnu2**2).sum())

            if ite > 0 and np.abs(current_F - F[ite - 1]) < tol:
                if is_trace:
                    print(current_F, (dFdnu1**2).sum(), (dFdnu2**2).sum())
                break
            else:
                F.append(current_F)
            pass

        self.F_ = F
        self.mean_ = est_mean
        self.sigma_ = est_sigma
        self.pri_beta_ = est_pri_beta

        return self
        pass

    def predict(self, test_X: np.ndarray):
        if not hasattr(self, "mean_"):
            raise ValueError(
                "fit has not finished yet, should fit before predict.")
        return test_X @ self.mean_
        pass

    pass


class VBApproxLaplace(BaseEstimator, RegressorMixin):
    def __init__(
        self, pri_beta: float = 20, pri_opt_flag: bool = True,
        seed: int = -1, iteration: int = 1000,
        tol: float = 1e-8, step: float = 0.1,
        is_trace: bool = False, trace_step: int = 20
    ):
        """
        Laplace prior is approximated by normal distribution,
        and approximated posterior distribution is
        obtained by the approximated laplace prior.

        This is originated from M. Seeger.,
        "Bayesian Inference and Optimal Design in the Sparse Linear Model",
        2012.

        """
        self.pri_beta = pri_beta
        self.pri_opt_flag = pri_opt_flag
        self.seed = seed
        self.iteration = iteration
        self.tol = tol
        self.step = step
        self.is_trace = is_trace
        self.trace_step = trace_step
        pass

    def _initialization(self, M: int):
        seed = self.seed

        if seed > 0:
            np.random.seed(seed)

        mean = np.random.normal(size=M)
        sigma = invwishart.rvs(df=M + 2, scale=np.eye(M), size=1)
        pri_beta = np.random.gamma(
            shape=3, size=1) if self.pri_opt_flag else self.pri_beta

        self.mean_ = mean
        self.sigma_ = sigma
        self.pri_beta_ = pri_beta
        pass

    def _obj_func(self, y: np.ndarray, pri_beta: float,
                  mean: np.ndarray, inv_sigma: np.ndarray,
                  h_xi: np.ndarray, v_xi: np.ndarray) -> float:
        """
        Calculate objective function.

        + Input:
            1. X: input matrix (n, M) matrix
            2. y: output vector (n, ) matrix
            3. pri_beta: hyperparameter of laplace prior distribution
            4. mean: mean parameter of vb posterior
            5. inv_sigma: covariance inverse matrix of vb posterior
            6. h_xi: complementary elements to approximate the Laplace prior
            7. v_xi: Complementary elements to approximate the Laplace prior

        + Output:
            value of the objective function.

        """

        F = 0
        F += pri_beta / 2 * np.sqrt(h_xi).sum() + \
            v_xi @ h_xi - M * np.log(pri_beta / 2)
        F += n / 2 * np.log(2 * np.pi) + train_Y @ train_Y / 2 - \
            mean @ (inv_sigma @ mean) / 2 + np.linalg.slogdet(inv_sigma)[0] / 2
        return F

    def fit(self, train_X: np.ndarray, train_Y: np.ndarray):
        iteration = self.iteration
        step = self.step
        tol = self.tol

        is_trace = self.is_trace
        trace_step = self.trace_step

        M = train_X.shape[1]

        if not hasattr(self, "mean_"):
            self._initialization(M)

        est_mean = self.mean_
        est_sigma = self.sigma_
        est_pri_beta = self.pri_beta_

        F = []
        X_cov = train_X.T @ train_X
        XY_cov = train_X.T @ train_Y

        for ite in range(iteration):
            # update form of approximated laplace prior
            est_h_xi = est_mean**2 + np.diag(est_sigma)
            est_v_xi = -est_pri_beta / 2 / np.sqrt(est_h_xi)

            # update posterior distribution
            inv_sigma = X_cov - 2 * np.diag(est_v_xi)
            est_mean = np.linalg.solve(inv_sigma, XY_cov)
            est_sigma = np.linalg.inv(inv_sigma)

            # update pri_beta by extreme value
            est_pri_beta = M / ((est_mean**2 + np.diag(est_sigma)) /
                                (2 * np.sqrt(est_h_xi))).sum() if self.pri_opt_flag else pri_beta

            current_F = self._obj_func(
                train_Y, est_pri_beta, est_mean, inv_sigma, est_h_xi, est_v_xi)
            if is_trace and ite % trace_step == 0:
                print(current_F)

            if ite > 0 and np.abs(current_F - F[ite - 1]) < tol:
                if is_trace:
                    print(current_F, (dFdnu1**2).sum(), (dFdnu2**2).sum())
                break
            else:
                F.append(current_F)
            pass

        self.F_ = F
        self.mean_ = est_mean
        self.sigma_ = est_sigma
        self.pri_beta_ = est_pri_beta

        return self
        pass

    def predict(self, test_X: np.ndarray):
        if not hasattr(self, "mean_"):
            raise ValueError(
                "fit has not finished yet, should fit before predict.")
        return test_X @ self.mean_
        pass

    pass
