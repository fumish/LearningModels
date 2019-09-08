library(rstan)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

rdirichlet <- function(phi) {
  numerator <- rgamma(n = length(phi), shape=phi, rate=1)
  return(numerator / sum(numerator))
}

seed.list <- list(true.seed = 72, sample.seed = 251, learning.seed = 25, test.seed = 50)

library(MASS)
source("logistic_library.R")

## problem setting
M <- 3 ## data dimension, including bias term
n <- 1000 # sample number

## true setting
K0 <- 5 # number of true componenet
set.seed(seed.list$true.seed)
true.ratio <- round(rdirichlet(rep(10,K0)),digits = 2)
true.ratio[K0] <- 1-sum(true.ratio[1:(K0-1)])
true.weight <- round(matrix(5*rnorm(M*K0), nrow=M, ncol=K0), digits = 1)
true.param <- list(ratio=true.ratio, weight=true.weight)

input.preprocessor = input.preprocessing.quad
base.dim = 1
training.info <- rlogistic.mixture(n=n, base.dim=base.dim, x.range=c(-5,5),
                  input.preprocessor=input.preprocessor,
                  param = true.param, seed = seed.list$sample.seed)

training.x <- training.info$x
training.y <- training.info$y

####### learning setting
K <- 8 # number of learning component
## hyperparameter
phi <- 0.1 ## hyperparameter for dirichlet distribution
beta <- 0.011 ## hyperparameter for normal distribution
prior.hyperparameter <- list(phi=phi, beta=beta)

## initializing relavance
init.phi <- 1
init.df <- 10
init.Sigma <- diag(M)

## framework relavance
restart.lva <- 5
restart.em <- 5
tol <- 1e-5 ## convergence tolerance
iteration.lva <- 1000 ## number of learning
iteration.em <- 1000
update.order <- c(lva.update.auxilirary.variable.fast, lva.update.parameters.fast, lva.update.latent.variable.fast)


###### Learning phase

#### learning by stan
LRMM.stan.model <- stan_model(file="LRMM.stan")
LRMM.stan.data <- list(n = n, M=M, K=K, x=training.x, y=training.y, alpha=prior.hyperparameter$phi, beta=prior.hyperparameter$beta)
LRMM.stan.vb <- vb(LRMM.stan.model, data=LRMM.stan.data,
                   seed = seed.list$learning.seed)
est.LRMM.stan.vb <- get_posterior_mean(LRMM.stan.vb)
est.LRMM.stan.ratio <- est.LRMM.stan.vb[1:K]
est.LRMM.stan.weight <- matrix(est.LRMM.stan.vb[(K+1):(K+K*M)], nrow=K, ncol=M, byrow=T)
stan.vb.mean.param <- list(ratio=est.LRMM.stan.ratio, weight=t(est.LRMM.stan.weight))

#### learning by lva
lva.batch.result <- lva.estimation.main(update.order = update.order,
                                        x = training.x,
                                        y = training.y,
                                        K = K,
                                        prior.hyperparameter = prior.hyperparameter,
                                        init.Sigma = init.Sigma,
                                        init.phi = init.phi,
                                        init.df = init.df,
                                        iteration = iteration.lva,
                                        restart = restart.lva,
                                        learning.seed = seed.list$learning.seed,
                                        trace.on = F)
lva.mean.param <- list(ratio=lva.batch.result$mean.estimator$ratio, weight=lva.batch.result$mean.estimator$weight)

#### test data
input.preprocessor = input.preprocessing.quad
base.dim = 1
test.num <- 10000
test.info <- rlogistic.mixture(n=test.num, base.dim=base.dim, x.range=c(-5,5),
                               input.preprocessor=input.preprocessor,
                               param = true.param, seed = seed.list$test.seed,
                               is.label = FALSE)
test.x <- test.info$x
test.y <- test.info$y
test.prob.y <- test.info$prob.y

func.entropy <- function(x,y) -mean(log(dlogistic.mixture(x,y,true.param)))
entropy <- func.entropy(test.x, test.y)
lva.gerror <- pr.generalization.error(test.x, test.y, pred.func.lva) - entropy
stan.vb.gerror <- pr.generalization.error(test.x, test.y, pred.func.stan.vb) - entropy


pred.func.lva <- function(x) prediction.logistic.mixture.em(x, lva.mean.param)
lva.pred.y <- pred.func.lva(test.x)

pred.func.stan.vb <- function(x) prediction.logistic.mixture.em(x, stan.vb.mean.param)
vb.stan.pred.y <- pred.func.stan.vb(test.x)

sort.test.x2 <- order(test.x[,2])

matplot(x = data.frame(test.x[sort.test.x2,2], test.x[sort.test.x2,2], test.x[sort.test.x2,2]),
        y = data.frame(test.prob.y[sort.test.x2], pred.func.lva(test.x)[sort.test.x2],
                       pred.func.stan.vb(test.x)[sort.test.x2]),
        col = c("black", "red", "green"),
        pch = c(15,0), cex = 2, lwd=3, lty=c(1,2,4),
        type = "l", xlab = "x_1", ylab="p(y=1|x_0,x_1,x_2)")
legend("bottomleft", legend=c("true",
                              paste("LVA batch error:", as.character(round(lva.gerror, digits = 4)), sep=""),
                              paste("Stan VB:", as.character(round(stan.vb.gerror, digits = 4)), sep="")),
       col=c("black", "red", "green"),
       lwd=3,
       lty=c(1,2,4))

# 
# pred.func.batch.lva <- function(x) prediction.logistic.mixture.fast(x,lva.batch.result)
