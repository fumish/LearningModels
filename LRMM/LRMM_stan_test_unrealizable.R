library(rstan)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

rdirichlet <- function(phi) {
  numerator <- rgamma(n = length(phi), shape=phi, rate=1)
  return(numerator / sum(numerator))
}

seed.list <- list(true.seed = 72, sample.seed = 210, learning.seed = 30, test.seed = 400)

library(MASS)
source("logistic_library.R")

## problem setting
M <- 3 ## data dimension, including bias term
n <- 4000 # sample number

## true setting
range.func <- function(x, min=0, max=1) as.numeric(min < x & x < max)
dprob <- function(x) sin(0.25*abs(x))
# dprob <- function(x) 0.8*(sin(0.5*x)+1)/2
# dprob.v5 <- function(x) (sin(0.5*x)+1)/2
# dprob <- function(x) 0.4*(sin(x)+1)/2
dprob.v4 <- function(x) (0.6*(range.func(x, min=-7, max=-3) + range.func(x, min=3, max=7)) + 0.2 ) 
dprob.v3 <- function(x) (range.func(x, min=-10, max=-6)/2 + range.func(x, min=-2, max=2)/2 + range.func(x, min=6, max=10)/2)
dprob.v2 <- function(x)  ifelse((-1 < x & x < 1), 0.5 , ((sin(10/x )+1)/2))
dprob.v1 <- function(x)  abs(sin(3*x/4)/sqrt(abs(x)))
## sample data generation
set.seed(seed=seed.list$sample.seed)
#input
x.range <- c(-10,10)
training.x <- matrix(0, nrow=n, ncol=M)
training.x[,1] <- 1
training.x[,2] <- runif(n, min = x.range[1], max = x.range[2])
training.x[,3] <- training.x[,2]^2
training.prob.y <- dprob(training.x[,2])
training.y <- as.numeric(runif(n) < training.prob.y)

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

em.result <- lmm.em(x = training.x,
                    y = training.y,
                    K = K,
                    iteration = iteration.em,
                    restart = restart.em,
                    learning.seed = seed.list$learning.seed,
                    stan.filename = "LMM_generalization_honban.stan")

#### test data
set.seed(seed=(seed.list$test.seed))
test.num <- 10000
test.x <- matrix(0, nrow=test.num, ncol = M)
test.x[,1] <- rep(1,test.num)
test.x[,2] <- runif(test.num, min = x.range[1], max = x.range[2])
test.x[,3] <- test.x[,2]^2
test.prob.y <- dprob(test.x[,2])
test.y <- as.numeric(runif(test.num) < test.prob.y)

func.entropy <- function(x,y) -mean(ifelse(y==1,log(dprob(x[,2])), log(1-dprob(x[,2]))))
entropy <- func.entropy(test.x, test.y)  

pred.func.lva <- function(x) prediction.logistic.mixture.em(x, lva.mean.param)
lva.pred.y <- pred.func.lva(test.x)

pred.func.stan.vb <- function(x) prediction.logistic.mixture.em(x, stan.vb.mean.param)
vb.stan.pred.y <- pred.func.stan.vb(test.x)

pred.func.em <- function(x) prediction.logistic.mixture.em(x, em.result)
em.pred.y <- pred.func.em(test.x)

lva.gerror <- pr.generalization.error(test.x, test.y, pred.func.lva) - entropy
stan.vb.gerror <- pr.generalization.error(test.x, test.y, pred.func.stan.vb) - entropy
em.gerror <- pr.generalization.error(test.x, test.y, pred.func.em) - entropy

sort.test.x2 <- order(test.x[,2])

par(cex.axis=1.25)
matplot(x = data.frame(test.x[sort.test.x2,2]),
        y = data.frame(test.prob.y[sort.test.x2],
                       pred.func.lva(test.x)[sort.test.x2],
                       pred.func.stan.vb(test.x)[sort.test.x2],
                       pred.func.em(test.x)[sort.test.x2]),
        col = c("black", "red", "green", "blue"),
        pch = c(15,0), cex = 4, lwd=4.5, lty=c(1,2,4,8),
        xlim=c(x.range[1], x.range[2]), ylim=c(0,1),
        type = "l", xlab = "x_1", ylab="p(y=1|x_0,x_1,x_2)")


legend("bottomleft", legend=c("true",
                              paste("LVA batch error:", as.character(round(lva.gerror, digits = 4)), sep=""),
                              paste("Stan VB:", as.character(round(stan.vb.gerror, digits = 4)), sep=""),
                              paste("EM:", as.character(round(em.gerror, digits = 4)), sep="")),
       col=c("black", "red", "green", "blue"),
       lwd=3,
       lty=c(1,2,4))

# matplot(x = data.frame(test.x[sort.test.x2,2]),
#         y = data.frame(test.prob.y[sort.test.x2],
#                        pred.func.lva(test.x)[sort.test.x2],
#                        pred.func.stan.vb(test.x)[sort.test.x2]),
#         col = c("black", "red", "green"),
#         pch = c(15,0), cex = 2, lwd=3, lty=c(1,2,4,8),
#         xlim=c(x.range[1], x.range[2]), ylim=c(0,1),
#         type = "l", xlab = "x_1", ylab="p(y=1|x_0,x_1,x_2)")
# legend("topleft", legend=c("true",
#                            paste("LVA batch error:", as.character(round(lva.gerror, digits = 4)), sep=""),
#                            paste("Stan VB:", as.character(round(stan.vb.gerror, digits = 4)), sep="")),
#        col=c("black", "red", "green", "blue"),
#        lwd=3,
#        lty=c(1,2,4))

# 
# pred.func.batch.lva <- function(x) prediction.logistic.mixture.fast(x,lva.batch.result)
