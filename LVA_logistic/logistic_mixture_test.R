

## problem setting
source("logistic.R")
n <- 400
M <- 4
K0 <- 3

##input
input.seed <- 5
xrange <- c(-5,5)
x <- input.unif.generate(n,M,input.seed, xrange)

##generate parameter
param.seed <- 3
# true.param <- param.mixture.norm.generate(M, K=K0, seed = param.seed)
true.ratio <- rep(1/K0,K0)
true.weight <- matrix(c(0,0,2,2,0,-1,1,0,-2,0,-2,2),nrow=M,ncol=K0)
true.param <- list(ratio=true.ratio, weight=true.weight)

##output
output.seed <- 7
output.info <- output.mixture.logistic.generate(n,x,true.param,seed=output.seed)
y <- output.info$output
label <- output.info$label

##model settings
beta <- 0.05 ##inverse covariance
phi <- 0.1
K <- 5
learning.seed <- 15
init.mean <- 0
init.sd <- 1
init.phi <- 1
iteration <- 2000

result <- LVA.mixture.logistic.regression.debug(x,y,K,beta = beta, phi = phi, iteration=iteration, seed=learning.seed,
                                          init.mean=init.mean, init.sd=init.sd, init.phi = init.phi)

