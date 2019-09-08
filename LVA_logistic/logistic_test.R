

## problem setting
source("logistic.R")
n <- 100
M <- 5

##input
input.seed <- 2
xrange <- c(-5,5)
x <- input.unif.generate(n,M,input.seed, xrange)

##generate parameter
# param.seed <- 2
# true.param <- param.norm.generate(M,seed = param.seed)
true.param <- c(0,0,0,0,0)

##output
output.seed <- 4
y <- output.logistic.generate(n,x,true.param,seed=output.seed)

##model settings
beta <- 0.01 ##inverse covariance
learning.seed <- 4
init.mean <- 0
init.sd <- 1
iteration <- 100

result <- LVA.logistic.regression(x,y,beta = beta,iteration=iteration, seed=learning.seed, init.mean=init.mean, init.sd=init.sd)
