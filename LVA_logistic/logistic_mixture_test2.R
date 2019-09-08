

## problem setting
source("logistic.R")
n <- 4000
M <- 4
K0 <- 3

##input
input.seed <- 2
xrange <- c(-5,5)
x <- input.unif.generate(n,M,input.seed, xrange)

##generate parameter
param.seed <- 3
# true.param <- param.mixture.norm.generate(M, K=K0, seed = param.seed)
true.ratio <- rep(1/K0,K0)
true.weight <- matrix(c(0,0,2,2,0,-1,1,0,-2,0,-2,2),nrow=M,ncol=K0)
true.param <- list(ratio=true.ratio, weight=true.weight)

##output
output.seed <- 17
output.info <- output.mixture.logistic.generate(n,x,true.param,seed=output.seed)
y <- output.info$output
label <- output.info$label

##model settings
beta <- 0.05 ##inverse covariance
phi <- 0.1
K <- 5
learning.seed <- 10
init.mean <- 0
init.sd <- 1
init.phi <- 1
iteration <- 1000

## test setting
test.num <- 10000
test.seed <- output.seed*2


result <- LVA.mixture.logistic.regression.analysis(x,y,K, true.param,beta = beta,iteration=iteration, seed=learning.seed,
                                                init.mean=init.mean, init.sd=init.sd, init.phi = init.phi,
                                                test.num=test.num, test.seed=test.seed)


## training error
# print(crossentropy(x,y,result))

## generalization error