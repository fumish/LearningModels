
seed.list <- list(input.seed = 402, output.seed = 805, learning.seed = 201)

## problem setting
source("logistic.R")
n <- 400
M <- 4
K0 <- 3

##generate parameter
# param.seed <- 3
# true.param <- param.mixture.norm.generate(M, K=K0, seed = param.seed)
true.ratio <- rep(1/K0,K0)
true.weight <- matrix(c(0,0,2,2,0,-1,1,0,-2,0,-2,2),nrow=M,ncol=K0)
true.param <- list(ratio=true.ratio, weight=true.weight)


##model settings
beta <- 0.05 ##inverse covariance
phi <- c(0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10)
# phi <- 0.1

K <- 5
learning.seed <- 12
init.df <- M+3
init.Sigma <- diag(M)
init.phi <- 1
iteration <- 1000
restart <- 10
update.order <- c(lva.update.parameters, lva.update.latent.variable, lva.update.auxilirary.variable)

dataset.num <- 20

result.list <- list()
# energies <- matrix(0, nrow=length(phi), ncol=iteration)
energies <- matrix(0, nrow = length(phi), ncol=dataset.num)
for(a.dataset.num in 1:dataset.num){
  a.seed.list <- lapply(seed.list, function(x)(x+a.dataset.num))
  
  ##input
  xrange <- c(-5,5)
  x <- input.unif.generate(n,M,a.seed.list$input.seed, xrange)
  
  ##output
  output.info <- output.mixture.logistic.generate(n, x, true.param, seed=a.seed.list$output.seed)
  y <- output.info$output
  label <- output.info$label
  
  for(i in 1:length(phi)){
    i.prior.hyperparameter <- list(phi=phi[i], beta=beta)
    
    lva.result <- lva.estimation.main(update.order = update.order,
                                      x = x,
                                      y = y,
                                      K = K,
                                      prior.hyperparameter = i.prior.hyperparameter,
                                      init.Sigma = init.Sigma,
                                      init.phi = init.phi,
                                      init.df = init.df,                                
                                      iteration = iteration,
                                      restart = restart,                                
                                      learning.seed = a.seed.list$learning.seed,
                                      trace.on = F)
    result.list <- c(result.list, list(lva.result))
    
    energies[i, a.dataset.num] <- lva.result$energy.trace
  }
}

matplot.x <- matrix(rep(phi,dataset.num),nrow=length(phi), ncol=dataset.num)
matplot(x = matplot.x, y = energies, "l")

mean.energy <- rowMeans(energies)
plot(x = phi, y = mean.energy, "b")
