input.unif.generate <- function(n,M,seed=1, xrange=c(0,1)){
  set.seed(seed)
  xrange <- c(-5,5)
  x <- matrix(runif(n*M,min=xrange[1],max=xrange[2]), nrow = n, ncol=M)
  return(x)
}

param.norm.generate <- function(M,seed=1,mean=0,sd=1){
  set.seed(seed)
  true.param <- rnorm(M,mean = mean, sd = sd)
  return(true.param)
}

param.mixture.norm.generate <- function(M, K, seed=1, mean=0, sd=1, phi=1){
  set.seed(seed)
  tmp <- rgamma(K, shape=phi)
  ratio <- tmp/sum(tmp)
  
  weight <- matrix(rnorm(M*K, mean=mean, sd=sd), nrow = M, ncol=K)
  return(list(ratio=ratio, weight=weight))
}

output.logistic.generate <- function(n,x,param,seed=1){
  set.seed(seed)
  # y <- numeric(n)
  
  p <- 1/(1 + exp(-x %*% param))
  random.val <- runif(n)
  # print(data.frame(p,random.val,random.val<p))
  return(as.integer(random.val < p))
  
}

output.mixture.logistic.generate <- function(n, x, param, seed=1){
  set.seed(seed)
  
  y <- numeric(n)
  label.realization <- rmultinom(n, size=1, prob=param$ratio)
  label <- apply(label.realization, 2, which.max)
  logit.prob <- numeric(n)
  for(i in 1:n){
    i.weight <- param$weight[,label[i]]
    i.p <- 1/(1+exp(-t(x[i,])%*%i.weight))
    logit.prob[i] <- i.p
    random.val <- runif(1)
    y[i] <- as.integer(random.val < i.p)
  }
  
  return(list(output=y, label=label, logit.prob=logit.prob))
}

LVA.logistic.regression <- function(x,y,beta = 0.001,iteration=100, seed=1, init.mean=0, init.sd=1){
  set.seed(seed)
  
  n <- nrow(x)
  M <- ncol(x)
  
  ##initial setting
  h.ksi <- rnorm(n,mean=init.mean,sd=init.sd)^2
  
  for(ite in 1:iteration){
    ##update parameter
    est.beta <- t((tanh(sqrt(h.ksi)/2)/(2*sqrt(h.ksi)))*x)%*%x+beta*eye(M)
    est.mean <- solve(est.beta) %*% apply((y-0.5)*x,2,sum)
    
    ##update latent variable
    h.ksi <- diag(x %*% (solve(est.beta)+est.mean%*%t(est.mean)) %*% t(x))
    
    ##calculate energy
    energy <- as.numeric(-(determinant(beta*eye(M),logarithm = T)$modulus-determinant(est.beta,logarithm = T)$modulus)/2-
      t(est.mean)%*%est.beta%*%est.mean/2+n*log(2)+
      sum(log(cosh(sqrt(h.ksi)/2))-tanh(sqrt(h.ksi)/2)/(4*sqrt(h.ksi))*h.ksi))
    
    print(energy)
  }
  return(list(param = est.mean, beta = est.beta, energy=energy))
}

LVA.mixture.logistic.regression <- function(x,y,K,beta = 0.001, phi=1,
                                    iteration=100, seed=1,
                                    init.mean=0, init.sd=1, init.phi=1){
  set.seed(seed)
  
  n <- nrow(x)
  M <- ncol(x)
  
  ##initial setting
  g.ksi <- matrix(rnorm(n*K,mean=init.mean,sd=init.sd)^2,nrow=n,ncol=K)
  label.prob <- matrix(0,nrow = n, ncol=K)
  for(i in 1:n){
    label.prob[i,] <- rdirichlet(rep(init.phi, K))
  }
  
  est.beta <- array(0, dim=c(M,M,K))
  est.mean <- matrix(0, nrow=M, ncol=K)
  est.phi <- numeric(K)
  
  for(ite in 1:iteration){
    ##update parameter
    est.phi <- apply(label.prob,2,sum)+phi
    eta.ksi <- -tanh(sqrt(g.ksi)/2)/(4*sqrt(g.ksi))
    for(k in 1:K){
      est.beta[,,k] <- t((label.prob[,k]*(-2*eta.ksi[,k]))*x)%*%x + beta*eye(M)
      est.mean[,k] <- solve(est.beta[,,k])%*%apply(label.prob[,k]*(y-0.5)*x,2,sum)
    }
    
    ##update latent variable
    prev.g.ksi <- g.ksi
    for(k in 1:K){
      g.ksi[,k] <- diag(x %*% (solve(est.beta[,,k])+est.mean[,k]%*%t(est.mean[,k])) %*% t(x))
    }    
    
    ##update label probability
    h.tau <- rep(1,n) %*% t(digamma(est.phi)-digamma(sum(est.phi)))+
      ((y-0.5)*x)%*%est.mean-log(2)-log(cosh(sqrt(g.ksi)/2))+
      eta.ksi*(g.ksi-prev.g.ksi)
    max.h.tau <- apply(h.tau,1,max)
    h.tau.dash <- h.tau - max.h.tau%*%t(rep(1,K))
    label.prob <- exp(h.tau.dash)/(apply(exp(h.tau.dash),1,sum)%*%t(rep(1,K)))
    
    ##calculate energy
    energy <- 0
    energy <- energy -sum(lgamma(est.phi))+lgamma(sum(est.phi))+K*lgamma(phi)-lgamma(K*phi)
    for(k in 1:K){
      energy <- energy + (t(est.mean[,k])%*%est.beta[,,k]%*%est.mean[,k] + as.numeric(determinant(est.beta[,,k])$modulus) - M*log(beta))/2
    }
    energy <- energy + sum((eta.ksi*g.ksi+h.tau)*label.prob)
    energy <- energy + sum(label.prob*log(cosh(sqrt(g.ksi)/2)))
    energy <- energy - sum(log(apply(exp(h.tau.dash),1,sum))+max.h.tau)
    # energy <- as.numeric(-(determinant(beta*eye(M),logarithm = T)$modulus-determinant(est.beta,logarithm = T)$modulus)/2-
    #                        t(est.mean)%*%est.beta%*%est.mean/2+n*log(2)+
    #                        sum(log(cosh(sqrt(g.ksi)/2))-tanh(sqrt(g.ksi)/2)/(4*sqrt(g.ksi))*g.ksi))
    # 
    print(energy)
  }
  energy <- 0
  return(list(param=list(ratio = est.phi/sum(est.phi), weight = est.mean), beta = est.beta, energy=energy))
}

ZRP.logistic.mixture <- function(x,y,time.length,K,beta = 0.001, phi=1,
                                            iteration=100, seed=1,
                                            init.mean=0, init.sd=1, init.phi=1){
  set.seed(seed)
  
  n <- nrow(x)
  M <- ncol(x)
  
  ##initial setting
  g.ksi <- matrix(rnorm(n*K,mean=init.mean,sd=init.sd)^2,nrow=n,ncol=K)
  label.prob <- matrix(0,nrow = n, ncol=K)
  for(i in 1:n){
    label.prob[i,] <- rdirichlet(rep(init.phi, K))
  }
  
  est.beta <- array(0, dim=c(M,M,K))
  est.mean <- matrix(0, nrow=M, ncol=K)
  est.phi <- numeric(K)
  
  for(ite in 1:iteration){
    ##update parameter
    est.phi <- apply(label.prob,2,sum)+phi
    eta.ksi <- -tanh(sqrt(g.ksi)/2)/(4*sqrt(g.ksi))
    for(k in 1:K){
      est.beta[,,k] <- t((label.prob[,k]*(-2*eta.ksi[,k]))*x)%*%x + beta*eye(M)
      est.mean[,k] <- solve(est.beta[,,k])%*%apply(label.prob[,k]*(y-0.5)*x,2,sum)
    }
    
    ##update latent variable
    prev.g.ksi <- g.ksi
    for(k in 1:K){
      g.ksi[,k] <- diag(x %*% (solve(est.beta[,,k])+est.mean[,k]%*%t(est.mean[,k])) %*% t(x))
    }    
    
    ##update label probability
    h.tau <- rep(1,n) %*% t(digamma(est.phi)-digamma(sum(est.phi)))+
      ((y-0.5)*x)%*%est.mean-log(2)-log(cosh(sqrt(g.ksi)/2))+
      eta.ksi*(g.ksi-prev.g.ksi)
    max.h.tau <- apply(h.tau,1,max)
    h.tau.dash <- h.tau - max.h.tau%*%t(rep(1,K))
    label.prob <- exp(h.tau.dash)/(apply(exp(h.tau.dash),1,sum)%*%t(rep(1,K)))
    
    ##calculate energy
    energy <- 0
    energy <- energy -sum(lgamma(est.phi))+lgamma(sum(est.phi))+K*lgamma(phi)-lgamma(K*phi)
    for(k in 1:K){
      energy <- energy + (t(est.mean[,k])%*%est.beta[,,k]%*%est.mean[,k] + as.numeric(determinant(est.beta[,,k])$modulus) - M*log(beta))/2
    }
    energy <- energy + sum((eta.ksi*g.ksi+h.tau)*label.prob)
    energy <- energy + sum(label.prob*log(cosh(sqrt(g.ksi)/2)))
    energy <- energy - sum(log(apply(exp(h.tau.dash),1,sum))+max.h.tau)
    # energy <- as.numeric(-(determinant(beta*eye(M),logarithm = T)$modulus-determinant(est.beta,logarithm = T)$modulus)/2-
    #                        t(est.mean)%*%est.beta%*%est.mean/2+n*log(2)+
    #                        sum(log(cosh(sqrt(g.ksi)/2))-tanh(sqrt(g.ksi)/2)/(4*sqrt(g.ksi))*g.ksi))
    # 
    print(energy)
  }
  energy <- 0
  return(list(param=list(ratio = est.phi/sum(est.phi), weight = est.mean), beta = est.beta, energy=energy))
}


LVA.mixture.logistic.regression.debug <- function(x,y,K,beta = 0.001, phi=1,
                                            iteration=100, seed=1,
                                            init.mean=0, init.sd=0.1,
                                            init.df = 10, init.sigma = 1,
                                            init.phi=1){
  print(phi)
  library(mvtnorm)
  set.seed(seed)
  
  n <- nrow(x)
  M <- ncol(x)
  
  in.out.matrix <- matrix( rep((y-0.5),M),nrow=n, ncol=M)*x
  
  ##initial setting
  est.beta <- array(0, dim=c(M,M,K))
  est.mean <- matrix(0, nrow=M, ncol=K)
  est.phi <- numeric(K)
  est.phi <- rdirichlet(rep(init.phi, K))
  wishart <- rWishart(K, df=init.df, Sigma=init.sigma*eye(M))
  for(k in 1:K){
    est.beta[,,k] <- solve(wishart[,,k])
    est.mean[,k] <- rmvnorm(1,mean=rep(init.mean,M),sigma = wishart[,,k]/init.sd)
  }
  
  latent.prob <- matrix(0,nrow = n, ncol=K)
  for(i in 1:n){
    latent.prob[i,] <- rdirichlet(rep(init.phi, K))
  }
  
  # est.beta <- array(0, dim=c(M,M,K))
  # est.mean <- matrix(0, nrow=M, ncol=K)
  # est.phi <- numeric(K)
  g.ksi <- matrix(rnorm(n*K,mean=init.mean,sd=init.sd)^2,nrow=n,ncol=K)
  # g.ksi <- matrix(0, nrow=n,ncol=K)
  
  energy.trace <- numeric(iteration)
  cross.entropy.trace <- numeric(iteration)
  
  for(ite in 1:iteration){
    ##update parameter
    est.phi <- colSums(latent.prob)+phi
    eta.ksi <- -tanh(sqrt(g.ksi)/2)/(4*sqrt(g.ksi))
    for(k in 1:K){
      est.beta[,,k] <- t((latent.prob[,k]*(-2*eta.ksi[,k]))*x)%*%x+beta*eye(M)
      est.mean[,k] <- solve(est.beta[,,k])%*% t(in.out.matrix) %*% latent.prob[,k]
    }    
    
    ##update auxilary variable
    for(k in 1:K){
      g.ksi[,k] <- rowSums((x %*% (solve(est.beta[,,k])+est.mean[,k]%*%t(est.mean[,k]))) * x)
      # g.ksi[,k] <- diag(x %*% (solve(est.beta[,,k])+est.mean[,k]%*%t(est.mean[,k])) %*% t(x))
    }    
    
    ##update label probability
    # eta.ksi <- -tanh(sqrt(g.ksi)/2)/(4*sqrt(g.ksi))    
    # h.tau <- rep(1,n) %*% t(digamma(est.phi)-digamma(sum(est.phi)))+
    #   ((y-0.5)*x)%*%est.mean-log(2*cosh(sqrt(g.ksi)/2))+
    #   eta.ksi*(g.ksi-prev.g.ksi)
    h.tau <- rep(1,n) %*% t(digamma(est.phi)-digamma(sum(est.phi)))+
      in.out.matrix%*%est.mean-log(2*cosh(sqrt(g.ksi)/2))
    max.h.tau <- apply(h.tau,1,max)
    h.tau.dash <- h.tau - matrix(rep(max.h.tau,K), nrow = n, ncol=K)
    latent.prob <- exp(h.tau.dash)/matrix(rep(rowSums(exp(h.tau.dash)),K), nrow=n, ncol=K)    
    
    ##calculate energy
    energy <- 0
    energy <- energy - sum(log(rowSums(exp(h.tau.dash)))+max.h.tau)

    for(k in 1:K){
      energy <- energy + sum(eta.ksi[,k]*rowSums( (x %*% (solve(est.beta[,,k])+est.mean[,k]%*%t(est.mean[,k]))) * x)*latent.prob[,k])
    }
    energy <- energy + sum(latent.prob * matrix( rep(digamma(est.phi)-digamma(sum(est.phi)),n), nrow=n, ncol=K, byrow=T))
    energy <- energy + sum((in.out.matrix%*%est.mean)*latent.prob)

    energy <- energy -sum(lgamma(est.phi))+lgamma(sum(est.phi))+K*lgamma(phi)-lgamma(K*phi)
    for(k in 1:K){
      energy <- energy + (-t(est.mean[,k])%*%est.beta[,,k]%*%est.mean[,k] + as.numeric(determinant(est.beta[,,k], logarithm=T)$modulus) - M*log(beta))/2

    }

    current.result <- list(param=list(ratio = est.phi/sum(est.phi), weight = est.mean), beta = est.beta, latent.prob = latent.prob)
    
    energy.trace[ite] <- energy
    cross.entropy.trace[ite] <- crossentropy(x, y, current.result)
    
    print(energy)
    print(cross.entropy.trace[ite])        
    
    print(crossentropy(x, y, current.result))
  }
  energy <- 0
  return(list(param=list(ratio = est.phi/sum(est.phi), weight = est.mean), beta = est.beta, energy.trace=energy.trace, cross.entropy.trace = cross.entropy.trace, latent.prob = latent.prob))
}



LVA.mixture.logistic.regression <- function(x,y,K,beta = 0.001, phi=1,
                                            iteration=100, seed=1,
                                            init.mean=0, init.sd=0.1,
                                            init.df = 10, init.sigma = 1,
                                            init.phi=1,
                                            trace.on = F){
  print(phi)
  library(mvtnorm)
  set.seed(seed)
  
  n <- nrow(x)
  M <- ncol(x)
  
  in.out.matrix <- matrix( rep((y-0.5),M),nrow=n, ncol=M)*x
  
  ##initial setting
  est.beta <- array(0, dim=c(M,M,K))
  est.mean <- matrix(0, nrow=M, ncol=K)
  est.phi <- numeric(K)
  est.phi <- rdirichlet(rep(init.phi, K))
  wishart <- rWishart(K, df=init.df, Sigma=init.sigma*eye(M))
  for(k in 1:K){
    est.beta[,,k] <- solve(wishart[,,k])
    est.mean[,k] <- rmvnorm(1,mean=rep(init.mean,M),sigma = wishart[,,k]/init.sd)
  }
  
  latent.prob <- matrix(0,nrow = n, ncol=K)
  for(i in 1:n){
    latent.prob[i,] <- rdirichlet(rep(init.phi, K))
  }
  
  # est.beta <- array(0, dim=c(M,M,K))
  # est.mean <- matrix(0, nrow=M, ncol=K)
  # est.phi <- numeric(K)
  g.ksi <- matrix(rnorm(n*K,mean=init.mean,sd=init.sd)^2,nrow=n,ncol=K)
  # g.ksi <- matrix(0, nrow=n,ncol=K)
  
  energy.trace <- numeric(iteration)
  cross.entropy.trace <- numeric(iteration)
  
  for(ite in 1:iteration){
    ##update parameter
    est.phi <- colSums(latent.prob)+phi
    eta.ksi <- -tanh(sqrt(g.ksi)/2)/(4*sqrt(g.ksi))
    for(k in 1:K){
      est.beta[,,k] <- t((latent.prob[,k]*(-2*eta.ksi[,k]))*x)%*%x+beta*eye(M)
      est.mean[,k] <- solve(est.beta[,,k])%*% t(in.out.matrix) %*% latent.prob[,k]
    }    
    
    ##update auxilary variable
    for(k in 1:K){
      g.ksi[,k] <- rowSums((x %*% (solve(est.beta[,,k])+est.mean[,k]%*%t(est.mean[,k]))) * x)
      # g.ksi[,k] <- diag(x %*% (solve(est.beta[,,k])+est.mean[,k]%*%t(est.mean[,k])) %*% t(x))
    }    
    
    ##update label probability
    h.tau <- rep(1,n) %*% t(digamma(est.phi)-digamma(sum(est.phi)))+
      in.out.matrix%*%est.mean-log(2*cosh(sqrt(g.ksi)/2))
    max.h.tau <- apply(h.tau,1,max)
    h.tau.dash <- h.tau - matrix(rep(max.h.tau,K), nrow = n, ncol=K)
    latent.prob <- exp(h.tau.dash)/matrix(rep(rowSums(exp(h.tau.dash)),K), nrow=n, ncol=K)    
    
    ##calculate energy
    energy <- 0
    energy <- energy - sum(log(rowSums(exp(h.tau.dash)))+max.h.tau)
    
    for(k in 1:K){
      energy <- energy + sum(eta.ksi[,k]*rowSums( (x %*% (solve(est.beta[,,k])+est.mean[,k]%*%t(est.mean[,k]))) * x)*latent.prob[,k])
    }
    energy <- energy + sum(latent.prob * matrix( rep(digamma(est.phi)-digamma(sum(est.phi)),n), nrow=n, ncol=K, byrow=T))
    energy <- energy + sum((in.out.matrix%*%est.mean)*latent.prob)
    
    energy <- energy -sum(lgamma(est.phi))+lgamma(sum(est.phi))+K*lgamma(phi)-lgamma(K*phi)
    for(k in 1:K){
      energy <- energy + (-t(est.mean[,k])%*%est.beta[,,k]%*%est.mean[,k] + as.numeric(determinant(est.beta[,,k], logarithm=T)$modulus) - M*log(beta))/2
      
    }
    
    current.result <- list(param=list(ratio = est.phi/sum(est.phi), weight = est.mean), beta = est.beta, latent.prob = latent.prob)
    
    energy.trace[ite] <- energy
    cross.entropy.trace[ite] <- crossentropy(x, y, current.result)
    
    print(energy)
    print(cross.entropy.trace[ite])        
    
    print(crossentropy(x, y, current.result))
  }
  energy <- 0
  return(list(param=list(ratio = est.phi/sum(est.phi), weight = est.mean), beta = est.beta, energy.trace=energy.trace, cross.entropy.trace = cross.entropy.trace, latent.prob = latent.prob))
}

LVA.mixture.logistic.regression.analysis <- function(x,y,K,true.param,
                                                     beta = 0.001, phi=1,
                                                     iteration=100, seed=1,
                                                     init.mean=0, init.sd=0.1,
                                                     init.df = 10, init.sigma = 1,
                                                     init.phi=1,
                                                     test.num = length(y),
                                                     test.seed = seed * 2
                                                     ){
  
  library(mvtnorm)
  set.seed(seed)
  
  n <- nrow(x)
  M <- ncol(x)
  
  in.out.matrix <- matrix( rep((y-0.5),M),nrow=n, ncol=M)*x
  
  ##initial setting
  est.beta <- array(0, dim=c(M,M,K))
  est.mean <- matrix(0, nrow=M, ncol=K)
  est.phi <- numeric(K)
  est.phi <- rdirichlet(rep(init.phi, K))
  wishart <- rWishart(K, df=init.df, Sigma=init.sigma*eye(M))
  for(k in 1:K){
    est.beta[,,k] <- solve(wishart[,,k])
    est.mean[,k] <- rmvnorm(1,mean=rep(init.mean,M),sigma = wishart[,,k]/init.sd)
  }
  
  latent.prob <- matrix(0,nrow = n, ncol=K)
  for(i in 1:n){
    latent.prob[i,] <- rdirichlet(rep(init.phi, K))
  }
  
  # est.beta <- array(0, dim=c(M,M,K))
  # est.mean <- matrix(0, nrow=M, ncol=K)
  # est.phi <- numeric(K)
  # g.ksi <- matrix(rnorm(n*K,mean=init.mean,sd=init.sd)^2,nrow=n,ncol=K)
  g.ksi <- matrix(0, nrow=n,ncol=K)
  
  crossentropies <- matrix(0, nrow=iteration, ncol=2)

  energy.trace <- numeric(iteration)
  
  ##cross entropy for test data
  xrange <- c(-5,5)
  test.x <- input.unif.generate(test.num, M, seed = 2*(test.seed)-1, xrange)
  test.y <- output.mixture.logistic.generate(test.num, test.x, true.param, seed=2*test.seed)$output  
  
  for(ite in 1:iteration){
    ##update auxilary variable
    for(k in 1:K){
      g.ksi[,k] <- diag(x %*% (solve(est.beta[,,k])+est.mean[,k]%*%t(est.mean[,k])) %*% t(x))
    }    
    
    ##update label probability
    # eta.ksi <- -tanh(sqrt(g.ksi)/2)/(4*sqrt(g.ksi))    
    # h.tau <- rep(1,n) %*% t(digamma(est.phi)-digamma(sum(est.phi)))+
    #   ((y-0.5)*x)%*%est.mean-log(2*cosh(sqrt(g.ksi)/2))+
    #   eta.ksi*(g.ksi-prev.g.ksi)
    h.tau <- rep(1,n) %*% t(digamma(est.phi)-digamma(sum(est.phi)))+
      in.out.matrix%*%est.mean-log(2*cosh(sqrt(g.ksi)/2))
    max.h.tau <- apply(h.tau,1,max)
    h.tau.dash <- h.tau - max.h.tau%*%t(rep(1,K))
    latent.prob <- exp(h.tau.dash)/(apply(exp(h.tau.dash),1,sum)%*%t(rep(1,K)))    
    
    ##update parameter
    est.phi <- apply(latent.prob,2,sum)+phi
    eta.ksi <- -tanh(sqrt(g.ksi)/2)/(4*sqrt(g.ksi))
    for(k in 1:K){
      est.beta[,,k] <- t((latent.prob[,k]*(-2*eta.ksi[,k]))*x)%*%x+beta*eye(M)
      est.mean[,k] <- solve(est.beta[,,k])%*% t(in.out.matrix) %*% latent.prob[,k]
    }
    
    ##calculate energy
    energy <- 0
    energy <- energy - sum(log(apply(exp(h.tau.dash),1,sum))+max.h.tau)

    for(k in 1:K){
      energy <- energy - sum(eta.ksi[,k]*diag(x %*% (solve(est.beta[,,k])+est.mean[,k]%*%t(est.mean[,k])) %*% t(x))*latent.prob[,k])
    }
    energy <- energy + sum(latent.prob * matrix( rep(digamma(est.phi)-digamma(sum(est.phi)),n), nrow=n, ncol=K, byrow=T))
    energy <- energy + sum((in.out.matrix%*%est.mean)*latent.prob)

    energy <- energy -sum(lgamma(est.phi))+lgamma(sum(est.phi))+K*lgamma(phi)-lgamma(K*phi)
    for(k in 1:K){
      energy <- energy + (-t(est.mean[,k])%*%est.beta[,,k]%*%est.mean[,k] + as.numeric(determinant(est.beta[,,k], logarithm=T)$modulus) - M*log(beta))/2

    }

    print(energy)
    energy.trace[ite] <- energy
    
    current.result <- list(param=list(ratio = est.phi/sum(est.phi), weight = est.mean), beta = est.beta, latent.prob = latent.prob)
    
    crossentropies[ite,] <- c(crossentropy(x, y, current.result), crossentropy(test.x, test.y, current.result))
    
    print(c(crossentropy(x, y, current.result), crossentropy(test.x, test.y, current.result)))
  }

  return(list(param=list(ratio = est.phi/sum(est.phi), weight = est.mean), beta = est.beta, energy.trace = energy.trace, crossentropies = crossentropies, latent.prob = latent.prob))
}

eye <- function(M){
  return(diag(rep(1,M)))
}

ones <- function(n,m){
  return(matrix(1,nrow=n,ncol=m))
}

rdirichlet <- function(phi) {
  numerator <- rgamma(n = length(phi), shape=phi, rate=1)
  return(numerator / sum(numerator))
}

sigmoid <- function(a){ return(1/(1+exp(-a))) }

predict.dist <- function(x, y, result){

}

crossentropy <- function(x,y, result){
  est.ratio <- result$param$ratio
  est.sigmoid.approx <- integrated.sigmoid.approx(x,result)
  
  return(-sum(log( est.sigmoid.approx %*% est.ratio )[y == 1]) - sum(log((1 - est.sigmoid.approx) %*% est.ratio)[y == 0]))
}

integrated.sigmoid.approx <- function(x,result){
  n <- dim(x)[1]
  K <- length(result$param$ratio)
  mu.a <- x %*% result$param$weight
  S.a <- matrix(0, nrow=n, ncol=K)
  for(k in 1:K){
    S.a[,k] <- rowSums( (x %*% solve(result$beta[,,k])) * x)
    
    # S.a[,k] <- diag(x %*% solve(result$beta[,,k]) %*% t(x))
  }
  return(sigmoid( mu.a / sqrt((1+pi*S.a)/8) ))
}





initialize.logistic.mixture <- function(n, K, M, init.Sigma,
                                             init.phi=1, init.df = 10, seed=1){
  library(MASS)
  set.seed(seed=seed)  
  
  # for hyperparameter of parameter distribution
  if(length(init.phi) == 1){
    init.phi <- rep(init.phi,K)
  }
  
  if(init.df < M){
    init.df <- M
  }
  phi <- rdirichlet(init.phi)
  beta <- rWishart(K,df=init.df, Sigma=init.Sigma)
  inv.beta <- array(0,dim=c(M,M,K))
  mu <- matrix(0, nrow=M, ncol=K)
  mu.cov <- array(0, dim = c(M,M,K))
  for(k in 1:K){
    inv.beta <- solve(beta[,,k])
    mu[,k] <- mvrnorm(n=1, mu=rep(0,M), Sigma=inv.beta)
    mu.cov[,,k] <- inv.beta + tcrossprod(mu[,k])  
  }
  
  
  # for latent variables
  est.latent.variable <- matrix(0,nrow=n,ncol=K)
  for(i in 1:n){
    est.latent.variable[i,] <- rdirichlet(rep(1,K))
  }
  
  # for auxiliary variables
  est.auxilirary.variable <- matrix(abs(rnorm(n*K)), nrow=n, ncol=K)
  est.sq.g.eta <- sqrt(est.auxilirary.variable)  
  est.v.eta <- -tanh(est.sq.g.eta/2)/(4*est.sq.g.eta)  
  
  return(list(param = list(phi=phi, mu=mu, beta=beta, mu.cov = mu.cov),
              latent = list(latent.variable=est.latent.variable),
              auxilirary = list(auxilirary.variable=est.auxilirary.variable, v.eta=est.v.eta, sq.g.eta=est.sq.g.eta) ))
}

lva.free.energy <- function(x, in.out.matrix, prior.hyperparameter,
                                  param.updated.val, latent.updated.val, auxilirary.updated.val){
  
  ## transform argument to actual values used here
  post.hyperparameter <- param.updated.val
  log.lantent.variable.partial <- latent.updated.val$log.latent.partial
  latent.variable <- latent.updated.val$latent.variable
  
  auxilirary.variable <- auxilirary.updated.val$auxilirary.variable
  est.v.eta <- auxilirary.updated.val$v.eta
  
  est.mu.cov <- post.hyperparameter$mu.cov
  
  phi <- prior.hyperparameter$phi
  beta <- prior.hyperparameter$beta
  K <- length(post.hyperparameter$phi)
  
  ## latent variable is relatively small, so normailized here.
  max.log.latent.variable.partial <- apply(log.lantent.variable.partial,1,max)
  norm.log.latent.variable.partial <- log.lantent.variable.partial - max.log.latent.variable.partial
  
  ##calculate energy
  energy <- 0
  energy <- energy - sum(log(rowSums(exp(norm.log.latent.variable.partial)))+max.log.latent.variable.partial)
  
  for(k in 1:K){
    energy <- energy + sum(est.v.eta[,k]*auxilirary.variable[,k]*latent.variable[,k])
  }
  energy <- energy + sum(latent.variable * matrix( rep(digamma(post.hyperparameter$phi)-digamma(sum(post.hyperparameter$phi)),n), nrow=n, ncol=K, byrow=T))
  energy <- energy + sum((in.out.matrix%*%post.hyperparameter$mu)*latent.variable)
  
  energy <- energy -sum(lgamma(post.hyperparameter$phi))+lgamma(sum(post.hyperparameter$phi))+K*lgamma(phi)-lgamma(K*phi)
  for(k in 1:K){
    energy <- energy + (-t(post.hyperparameter$mu[,k])%*%post.hyperparameter$beta[,,k]%*%post.hyperparameter$mu[,k] + as.numeric(determinant(post.hyperparameter$beta[,,k], logarithm=T)$modulus) - M*log(beta))/2
    
  }
  
  return(energy)
}
# 
# 
# lva.free.energy <- function(x, in.out.matrix, prior.hyperparameter,
#                                  param.updated.val, latent.updated.val, auxilirary.updated.val){
#   
#   ## transform argument to actual values used here
#   post.hyperparameter <- param.updated.val
#   log.lantent.variable.partial <- latent.updated.val$log.latent.partial
#   latent.variable <- latent.updated.val$latent.variable
#   
#   auxilirary.variable <- auxilirary.updated.val$auxilirary.variable
#   est.v.eta <- auxilirary.updated.val$v.eta
#   
#   est.mu.cov <- post.hyperparameter$mu.cov
#   
#   phi <- prior.hyperparameter$phi
#   beta <- prior.hyperparameter$beta
#   K <- length(post.hyperparameter$phi)
#   
#   ## latent variable is relatively small, so normailized here.
#   max.log.latent.variable.partial <- apply(log.lantent.variable.partial,1,max)
#   norm.log.latent.variable.partial <- log.lantent.variable.partial - max.log.latent.variable.partial
#   
#   energy <- 0
#   energy <- energy - sum(log(rowSums(exp(norm.log.latent.variable.partial)))+max.log.latent.variable.partial)
#   for(k in 1:K){
#     energy <- energy + sum(latent.variable[,k]*(
#       digamma(post.hyperparameter$phi[k]) - digamma(sum(post.hyperparameter$phi)) + 
#         in.out.matrix %*% post.hyperparameter$mu[,k] + est.v.eta[,k]* auxilirary.variable[,k]
#     ))
#     
#     energy <- energy + determinant(post.hyperparameter$beta[,,k]/(2*pi), logarithm=T)$modulus[1]/2 - t(post.hyperparameter$mu[,k]) %*% post.hyperparameter$beta[,,k] %*% post.hyperparameter$mu[,k]/2
#   }
#   energy <- energy + lgamma(sum(post.hyperparameter$phi)) - sum(lgamma(post.hyperparameter$phi)) + K*lgamma(prior.hyperparameter$phi) - lgamma(K*prior.hyperparameter$phi) - K*determinant(prior.hyperparameter$beta*diag(M)/(2*pi), logarithm=T)$modulus[1]/2
#   return(energy)
# }

lva.estimation.main <- function(update.order=c(lva.update.parameters, lva.update.latent.variable, lva.update.auxilirary.variable),
                                x,
                                y,
                                K,
                                prior.hyperparameter,
                                init.Sigma,
                                init.phi = 1,
                                init.df = 10,                                
                                iteration = 1000,
                                restart = 1,                                
                                learning.seed = 1,
                                trace.on = FALSE,
                                save.file = NA
){
  n <- dim(x)[1]
  M <- dim(x)[2]
  
  est.result <- list()
  est.result$energy.trace <- Inf
  
  for(i.restart in 1:restart){
    i.initial.result <- initialize.logistic.mixture(n, K, M, init.Sigma, init.phi, init.df, seed = learning.seed+i.restart)
    
    i.restart.result <- lva.estimation.learning(update.order, x, y, K, prior.hyperparameter,
                                                i.initial.result, iteration, trace.on=trace.on,
                                                step.savefile = paste("result", i.restart, "_tmp.rds", sep=""))
    print(c(i.restart, i.restart.result$energy.trace[length(est.result$energy.trace)]))
    
    if(i.restart.result$energy.trace[length(i.restart.result$energy.trace)] < est.result$energy.trace[length(est.result$energy.trace)]){
      est.result <- i.restart.result
    }
    
    if(!is.na(save.file)){
      save(i.restart.result, file = paste("result",as.character(i.restart), "_", save.file, sep = ""))
    }
    
  }
  return(est.result)
}

lva.estimation.learning <- function(update.order=c(lva.update.parameters, lva.update.latent.variable, lva.update.auxilirary.variable),
                                    x,
                                    y,
                                    K,
                                    prior.hyperparameter,
                                    initial.result,
                                    iteration = 1000,
                                    tol = 1e-5,
                                    trace.on = FALSE,
                                    step.savefile = NA
){
  n <- dim(x)[1]
  M <- dim(x)[2]
  
  in.out.matrix <- matrix(rep((y - 0.5),M),nrow=n,ncol=M) * x
  
  est.result <- initial.result
  if(trace.on == TRUE){
    est.result$energy.trace <- numeric(iteration)-1
  }else{
    est.result$energy.trace <- 0
  }
  
  for(ite in 1:iteration){
    for(j in 1:length(update.order)){
      j.update.result <- update.order[[j]](x, in.out.matrix, K, prior.hyperparameter, est.result)
      est.result[[j.update.result$name]] <- j.update.result$updated.val
    }
    
    if(trace.on == TRUE){
      est.result$energy.trace[ite] <- lva.free.energy(x, in.out.matrix, prior.hyperparameter,
                                                           param.updated.val = est.result$param,
                                                           latent.updated.val = est.result$latent, 
                                                           auxilirary.updated.val = est.result$auxilirary)      
      print(c(ite, est.result$energy.trace[ite]))
    }else{
      # print(ite)
    }
    
    if(!is.na(step.savefile)){
      saveRDS(est.result, file = step.savefile)
    }
  }
  if(trace.on == FALSE){
    est.result$energy.trace <- lva.free.energy(x, in.out.matrix, prior.hyperparameter,
                                                    param.updated.val = est.result$param,
                                                    latent.updated.val = est.result$latent, 
                                                    auxilirary.updated.val = est.result$auxilirary) 
  }
  est.result$mean.estimator <- list(ratio=est.result$param$phi/sum(est.result$param$phi), weight=est.result$param$mu)
  return(est.result)
}


lva.update.parameters <- function(x, in.out.matrix, K, prior.hyperparameter, current.est.result){
  ##pre-processing for this function
  n <- dim(x)[1]
  M <- dim(x)[2]
  phi <- prior.hyperparameter$phi
  beta <- prior.hyperparameter$beta
  est.latent.variable <- current.est.result$latent$latent.variable
  est.v.eta <- current.est.result$auxilirary$v.eta
  
  est.beta <- array(0, dim=c(M,M,K))
  est.mu <- matrix(0, nrow=M, ncol=K)
  est.mu.cov <- array(0, dim=c(M,M,K))
  
  est.phi <- colSums(est.latent.variable)+phi
  for(k in 1:K){
    extend.est.v.eta <- -2 * matrix(rep(est.v.eta[,k],M),nrow=n,ncol=M) ## -2 is added to adjust to necessary calculaion
    extend.est.latent.variable <- matrix(rep(est.latent.variable[,k],M), nrow=n, ncol=M)
    est.beta[,,k] <- crossprod(x, (extend.est.latent.variable * extend.est.v.eta) * x) + beta*diag(M)
    
    inv.est.beta <- solve(est.beta[,,k])
    
    est.mu[,k] <- inv.est.beta %*% colSums(extend.est.latent.variable * in.out.matrix)
    
    est.mu.cov[,,k] <- inv.est.beta + tcrossprod(est.mu[,k])
  }
  
  return(list(name="param", updated.val=list(phi=est.phi, beta=est.beta, mu=est.mu, mu.cov = est.mu.cov)))
}

lva.update.latent.variable <- function(x, in.out.matrix, K, prior.hyperparameter, current.est.result){
  ##pre-processing for this function
  n <- dim(x)[1]
  M <- dim(x)[2]
  est.phi <- current.est.result$param$phi
  est.mu <- current.est.result$param$mu
  est.beta <- current.est.result$param$beta
  
  est.mu.cov <- current.est.result$param$mu.cov
  
  est.v.eta <- current.est.result$auxilirary$v.eta
  est.sq.g.eta <- current.est.result$auxilirary$sq.g.eta
  est.g.eta  <- current.est.result$auxilirary$auxilirary.variable
  
  est.h.xi <- matrix(0, nrow=n, ncol=K)
  for(k in 1:K){
    est.h.xi[,k] <- digamma(est.phi[k]) - digamma(sum(est.phi)) +
      in.out.matrix %*% est.mu[,k] - log(2*cosh(est.sq.g.eta[,k]/2)) + est.v.eta[,k]*(rowSums((x %*% est.mu.cov[,,k]) * x) - est.g.eta[,k])
    # est.h.xi[,k] <- digamma(est.phi[k]) - digamma(sum(est.phi)) +
    #   in.out.matrix %*% est.m[,k] - log(2*cosh(sqrt(est.g.eta[,k])/2))
  }
  max.est.h.xi <- apply(est.h.xi,1,max)
  exp.norm.est.h.xi <- exp(est.h.xi - max.est.h.xi)
  est.latent.variable <-  exp.norm.est.h.xi / matrix(rep(rowSums(exp.norm.est.h.xi),K),nrow=n,ncol=K)
  return(list(name="latent", updated.val=list(latent.variable=est.latent.variable, log.latent.partial=est.h.xi)))
}

lva.update.auxilirary.variable <- function(x, in.out.matrix, K, prior.hyperparameter, current.est.result){
  ##pre-processing for this function
  n <- dim(x)[1]
  
  est.mu.cov <- current.est.result$param$mu.cov
  est.g.eta <- matrix(0, nrow=n, ncol=K)
  for(k in 1:K){
    est.g.eta[,k] <- rowSums((x %*% est.mu.cov[,,k]) * x)
  }
  est.sq.g.eta <- sqrt(est.g.eta)  
  est.v.eta <- -tanh(est.sq.g.eta/2)/(4*est.sq.g.eta)
  
  return(list(name="auxilirary", updated.val=list(auxilirary.variable=est.g.eta, v.eta=est.v.eta, sq.g.eta=est.sq.g.eta)))
}

zrp.forward.prob <- function(dx, weight){
  vehicle.num <- length(dx)
  hat.dx <- matrix(1, nrow=vehicle.num, ncol=2)
  hat.dx[,2] <- dx
  is.hop <- rep(FALSE, vehicle.num)
  forward.flag <- dx != 0
  
  forward.prob <- 1/(1+exp(-rowSums(hat.dx * weight)))
  # print(forward.prob)
  is.hop[forward.flag] <- runif(sum(forward.flag)) < forward.prob[forward.flag]
  
  return(is.hop)
}

zrp.generate.vehicle.info <- function(vehicle.num, param){
  ## set weight for each vehicle
  label.realization <- rmultinom(vehicle.num, size=1, prob=param$ratio)
  label <- apply(label.realization, 2, which.max)
  vehicle.forward.info <- matrix(0, nrow=vehicle.num, ncol=dim(param$weight)[1])
  for(i in 1:vehicle.num){
    vehicle.forward.info[i,] <- param$weight[,label[i]]
  }
  return(vehicle.forward.info)
}

zrp.ca.synthetic <- function(ring.size, vehicle.num, time.length, vehicle.forward.info){
  ca.pos <- matrix(0, nrow=vehicle.num, ncol=time.length)
  is.hop <- matrix(0, nrow=vehicle.num, ncol=time.length)
  ### initial pos
  current.pos <- ca.synthetic.initialize.deterministic(ring.size, vehicle.num)
  ca.pos[,1] <- current.pos
  # current.ca <- current.info$current.ca
  for(t in 1:time.length){
    ### calc dx
    current.pos <- ca.pos[,t]
    front.ind <- (1:vehicle.num) %% (vehicle.num)+1
    current.dx <- ((current.pos[front.ind] - current.pos) %% cell.num) - 1
    
    current.hop <- zrp.forward.prob(current.dx, vehicle.forward.info[,1:2])
    is.hop[,t] <- current.hop
    
    next.pos <- current.pos
    next.pos[current.hop] <- current.pos[current.hop] %% cell.num + 1
    if(t != time.length){
      ca.pos[,t+1] <- next.pos
    }
    t <- t + 1
  }
  return(list(ca.pos = ca.pos, is.hop = is.hop))
}

ca.synthetic.initialize.deterministic <- function(ring.size, vehicle.num){
  vehicle.pos.ind <- floor(ring.size / vehicle.num)
  vehicle.ind <- seq(from=1, to=ring.size, by=vehicle.pos.ind)
  # current.ca <- numeric(ring.size)
  # current.ca[vehicle.ind] <- 1:vehicle.num
  return(vehicle.ind)
}

ca.synthetic.initialize.random <- function(ring.size, vehicle.num){
  
}