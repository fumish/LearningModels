source("lib_logistic_zrp.R")

#### problem setting
true.ratio <- c(0.33, 0.33, 0.33)
true.weight <- matrix(c(-3,1, -5,3, -1,7 ), nrow=2, ncol=length(true.ratio))
true.param <- list(ratio=true.ratio, weight=true.weight)

#### model setting
cell.num <- 200
vehicle.num <- 100
time.length <- 100
vehicle.forward.info <- zrp.generate.vehicle.info(vehicle.num, true.param)
ca.sim.info <- zrp.ca.synthetic(ring.size = cell.num, vehicle.num = vehicle.num,
                                time.length = time.length, vehicle.forward.info = vehicle.forward.info)

### cellular automaton
ca.pos <- ceiling(data.matrix/re.ring.size*cell.num)

### vehicular gap
front.ind <- (1:vehicle.num) %% (vehicle.num)+1
ca.dx <- ((ca.pos[,front.ind] - ca.pos) %% cell.num) - 1
ca.dx <- ca.dx[-time.length,]
### vehicle forward or not
### ban backward
back.threshold <- -3
diff.ca <- ca.pos[2:time.length,] - ca.pos[1:(time.length-1),]
ca.dv <- ifelse(back.threshold < diff.ca & diff.ca < 0, 0, diff.ca %% cell.num)
is.hop <- ifelse(ca.dv > 0, 1, 0)

### eliminate ca.dv = 0 when ca.dx = 0
eff.time.length <- numeric(vehicle.num)
eliminate.ca.info <- matrix(-1, nrow = dim(ca.dx)[1], ncol= dim(ca.dx)[2])
eliminate.ca.dv <- matrix(-1, nrow=dim(is.hop)[1], ncol=dim(is.hop)[2])
for(i in 1:vehicle.num){
  i.target.ind <- ca.dx[,i] != 0
  eff.time.length[i] <- sum(i.eliminated.ind)
  eliminate.ca.info[1:eff.time.length[i],i] <- ca.dx[i.eliminated.ind,i]
  eliminate.ca.dv[1:eff.time.length[i],i] <- is.hop[i.eliminated.ind,i]
}

### transform data to input and output
training.x <- array(1,dim = c(2,vehicle.num,time.length-1))
training.x[1,,] <- eliminate.ca.info
training.y <- eliminate.ca.dv





