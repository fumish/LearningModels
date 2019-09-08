cell.num <- 52
re.ring.size <- 230
discrete.width <- re.ring.size/cell.num

vehicle.data <- read.table("./data/case1.data")
index.last <- which(vehicle.data$V2 == 250)
vehicle.num <- length(index.last)
index.first <- c(1,(1+index.last[1:(vehicle.num-1)]))

time.length <- index.last[1] - index.first[1] + 1

### spaciotemporal diagram
data.matrix <- matrix(0, nrow=time.length, ncol=vehicle.num)
for(i in 1:vehicle.num){
  data.matrix[,i] <- vehicle.data$V1[index.first[i]:index.last[i]]
}

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





