#PC algorithm
args = commandArgs(trailingOnly=T)
tag = as.integer(args[1])
n = as.integer(args[2])
alpha = as.double(args[3])
mmax = as.integer(args[4])
if (mmax<0){
  mmax=Inf
}

library(pcalg)
C = read.csv(paste("./data/C_", tag, ".csv", sep=""), header=F)
C = as.matrix(C)
p = nrow(C)

t0 = proc.time()[[3]]
suffStat <- list(C = C, n = n)
varNames = as.character(0:(p-1))
pc.C <- pc(suffStat, indepTest = gaussCItest, labels = varNames,
           alpha = alpha, m.max = mmax, skel.method = "stable.fast")
# the default m.max = inf will lead to infeasible running time for some data (>8hours)
t1 = proc.time()[[3]]
runtime = t1-t0

#plot(pc.C)
#showEdgeList(pc.C)
A = as(pc.C, "amat")
write.csv(A, paste("./data/A_cpdag_", tag, ".csv", sep=""))
txtfile<-file(paste("./data/runtime_", tag, ".txt", sep=""))
writeLines(paste(runtime), txtfile)
close(txtfile)
