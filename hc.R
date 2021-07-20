#hill-climbing
args = commandArgs(trailingOnly=T)
tag = as.integer(args[1])

library(bnlearn)
X = read.csv(paste("./data/X_", tag, ".csv", sep=""), header=F)
X.df = data.frame(X)

t0 = proc.time()[[3]]
X.hc = hc(X.df)
t1 = proc.time()[[3]]
runtime = t1-t0

# need to transpose for usual convention
A = t(amat(cpdag(X.hc)))
write.csv(A, paste("./data/A_cpdag_", tag, ".csv", sep=""))
txtfile<-file(paste("./data/runtime_", tag, ".txt", sep=""))
writeLines(paste(runtime), txtfile)
close(txtfile)
