args = commandArgs(trailingOnly=T)
tag = as.integer(args[1])

library(bnlearn)
A.df = read.csv(paste("./data/A_",tag,".csv",sep=""),header=FALSE)
p = ncol(A.df)
var_names = names(A.df)
G = empty.graph(var_names)
amat(G) = t(as.matrix(A.df)) # A is transposed from R's convention
# graphviz.plot(cpdag(G))
A_cpdag = t(amat(cpdag(G))) # A is transposed from R's convention
write.csv(A_cpdag, paste("./data/A_cpdag_", tag, ".csv", sep=""))
