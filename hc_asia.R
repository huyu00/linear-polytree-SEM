#hill-climbing
args = commandArgs(trailingOnly=T)
tag = as.integer(args[1])
n_use = as.integer(args[2])

library(bnlearn)
library(Rgraphviz)
# str(asia)
p = ncol(asia)
n_total = nrow(asia)
# X.df = asia[sample(1:n_total,n_use,replace=F),]
X.df = asia[sample(1:n_total,n_use,replace=T),] #bootstrap
# X.df = asia

# true graph
asia.dag = model2network("[A][S][T|A][L|S][B|S][D|B:E][E|T:L][X|E]")
graphviz.plot(cpdag(asia.dag),layout = "dot")
# need to transpose for usual convention
A0 = t(amat(cpdag(asia.dag)))
# re-ordering....
nls = names(asia)
id_sort = order(nls)
id_sort_rev = 1:p
id_sort_rev[id_sort] = 1:p
nls_sort = sort(nls)
# print(nls_sort)
# print(nls_sort[id_sort_rev])
A0 = A0[id_sort_rev,]
A0 = A0[,id_sort_rev]
write.csv(A0, paste("./data/A_asia_cpdag.csv", sep=""))

# hc
t0 = proc.time()[[3]]
X.hc = hc(X.df)
t1 = proc.time()[[3]]
runtime = t1-t0
# graphviz.plot(cpdag(X.hc),layout = "dot")

# need to transpose for usual convention
A = t(amat(cpdag(X.hc)))
write.csv(A, paste("./data/A_cpdag_", tag, ".csv", sep=""))
txtfile<-file(paste("./data/runtime_", tag, ".txt", sep=""))
writeLines(paste(runtime), txtfile)
close(txtfile)

# save numerical X
Y = X.df
indx <- sapply(Y, is.factor)
Y[indx] <- lapply(Y[indx], function(x) as.numeric(x))
write.csv(Y, paste("./data/X_", tag, ".csv", sep=""))
