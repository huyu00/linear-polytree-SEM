# sample from baysian network and save as csv
args = commandArgs(trailingOnly=T)
# tag = as.integer(args[1])
# n_use = as.integer(args[2])

library(bnlearn)
library(Rgraphviz)
earthquake.bif = read.bif("./data/earthquake.bif")
X.df = rbn(earthquake.bif, n=100000) # sample from the baysian network
# X.df = rbn(earthquake.bif, n=n_use) # sample from the baysian network
p = ncol(X.df)

# save numerical X
Y = X.df
indx <- sapply(Y, is.factor)
Y[indx] <- lapply(Y[indx], function(x) as.numeric(x))
# write.csv(Y, paste("./data/X_", tag, ".csv", sep=""))
write.csv(Y, paste("./data/X_earthquake_100000, ".csv", sep=""))
