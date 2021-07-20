#hill-climbing
args = commandArgs(trailingOnly=T)
tag = as.integer(args[1])
file_name = args[2]
# print(file_name)


# load data
# file_name = "./data/alarm_data/Alarm1_s5000_v1.csv"
X.df = read.csv(file_name,header=FALSE)
# # convert to factor
# names = seq(37)
# X.df[,names] <- lapply(X.df[,names] , factor)
# str(X.df)

p = 37

library(bnlearn)
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


# true graph
# A.df = read.csv("./data/alarm_data/Alarm1_graph.csv",header=FALSE)
# var_names = paste(seq(p))
# var_names[26] = "MVS"
# var_names[25] = "VMCH"
# var_names[24] = "DISC"
# var_names[29] = "INT"
# var_names[2] = "PRSS"
# var_names[23] = "VTUB"
# var_names[27] = "KINK"
# var_names[22] = "VLNG"
# var_names[28] = "SHNT"
# var_names[30] = "PMB"
# var_names[3] = "PAP"
# var_names[18] = "SAO2"
# var_names[1] = "MINV"
# var_names[4] = "ECO2"
# var_names[15] = "ACO2"
# var_names[21] = "VALV"
# var_names[19] = "PVS"
# var_names[20] = "FIO2"
# var_names[14] = "CCHL"
# var_names[31] = "ANES"
# var_names[16] = "TPR"
# var_names[17] = "VAPL"
# var_names[10] = "BP"
# var_names[13] = "HR"
# var_names[9] = "ERCA"
# var_names[8] = "HREK"
# var_names[7] = "HRSA"
# var_names[12] = "CO"
# var_names[34] = "STKV"
# var_names[6] = "ERLO"
# var_names[5] = "HRBP"
# var_names[11] = "HIST"
# var_names[36] = "LVF"
# var_names[35] = "LVV"
# var_names[37] = "HYP"
# var_names[32] = "CVP"
# var_names[33] = "PCWP"

# cat(paste(shQuote(var_names, type="cmd"), collapse=", "))
