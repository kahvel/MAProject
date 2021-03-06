

# Raw EEG
setwd("../data/")

eeg_data <- read.csv("test5_1.csv", header=TRUE, sep=" ")
eeg_data <- eeg_data[c("P7", "O1", "O2", "P8")]

par(mfrow=c(2,2))
plot(eeg_data$O1, type="l")
plot(eeg_data$O2, type="l")
plot(eeg_data$P7, type="l")
plot(eeg_data$P8, type="l")




setwd("../data/")

trial_data <- read.csv("test5_targets_2.csv", header=TRUE, sep=" ")
trial <- trial_data[5,]

x <- trial$Start:trial$Stop
trial_eeg <- eeg_data[trial$Start:trial$Stop,]
trial_eeg <- as.data.frame(apply(trial_eeg, 2, function(x){scale(x, scale=FALSE)}))
par(mfrow=c(3,2))
plot(x, trial_eeg$O1, type="l")
plot(x, trial_eeg$O2, type="l")
plot(x, trial_eeg$P7, type="l")
plot(x, trial_eeg$P8, type="l")


fit <- factanal(trial_eeg, 1, scores="regression")
plot(x, fit$scores, type="l")


#par(mfrow=c(1,1))
#plot(x, fit$scores, type="l")
#lines(x, trial_eeg$O1, type="l", col="red")




step <- 1
packet_to_result <- function(packet) {
    if (packet < 256) {
        0
    } else {
        ceiling((packet-256+step)/step)
    }
}

setwd("../data/")

result_data <- read.csv("test5_results_2_all.csv", header=TRUE, sep=" ")
psdasum <- c("PSDA_sum_f1","PSDA_sum_f2","PSDA_sum_f3")
psda2 <- c("PSDA_h2_f1","PSDA_h2_f2","PSDA_h2_f3")
f <- c("f1", "f2","f3")
cca <- c("CCA_f1", "CCA_f2", "CCA_f3")
f1 <- c("PSDA_sum_f1", "PSDA_h1_f1", "CCA_f1")
result_data <- result_data[,!(names(result_data) %in% psdasum)]

result_data <- result_data[c(f1)]


trial <- trial_data[3,]

indices <- packet_to_result(trial$Start):packet_to_result(trial$Stop)
trial_result <- result_data[indices,]

trial_result <- as.data.frame(apply(trial_result, 2, function(x){scale(x)}))
#plot(trial_result$PSDA_sum_f1, type="l")
#lines(trial_result$CCA_f1, col="red")
#plot(trial_result$PSDA_sum_f2, type="l")
#lines(trial_result$CCA_f2, col="red")
#plot(fit$scores, type="l")

fit <- factanal(trial_result, 1, rotation="varimax", scores="regression")
fit
trial




new_cols <- matrix(NA, ncol=3, nrow=nrow(result_data))
trial_index <- 1

step <- 1

for (i in 1:nrow(result_data)) {
    trial <- trial_data[trial_index,]
    if (256+i*step >= trial$Stop) {
        trial_index <- trial_index + 1
    }
    new_cols[i,1] <- as.numeric(trial$Target == 1)
    new_cols[i,2] <- as.numeric(trial$Target == 2)
    new_cols[i,3] <- as.numeric(trial$Target == 3)
}
result_data["t1"] <- new_cols[,1]
result_data["t2"] <- new_cols[,2]
result_data["t3"] <- new_cols[,3]

result_data <- result_data[1:(length(result_data)-1),]




loadings <- list()#matrix(NA, nrow=nrow(trial_data), ncol=ncol(result_data))
scores <- list()
for (i in 2:nrow(trial_data)) {
    trial <- trial_data[i,]
    indices <- packet_to_result(trial$Start):packet_to_result(trial$Stop)
    trial_result <- result_data[indices,]
    
    fit <- factanal(trial_result, 3, rotation="varimax", scores="regression")
    loadings[[i]] <- fit$loadings[,1:3]
    scores[[i]] <- fit$scores
}
#colnames(loadings) <- names(fit$loadings[,1])
#cbind(round(loadings,2), trial_data$Target)





library(TTR)

setwd("C:\\Users\\Anti\\Desktop\\PycharmProjects\\MAProject\\src\\result")
result_data <- read.csv("test5_results_1.csv", header=TRUE, sep=" ")
result_data <- as.data.frame(apply(result_data, 2, function(x){scale(x)}))


plot(SMA(result_data$PSDA_sum_f1,64), type="l", ylim=c(-3,3))
lines(SMA(result_data$PSDA_sum_f2,64), col="red")
lines(SMA(result_data$PSDA_sum_f3,64), col="blue")
points(new_cols[,1]-4)


par(mfrow=c(3,1))
plot(SMA(result_data$PSDA_sum_f1,7), type="l", ylim=c(-3,3))
lines(SMA(result_data$PSDA_h1_f1,7), col="red")
lines(SMA(result_data$CCA_f1,7), col="blue")
points(new_cols[,1])

#par(mfrow=c(1,1))
plot(SMA(result_data$PSDA_sum_f2,7), type="l", ylim=c(-3,3))
lines(SMA(result_data$PSDA_h1_f2,7), col="red")
lines(SMA(result_data$CCA_f2,7), col="blue")
points(new_cols[,2])

#par(mfrow=c(1,1))
plot(SMA(result_data$PSDA_sum_f3,7), type="l", ylim=c(-3,3))
lines(SMA(result_data$PSDA_h1_f3,7), col="red")
lines(SMA(result_data$CCA_f3,7), col="blue")
points(new_cols[,3])




#I realized we cannot centralise (and normalise) all the features one by one. We have to combine some of them, otherwise we lose the ordering which is essential.??


setwd("C:\\Users\\Anti\\Desktop\\PycharmProjects\\MAProject\\src\\result")
result_data <- read.csv("test5_results_1_all.csv", header=TRUE, sep=" ")

result_data <-
  data.frame(PSDA_sum=c(result_data$PSDA_sum_f1, result_data$PSDA_sum_f2, result_data$PSDA_sum_f3),
             PSDA_h1=c(result_data$PSDA_h1_f1, result_data$PSDA_h1_f2, result_data$PSDA_h1_f3),
             CCA=c(result_data$CCA_f1, result_data$CCA_f2, result_data$CCA_f3))

fit <- factanal(result_data, 3, rotation="promax", scores="regression")

l <- nrow(result_data)/3

par(mfrow=c(1,1))
plot(result_data$PSDA_sum[1:l], type="l", ylim=c(-3,3))
lines(result_data$PSDA_h1[1:l], col="red")
lines(result_data$CCA[1:l], col="blue")
points(new_cols[,1])

points(scores[[2]])
points(fit$scores[1:l])

plot((fit$scores[1:381,1]), type="l")
lines((fit$scores[1:381,2]), type="l", col="red")
lines((fit$scores[1:381,3]), type="l", col="blue")
points(new_cols[7:388,1]-2.5)
points(new_cols[7:388,2]-2.5, col="red")
points(new_cols[7:388,3]-2.5, col="blue")

plot((fit$scores[,2]), type="l")
lines((fit$scores[,1]), type="l", col="red")
lines((fit$scores[,3]), type="l", col="blue")
points(new_cols[,1]-3)
points(new_cols[,2]-3, col="red")
points(new_cols[,3]-3, col="blue")

# try to factanal the scores

fit2 <- factanal(fit$scores[indices,], 1, scores="regression")



# Let's try to use moving average filter on the results before factanal.


setwd("C:\\Users\\Anti\\Desktop\\PycharmProjects\\MAProject\\src\\result")
result_data <- read.csv("test5_results_1_all.csv", header=TRUE, sep=" ")
psda2 <- c("PSDA_h1_f1","PSDA_h1_f2","PSDA_h1_f3")
psda2 <- c("PSDA_sum_f1","PSDA_sum_f2","PSDA_sum_f3")
psda2 <- c("PSDA_h2_f1","PSDA_h2_f2","PSDA_h2_f3")
result_data <- result_data[,!(names(result_data) %in% psda2)]
library(TTR)
result_data <- as.data.frame(apply(result_data, 2, function(x){SMA(x, 64)}))[64:nrow(result_data),]

fit <- factanal(result_data, 5, rotation="varimax", scores="regression")

difference1 <- fit$scores[,1]-fit$scores[,2]-fit$scores[,3]
difference2 <- fit$scores[,2]-fit$scores[,1]-fit$scores[,3]
difference3 <- fit$scores[,3]-fit$scores[,2]-fit$scores[,1]
plot(difference1, type="l")
lines(difference2, col="red")
lines(difference3, col="blue")
points(new_cols[7:200,2]-6)
points(new_cols[7:200,3]-6, col="red")
points(new_cols[7:200,1]-6, col="blue")



# Combining EEG with results just puts the signal into one factor. Useless?



result_data1 <- read.csv("test5_results_1_all.csv", header=TRUE, sep=" ")
result_data2 <- read.csv("test5_results_2_all.csv", header=TRUE, sep=" ")
result_data3 <- read.csv("test5_results_3_all.csv", header=TRUE, sep=" ")
result_data <- rbind(result_data1, result_data2, result_data3)






#F1 -> PSDA_sum_f1, a3, NA
#F2 -> PSDA_sum_f2, b3, NA
#F3 -> PSDA_sum_f3, c3, NA

library(sem)
result_data_cov <- cov(result_data)
model <- specify.model()
F1 -> PSDA_h1_f1, a1, NA
F1 -> PSDA_h2_f1, a2, NA
F1 -> CCA_f1, a3, NA
F2 -> PSDA_h1_f2, b1, NA
F2 -> PSDA_h2_f2, b2, NA
F2 -> CCA_f2, b3, NA
F3 -> PSDA_h1_f3, c1, NA
F3 -> PSDA_h2_f3, c2, NA
F3 -> CCA_f3, c3, NA
PSDA_h1_f1 <-> PSDA_h1_f1, e11,   NA 
PSDA_h2_f1 <-> PSDA_h2_f1, e12,   NA 
CCA_f1 <-> CCA_f1, e13,   NA 
PSDA_h1_f2 <-> PSDA_h1_f2, e21,   NA 
PSDA_h2_f2 <-> PSDA_h2_f2, e22,   NA 
CCA_f2 <-> CCA_f2, e23,   NA 
PSDA_h1_f3 <-> PSDA_h1_f3, e31,   NA 
PSDA_h2_f3 <-> PSDA_h2_f3, e32,   NA 
CCA_f3 <-> CCA_f3, e33,   NA 
F1 <-> F1, NA,    1 
F2 <-> F2, NA,    1
F3 <-> F3, NA,    1
F1 <-> F2, F1F2, NA
F1 <-> F3, F1F3, NA
F3 <-> F2, F3F2, NA

r <- sem(model, result_data_cov, nrow(result_data), maxiter=50000, iterlim=10000)





setwd("C:\\Users\\Anti\\Desktop\\sas\\SASUniversityEdition\\myfolders")

result_data <- read.csv("RESULT.csv", header=TRUE)

range <- 1:10370
average <- 64
plot(SMA(result_data$Factor1[range], average), type="l", ylim=c(-3, 3))
lines(SMA(result_data$Factor3[range], average), type="l", col="red")
lines(SMA(result_data$Factor2[range], average), type="l", col="blue")

maximums <- unlist(apply(cbind(result_data$Factor1,result_data$Factor3,result_data$Factor2), 1, which.max))
minimums <- unlist(apply(cbind(result_data$Factor1,result_data$Factor3,result_data$Factor2), 1, which.min))

plot(SMA(maximums[range], 256), type="l")
lines(SMA(minimums[range], 256), col="red")
offs <- 0
points(new_cols[,1]+offs)
points(new_cols[,2]+offs, col="red")
points(new_cols[,3]+offs, col="blue")



setwd("../data/")
result_data <- read.csv("test5_results_2_all.csv", header=TRUE, sep=" ")
setwd("../data/")
trial_data <- read.csv("test5_targets_2.csv", header=TRUE, sep=" ")

for (i in 1:nrow(new_cols)) {
    new_cols[i,] <- new_cols[i,] + c(rnorm(1, sd=0.5), rnorm(1, sd=0.5), rnorm(1, sd=0.5))
}

result_data["t1"] <- new_cols[,1]
result_data["t2"] <- new_cols[,2]
result_data["t3"] <- new_cols[,3]

write.csv(result_data, "C:\\Users\\Anti\\Desktop\\sas\\SASUniversityEdition\\myfolders\\result_with_cols2.csv")




setwd("C:\\Users\\Anti\\Desktop\\sas\\SASUniversityEdition\\myfolders")

scores <- read.csv("SCORES.csv", header=TRUE)

colnames(scores) <- c("f1", "f2", "f3")

range <- 1:10433

library(TTR)
plot(SMA(scores$f1[range],256), type="l")
lines(SMA(scores$f2[range],256), col="red")
lines(SMA(scores$f3[range],256), col="blue")
points(new_cols[range,][,1]-2)
points(new_cols[range,][,2]-2, col="red")
points(new_cols[range,][,3]-2, col="blue")

par(mfrow=c(1,1))
plot(SMA(scores$PSDA_sum_f1[range],64), type="l", ylim=c(0,6))
lines(SMA(scores$PSDA_h1_f1[range],64))
lines(SMA(scores$CCA_f1[range],64))
points(new_cols[range,][,1]-1)
points(new_cols[range,][,2]-1, col="red")
points(new_cols[range,][,3]-1, col="blue")

lines(SMA(scores$PSDA_sum_f2[range],64), type="l", ylim=c(0,6), col="red")
lines(SMA(scores$PSDA_h1_f2[range],64), col="red")
lines(SMA(scores$CCA_f2[range],64), col="red")
points(new_cols[range,][,1]-1)
points(new_cols[range,][,2]-1, col="red")
points(new_cols[range,][,3]-1, col="blue")





setwd("../data/")
result_data <- read.csv("test5_results_2_all.csv", header=TRUE, sep=" ")

result_data <- apply(result_data, 2, function(x) {SMA(x, 64)})[64:nrow(result_data),]

write.csv(result_data, "C:\\Users\\Anti\\Desktop\\sas\\SASUniversityEdition\\myfolders\\moving_average.csv")




library(MARSS)
Z.vals = list(
  "z11", 0, 0,
  "z21", "z22", 0,
  "z31", "z32", "z33",
  "z41", "z42", "z43",
  "z51", "z52", "z53",
  "z61", "z62", "z63"
)
N.ts = 6
Z = matrix(Z.vals, nrow=N.ts, ncol=3, byrow=TRUE)
Q = B = diag(1,3)
R.vals = list(
"r11",0,0,0,0,0,
0,"r22",0,0,0,0,
0,0,"r33",0,0,0,
0,0,0,"r44",0,0,
0,0,0,0,"r55",0,
0,0,0,0,0,"r66")
R = matrix(R.vals, nrow=N.ts, ncol=N.ts, byrow=TRUE)
x0 = U = matrix(0, nrow=3, ncol=1)
A = matrix(0, nrow=6, ncol=1)
V0 = diag(5,3)
dfa.model = list(Z=Z, A="zero", R=R, B=B, U=U, Q=Q, x0=x0, V0=V0)
cntl.list = list(maxit=50)
kemz.3 = MARSS(t(as.matrix(result_data)), model=dfa.model, control=cntl.list)



# load the data (there are 3 datasets contained here)
data(lakeWAplankton)
# we want lakeWAplanktonTrans, which has been transformed
# so the 0s are replaced with NAs and the data z-scored
dat = lakeWAplanktonTrans
# use only the 10 years from 1980-1989
plankdat = dat[dat[,"Year"]>=1980 & dat[,"Year"]<1990,]
# create vector of phytoplankton group names
phytoplankton = c("Cryptomonas", "Diatoms", "Greens",
"Unicells", "Other.algae")
# get only the phytoplankton
dat.spp.1980 = plankdat[,phytoplankton]
dat.spp.1980 = t(dat.spp.1980)





library(TTR)
setwd("../data/")
result_data <- read.csv("test5_results_3_all.csv", header=TRUE, sep=" ")
setwd("../data/")
trial_data <- read.csv("test5_targets_3.csv", header=TRUE, sep=" ")

classify_results <- function(method_data) {
    result <- matrix(NA, ncol=3, nrow=nrow(method_data))
    for (i in 1:nrow(method_data)) {
        index <- order(method_data[i,], decreasing=TRUE)
        #result[i,] <- c(i, index[1], method_data[i,index[1]]-method_data[i,index[2]])
        result[i,] <- c(i, index[1], method_data[i,index[1]]/sum(method_data[i,]))
    }
    result
}

step <- 1
sma <- 1
plot_roc <- function(result, rep, target=0){
    index <- order(result[,3], decreasing=TRUE)
    roc_x <- c(0)
    roc_y <- c(0)
    x_count <- 0
    y_count <- 0
    for (i in 1:nrow(result)) {
        r <- result[index[i],]
        for (j in 1:nrow(trial_data)) {
            row <- trial_data[j,]
            packet <- r[1]*step+256-step + sma - 1
            if (packet <= row$Stop && packet >= row$Start) {
                if (rep || index[i] == 1 || index[i] != 1 && result[index[i]-1,2] != r[2]) { # no repeating
                    len <- y_count+x_count+1
                    if (target != 0) {
                        expected_target = target
                    } else {
                        expected_target = row$Target
                    }
                    if (r[2] == expected_target) {
                        roc_x <- c(roc_x, roc_x[len])
                        roc_y <- c(roc_y, roc_y[len]+1)
                        y_count <- y_count + 1
                    } else {
                        roc_x <- c(roc_x, roc_x[len]+1)
                        roc_y <- c(roc_y, roc_y[len])
                        x_count <- x_count + 1
                    }
                }
                break
            }
        }
    }
    roc_x <- roc_x/x_count
    roc_y <- roc_y/y_count
    list(x=roc_x, y=roc_y)
}



result_avg <- as.data.frame(apply(result_data, 2, function(x){SMA(x, 64)}))[64:nrow(result_data),]

plot(plot_roc(classify_results(result_data[c("CCA_f1", "CCA_f2", "CCA_f3")]), TRUE), type="l")
lines(plot_roc(classify_results(result_data[c("PSDA_sum_f1", "PSDA_sum_f2", "PSDA_sum_f3")]), TRUE),
      type="l", col="blue")
lines(plot_roc(classify_results(result_data[c("PSDA_h1_f1", "PSDA_h1_f2", "PSDA_h1_f3")]), TRUE),
      type="l", col="red")
lines(plot_roc(classify_results(result_data[c("PSDA_h2_f1", "PSDA_h2_f2", "PSDA_h2_f3")]), FALSE),
      type="l", col="green")
abline(0,1)

classification <- classify_results(result_data)

line <- plot_roc(classify_results(scores))
lines(plot_roc(classify_results(scores)), type="l", col="blue")


scores <- read.csv("SCORES.csv", header=TRUE)
colnames(scores) <- c("f1", "f2", "f3")
lines(plot_roc(classify_results(scores)), type="l", col="orange")



# No repeating made h2 better.


s <- read.table("C:\\Users\\Anti\\Desktop\\PycharmProjects\\MAProject\\R\\s.txt")

test <- as.data.frame(apply(result_data[c("PSDA_h2_f1","PSDA_h2_f2","PSDA_h2_f3",
                                          "CCA_f1", "CCA_f2", "CCA_f3")], 2, function(x){scale(x)}))
factor_scores <- as.matrix(test) %*% (as.matrix(s))

lines(plot_roc(classify_results(factor_scores), TRUE), type="l", col="red")


# Lets try time series factor analysis (TSFA)


library(tsfa)

target <- rbind(c(1,0,0),
                c(0,1,0),
                c(0,0,1),
                c(1,0,0),
                c(0,1,0),
                c(0,0,1))

fit = estTSFmodel(ts(result_data[c("CCA_f1", "CCA_f2", "CCA_f3", "PSDA_h1_f1","PSDA_h1_f2","PSDA_h1_f3")]), 3, rotation="targetQ", GPFargs = list(Target=target))

scores = as.ts(apply(fit$f, 2, function(x){scale(x)}))
#plot(scores)

classified <- classify_results(scores)
plot(plot_roc(classified, FALSE, 1), type="l")
lines(plot_roc(classified, FALSE, 2), type="l", col="red")
lines(plot_roc(classified, FALSE, 3), type="l", col="blue")
abline(0, 1)

library("ROCR")


pred <- prediction(classified[,3], true_labels == 1)
perf <- performance(pred, "tpr", "fpr")
plot(perf, add=T)
abline(0,1)


# Classification, let's start out simple and move on to more complex


setwd("../data/")
result_data <- read.csv("test5_results_2_all.csv", header=TRUE, sep=" ")
setwd("../data/")
trial_data <- read.csv("test5_targets_2.csv", header=TRUE, sep=" ")

get_true_labels = function(trial_data, step=1, sma=1) {
    true_labels = c()
    trial_index <- 1
    row <- trial_data[trial_index,]
    packet_count = trial_data[nrow(trial_data),]$Stop
    for (i in seq(256-step, packet_count, step)) {
        if (i > row$Stop) {
            trial_index <- trial_index + 1
            row <- trial_data[trial_index,]
        }
        if (i != packet_count) {
            true_labels = c(true_labels, row$Target)
        }
    }
    true_labels
}

true_labels = get_true_labels(trial_data)

color = c("black", "red", "blue")
psda1 <- c("PSDA_h1_f1","PSDA_h1_f2","PSDA_h1_f3")
psdasum <- c("PSDA_sum_f1","PSDA_sum_f2","PSDA_sum_f3")
psda2 <- c("PSDA_h2_f1","PSDA_h2_f2","PSDA_h2_f3")
cca <- c("CCA_f1", "CCA_f2", "CCA_f3")

method_names = list(PSDA_h1 = psda1,
                    PSDA_sum = psdasum,
                    PSDA_h2 = psda2,
                    CCA=cca)
library(ROCR)
par(mfrow=c(2,2))
plot_rocs = function(input_data, labels, method_names) {
    result=input_data[["result"]]
    if (is.null(result)) {
        result=input_data
    }
    overall=input_data[["overall"]]
    for (j in 1:length(method_names)){
        for (i in 1:length(method_names[[j]])) {
            pred <- prediction(result[method_names[[j]][i]], labels == i)
            perf <- performance(pred, "tpr", "fpr")
            plot(perf, add=i!=1, col=color[i], main=names(method_names)[j])
        }
        if (!is.null(overall)) {
            pred <- prediction(overall[,2], labels == overall[,1])
            perf <- performance(pred, "tpr", "fpr")
            plot(perf, add=T, main=names(method_names)[j], col="green")
        }
        abline(0,1)
        #legend("bottomright", legend=c(6.67, 7.5, 8.57), col=color[i])
    }
}
#         roc_x <- c(0)
#         roc_y <- c(0)
#         x_count <- 0
#         y_count <- 0
#         sorted <- apply(result[method_names[[j]]], 1, function(x){sort(x, decreasing=TRUE)})
#         index <- order(sorted[])
#         for (i in 1:nrow(result)){
#             r = result[index[i], method_names[[j]]]
#             index2 <- order(r, decreasing=TRUE)
#             len <- y_count+x_count+1
#             if (index2[1] == labels[i]) {
#                 roc_x <- c(roc_x, roc_x[len])
#                 roc_y <- c(roc_y, roc_y[len]+1)
#                 y_count <- y_count + 1
#             } else {
#                 roc_x <- c(roc_x, roc_x[len]+1)
#                 roc_y <- c(roc_y, roc_y[len])
#                 x_count <- x_count + 1
#             }
#         }
#         roc_x <- roc_x/x_count
#         roc_y <- roc_y/y_count
#         lines(roc_x, roc_y, col="green")

setwd("C:\\Users\\Anti\\Desktop\\sas\\SASUniversityEdition\\myfolders")
sas_factor1 <- read.csv("RESULT1.csv", header=TRUE)
sas_factor2 <- read.csv("RESULT2.csv", header=TRUE)
sas_factor3 <- read.csv("RESULT3.csv", header=TRUE)
sas_factor <- data.frame(sas_factor1["Factor1"],
                         sas_factor2["Factor1"],
                         sas_factor3["Factor1"])
names(sas_factor) <- c("Factor1","Factor2","Factor3")
plot_rocs(sas_factor, true_labels, list(sas=c("Factor1","Factor2","Factor3")))

sma_sas <- as.data.frame(apply(sas_factor, 2, function(x){SMA(x, 64)}))[64:nrow(sas_factor),]
plot_rocs(sma_sas, true_labels[64:nrow(sas_factor)], list(sas=c("Factor1","Factor2","Factor3")))


sas_factor <- read.csv("RESULT.csv", header=TRUE)
sas_factor["Factor2"] <- sas_factor["Factor1"]
sas_factor["Factor1"] <- sas_factor["Factor3"]
sas_factor["Factor3"] <- sas_factor["Factor4"]

frequency_combinations = list(f1=c("CCA_f1", "PSDA_h1_f1", "PSDA_h2_f1"),
                              f2=c("CCA_f2", "PSDA_h1_f2", "PSDA_h2_f2"),
                              f3=c("CCA_f3", "PSDA_h1_f3", "PSDA_h2_f3"))

frequency_names = list(TSFA=c("V1","V2","V3"))

scores = find_scores(result_data, frequency_combinations, 1)
plot_rocs(scores, true_labels, frequency_names)

positive_data = apply(sas_factor, 2, function(x){x-min(x)+1})
classes = classify_results(positive_data, list(sas=c("Factor1","Factor2","Factor3")))
plot_rocs(classes, true_labels[1:11884], list(sas=c("Factor1","Factor2","Factor3")))



plot_rocs(result_data, true_labels, method_names)
library(TTR)
sma_result <- as.data.frame(apply(result_data, 2, function(x){SMA(x, 64)}))[64:nrow(result_data),]
plot_rocs(sma_result, true_labels[64:nrow(result_data)], method_names)

library(tsfa)
target2 <- rbind(c(1,0,0),c(0,1,0),c(0,0,1),c(1,0,0),c(0,1,0),c(0,0,1))
target3 <- rbind(c(1,0,0),c(0,1,0),c(0,0,1),c(1,0,0),c(0,1,0),c(0,0,1),c(1,0,0),c(0,1,0),c(0,0,1))

method_combinations = list(CCA_PSDA_h1=c("CCA_f1", "CCA_f2", "CCA_f3", "PSDA_h1_f1","PSDA_h1_f2","PSDA_h1_f3"),
                           CCA_PSDA_h2=c("CCA_f1", "CCA_f2", "CCA_f3", "PSDA_h2_f1","PSDA_h2_f2","PSDA_h2_f3"),
                           CCA_PSDA_sum=c("CCA_f1", "CCA_f2", "CCA_f3", "PSDA_sum_f1","PSDA_sum_f2","PSDA_sum_f3"),
                           CCA_PSDA_h1_h2=c("CCA_f1", "CCA_f2", "CCA_f3", "PSDA_h1_f1","PSDA_h1_f2","PSDA_h1_f3","PSDA_h2_f1","PSDA_h2_f2","PSDA_h2_f3"))


factor_names = list(CCA_PSDA_h1=c("V1","V2","V3"),#c("CCA_PSDA_h11","CCA_PSDA_h12","CCA_PSDA_h13"),
                    CCA_PSDA_h2=c("V4","V5","V6"),#c("CCA_PSDA_h21","CCA_PSDA_h22","CCA_PSDA_h23"),
                    CCA_PSDA_sum=c("V7","V8","V9"),#c("CCA_PSDA_sum1","CCA_PSDA_sum2","CCA_PSDA_sum3"),
                    CCA_PSDA_h1_h2=c("V10","V11","V12"))#c("CCA_PSDA_h1_h21","CCA_PSDA_h1_h22","CCA_PSDA_h1_h23"))


find_scores = function(input_data, method_combinations, nfact=3) {
    result = matrix(NA, nrow=nrow(input_data), ncol=length(method_combinations)*nfact)
    for (i in 1:length(method_combinations)) {
        if (length(method_combinations[[i]])==6) {
            target = target2
        } else {
            target = target3
        }
        if (nfact==3) {
            fit = estTSFmodel(ts(input_data[method_combinations[[i]]]),
                              3, rotation="targetQ", GPFargs = list(Target=target, maxit=1000))
        } else if (nfact==1) {
            fit = estTSFmodel(ts(input_data[method_combinations[[i]]]), 1)
        }
        scores = as.data.frame(apply(fit$f, 2, function(x){scale(x)}))
        for (j in 1:nfact) {
            result[,(i-1)*nfact+j] = scores[,j]
        }
    }
    as.data.frame(result)
}

scores = find_scores(result_data, method_combinations)
plot_rocs(scores, true_labels, factor_names)
scores = find_scores(sma_result, method_combinations)
plot_rocs(scores, true_labels[64:nrow(result_data)], factor_names)


fit = estTSFmodel(ts(sma_result[c("PSDA_h2_f1","PSDA_h2_f2","PSDA_h2_f3", "PSDA_h1_f1","PSDA_h1_f2","PSDA_h1_f3")]), 3, rotation="targetQ", GPFargs = list(Target=target2))
scores = as.data.frame(apply(fit$f, 2, function(x){scale(x)}))
plot_rocs(scores, true_labels[64:nrow(result_data)], list(Faktorid=c("Factor 1", "Factor 2", "Factor 3")))
fit = estTSFmodel(ts(result_data[c("PSDA_h2_f1","PSDA_h2_f2","PSDA_h2_f3", "PSDA_h1_f1","PSDA_h1_f2","PSDA_h1_f3")]), 3, rotation="targetQ", GPFargs = list(Target=target2))
scores = as.data.frame(fit$f)
plot_rocs(scores, true_labels, list(Faktorid=c("Factor 1", "Factor 2", "Factor 3")))




classify_results <- function(input_data, method_names) {
    result = matrix(NA, nrow=nrow(input_data), ncol=length(method_names)*3)
    overall = matrix(NA, nrow=nrow(input_data), 2)
    for (j in 1:length(method_names)) {
        for (i in 1:nrow(input_data)) {
            method_data <- input_data[i, method_names[[j]]]
            index <- order(method_data, decreasing=TRUE)
            result[i,(j-1)*3+index[1]] <- method_data[index[1]]/sum(method_data)+2
            result[i,(j-1)*3+index[2]] <- method_data[index[2]]/sum(method_data)+1
            result[i,(j-1)*3+index[3]] <- method_data[index[3]]/sum(method_data)
            overall[i,] <- c(index[1], method_data[index[1]]/sum(method_data))
        }
    }
    colnames(result) <- unlist(method_names)
    list(result=as.data.frame(result), overall=overall)
}

positive_data = apply(result_data, 2, function(x){x-min(x)+1})
classes = classify_results(positive_data, method_names)
plot_rocs(classes, true_labels, method_names)

positive_data = apply(result_data, 2, function(x){x-min(x)+1})
classes = classify_results(positive_data, method_names)
classes = as.data.frame(apply(classes, 2, function(x){SMA(x, 64)})[64:nrow(classes),])
plot_rocs(classes, true_labels[64:length(true_labels)], method_names)

positive_sma_data = apply(sma_result, 2, function(x){x-min(x)+1})
classes = classify_results(positive_sma_data, method_names)
plot_rocs(classes, true_labels[64:length(true_labels)], method_names)

scores = find_scores(result_data, method_combinations)
positive_data = apply(scores, 2, function(x){x-min(x)+1})
classes = classify_results(positive_data, factor_names)
plot_rocs(classes, true_labels, factor_names)

scores = find_scores(sma_result, method_combinations)
positive_data = apply(scores, 2, function(x){x-min(x)+1})
classes = classify_results(positive_data, factor_names)
plot_rocs(classes, true_labels[64:length(true_labels)], factor_names)



# Lähtetunnuste dispersioonid on ühed. Kommunaliteetide summa jagatud tunnuste arvuga. Peakomponendid - maksimeerib dispersiooni. Faktormeetod - kirjelda ära korrelatsioonimaatriks (see esikohal ja siis vaadata ka dispersiooni). 

# Kinnitavas p-value võib olla liiga tundlik, liiga väike.

