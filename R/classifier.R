
setwd("../data/")
trial_data <- read.csv("test5_targets_2.csv", header=TRUE, sep=" ")
#eeg_data <- read.csv("test5_2.csv", header=TRUE, sep=" ")[c("O1", "O2", "P7", "P8")]
frequency_data <- read.csv("test5_freq_2.csv", header=TRUE, sep=" ")
result_data <- read.csv("test5_results_2_all.csv", header=TRUE, sep=" ")




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




library(ROCR)
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




library(TTR)
classify_results <- function(input_data, method_names, sma=1) {
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
    if (sma > 1) {
        result <- apply(result, 2, function(x){SMA(x, 64)})
        overall <- SMA(overall, 64)
    }
    colnames(result) <- unlist(method_names)
    list(result=as.data.frame(result), overall=overall)
}




