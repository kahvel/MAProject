from main import normaliseAndScale, readData, binariseLabels, getTrueLabels, buildDataMatrix, removePacketsAfterChange, printMetrics
from voting import Voting
import pickle
from sklearn import metrics
from matplotlib import pyplot as plt
import matplotlib2tikz

# models = {
#     "boost": dict(),
#     "logistic": dict(),
#     "SVM_linear": dict(),
#     "SVM_RBF": dict(),
#     "SVM_poly": dict(),
#     "SVM_sigmoid": dict()
# }

test_set = 2
number_of_models = {
    "boost": 4,
    "logistic": 4,
}

print "Test set:", (test_set+1)
print "Used models:", number_of_models

test_data = normaliseAndScale(readData("..\\data\\test5_results_" + str(test_set + 1) + "_all.csv"))

test_labels = readData("..\\data\\test5_targets_" + str(test_set + 1) + ".csv")

labels = getTrueLabels(test_labels)

binarised_labels = [binariseLabels(labels, target) for target in [1,2,3]]

data_matrix = [buildDataMatrix(test_data, target) for target in [0,1,2]]

for target in [0,1,2]:
    data_matrix[target], binarised_labels[target] =\
        removePacketsAfterChange(data_matrix[target], binarised_labels[target], test_labels, 256)

print "The data has been read in!"

estimators = {
    1: [],
    2: [],
    3: []
}

for target in [1, 2, 3]:
    for dataset in [0, 1, 2]:
        if dataset != test_set:
            for model_name in number_of_models:
                for k in range(number_of_models[model_name]):
                    file_name = model_name + "_dataset_" + str(dataset) + "_target_" + str(target) + "_" + str(k) + ".pkl"
                    estimators[target].append((file_name, pickle.load(file('../pickle/models/' + file_name))))


type = "soft"

voters = {
    1: Voting(estimators[1], voting=type), #, weights={True: 0.8, False: 0.2}
    2: Voting(estimators[2], voting=type),
    3: Voting(estimators[3], voting=type)
}
# voters = { # Estimators used in DM project
#     1: Voting(estimators[1][8:10]+estimators[1][23:24], voting=type), #, weights={True: 0.8, False: 0.2}
#     2: Voting(estimators[2][0:8]+estimators[2][21:22], voting=type),
#     3: Voting(estimators[3][0:8]+estimators[3][23:24], voting=type)
# }

print "Models have been read in!"

for target in [1, 2, 3]:
    decision = voters[target].transform(data_matrix[target-1])
    if type == "soft":
        decision = sum(decision).transpose()[0]
    elif type == "hard":
        decision = sum(decision.transpose())
    fpr, tpr, threshold = metrics.roc_curve(binarised_labels[target-1], decision, pos_label=True)
    # printMetrics(fpr, tpr, threshold, 0.99, decision[0], binarised_labels[target-1])
    # printMetrics(fpr, tpr, threshold, 1, decision[0], binarised_labels[target-1])
    prediction = printMetrics(fpr, tpr, threshold, 0.01, decision, binarised_labels[target-1])
    printMetrics(fpr, tpr, threshold, 0, decision, binarised_labels[target-1])
    plt.subplot(2, 2, 1)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot((0, 1), (0, 1))
    plt.subplot(2, 2, target+1)
    axes = plt.gca()
    axes.set_ylim([-0.1, 1.1])
    plt.plot(map(lambda x: x, prediction))
    plt.plot(binarised_labels[target-1], "--")

matplotlib2tikz.save("roc.tex")
plt.show()
