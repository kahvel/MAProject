import numpy as np
from main import readData, getTrueLabels, normaliseAndScale, binariseLabels, buildDataMatrix, removePacketsAfterChange
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.svm import SVC
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pickle


np.random.seed(99)

data = list()
data.append(normaliseAndScale(readData("..\\data\\test5_results_1_all.csv")))
data.append(normaliseAndScale(readData("..\\data\\test5_results_2_all.csv")))
data.append(normaliseAndScale(readData("..\\data\\test5_results_3_all.csv")))

label_data = list()
label_data.append(readData("..\\data\\test5_targets_1.csv"))
label_data.append(readData("..\\data\\test5_targets_2.csv"))
label_data.append(readData("..\\data\\test5_targets_3.csv"))

labels = [getTrueLabels(label) for label in label_data]

binarised_labels = dict()
binarised_labels[1] = [binariseLabels(label, 1) for label in labels]
binarised_labels[2] = [binariseLabels(label, 2) for label in labels]
binarised_labels[3] = [binariseLabels(label, 3) for label in labels]

data_matrix = dict()
data_matrix[1] = [buildDataMatrix(d, 0) for d in data]
data_matrix[2] = [buildDataMatrix(d, 1) for d in data]
data_matrix[3] = [buildDataMatrix(d, 2) for d in data]

for target in [1, 2, 3]:
    for dataset in [0, 1, 2]:
        data_matrix[target][dataset], binarised_labels[target][dataset] =\
            removePacketsAfterChange(data_matrix[target][dataset], binarised_labels[target][dataset], label_data[dataset], 256)
        print len(data_matrix[target][dataset]), len(data_matrix[target][dataset][0])
        print len(binarised_labels[target][dataset])


def save_model(model_class, name, **kwargs):
    model = model_class(**kwargs)
    model.classes_ = [True, False]
    model.fit(train_data, train_labels)
    model_name = name + "_dataset_" + str(dataset) + "_target_" + str(target) + "_" + str(k) + ".pkl"
    pickle.Pickler(file("../pickle/models/" + model_name, "w")).dump(model)


for dataset in [0, 1, 2]:
    for target in [1, 2, 3]:
        for k in range(4):
            positive_train_labels = [i for i in range(len(binarised_labels[target][dataset])) if binarised_labels[target][dataset][i]]
            positive_train_labels = list(np.random.choice(positive_train_labels, size=int(len(positive_train_labels)*0.8), replace=False))
            negative_train_labels = list(np.random.choice([i for i in range(len(binarised_labels[target][dataset])) if i not in positive_train_labels], replace=False, size=len(positive_train_labels)))
            train_data = [data_matrix[target][dataset][i] for i in positive_train_labels + negative_train_labels]
            train_labels = [binarised_labels[target][dataset][i] for i in positive_train_labels + negative_train_labels]
            save_model(AdaBoostClassifier, "boost", n_estimators=100, learning_rate=1,
                       base_estimator=DecisionTreeClassifier(max_depth=1, class_weight={True: 0.8, False: 0.2}))
            save_model(SVC, "SVM_RBF", kernel="rbf", probability=True, class_weight={True: 0.8, False: 0.2})
            save_model(SVC, "SVM_poly", kernel="poly", probability=True, class_weight={True: 0.8, False: 0.2}, degree=2)
            save_model(SVC, "SVM_linear", kernel="linear", probability=True, class_weight={True: 0.8, False: 0.2})
            save_model(SVC, "SVM_sigmoid", kernel="sigmoid", probability=True, class_weight={True: 0.8, False: 0.2})
            save_model(LogisticRegression, "logistic", class_weight={True: 0.8, False: 0.2})
