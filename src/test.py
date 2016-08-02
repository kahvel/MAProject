from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, RFECV
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cross_validation
import sklearn.metrics
import scipy
import pickle


target_count = 3
group_names = ["CCA", "PSDA_h1", "PSDA_h2"]#, "PSDA_sum"]
col_names = {
    "CCA": ["CCA_f1", "CCA_f3", "CCA_f5"],
    "PSDA_h1": ["PSDA_h1_f1", "PSDA_h1_f3", "PSDA_h1_f5"],
    "PSDA_h2": ["PSDA_h2_f1", "PSDA_h2_f3", "PSDA_h2_f5"],
    # "PSDA_sum": ["PSDA_sum_f1", "PSDA_sum_f2", "PSDA_sum_f3"],
    "SNR_h1": ["SNR_h1_f1", "SNR_h1_f3", "SNR_h1_f5"],
    "SNR_h2": ["SNR_h2_f1", "SNR_h2_f3", "SNR_h2_f5"],
    "LRT": ["LRT_f1", "LRT_f3", "LRT_f5"],
}
# col_names = {
#     "CCA": ["CCA_f1", "CCA_f2", "CCA_f3"],
#     "PSDA_h1": ["PSDA_h1_f1", "PSDA_h1_f2", "PSDA_h1_f3"],
#     "PSDA_h2": ["PSDA_h2_f1", "PSDA_h2_f2", "PSDA_h2_f3"],
#     # "PSDA_sum": ["PSDA_sum_f1", "PSDA_sum_f2", "PSDA_sum_f3"],
#     "SNR_h1": ["SNR_h1_f1", "SNR_h1_f2", "SNR_h1_f3"],
#     "SNR_h2": ["SNR_h2_f1", "SNR_h2_f2", "SNR_h2_f3"],
#     "LRT": ["LRT_f1", "LRT_f2", "LRT_f3"],
# }
# col_names = {
#     "CCA": ["CCA_f1", "CCA_f2", "CCA_f3", "CCA_f4", "CCA_f5"],
#     "PSDA_h1": ["PSDA_h1_f1", "PSDA_h1_f2", "PSDA_h1_f3", "PSDA_h1_f4", "PSDA_h1_f5"],
#     "PSDA_h2": ["PSDA_h2_f1", "PSDA_h2_f2", "PSDA_h2_f3", "PSDA_h2_f4", "PSDA_h2_f5"],
#     # "PSDA_sum": ["PSDA_sum_f1", "PSDA_sum_f2", "PSDA_sum_f3"],
#     "SNR_h1": ["SNR_h1_f1", "SNR_h1_f2", "SNR_h1_f3", "SNR_h1_f4", "SNR_h1_f5"],
#     "SNR_h2": ["SNR_h2_f1", "SNR_h2_f2", "SNR_h2_f3", "SNR_h2_f4", "SNR_h2_f5"],
#     "LRT": ["LRT_f1", "LRT_f2", "LRT_f3", "LRT_f4", "LRT_f5"],
# }


def readData(file_name, sep=" "):  # Is there ID???
    content = open(file_name).readlines()
    names = content[0].strip().split(sep)
    result = {name: [] for name in names}
    for line in content[1:]:
        results = line.strip().split(sep)[1:]
        for i in range(len(names)):
            result[names[i]].append(float(results[i]))
    return result


def buildDataMatrix(data):
    matrix = []
    for group_name in group_names:
        for name in col_names[group_name]:
            matrix.append(data[name])
    return np.transpose(matrix)


def buildDataMatrixPerTarget(data):
    matrices = [[] for _ in range(target_count)]
    for group_name in group_names:
        for i, name in enumerate(col_names[group_name]):
            matrices[i].append(data[name])
    for target in range(len(matrices)):
        matrices[target] = np.transpose(matrices[target])
    return matrices


def combineSamples(data, labels, sample_count):
    new_data = []
    new_labels = []
    previous_labels = []
    new_sample = []
    for sample, label in zip(data, labels):
        if len(previous_labels) == sample_count:
            if all(map(lambda x: x == previous_labels[0], previous_labels)):
                new_data.append(np.concatenate(new_sample))
                new_labels.append(label)
            del previous_labels[0]
            del new_sample[0]
            previous_labels.append(label)
            new_sample.append(sample)
        else:
            previous_labels.append(label)
            new_sample.append(sample)
    return np.array(new_data), new_labels
    # for i, sample in enumerate(data[sample_count-1:]):
    #     new_data.append(np.concatenate(data[i:i+sample_count]))
    # return new_data


def getMinMax(data):
    min_max = {}
    for group in group_names:
        mins = []
        maxs = []
        for name in col_names[group]:
            mins.append(min(data[name]))
            maxs.append(max(data[name]))
        minimum = min(mins)
        maximum = max(maxs)
        min_max[group] = (minimum, maximum)
    return min_max


def getScalingFunction(min_max):
    functions = {}
    for group in group_names:
        minimum, maximum = min_max[group]
        functions[group] = lambda x: (x-minimum)/(maximum-minimum)
    return functions


def scale(data, functions):
    scaled_data = {}
    for group in group_names:
        for name in col_names[group]:
            scaled_data[name] = list(map(functions[group], data[name]))
    return scaled_data


def addRatio(data_matrices, data_matrix):
    for i in range(target_count):
        data_matrices[i] = data_matrices[i].tolist()
    for i, rows in enumerate(zip(*data_matrices)):
        for features in zip(*rows):
            feature_sum = sum(features)
            feature = [features[j]/feature_sum for j in range(target_count)]
            for j in range(target_count):
                data_matrices[j][i].append(feature[j])
            data_matrix[i].extend([feature[0], feature[1], feature[2]])


def readDataMultipleFiles(file_numbers):
    all_raw_data = {}
    for file in file_numbers:
        raw_data = readData("U:\\data\\test\\5_targets\\result" + str(file) + ".csv")
        # raw_data = readData("U:\\data\\my\\3_75_results\\results01.csv")
        for key in raw_data:
            if key in all_raw_data:
                all_raw_data[key].extend(raw_data[key])
            else:
                all_raw_data[key] = raw_data[key]
    labels = map(int, all_raw_data["class"])
    return all_raw_data, labels


def buildMatricesAndLabels(all_raw_data, labels, scaling_functions):
    raw_data = scale(all_raw_data, scaling_functions)

    data_matrices = buildDataMatrixPerTarget(raw_data)

    data_matrix = buildDataMatrix(raw_data).tolist()
    addRatio(data_matrices, data_matrix)

    to_delete = 9
    data_matrix = np.delete(data_matrix, [i for i in range(to_delete)], 1)
    for j in range(target_count):
        data_matrices[j] = np.delete(data_matrices[j], [i for i in range(to_delete/target_count)], 1)

    look_back = 10

    for i in range(target_count):
        data_matrices[i], _ = combineSamples(data_matrices[i], labels, look_back)

    data_matrix, labels = combineSamples(data_matrix, labels, look_back)

    labels_binary = []
    for i in range(target_count):
        labels_binary.append(map(lambda x: x == i+1, labels))
    return data_matrix, data_matrices, labels, labels_binary


raw_data, labels = readDataMultipleFiles([2,3,4])
min_max = getMinMax(raw_data)
scaling_functions = getScalingFunction(min_max)
data_matrix, data_matrices, labels, labels_binary = buildMatricesAndLabels(raw_data, labels, scaling_functions)

print data_matrix.shape
print data_matrices[0].shape, data_matrices[1].shape

for i in range(target_count):
    print sum(labels_binary[i])

models = []
for i in range(target_count):
    # models.append(RandomForestClassifier(n_estimators=10, max_depth=int(sum(labels_binary[i])/100)+1))#, class_weight={True: 0.2, False:0.2})
    # models.append(AdaBoostClassifier(n_estimators=50))#, class_weight={True: 0.2, False:0.2})
    # models.append(KNeighborsClassifier(n_neighbors=10))
    models.append(LinearDiscriminantAnalysis())
    models[-1].classes_ = [False, True]

sample_weights = []
for i in range(target_count):
    sample_weights.append(np.array(map(lambda x: 0.5 if x == 1 else 0.2, labels_binary[i])))

for i in range(target_count):
    models[i].fit(data_matrices[i], labels_binary[i])#, sample_weight=sample_weights[i])

predicted = []
for i in range(target_count):
    predicted.append(models[i].predict(data_matrices[i]))

for i in range(target_count):
    print i
    print sklearn.metrics.confusion_matrix(labels_binary[i], predicted[i])

prediction = []
for i in range(target_count):
    prediction.append(sklearn.cross_validation.cross_val_predict(models[i], data_matrices[i], labels_binary[i], cv=5))#, fit_params={"sample_weight": sample_weight1})
    print "CV " + str(i)
    print sklearn.metrics.confusion_matrix(labels_binary[i], prediction[i])

# use_prediction = True
# test_data_matrix, test_data_matrices, test_labels, test_labels_binary = buildMatricesAndLabels([11])
# test_predictions = []
# for i, target_data in enumerate(test_data_matrices):
#     test_prediction = []
#     for features in target_data:
#         if not use_prediction:
#             test_prediction.append(models[i].decision_function([features])[0])  # score for classes_[1]
#         else:
#             test_prediction.append(models[i].predict_proba([features])[0])
#     test_predictions.append(test_prediction)
#
# if use_prediction:
#     test_predictions = map(lambda matrix: map(lambda x: x[1], matrix), test_predictions)  # take proba for classes_[1]
# print test_predictions
#
# plt.figure()
# thresholds = []
# for i in range(target_count):
#     fpr, tpr, threshold = sklearn.metrics.roc_curve(test_labels_binary[i], test_predictions[i], pos_label=True)
#     plt.subplot(2, 2, i+1)
#     plt.plot(fpr, tpr)
# #     thresholds.append(threshold)
# #     print len(threshold)
# #
# # for i, pred in enumerate(zip(*test_predictions)):
# #     print i, pred, test_labels[i], np.argmax(pred)


def test_model(model):
    model.fit(data_matrix, labels)
    # feature_selector = RFECV(estimator=model)
    # # feature_selector = SelectKBest(score_func=chi2, k=20)
    # feature_selector.fit(data_matrix, labels)
    # data_matrix = feature_selector.transform(data_matrix)
    # print data_matrix.shape
    # model.fit(data_matrix, labels)
    predicted = model.predict(data_matrix)
    print sklearn.metrics.confusion_matrix(labels, predicted)
    prediction = sklearn.cross_validation.cross_val_predict(model, data_matrix, labels, cv=10)
    print sklearn.metrics.confusion_matrix(labels, prediction)


def plot_lda(model, labels, data_matrix):
    if target_count == 3:
        x = model.transform(data_matrix)
        labels = np.array(labels)
        plt.figure()
        for c, i, target_name in zip("rgb", [1, 2, 3], [1, 2, 3]):
            plt.scatter(x[labels == i, 0], x[labels == i, 1], c=c, label=target_name, marker="o")
        plt.legend()
        plt.title('LDA')


def multiclassRoc(test_predictions, test_labels_binary):
    test_predictions = np.transpose(test_predictions)
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    thresholds_for_bci = []
    for i in range(target_count):
        fpr[i], tpr[i], thresholds[i] = sklearn.metrics.roc_curve(test_labels_binary[i], test_predictions[i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
    for i in range(target_count):
        print i
        for false_positive_rate, true_positive_rate, threshold in zip(fpr[i], tpr[i], thresholds[i]):
            if false_positive_rate > 0.05:
                print false_positive_rate, true_positive_rate, threshold
                thresholds_for_bci.append(threshold)
                break
    fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(np.array(test_labels_binary).ravel(), test_predictions.ravel())
    roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(target_count)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(target_count):
        mean_tpr += scipy.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= target_count

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             linewidth=2)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             linewidth=2)
    for i in range(target_count):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    return thresholds_for_bci


model = RandomForestClassifier(n_estimators=10, max_depth=3)
print "Random Forest"
test_model(model)

model_lda = LinearDiscriminantAnalysis()
print "LDA"
test_model(model_lda)

use_prediction = False
raw_test_data, test_labels = readDataMultipleFiles([3])
test_data_matrix, test_data_matrices, test_labels, test_labels_binary = buildMatricesAndLabels(raw_test_data, test_labels, scaling_functions)
test_predictions = []
for features in test_data_matrix:
    if not use_prediction:
        test_predictions.append(model_lda.decision_function([features])[0])  # score for classes_[1]
    else:
        test_predictions.append(model_lda.predict_proba([features])[0])

for i in range(target_count):
    print sum(test_labels_binary[i])

thresholds_for_bci = multiclassRoc(test_predictions, test_labels_binary)

# model = SVC(C=1000, kernel="poly", degree=2)
# print "SVM"
# test_model(model)

# pickle.Pickler(file("U:\\data\\test\\5_targets\\model0.pkl", "w")).dump(model_lda)
# pickle.Pickler(file("U:\\data\\test\\5_targets\\model0_mm.pkl", "w")).dump(min_max)
# pickle.Pickler(file("U:\\data\\test\\5_targets\\model0_thresh.pkl", "w")).dump(thresholds_for_bci)

# print model_lda.coef_
# plt.figure()
# # plt.subplot(2,2,1)
# plt.scatter(range(len(model_lda.coef_[0])), model_lda.coef_[0], c="b")
# # plt.subplot(2,2,2)
# plt.scatter(range(len(model_lda.coef_[1])), model_lda.coef_[1], c="g")
# # plt.subplot(2,2,3)
# plt.scatter(range(len(model_lda.coef_[2])), model_lda.coef_[2], c="r")
plot_lda(model_lda, labels, data_matrix)
plot_lda(model_lda, test_labels, test_data_matrix)
plt.show()

