from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, RFECV
import numpy as np


group_names = ["CCA", "PSDA_h1", "PSDA_h2", "SNR_h1", "SNR_h2"]#, "PSDA_sum"]
col_names = {
    "CCA": ["CCA_f1", "CCA_f2", "CCA_f3"],
    "PSDA_h1": ["PSDA_h1_f1", "PSDA_h1_f2", "PSDA_h1_f3"],
    "PSDA_h2": ["PSDA_h2_f1", "PSDA_h2_f2", "PSDA_h2_f3"],
    # "PSDA_sum": ["PSDA_sum_f1", "PSDA_sum_f2", "PSDA_sum_f3"],
    "SNR_h1": ["SNR_h1_f1", "SNR_h1_f2", "SNR_h1_f3"],
    "SNR_h2": ["SNR_h2_f1", "SNR_h2_f2", "SNR_h2_f3"],
}


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
    matrices = {target: [] for target in range(3)}
    for group_name in group_names:
        for i, name in enumerate(col_names[group_name]):
            matrices[i].append(data[name])
    for target in matrices:
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


def scale(data):
    scaled_data = {}
    for group in group_names:
        mins = []
        maxs = []
        for name in col_names[group]:
            mins.append(min(data[name]))
            maxs.append(max(data[name]))
        minimum = min(mins)
        maximum = max(maxs)
        for name in col_names[group]:
            scaled_data[name] = list(map(lambda x: (x-minimum)/(maximum-minimum), data[name]))
    return scaled_data


def addRatio(data_matrices, data_matrix):
    data_matrices[0] = data_matrices[0].tolist()
    data_matrices[1] = data_matrices[1].tolist()
    data_matrices[2] = data_matrices[2].tolist()
    for i, rows in enumerate(zip(data_matrices[0], data_matrices[1], data_matrices[2])):
        for features in zip(rows[0], rows[1], rows[2]):
            feature_sum = sum(features)
            feature1 = features[0]/feature_sum
            feature2 = features[1]/feature_sum
            feature3 = features[2]/feature_sum
            data_matrices[0][i].append(feature1)
            data_matrices[1][i].append(feature2)
            data_matrices[2][i].append(feature3)
            data_matrix[i].extend([feature1, feature2, feature3])


raw_data = readData("U:\\data\\my\\3_75_results\\results01.csv")
# raw_data = readData("U:\\data\\my\\results1_2_target\\results4.csv")
print raw_data["CCA_f1"]
print raw_data["CCA_f2"]
print raw_data["CCA_f3"]

labels = map(int, raw_data["class"])

raw_data = scale(raw_data)
print raw_data["CCA_f1"]
print raw_data["CCA_f2"]
print raw_data["CCA_f3"]

data_matrices = buildDataMatrixPerTarget(raw_data)
print data_matrices[0][0]
print data_matrices[1][0]
print data_matrices[2][0]

data_matrix = buildDataMatrix(raw_data).tolist()

addRatio(data_matrices, data_matrix)
print data_matrices[0][0]
print data_matrices[1][0]
print data_matrices[2][0]

print data_matrix[0]

look_back = 5

data_matrices[0], _ = combineSamples(data_matrices[0], labels, look_back)
data_matrices[1], _ = combineSamples(data_matrices[1], labels, look_back)
data_matrices[2], _ = combineSamples(data_matrices[2], labels, look_back)

data_matrix, labels = combineSamples(data_matrix, labels, look_back)

print data_matrices[0][-1]
print data_matrices[1][-1]
print data_matrices[2][-1]
print labels
print len(labels)
print data_matrix.shape

labels1 = map(lambda x: x == 1, labels)
labels2 = map(lambda x: x == 2, labels)
labels3 = map(lambda x: x == 3, labels)

print map(int, labels1)
print map(int, labels2)
print map(int, labels3)

print sum(labels1)
print sum(labels2)
print sum(labels3)

# model1 = LinearDiscriminantAnalysis()
# model2 = LinearDiscriminantAnalysis()
# model3 = LinearDiscriminantAnalysis()

model1 = RandomForestClassifier(n_estimators=10, max_depth=3)#, class_weight={True: 0.2, False:0.2})
model2 = RandomForestClassifier(n_estimators=10, max_depth=3)#, class_weight={True: 0.2, False:0.2})
model3 = RandomForestClassifier(n_estimators=10, max_depth=3)#, class_weight={True: 0.2, False:0.2})

# model1 = AdaBoostClassifier(n_estimators=100)#, class_weight={True: 0.2, False:0.2})
# model2 = AdaBoostClassifier(n_estimators=100)#, class_weight={True: 0.2, False:0.2})
# model3 = AdaBoostClassifier(n_estimators=100)#, class_weight={True: 0.2, False:0.2})

# model1 = KNeighborsClassifier(n_neighbors=20)
# model2 = KNeighborsClassifier(n_neighbors=20)
# model3 = KNeighborsClassifier(n_neighbors=20)

sample_weight1 = np.array(map(lambda x: 0.5 if x == 1 else 0.2, labels1))
sample_weight2 = np.array(map(lambda x: 0.5 if x == 1 else 0.2, labels2))
sample_weight3 = np.array(map(lambda x: 0.5 if x == 1 else 0.2, labels3))
# print sample_weight1

model1.fit(data_matrices[0], labels1)#, sample_weight=sample_weight1)
model2.fit(data_matrices[1], labels2)#, sample_weight=sample_weight2)
model3.fit(data_matrices[2], labels3)#, sample_weight=sample_weight3)

predicted1 = model1.predict(data_matrices[0])
predicted2 = model2.predict(data_matrices[1])
predicted3 = model3.predict(data_matrices[2])

import sklearn.metrics

# print predicted
print "1"
print sklearn.metrics.confusion_matrix(labels1, predicted1)
print "2"
print sklearn.metrics.confusion_matrix(labels2, predicted2)
print "3"
print sklearn.metrics.confusion_matrix(labels3, predicted3)

import sklearn.cross_validation

prediction21 = sklearn.cross_validation.cross_val_predict(model1, data_matrices[0], labels1, cv=5)#, fit_params={"sample_weight": sample_weight1})
prediction22 = sklearn.cross_validation.cross_val_predict(model2, data_matrices[1], labels2, cv=5)#, fit_params={"sample_weight": sample_weight2})
prediction23 = sklearn.cross_validation.cross_val_predict(model3, data_matrices[2], labels3, cv=5)#, fit_params={"sample_weight": sample_weight3})
print "CV 1"
print sklearn.metrics.confusion_matrix(labels1, prediction21)
print "CV 2"
print sklearn.metrics.confusion_matrix(labels2, prediction22)
print "CV 3"
print sklearn.metrics.confusion_matrix(labels3, prediction23)


import matplotlib.pyplot as plt


model = LinearDiscriminantAnalysis()
model.fit(data_matrix, labels)

# feature_selector = RFECV(estimator=model)
# # feature_selector = SelectKBest(score_func=chi2, k=20)
# feature_selector.fit(data_matrix, labels)
# data_matrix = feature_selector.transform(data_matrix)
# print data_matrix.shape
# model.fit(data_matrix, labels)

predicted = model.predict(data_matrix)
print sklearn.metrics.confusion_matrix(labels, predicted)

prediction = sklearn.cross_validation.cross_val_predict(model, data_matrix, labels, cv=5)
print sklearn.metrics.confusion_matrix(labels, prediction)

x = model.transform(data_matrix)

labels = np.array(labels)

plt.figure()
for c, i, target_name in zip("rgb", [1, 2, 3], [1, 2, 3]):
    plt.scatter(x[labels == i, 0], x[labels == i, 1], c=c, label=target_name)
plt.legend()
plt.title('LDA')

plt.show()
