from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import classifiers
import numpy as np


group_names = ["CCA", "PSDA_h1", "PSDA_h2"]#, "PSDA_sum"]
col_names = {
    "CCA": ["CCA_f1", "CCA_f2", "CCA_f3"],
    "PSDA_h1": ["PSDA_h1_f1", "PSDA_h1_f2", "PSDA_h1_f3"],
    "PSDA_h2": ["PSDA_h2_f1", "PSDA_h2_f2", "PSDA_h2_f3"],
    # "PSDA_sum": ["PSDA_sum_f1", "PSDA_sum_f2", "PSDA_sum_f3"]
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


def addRatio(data):
    data[0] = data[0].tolist()
    data[1] = data[1].tolist()
    data[2] = data[2].tolist()
    for i, rows in enumerate(zip(data[0], data[1], data[2])):
        for features in zip(rows[0], rows[1], rows[2]):
            feature_sum = sum(features)
            data[0][i].append(features[0]/feature_sum)
            data[1][i].append(features[1]/feature_sum)
            data[2][i].append(features[2]/feature_sum)


raw_data = readData("U:\\data\\my\\results1_2_target\\results0.csv")
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

addRatio(data_matrices)
print data_matrices[0][0]
print data_matrices[1][0]
print data_matrices[2][0]

data_matrices[0], _ = combineSamples(data_matrices[0], labels, 2)
data_matrices[1], _ = combineSamples(data_matrices[1], labels, 2)
data_matrices[2], _ = combineSamples(data_matrices[2], labels, 2)

data_matrix = buildDataMatrix(raw_data)
data_matrix, labels = combineSamples(data_matrix, labels, 2)

print data_matrix[-1]
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

model1 = RandomForestClassifier(n_estimators=10, max_depth=3)#, class_weight={True: 0.2, False:0.2})
model2 = RandomForestClassifier(n_estimators=10, max_depth=3)#, class_weight={True: 0.2, False:0.2})
model3 = RandomForestClassifier(n_estimators=10, max_depth=3)#, class_weight={True: 0.2, False:0.2})

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
