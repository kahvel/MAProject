from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, RFECV
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cross_validation
import sklearn.metrics


target_count = 5
group_names = ["CCA", "LRT", "PSDA_h1", "PSDA_h2", "SNR_h1", "SNR_h2"]#, "PSDA_sum"]
col_names = {
    "CCA": ["CCA_f1", "CCA_f3", "CCA_f5"],
    "PSDA_h1": ["PSDA_h1_f1", "PSDA_h1_f3", "PSDA_h1_f5"],
    "PSDA_h2": ["PSDA_h2_f1", "PSDA_h2_f3", "PSDA_h2_f5"],
    # "PSDA_sum": ["PSDA_sum_f1", "PSDA_sum_f2", "PSDA_sum_f3"],
    "SNR_h1": ["SNR_h1_f1", "SNR_h1_f3", "SNR_h1_f5"],
    "SNR_h2": ["SNR_h2_f1", "SNR_h2_f3", "SNR_h2_f5"],
    "LRT": ["LRT_f1", "LRT_f3", "LRT_f5"],
}
col_names = {
    "CCA": ["CCA_f1", "CCA_f3", "CCA_f5"],
    "PSDA_h1": ["PSDA_h1_f1", "PSDA_h1_f2", "PSDA_h1_f3", "PSDA_h1_f4", "PSDA_h1_f5"],
    "PSDA_h2": ["PSDA_h2_f1", "PSDA_h2_f2", "PSDA_h2_f3", "PSDA_h2_f4", "PSDA_h2_f5"],
    # "PSDA_sum": ["PSDA_sum_f1", "PSDA_sum_f2", "PSDA_sum_f3"],
    "SNR_h1": ["SNR_h1_f1", "SNR_h1_f2", "SNR_h1_f3", "SNR_h1_f4", "SNR_h1_f5"],
    "SNR_h2": ["SNR_h2_f1", "SNR_h2_f2", "SNR_h2_f3", "SNR_h2_f4", "SNR_h2_f5"],
    "LRT": ["LRT_f1", "LRT_f2", "LRT_f3", "LRT_f4", "LRT_f5"],
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
    for i in range(target_count):
        data_matrices[i] = data_matrices[i].tolist()
    for i, rows in enumerate(zip(*data_matrices)):
        for features in zip(*rows):
            feature_sum = sum(features)
            feature = [features[j]/feature_sum for j in range(target_count)]
            for j in range(target_count):
                data_matrices[j][i].append(feature[j])
            data_matrix[i].extend([feature[0], feature[1], feature[2]])

all_data_matrices = []
all_data_matrix = []
all_labels = []
all_labels_binary = [[] for _ in range(target_count)]

for file in [1,2]:
    index = len(all_data_matrices)
    raw_data = readData("U:\\data\\my\\results1_2_target\\results" + str(file) + ".csv")
    # raw_data = readData("U:\\data\\my\\results1_2_target\\results4.csv")
    # print raw_data["CCA_f1"]
    # print raw_data["CCA_f2"]
    # print raw_data["CCA_f3"]

    labels = map(int, raw_data["class"])

    raw_data = scale(raw_data)
    # print raw_data["CCA_f1"]
    # print raw_data["CCA_f2"]
    # print raw_data["CCA_f3"]

    all_data_matrices.append(buildDataMatrixPerTarget(raw_data))
    # print all_data_matrices[file][0][0]
    # print all_data_matrices[file][1][0]
    # print all_data_matrices[file][2][0]

    all_data_matrix.append(buildDataMatrix(raw_data).tolist())
    # addRatio(all_data_matrices[index], all_data_matrix[index])
    # print all_data_matrices[index][0][0]
    # print all_data_matrices[index][1][0]
    # print all_data_matrices[index][2][0]
    #
    # print all_data_matrix[index][0]

    look_back = 1

    for i in range(target_count):
        all_data_matrices[index][i], _ = combineSamples(all_data_matrices[index][i], labels, look_back)

    all_labels.append(None)
    all_data_matrix[index], all_labels[index] = combineSamples(all_data_matrix[index], labels, look_back)

    # print all_data_matrices[index][0][0]
    # print all_data_matrices[index][1][0]
    # print all_data_matrices[index][2][0]
    # print all_labels[index]
    # print len(all_labels[index])
    # print all_data_matrix[index].shape

    for i in range(target_count):
        all_labels_binary[i].append(map(lambda x: x == i+1, all_labels[index]))

    # raw_input("done")

data_matrix = np.concatenate(all_data_matrix)
data_matrices = [
    np.concatenate(map(lambda x: x[0], all_data_matrices)) for i in range(target_count)
]
labels = np.concatenate(all_labels)
labels_binary = [np.concatenate(all_labels_binary[i]) for i in range(target_count)]

print data_matrix.shape

for i in range(target_count):
    print sum(labels_binary[i])

# model1 = LinearDiscriminantAnalysis()
# model2 = LinearDiscriminantAnalysis()
# model3 = LinearDiscriminantAnalysis()

models = []
for i in range(target_count):
    models.append(RandomForestClassifier(n_estimators=10, max_depth=7))#, class_weight={True: 0.2, False:0.2})

# model1 = AdaBoostClassifier(n_estimators=100)#, class_weight={True: 0.2, False:0.2})
# model2 = AdaBoostClassifier(n_estimators=100)#, class_weight={True: 0.2, False:0.2})
# model3 = AdaBoostClassifier(n_estimators=100)#, class_weight={True: 0.2, False:0.2})

# model1 = KNeighborsClassifier(n_neighbors=20)
# model2 = KNeighborsClassifier(n_neighbors=20)
# model3 = KNeighborsClassifier(n_neighbors=20)

sample_weights = []
for i in range(target_count):
    sample_weights.append(np.array(map(lambda x: 0.5 if x == 1 else 0.2, labels_binary[i])))
# print sample_weight1

for i in range(target_count):
    models[i].fit(data_matrices[i], labels_binary[i])#, sample_weight=sample_weight1)

predicted = []
for i in range(target_count):
    predicted.append(models[i].predict(data_matrices[i]))


# print predicted
for i in range(target_count):
    print i
    print sklearn.metrics.confusion_matrix(labels_binary[i], predicted[i])


prediction = []
for i in range(target_count):
    prediction.append(sklearn.cross_validation.cross_val_predict(models[i], data_matrices[i], labels_binary[i], cv=5))#, fit_params={"sample_weight": sample_weight1})
    print "CV " + str(i)
    print sklearn.metrics.confusion_matrix(labels_binary[i], prediction[i])


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

if target_count == 3:
    x = model.transform(data_matrix)

    labels = np.array(labels)

    plt.figure()
    for c, i, target_name in zip("rgb", [1, 2, 3], [1, 2, 3]):
        plt.scatter(x[labels == i, 0], x[labels == i, 1], c=c, label=target_name)
    plt.legend()
    plt.title('LDA')

    plt.show()
