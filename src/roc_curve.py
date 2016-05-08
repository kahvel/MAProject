
group_names = ["CCA", "PSDA_h1", "PSDA_h2", "PSDA_sum"]
col_names = {
    "CCA": ["CCA_f1", "CCA_f2", "CCA_f3"],
    "PSDA_h1": ["PSDA_h1_f1", "PSDA_h1_f2", "PSDA_h1_f3"],
    "PSDA_h2": ["PSDA_h2_f1", "PSDA_h2_f2", "PSDA_h2_f3"],
    "PSDA_sum": ["PSDA_sum_f1", "PSDA_sum_f2", "PSDA_sum_f3"]
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


def rocLine(classification, true_labels):
    assert len(true_labels) == len(classification)
    index = map(lambda x: x[0], sorted(classification, key=lambda x: -x[2]))
    roc_x = [0.0]
    roc_y = [0.0]
    x_count = 0
    y_count = 0
    for i in index:
        length = y_count+x_count
        if classification[i][1] == true_labels[i]:
            roc_x.append(roc_x[length])
            roc_y.append(roc_y[length]+1)
            y_count += 1
        else:
            roc_x.append(roc_x[length]+1)
            roc_y.append(roc_y[length])
            x_count += 1
    roc_x = map(lambda x: x/x_count, roc_x)
    roc_y = map(lambda y: y/y_count, roc_y)
    return roc_x, roc_y


def getTrueLabels(trial_data, length=256, step=1, average=1):
    true_labels = []
    trial_index = 0
    packet_count = int(trial_data["Stop"][-1])
    for i in range(length-step, packet_count, step):
        if i > trial_data["Stop"][trial_index]:
            trial_index += 1
        if i != packet_count:
            true_labels.append(trial_data["Target"][trial_index])
    return true_labels


def makeDataNonNegative(data):
    for group in col_names:
        mins = []
        for name in col_names[group]:
            mins.append(min(data[name]))
        for name in col_names[group]:
            data[name] = list(map(lambda x: x-min(mins), data[name]))
    return data


def binariseClassification(classification, target):
    pass


def buildDataMatrix(data):
    average = [10, 100, 1000]
    x = []
    for name in group_names:
        for avg in average:
            classification = classifiers.ThresholdClassification(data, col_names[name], 0).classifyByAverage(avg)
            x.append(map(lambda x: x[2], classification))
            # roc_curve = rocLine(classification, true_labels)
            # plt.plot(roc_curve[0], roc_curve[1])
    x = np.array(x).transpose()
    return x


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import classifiers
    from sklearn import qda, lda
    import numpy as np
    model = qda.QDA()
    # model = lda.LDA()
    train_labels = getTrueLabels(readData("..\\data\\test5_targets_1.csv"))
    data = readData("..\\data\\test5_results_1_all.csv")
    data = makeDataNonNegative(data)
    # roc_curve = rocLine(classifiers.ClassifyByRatio(data, col_names["CCA"]).classify(), true_labels)
    # plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ClassifyByDifference(data, col_names["CCA"]).classify(), true_labels)
    # plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ClassifyByRatio(data, col_names["CCA"]).classifyByAverage(1000), true_labels[999:])
    # plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ClassifyByDifference(data, col_names["CCA"]).classifyByAverage(1000), true_labels[999:])
    # plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ThresholdClassification(data, col_names["CCA"], 0).classify(), true_labels)
    # plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ThresholdClassification(data, col_names["CCA"], 0).classifyByAverage(2000), true_labels[1999:])
    # plt.plot(roc_curve[0], roc_curve[1])

    train_data = buildDataMatrix(data)
    model.fit(train_data, train_labels)
    print model.score(train_data, train_labels)

    test_data = readData("..\\data\\test5_results_3_all.csv")
    test_data = makeDataNonNegative(test_data)
    test_data = buildDataMatrix(test_data)
    test_labels = getTrueLabels(readData("..\\data\\test5_targets_3.csv"))

    print model.score(test_data, test_labels)
    decision = model.decision_function(test_data)
    print decision.shape
    for i in [0, 1, 2]:
        classification = classifiers.ThresholdClassification(decision.transpose(), [0, 1, 2], i).classifyByAverage(1)
        roc_curve = rocLine(classification, test_labels)
        plt.plot(roc_curve[0], roc_curve[1])

    plt.plot((0,1), (0,1))
    plt.show()


