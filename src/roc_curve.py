

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import classifiers
    true_labels = getTrueLabels(readData("..\\data\\test5_targets_1.csv"))
    data = readData("..\\data\\test5_results_1_all.csv")
    col_names = ["CCA_f1", "CCA_f2", "CCA_f3"]
    # roc_curve = rocLine(classifiers.ClassifyByRatio(data, col_names).classify(), true_labels)
    # plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ClassifyByDifference(data, col_names).classify(), true_labels)
    # plt.plot(roc_curve[0], roc_curve[1])
    roc_curve = rocLine(classifiers.ThresholdClassification(data, col_names, 0).classify(), true_labels)
    plt.plot(roc_curve[0], roc_curve[1])
    roc_curve = rocLine(classifiers.ThresholdClassification(data, col_names, 0).classifyByAverage(2000), true_labels[1999:])
    plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ThresholdClassification(data, col_names, 0).averageClassifier(3000), true_labels[2999:])
    # plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ThresholdClassification(data, col_names, 0).averageClassifier(4000), true_labels[3999:])
    # plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ThresholdClassification(data, col_names, 1).classify(), true_labels)
    # plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ThresholdClassification(data, col_names, 2).classify(), true_labels)
    # plt.plot(roc_curve[0], roc_curve[1])
    plt.plot((0,1), (0,1))
    plt.show()


