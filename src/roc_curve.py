
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
    if x_count == 0 or y_count == 0:
        print "x_count = " + str(x_count)
        print "y_count = " + str(y_count)
    roc_x = map(lambda x: x/(x_count if x_count != 0 else 1), roc_x)
    roc_y = map(lambda y: y/(y_count if y_count != 0 else 1), roc_y)
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


def binariseLabels(labels, target):
    return [label == target for label in labels]


def buildDataMatrix(data, target_index):
    average = [1, 10, 100]
    x = []
    for name in group_names:
        for avg in average:
            classification = classifiers.ThresholdClassification(data, col_names[name], target_index).classifyByAverage(avg)
            x.append(map(lambda x: x[2], classification))
            # roc_curve = rocLine(classification, true_labels)
            # plt.plot(roc_curve[0], roc_curve[1])
            classification = classifiers.ThresholdRatioClassification(data, col_names[name], target_index).classifyByAverage(avg)
            x.append(map(lambda x: x[2], classification))
            classification = classifiers.ThresholdDifferenceClassification(data, col_names[name], target_index).classifyByAverage(avg)
            x.append(map(lambda x: x[2], classification))
    x = np.array(x).transpose()
    return x


def removePacketsAfterChange(data, labels, label_data, n_packets, length=256, step=1, average=1):
    new_data = []
    new_labels = []
    for start_packet, end_packet in zip(label_data["Start"], label_data["Stop"]):
        start_index = int(start_packet+n_packets+length)#-step
        end_index = int(end_packet)
        new_data.extend(data[start_index:end_index])
        new_labels.extend(labels[start_index:end_index])
    return new_data, new_labels


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import classifiers
    # from sklearn import qda, lda
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
    from sklearn.svm import LinearSVC, SVC
    import numpy as np

    train_label_data = readData("..\\data\\test5_targets_2.csv")
    train_labels = getTrueLabels(train_label_data)
    raw_train_data = readData("..\\data\\test5_results_2_all.csv")
    raw_train_data = makeDataNonNegative(raw_train_data)
    # roc_curve = rocLine(classifiers.ClassifyByRatio(data, col_names["CCA"]).classify(), true_labels)
    # plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ClassifyByDifference(data, col_names["CCA"]).classify(), true_labels)
    # plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ClassifyByRatio(data, col_names["CCA"]).classifyByAverage(1000), true_labels[999:])
    # plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ThresholdRatioClassification(data, col_names["CCA"], 0).classifyByAverage(1), train_labels[:])
    # plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ThresholdRatioClassification(data, col_names["CCA"], 1).classifyByAverage(1), train_labels[:])
    # plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ThresholdRatioClassification(data, col_names["CCA"], 2).classifyByAverage(1), train_labels[:])
    # plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ThresholdClassification(data, col_names["CCA"], 0).classify(), true_labels)
    # plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ThresholdClassification(data, col_names["CCA"], 0).classifyByAverage(2000), true_labels[1999:])
    # plt.plot(roc_curve[0], roc_curve[1])

    # model = qda.QDA()

    raw_test_data = readData("..\\data\\test5_results_3_all.csv")
    raw_test_data = makeDataNonNegative(raw_test_data)
    test_label_data = readData("..\\data\\test5_targets_3.csv")
    test_labels = getTrueLabels(test_label_data)

    multiclass = False

    if multiclass:
        # model = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=1000, max_samples=0.2, max_features=1.0)
        # model = AdaBoostClassifier(n_estimators=1000, learning_rate=0.1, base_estimator=DecisionTreeClassifier(max_depth=1))
        # model = RandomForestClassifier(n_estimators=1000, max_depth=2)
        model = LinearSVC()

        model.classes_ = [1, 2, 3]
        test_data = map(list, buildDataMatrix(raw_test_data, 0))
        for target_index in [1, 2]:
            result = buildDataMatrix(raw_test_data, target_index)
            for i in range(len(result)):
                test_data[i].extend(result[i])
        train_data = map(list, buildDataMatrix(raw_train_data, 0))
        for target_index in [1, 2]:
            result = buildDataMatrix(raw_train_data, target_index)
            for i in range(len(result)):
                train_data[i].extend(result[i])
        test_data, test_1_labels = removePacketsAfterChange(test_data, test_labels, test_label_data, 200)
        train_data, train_1_labels = removePacketsAfterChange(train_data, train_labels, train_label_data, 200)
        model.fit(train_data, train_1_labels)

        print len(test_data), len(test_data[0])

        print model.score(train_data, train_1_labels)
        print model.score(test_data, test_1_labels)

        decision = model.decision_function(test_data).transpose()

        # decision = model.predict_proba(test_data).transpose()
        classification = classifiers.ClassifyByDifference(decision, [0, 1, 2]).classifyByAverage(1)

        print decision
        print classification[:6]
        print test_1_labels[:6]

        roc_curve = rocLine(classification, test_1_labels)
        plt.plot(roc_curve[0], roc_curve[1])
    else:
        for target_index in [0, 1, 2]:
            # model = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=1000, max_samples=0.2, max_features=1.0)
            # model = AdaBoostClassifier(n_estimators=1000, learning_rate=0.1, base_estimator=DecisionTreeClassifier(max_depth=1))
            # model = RandomForestClassifier(n_estimators=1000, max_depth=1)
            # model = LinearSVC()
            # model = SVC(C=0.1)
            model = BaggingClassifier(base_estimator=SVC(), n_estimators=100, max_samples=0.2, max_features=1.0)

            model.classes_ = [1, 0]

            test_data = buildDataMatrix(raw_test_data, target_index)
            test_1_labels = binariseLabels(test_labels, target_index+1)
            train_data = buildDataMatrix(raw_train_data, target_index)
            train_1_labels = binariseLabels(train_labels, target_index+1)

            print test_data.shape

            test_data, test_1_labels = removePacketsAfterChange(test_data, test_1_labels, test_label_data, 200)
            train_data, train_1_labels = removePacketsAfterChange(train_data, train_1_labels, train_label_data, 200)

            model.fit(train_data, train_1_labels)
            print model.score(train_data, train_1_labels)
            print model.score(test_data, test_1_labels)

            decision = [model.decision_function(test_data)]  # with one target needs another list around it?
            # decision = model.predict_proba(test_data).transpose()  # everything except svm

            classification = classifiers.ThresholdClassification(decision, [0], 0).classifyByAverage(1)

            print decision
            print classification[:6]
            print test_1_labels[:6]

            # decision = map(lambda x: (x[0], 1, x[1]), enumerate(decision.transpose()[0]))
            roc_curve = rocLine(classification, test_1_labels)
            plt.plot(roc_curve[0], roc_curve[1])

    plt.plot((0,1), (0,1))
    plt.show()
