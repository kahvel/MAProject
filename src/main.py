
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


def normaliseAndScale(data):
    for group in col_names:
        mins = []
        maxs = []
        for name in col_names[group]:
            mins.append(min(data[name]))
            maxs.append(max(data[name]))
        minimum = min(mins)
        maximum = max(maxs)
        for name in col_names[group]:
            data[name] = list(map(lambda x: (x-minimum)/(maximum-minimum), data[name]))
    return data


def normaliseAndScale2(data, target_count=3, average_count=3):
    new_data = []
    for row in data:
        minimum = min(row)
        maximum = max(row)
        new_data.append(list(map(lambda x: (x-minimum)/(maximum-minimum), row)))
    return new_data


def binariseLabels(labels, target):
    return [label == target for label in labels]


def buildDataMatrix(data, target_index):
    average = [50, 100, 150]
    x = []
    for name in group_names:
        for avg in average:
            classification = classifiers.ThresholdClassification(data, col_names[name], target_index).classifyByAverage(avg)  # used for averaging
            x.append(map(lambda x: x[2], classification))
            # roc_curve = rocLine(classification, true_labels)
            # plt.plot(roc_curve[0], roc_curve[1])
            classification = classifiers.ThresholdRatioClassification(data, col_names[name], target_index).classifyByAverage(avg)
            x.append(map(lambda x: x[2], classification))
            classification = classifiers.ThresholdDifferenceClassification(data, col_names[name], target_index).classifyByAverage(avg)
            x.append(map(lambda x: x[2], classification))
    x = addDifferences(x)
    return np.array(x).transpose()


def removePacketsAfterChange(data, labels, label_data, n_packets, length=256, step=1, average=1):
    new_data = []
    new_labels = []
    i = 0
    for start_packet, end_packet in zip(label_data["Start"], label_data["Stop"]):
        if i == 0:
            start_index = 0#-step
        else:
            start_index = int(start_packet+n_packets-length)
        end_index = int(end_packet)-length
        data_segment = data[start_index:end_index+1]
        labels_segment = labels[start_index:end_index+1]
        new_data.extend(data_segment)
        new_labels.extend(labels_segment)
        # print all(map(lambda x: x == (label_data["Target"][i] == target_index+1), labels_segment))
        # print start_index, end_index
        # print start_packet, end_packet
        # print labels_segment
        i += 1
    return new_data, new_labels


def differenceTimeSeries(data):
    result = {}
    for group_name in group_names:
        for col_name in col_names[group_name]:
            row = []
            for i in range(1, len(data[col_name])):
                row.append(data[col_name][i]-data[col_name][i-1])
                # result.append(map(lambda x: x[1]-x[0], zip(data[i], data[i-1])))
            result[col_name] = row
    return result


def check_result(data):
    print data
    x = [i+1 for i in range(len(data[0]))]
    n = len(data)
    for i, row in enumerate(data):
        print all(map(lambda x: x == row[0], row))
        plt.subplot(n, 1, i+1)
        plt.plot(x, row)
    plt.show()


def addDifferences(data):
    new_data = []
    differences = [1, 2, 3]
    for row in data:
        for d in differences:
            new_row = [0 for _ in range(d)]
            for i in range(len(row[d:])):
                new_row.append(row[i]-row[i-d])
            new_data.append(new_row)
            new_data.append(row)
    return new_data


def printMetrics(fprs, tprs, thresholds, threshold):
    perfect_fpr = np.where(fprs <= threshold)[0]
    if len(perfect_fpr) > 0:
        fpr_index = perfect_fpr[-1]
        pos_threshold = thresholds[fpr_index]
        prediction = map(lambda x: x >= pos_threshold, decision[0])
        print list(np.where(prediction)[0])
        print fprs[fpr_index]
        print tprs[fpr_index]
        print pos_threshold
        print metrics.confusion_matrix(test_1_labels, prediction, labels=[True, False])
        print metrics.classification_report(test_1_labels, prediction, labels=[True, False])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import classifiers
    # from sklearn import qda, lda
    # from sklearn.lda import LDA
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import metrics
    from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import LinearSVC, SVC
    from sklearn.neural_network import BernoulliRBM
    import numpy as np
    from sklearn.pipeline import Pipeline
    # from sklearn.gaussian_process import GaussianProcess
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from svm import LinearSVM
    # from pipeline import MyPipeline
    from sklearn.externals import joblib
    import pickle
    from voting import Voting

    np.random.seed(99)

    # roc_curve = rocLine(classifiers.ClassifyByRatio(data, col_names["CCA"]).classify(), true_labels)
    # plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ClassifyByDifference(data, col_names["CCA"]).classify(), true_labels)
    # plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ClassifyByRatio(data, col_names["CCA"]).classifyByAverage(1000), true_labels[999:])
    # plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ThresholdRatioClassification(data, col_names["CCA"], 0).classifyByAverage(1), train_labels[:])
    # plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ThresholdClassification(data, col_names["CCA"], 0).classify(), true_labels)
    # plt.plot(roc_curve[0], roc_curve[1])
    # roc_curve = rocLine(classifiers.ThresholdClassification(data, col_names["CCA"], 0).classifyByAverage(2000), true_labels[1999:])
    # plt.plot(roc_curve[0], roc_curve[1])

    # model = qda.QDA()

    multiclass = False
    difference = False

    train_label_data = readData("..\\data\\test5_targets_1.csv")
    train_labels = getTrueLabels(train_label_data)
    raw_train_data = readData("..\\data\\test5_results_1_all.csv")

    raw_test_data = readData("..\\data\\test5_results_3_all.csv")
    test_label_data = readData("..\\data\\test5_targets_3.csv")
    test_labels = getTrueLabels(test_label_data)

    if difference:
        raw_train_data = differenceTimeSeries(raw_train_data)
        raw_test_data = differenceTimeSeries(raw_test_data)
        del train_labels[0]
        del test_labels[0]

    raw_test_data = normaliseAndScale(raw_test_data)
    raw_train_data = normaliseAndScale(raw_train_data)

    print len(raw_train_data), len(raw_train_data["CCA_f1"])
    print len(raw_test_data), len(raw_test_data["CCA_f1"])
    print len(train_labels)
    print len(test_labels)

    if multiclass:
        # model = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=1000, max_samples=0.2, max_features=1.0)
        # model = AdaBoostClassifier(n_estimators=1000, learning_rate=0.1, base_estimator=DecisionTreeClassifier(max_depth=1))
        model = RandomForestClassifier(n_estimators=1000, max_depth=10)
        # model = LinearSVC()
        svm = LinearSVM(C=1)
        # svm = SGDClassifier(shuffle=True, loss="hinge")
        # svm = LogisticRegression()

        svm.classes_ = [1, 2, 3]

        pre_model = BernoulliRBM(learning_rate=0.1, n_components=5, n_iter=20)
        # pipeline = MyPipeline(steps=[("rbm", pre_model), ("svm", svm)])
        # model = pipeline
        # model = BaggingClassifier(base_estimator=svm, n_estimators=10, max_samples=0.2, max_features=1.0)
        # model = AdaBoostClassifier(n_estimators=5, learning_rate=1., base_estimator=model)
        # model = GradientBoostingClassifier()

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
        test_data, test_1_labels = removePacketsAfterChange(test_data, test_labels, test_label_data, 256)
        train_data, train_1_labels = removePacketsAfterChange(train_data, train_labels, train_label_data, 256)
        # test_1_labels = test_labels
        # train_1_labels = train_labels

        # train_data = pre_model.fit_transform(train_data, train_1_labels)
        # test_data = pre_model.transform(test_data)
        # check_result(train_data.transpose())
        # check_result(test_data.transpose())

        model.fit(train_data, train_1_labels)

        print len(test_data), len(test_data[0])

        print model.score(train_data, train_1_labels)
        print model.score(test_data, test_1_labels)

        decision = model.predict_proba(test_data).transpose()
        check_result(decision)

        # decision = model.predict_proba(test_data).transpose()
        classification = classifiers.ClassifyByRatio(decision, [0, 1, 2]).classifyByAverage(1)
        roc_curve = rocLine(classification, test_1_labels)
        plt.plot(roc_curve[0], roc_curve[1])
        plt.plot((0, 1), (0, 1))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        for target_index in [0, 1, 2]:
            t_labels = binariseLabels(test_1_labels, target_index+1)
            fpr, tpr, threshold = metrics.roc_curve(t_labels, decision[target_index], pos_label=True)
            # fpr, tpr, threshold = metrics.roc_curve(t_labels, model.predict_proba(test_data).transpose()[0], pos_label=True)
            plt.plot(fpr, tpr)
    else:
        for target_index in [0,1,2]:
            for j, n_estimators in enumerate([1000]):#enumerate([10**(r) for r in range(-3, 1)]):
                # model = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=1000, max_samples=0.2, max_features=0.2)
                # model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=0.1, base_estimator=DecisionTreeClassifier(max_depth=1))
                # model = RandomForestClassifier(n_estimators=1000, max_depth=1)#, class_weight={True: 0.001, False: 0.9999}
                # model = GradientBoostingClassifier(n_estimators=1000, max_depth=3, random_state=99)
                # svm = LinearSVC(C=1)
                # # svm = LDA()
                # svm = LogisticRegression()
                # # model = SVC(C=1.0)
                # # model = BaggingClassifier(base_estimator=LinearSVC(), n_estimators=100, max_samples=0.2, max_features=1.0)
                # pre_model = BernoulliRBM(learning_rate=0.001, n_components=1, n_iter=200)
                #
                # svm.classes_ = [True, False]
                #
                # pipeline = Pipeline(steps=[("rbm", pre_model), ("svm", svm)])
                # bagging = BaggingClassifier(base_estimator=pipeline, n_estimators=5, max_samples=0.2, max_features=1.0)
                # model = AdaBoostClassifier(n_estimators=5, learning_rate=0.1, base_estimator=svm)
                # model = pipeline
                # # model = GaussianProcess()
                # # model = BaggingClassifier(base_estimator=gaussian, n_estimators=10, max_samples=1.0, max_features=1.0)

                # model.classes_ = [True, False]

                test_1_labels = binariseLabels(test_labels, target_index+1)
                train_1_labels = binariseLabels(train_labels, target_index+1)

                test_data = buildDataMatrix(raw_test_data, target_index)
                train_data = buildDataMatrix(raw_train_data, target_index)

                test_data, test_1_labels = removePacketsAfterChange(test_data, test_1_labels, test_label_data, 256)
                train_data, train_1_labels = removePacketsAfterChange(train_data, train_1_labels, train_label_data, 256)
                print len(test_data), len(test_data[0])

                estimators = []
                for k in range(10):
                    model = AdaBoostClassifier(n_estimators=100, learning_rate=0.1,
                                               base_estimator=DecisionTreeClassifier(max_depth=1, class_weight={True: 0.9, False: 0.1}))
                    # model = RandomForestClassifier(n_estimators=100, max_depth=1, class_weight={True: 0.9, False: 0.1})
                    # model = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, max_samples=1.0, max_features=1.0)
                    model.classes_ = [True, False]
                    # svm = LinearSVM(C=1)
                    # svm.classes_ = [True, False]
                    # pre_model = BernoulliRBM(learning_rate=0.1, n_components=10, n_iter=20)
                    # model = Pipeline(steps=[("rbm", pre_model), ("svm", svm)])

                    positive_train_labels = [i for i in range(len(train_1_labels)) if train_1_labels[i]]
                    # positive_test_labels = [i for i in range(len(test_1_labels)) if test_1_labels[i]]
                    negative_train_labels = list(np.random.choice([i for i in range(len(train_1_labels)) if i not in positive_train_labels], replace=False, size=len(positive_train_labels)))
                    # negative_test_labels = list(np.random.choice([i for i in range(len(test_1_labels)) if i not in positive_test_labels], replace=False, size=len(positive_test_labels)))

                    # test_data = [test_data[i] for i in positive_test_labels + negative_test_labels]
                    train_data = [train_data[i] for i in positive_train_labels + negative_train_labels]

                    # test_1_labels = [test_1_labels[i] for i in positive_test_labels + negative_test_labels]
                    train_1_labels = [train_1_labels[i] for i in positive_train_labels + negative_train_labels]
                    model.fit(train_data, train_1_labels)
                    estimators.append((str(k), model))
                voting = Voting(estimators, voting="soft")
                decision = voting.transform(test_data)
                # print decision
                decision = sum(decision).transpose()
                # decision = [decision]  # for svm
                # print decision

                # train_data = pre_model.fit_transform(train_data, train_1_labels)
                # test_data = pre_model.transform(test_data)
                # print len(test_data), len(test_data[0])

                # model.fit(train_data, train_1_labels)
                # print model.score(train_data, train_1_labels)
                # print model.score(test_data, test_1_labels)

                # print metrics.classification_report(test_1_labels, model.predict(test_data))

                # decision = [model.decision_function(test_data)]  # with one target needs another list around it?
                # decision = model.predict_proba(test_data).transpose()  # everything except svm
                # print all(map(lambda x: x==decision[0][0], decision[0]))

                # plt.subplot(2, 2, 2)
                # check_result(decision)

                fpr, tpr, threshold = metrics.roc_curve(test_1_labels, decision[0], pos_label=True)
                # fpr, tpr, threshold = metrics.roc_curve(test_1_labels, model.predict_proba(test_data).transpose()[0], pos_label=True)
                print threshold

                # classification = classifiers.ThresholdClassification(decision, [0], 0).classifyByAverage(1)
                # pred = model.predict(test_data)
                # print all(pred) or not any(pred), pred[0]
                printMetrics(fpr, tpr, threshold, 0.01)
                printMetrics(fpr, tpr, threshold, 0)

                # # decision = map(lambda x: (x[0], 1, x[1]), enumerate(decision.transpose()[0]))
                # roc_curve = rocLine(classification, test_1_labels)
                plt.subplot(2, 2, 1+j)
                # plt.plot(roc_curve[0], roc_curve[1])
                plt.plot(fpr, tpr)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.plot((0, 1), (0, 1))
                # joblib.dump(model, "boost1.pkl")
                # pickle.Pickler(file("../pickle/boost_1000_1a/boost.pkl", "w")).dump(model)
    plt.show()
