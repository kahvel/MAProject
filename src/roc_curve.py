

def readData(file_name, sep=" "):  # Is there ID???
    content = open(file_name).readlines()
    names = content[0].strip().split(sep)
    result = {name: [] for name in names}
    for line in content[1:]:
        results = line.strip().split(sep)[1:]
        for i in range(len(names)):
            result[names[i]].append(float(results[i]))
    return result


def rocLine(classification, trial_data, step=1, sma=0):
    index = map(lambda x: x[0], sorted(classification, key=lambda x: -x[2]))
    roc_x = [0.0]
    roc_y = [0.0]
    x_count = 0
    y_count = 0
    n = len(trial_data["Start"])
    for i in index:
        classification_row = classification[i]
        for j in range(n):
            packet_nr = classification_row[0]*step+256-step + sma
            if trial_data["Start"][j] <= packet_nr <= trial_data["Stop"][j]:
                length = y_count+x_count
                if classification_row[1] == trial_data["Target"][j]:
                    roc_x.append(roc_x[length])
                    roc_y.append(roc_y[length]+1)
                    y_count += 1
                else:
                    roc_x.append(roc_x[length]+1)
                    roc_y.append(roc_y[length])
                    x_count += 1
                break
    roc_x = map(lambda x: x/x_count, roc_x)
    roc_y = map(lambda y: y/y_count, roc_y)
    return roc_x, roc_y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import classifiers
    trial_data = readData("..\\data\\test5_targets_2.csv")

    data = readData("..\\data\\test5_results_2_all.csv")
    #predicted = classifyMax(data, ["CCA_f1", "CCA_f2", "CCA_f3"], classifyMaxRatio)
    #column = getColumn(data, 0, ["CCA_f1", "CCA_f2", "CCA_f3"])
    col_names = ["CCA_f1", "CCA_f2", "CCA_f3"]
    classifier = classifiers.ClassifyByRatio(data, col_names)
    classification = classifier.classify()
    roc_curve = rocLine(classification, trial_data)
    plt.plot(roc_curve[0], roc_curve[1])
    plt.plot((0,1), (0,1))
    plt.show()


