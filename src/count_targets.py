from main import readData, getTrueLabels, binariseLabels, removePacketsAfterChange


label_data = list()
label_data.append(readData("..\\data\\test5_targets_1.csv"))
label_data.append(readData("..\\data\\test5_targets_2.csv"))
label_data.append(readData("..\\data\\test5_targets_3.csv"))

labels = [getTrueLabels(label) for label in label_data]

binarised_labels = dict()
binarised_labels[1] = [binariseLabels(label, 1) for label in labels]
binarised_labels[2] = [binariseLabels(label, 2) for label in labels]
binarised_labels[3] = [binariseLabels(label, 3) for label in labels]

for target in [1,2,3]:
    for dataset in [0,1,2]:
        _, binarised_labels[target][dataset] =\
            removePacketsAfterChange(binarised_labels[target][dataset], binarised_labels[target][dataset], label_data[dataset], 256)

for target in [1,2,3]:
    for dataset in [0,1,2]:
        print "Dataset:", str(dataset+1), "Target:", str(target), "Count:", str(sum(binarised_labels[target][dataset]))
