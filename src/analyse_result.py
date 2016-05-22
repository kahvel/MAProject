from sklearn.externals import joblib
from sklearn import metrics
from roc_curve import readData, getTrueLabels, normaliseAndScale, buildDataMatrix, binariseLabels, removePacketsAfterChange
import numpy as np

target_index = 0
model = joblib.load('../pickle/boost_1000_1/boost1.pkl')
raw_test_data = readData("..\\data\\test5_results_3_all.csv")
test_label_data = readData("..\\data\\test5_targets_3.csv")
test_labels = getTrueLabels(test_label_data)
raw_test_data = normaliseAndScale(raw_test_data)
test_data = buildDataMatrix(raw_test_data, target_index)
test_1_labels = binariseLabels(test_labels, target_index+1)
test_data, test_1_labels = removePacketsAfterChange(test_data, test_1_labels, test_label_data, 256)
decision = model.decision_function(test_data)



