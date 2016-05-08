

class Classification(object):
    def __init__(self, data, col_names):
        self.data = data
        self.col_names = col_names

    def classify(self):
        raise NotImplementedError("classifyAll not implemented!")

    def classifyByAverage(self, window_length):
        raise NotImplementedError("classifyByAverage not implemented!")

    def orderResult(self, row):
        return map(lambda x: x[0], sorted(enumerate(row), key=lambda x: -x[1]))

    def getCol(self, i):
        return [self.data[name][i] for name in self.col_names]

    def calculateResult(self, row, order):
        raise NotImplementedError("calculateResult not implemented!")


class ComparativeClassification(Classification):
    def __init__(self, data, col_names):
        Classification.__init__(self, data, col_names)

    def classifyResult(self, row, order):
        return order[0]+1, self.calculateResult(row, order)

    def classifyData(self, data):
        classification = []
        n = len(data[self.col_names[0]])
        for i in range(n):
            row = self.getCol(i)
            prediction, result = self.classifyResult(row, self.orderResult(row))
            classification.append((i, prediction, result))
        return classification

    def classify(self):
        return self.classifyData(self.data)

    def classifyByAverage(self, window_length):
        classification = {}
        for i, name in enumerate(self.col_names):
            classifier = ThresholdClassification(self.data, self.col_names, i)
            classification[name] = map(lambda x: x[2], classifier.classifyByAverage(window_length))
        return self.classifyData(classification)


class RatioCalculator(object):
    def calculateResult(self, row, order):
        return row[order[0]]/sum(row)


class DifferenceCalculator:
    def calculateResult(self, row, order):
        return row[order[0]]-row[order[1]]


class ClassifyByRatio(RatioCalculator, ComparativeClassification):
    def __init__(self, data, col_names):
        ComparativeClassification.__init__(self, data, col_names)


class ClassifyByDifference(DifferenceCalculator, ComparativeClassification):
    def __init__(self, data, col_names):
        ComparativeClassification.__init__(self, data, col_names)


class ThresholdClassification(Classification):
    def __init__(self, data, col_names, target_index):
        Classification.__init__(self, data, col_names)
        self.target_index = target_index

    def classify(self):
        data = self.data[self.col_names[self.target_index]]
        return [(i, self.target_index+1, data[i]) for i in range(len(data))]

    def classifyByAverage(self, window_length):
        normal_classification = self.classify()
        average_classification = []
        results = []
        for i in range(len(normal_classification)):
            results.append(normal_classification[i])
            if len(results) <= window_length:
                average_classification.append((i-window_length+1, self.target_index+1, sum(map(lambda x: x[2], results))/len(results)))
            else:
                deleted_result = results[0][2]
                del results[0]
                res = average_classification[-1][2]+(-deleted_result+results[-1][2])/window_length
                average_classification.append((i-window_length+1, self.target_index+1, res))
        return average_classification


class AbstractThresholdClassification(ThresholdClassification):
    def __init__(self, data, col_names, target_index):
        ThresholdClassification.__init__(self, data, col_names, target_index)

    def classify(self):
        classification = []
        n = len(self.data[self.col_names[0]])
        for i in range(n):
            row = self.getCol(i)
            classification.append((i, self.target_index+1, self.calculateResult(row, self.orderResult(row))))
        return classification


class ThresholdRatioClassification(RatioCalculator, AbstractThresholdClassification):
    def __init__(self, data, col_names, target_index):
        AbstractThresholdClassification.__init__(self, data, col_names, target_index)


class ThresholdDifferenceClassification(DifferenceCalculator, AbstractThresholdClassification):
    def __init__(self, data, col_names, target_index):
        AbstractThresholdClassification.__init__(self, data, col_names, target_index)
