import torch
from torch.utils.data import TensorDataset, DataLoader





class Counter(object):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def count(self, predicts, ground_truth):
        self.correct += (predicts == ground_truth).sum()
        self.total += ground_truth.shape[0]

    def accuracy(self):
        return float(self.correct) / self.total

    def clear(self):
        self.__init__()


class BinaryCounter(Counter):
    def __init__(self):
        super(BinaryCounter, self).__init__()
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0

        self.recall = self.true_positive_rate
        self.selectivity = self.true_negative_rate
        self.miss_rate = self.false_negative_rate
        self.fall_out = self.false_positive_rate

    def count(self, predicts, ground_truth):
        super(BinaryCounter, self).count(predicts, ground_truth)
        true_positive = ((predicts == 1).int() * (ground_truth == 1).int()).sum()
        true_negative = ((predicts == 0).int() * (ground_truth == 0).int()).sum()
        false_positive = ((predicts == 1).int() * (ground_truth == 0).int()).sum()
        false_negative = ((predicts == 0).int() * (ground_truth == 1).int()).sum()
        self.true_positive += true_positive
        self.true_negative += true_negative
        self.false_positive += false_positive
        self.false_negative += false_negative

    def precision(self):
        return float(self.true_positive) / (self.true_positive + self.false_positive)

    def true_positive_rate(self):
        return float(self.true_positive) / (self.true_positive + self.false_negative)

    def true_negative_rate(self):
        return float(self.true_negative) / (self.true_negative + self.false_positive)

    def false_positive_rate(self):
        return 1 - self.true_negative_rate()

    def false_negative_rate(self):
        return 1 - self.true_positive_rate()
