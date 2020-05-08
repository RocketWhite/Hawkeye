import torch
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from utils import BinaryCounter


class BayesClassifier(QuadraticDiscriminantAnalysis):
    def __init__(self, device):
        super(BayesClassifier, self).__init__()
        self.device = device
        self.stat = BinaryCounter()

    def predict(self, X, y):
        output = torch.tensor(super(BayesClassifier, self).predict(X))
        self.stat.count(output, y)
        return output