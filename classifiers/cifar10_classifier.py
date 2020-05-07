import torch
import torch.nn
import torch.nn as nn
from classifiers  import NNClassifier 


class Cifar10Classifier(NNClassifier):
    def __init__(self):
        super(Cifar10Classifier, self).__init__()
        self.fc1 = nn.Linear(10, 1000, bias=True)
        self.fc2 = nn.Linear(1000, 100,bias=True)
        self.fc3 = nn.Linear(100, 10, bias=True)
        self.fc4 = nn.Linear(10, 2)
        self.relu = nn.ReLU(inplace=True)
