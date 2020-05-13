import torch.nn as nn
from classifiers import NNClassifier


class Cifar10Classifier(NNClassifier):
    def __init__(self, device, num_epochs=40, learning_rate=1e-4, batch_size=100):
        super().__init__(device, num_epochs, learning_rate, batch_size)
        self.fc1 = nn.Linear(10, 10000, bias=True)
        self.fc2 = nn.Linear(10000, 100, bias=True)
        self.fc3 = nn.Linear(100, 10, bias=True)
        self.fc4 = nn.Linear(10, 2)
