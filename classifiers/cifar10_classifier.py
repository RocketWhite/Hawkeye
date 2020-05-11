import torch
import torch.nn
import torch.nn as nn
from classifiers  import NNClassifier 


class Cifar10Classifier(NNClassifier):
    def __init__(self, device, num_epochs=10, learning_rate=2e-4, batch_size=100):
        super(Cifar10Classifier, self).__init__(device, num_epochs, learning_rate, batch_size)
        self.fc1 = nn.Linear(10, 100, bias=True)
        self.fc2 = nn.Linear(100, 1000,bias=True)
        self.fc3 = nn.Linear(1000, 100, bias=True)
        self.fc4 = nn.Linear(100, 10)
        self.fc5 = nn.Linear(10, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        out = self.relu(self.fc1(X))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.fc5(out)
        return out