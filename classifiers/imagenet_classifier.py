import torch
import torch.nn
import torch.nn as nn
from classifiers  import NNClassifier 


class ImageNetClassifier(NNClassifier):
    def __init__(self, device, num_epochs=10, learning_rate=2e-4, batch_size=100):
        super(ImageNetClassifier, self).__init__(device, num_epochs, learning_rate, batch_size)
        self.fc1 = nn.Linear(1000, 1000, bias=True)
        self.fc2 = nn.Linear(1000, 100,bias=True)
        self.fc3 = nn.Linear(100, 10, bias=True)
        self.fc4 = nn.Linear(10, 2)
        self.relu = nn.ReLU(inplace=True)
