import torch
import torch.nn
from utils import BinaryCounter
import matplotlib.pyplot as plt


class LinearClassifier(object):
    def __init__(self, device, **kw):
        super(LinearClassifier, self).__init__()
        self.threshold = 0
        self.stat = BinaryCounter()
        self.device = device
        self.kw = kw

    @staticmethod
    def forward(X):
        return torch.norm(X, p=1, dim=1)

    def fit(self, X, y):
        mode = self.kw["mode"]
        param = float(self.kw['param'])
        """
        :mode: the way to find the threshold 
               'FPR': set the training false positive rate; 
               'TPR': set the training false positive rate;
        """
        # fit
        output = self.forward(X)
        index = (y == 0).nonzero()
        legitimate_output = output[index[:, 0]]
        index = (y == 1).nonzero()
        malicious_output = output[index[:, 0]]
        plt.hist(legitimate_output, color='blue')
        plt.hist(malicious_output, color='red')
        plt.show()
        num_of_legitimate_output = legitimate_output.shape[0]
        num_of_malicious_output = malicious_output.shape[0]

        if mode == 'FPR':
            self.threshold = torch.topk(legitimate_output, round(param * num_of_legitimate_output)).values[-1]
        elif mode == 'TPR':
            self.threshold = torch.topk(legitimate_output, round(param * num_of_malicious_output)).values[-1]

    def predict(self, X, y):
        # Test the model
        output = (self.forward(X) > self.threshold).type(torch.LongTensor)
        self.stat.count(output, y)
        return output

