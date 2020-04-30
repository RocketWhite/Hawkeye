import torch
import torch.nn
import torch.nn as nn
from utils import BinaryCounter

class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.threshold = 0
        self.stat = BinaryCounter()

    
    def forward(self, x):
        return torch.norm(x, p=1, dim=1)

    # def training(self, device, data_loader, learning_rate=2e-4, num_epochs=10):
    #     for epoch in range(num_epochs):
    #         for i, (images, labels) in enumerate(data_loader):
    #             images = images.to(device)
    #             labels = labels.to(device)
    #             loss = self.fit(device, images, labels, learning_rate)
    #
    #             if (i + 1) % 10 == 0:
    #                 print("Epoch [{}/{}], Step [{}] Loss: {:.4f}"
    #                       .format(epoch + 1, num_epochs, i + 1, loss))

    def fit(self, device, x, y, mode='MAP', param=0.01):
        """
        :mode: the way to find the threshold 
               'FPR': set the training false positive rate; 
               'TPR': set the training false positive rate;
               'FNR': set the training false negative rate;
               'ACC': set the accuracy rate;
               'ERR': set the error rate;
               'MAP': maximum-a-posteriori;
               'ML':  maximum likelihood;
        """
        # fit
        x = x.to(device)
        y = y.to(device)
        if mode == 'FPR':
            index = (y == 0).nonzero()
            legitimate_x = x[index[:,0]]
            num_of_legitimate_x = legitimate_x.shape[0]
            self.threshold = torch.topk(legitimate_x, round(param*num_of_legitimate_x)).values[-1]


    # Save the model checkpoint
    # path = "./models_dict/%s.ckpt" % model.__class__.__name__
    # torch.save(model.state_dict(), path)

    
    def predict(self, device, x, y):
        # Test the model
        x = x.to(device)
        y = y.to(device)
        output = (self.forward(x) > self.threshold).type(torch.LongTensor).to(device)
        self.stat.count(output, y)
        return output

