import torch
from utils import BinaryCounter


class Hawkeye(object):
    def __init__(self, model, classifiers):
        self.model = model
        self.device = next(model.parameters()).device
        self.classifiers = []
        for classifier in classifiers:
            self.classifiers.append(classifier.to(self.device))
        self.stat = BinaryCounter()

    def train(self,  data_loader, squeezers, learning_rate=1e-4, num_epochs=10):
        """
        could write to multi processor for later update
        :param dataloader: combined with adversarial examples and natural examples; labels are 0 or 1 represent natural
        or adversarial exmaples.
        :return:
        """
        for epoch in range(num_epochs):
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images).detach()

                for i, squeezer in enumerate(squeezers):
                    logit_diff = logits - self.model(squeezer.transform(images)).detach()
                    loss = self.classifiers[i].fit(self.device, x=logit_diff, y=labels, learning_rate=learning_rate)
#                    print("Epoch [{}/{}], Step [{}] Loss: {:.4f}".format(epoch + 1, num_epochs, i + 1, loss))

    def test(self, data_loader, squeezers):
        for images, labels in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logit = self.model(images).detach()
            predicts = torch.ones_like(labels)
            for i, squeezer in enumerate(squeezers):
                logit_diff = logit - self.model(squeezer.transform(images))
                predicts = predicts & self.classifiers[i].predict(self.device, x=logit_diff, y=labels)
            self.stat.count(predicts, labels)
