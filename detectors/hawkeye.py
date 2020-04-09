import torch
from torch.utils.data import TensorDataset, DataLoader


class Hawkeye():
    def __init__(self, model, classifiers):
        self.model = model

        self.classifiers = classifiers
        self.correct = 0.
        self.total = 0.
        self.true_positive = 0.
        self.true_negative = 0.
        self.false_positive = 0.
        self.false_negative = 0.

    def training(self, device, data_loader, squeezers, learning_rate=2e-4, num_epochs=10):
        """
        could write to multi processor for later update
        :param dataloader: combined with adversarial examples and natural examples; labels are 0 or 1 represent natural
        or adversarial exmaples.
        :return:
        """
        for epoch in range(num_epochs):
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)
                logit = self.model(images).detach()

                for i, squeezer in enumerate(squeezers):
                    logit_diff = logit - self.model(squeezer.transform(images))
                    loss = self.classifiers[i].fit(device, x=logit_diff, y=labels, learning_rate=learning_rate)
                    print("Epoch [{}/{}], Step [{}] Loss: {:.4f}"
                          .format(epoch + 1, num_epochs, i + 1, loss))

    def test(self, device, data_loader, squeezers):
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logit = self.model(images).detach()
            predicts = torch.ones_like(labels)
            for i, squeezer in enumerate(squeezers):
                logit_diff = logit - self.model(squeezer.transform(images))
                predicts = predicts * self.classifiers[i].predict(device, x=logit_diff, y=labels)
            true_positive = torch.sum((predicts == 1).float() * (labels == 1).float())
            true_negative = torch.sum((predicts == 0).float() * (labels == 0).float())
            false_positive = torch.sum((predicts == 1).float() * (labels == 0).float())
            false_negative = torch.sum((predicts == 0).float() * (labels == 1).float())
            self.true_positive += true_positive
            self.true_negative += true_negative
            self.false_positive += false_positive
            self.false_negative += false_negative
            self.correct += true_positive + true_negative
            self.total += true_positive + true_negative + false_positive + false_negative

    def clear(self):
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
        self.correct = 0
        self.total = 0

        for classifier in self.classifiers:
            classifier.clear()