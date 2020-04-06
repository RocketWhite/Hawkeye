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

    def training(self, device, data_loader, squeezers, attackers, learning_rate=2e-4, num_epochs=10):
        """
        could write to multi processor for later update
        :param dataloader:
        :return:
        """
        for epoch in range(num_epochs):
            for image, label in data_loader:
                image = image.to(device)
                label = label.to(device)
                ae_image, ae_y = attackers.attack(image, label)
                x = torch.cat((image, ae_image), dim=0).detach()
                y = torch.cat((torch.zeros_like(ae_y), ae_y), dim=0).detach()
                logit = self.model(x).detach()

                for i, squeezer in enumerate(squeezers):
                    logit_diff = logit - self.model(squeezer.transform(x))
                    loss = self.classifiers[i].fit(device, x=logit_diff, y=y, learning_rate=learning_rate)
                    print("Epoch [{}/{}], Step [{}] Loss: {:.4f}"
                          .format(epoch + 1, num_epochs, i + 1, loss))

    def test(self, device, data_loader, squeezers, attackers):
        for image, label in data_loader:
            image = image.to(device)
            label = label.to(device)
            ae_image, ae_y = attackers.attack(image, label)
            x = torch.cat((image, ae_image), dim=0).detach()
            y = torch.cat((torch.zeros_like(ae_y), ae_y), dim=0).detach()
            logit = self.model(x).detach()
            predict = torch.ones_like(y)

            for i, squeezer in enumerate(squeezers):
                logit_diff = logit - self.model(squeezer.transform(x))
                predict = predict * self.classifiers[i].predict(device, x=logit_diff, y=y)
            true_positive = torch.sum((predict == 1).float() * (y == 1).float())
            true_negative = torch.sum((predict == 0).float() * (y == 0).float())
            false_positive = torch.sum((predict == 1).float() * (y == 0).float())
            false_negative = torch.sum((predict == 0).float() * (y == 1).float())
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