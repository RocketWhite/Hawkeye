import torch
from torch.utils.data import TensorDataset, DataLoader


class Hawkeye():
    def __init__(self, model, device, squeezers, classifiers):
        self.model = model
        self.device = device
        self.squeezers = squeezers
        self.classifiers = classifiers

    def fit(self, train_loader):
        """
        could write to multi processor for later update
        :param dataloader:
        :return:
        """

        logits_diff = [torch.zeros(0, dtype=torch.float32).to(self.device)] * len(self.squeezers)
        labels = torch.zeros(0, dtype=torch.float32).to(self.device)
        batch_size = train_loader.batch_size
        for image, label in train_loader:
            image = image.to(self.device)
            label = label.to(self.device)
            logit = self.model(image)
            labels = torch.cat((labels, label), 0)
            for i, squeezer in enumerate(self.squeezers):
                logits_diff[i] = torch.cat((logits_diff[i], logit - self.model(squeezer.transform(image))), 0)

        for i, classifier in enumerate(self.classifiers):
            logits_loader = DataLoader(TensorDataset(logits_diff[i].detach(), labels.detach()), batch_size=batch_size)
            classifier.fit(logits_loader)

    def test(self, test_loader):
        for image, label in test_loader:
            image = image.to(self.device)
            label = label.to(self.device)
            logit = self.model(image)
            predict = torch.ones_like(label)
            print(predict.shape)
            for i, squeezer in enumerate(self.squeezers):
                print(i)
                logits_diff = logit - self.model(squeezer.transform(image))
                predict = predict * self.classifiers[i].predict(logits_diff)
                print(self.classifiers[i].predict(logits_diff).shape)
                print(predict.shape)

            true_positive = torch.sum((predict==1).float() * (label==1).float())
            true_negative = torch.sum((predict==0).float() * (label==0).float())
            false_positive = torch.sum((predict==1).float() * (label==0).float())
            false_negative = torch.sum((predict==0).float() * (label==1).float())
            print(true_positive, true_negative, false_positive, false_negative)