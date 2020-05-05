import torch
from utils import BinaryCounter


class FeatureSqueezing(object):
    def __init__(self, model, classifiers):
        self.model = model
        self.device = next(model.parameters()).device
        self.classifiers = classifiers
        self.stat = BinaryCounter()

    def train(self,  data_loader, squeezers, learning_rate=1e-4, num_epochs=10):
        """
        could write to multi processor for later update
        :param dataloader: combined with adversarial examples and natural examples; labels are 0 or 1 represent natural
        or adversarial exmaples.
        :return:
        """

        logit_diff_list = [[], []]
        label_list = []
        for images, labels in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images).detach()
            label_list.append(labels)
            for i, squeezer in enumerate(squeezers):
                logit_diff = logits - self.model(squeezer.transform(images)).detach()
                logit_diff_list[i].append(logit_diff)
        labels = torch.cat(label_list)
        for i, squeezer in enumerate(squeezers):
            logit_diff = torch.cat(logit_diff_list[i])
            print(logit_diff.shape)
            self.classifiers[i].fit(self.device, x=logit_diff, y=labels, mode='FPR', param=0.1)

    def test(self, data_loader, squeezers):
        for images, labels in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logit = self.model(images).detach()
            predicts = torch.zeros_like(labels)
            for i, squeezer in enumerate(squeezers):
                logit_diff = logit - self.model(squeezer.transform(images))
                predicts = predicts | self.classifiers[i].predict(self.device, x=logit_diff, y=labels).to(self.device)
            self.stat.count(predicts, labels)
