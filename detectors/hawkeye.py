import torch
from .feature_squeezing import FeatureSqueezing


class Hawkeye(FeatureSqueezing):
    def __init__(self, model, classifiers, output_mode):
        super(Hawkeye, self).__init__(model, classifiers, output_mode)
        self.classifiers = classifiers

    def fit(self, data_loader, squeezers):
        outputs_diff_list = [[] for i in squeezers]
        labels_list = []
        for images, labels in data_loader:
            images = images.to(self.device)
            labels_list.append(labels)
            outputs = self.model_forward(images).detach().cpu()
            for i, squeezer in enumerate(squeezers):
                outputs_diff = outputs - self.model_forward(squeezer.transform(images)).detach().cpu()
                outputs_diff_list[i].append(outputs_diff)
        labels_all = torch.cat(labels_list)
        for i, squeezer in enumerate(squeezers):
            outputs_diff = torch.cat(outputs_diff_list[i])
            self.classifiers[i].fit(X=outputs_diff, y=labels_all)
    def predict(self, data_loader, squeezers):
        for images, labels in data_loader:
            images = images.to(self.device)
            outputs = self.model_forward(images).detach().cpu()
            predicts = torch.ones_like(labels)
            for i, squeezer in enumerate(squeezers):
                logits_diff = outputs - self.model_forward(squeezer.transform(images)).detach().cpu()
                predicts = predicts & self.classifiers[i].predict(X=logits_diff, y=labels)
            self.stat.count(predicts, labels)
