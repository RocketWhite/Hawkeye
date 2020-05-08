import warnings
import torch
from utils import BinaryCounter


class FeatureSqueezing(object):
    def __init__(self, model, classifiers, output_mode):
        self.model = model
        self.device = next(model.parameters()).device
        self.classifiers = classifiers
        self.output_mode = output_mode
        self.stat = BinaryCounter()

    def fit(self, data_loader, squeezers):
        """
        :param data_loader: from torchvision
        :param squeezers:
        :param learning_rate:
        :param num_epochs:
        :return:
        """
        outputs_diff_list = [[], []]
        label_list = []
        for images, labels in data_loader:
            images = images.to(self.device)
            outputs = self.model_forward(images).detach().cpu()
            label_list.append(labels)
            for i, squeezer in enumerate(squeezers):
                outputs_diff = outputs - self.model_forward(squeezer.transform(images)).detach().cpu()
                outputs_diff_list[i].append(outputs_diff)
        labels = torch.cat(label_list)
        for i, squeezer in enumerate(squeezers):
            outputs_diff = torch.cat(outputs_diff_list[i])
            self.classifiers[i].fit(X=outputs_diff, y=labels)

    def predict(self, data_loader, squeezers):
        for images, labels in data_loader:
            images = images.to(self.device)
            outputs = self.model_forward(images).detach().cpu()
            predicts = torch.zeros_like(labels)
            for i, squeezer in enumerate(squeezers):
                outputs_diff = outputs - self.model_forward(squeezer.transform(images)).detach().cpu()
                predicts = predicts | self.classifiers[i].predict(X=outputs_diff, y=labels)
            self.stat.count(predicts, labels)

    def model_forward(self, images):
        if self.output_mode == 'logits':
            return self.model(images).detach()
        elif self.output_mode == 'probabilities':
            return torch.nn.functional.softmax(self.model(images).detach(), dim=1)
        else:
            warnings.warn("output_mode is not set properly. "
                          "Please read config.ini to check; (currently using logits as default)")
            return self.model(images).detach()
