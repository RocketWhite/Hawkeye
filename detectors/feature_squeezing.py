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

    def train(self, data_loader, squeezers, learning_rate=1e-4, num_epochs=10):
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
            labels = labels.to(self.device)
            outputs = self.model_forward(images)
            label_list.append(labels)
            for i, squeezer in enumerate(squeezers):
                outputs_diff = outputs - self.model_forward(squeezer.transform(images))
                outputs_diff_list[i].append(outputs_diff)
        labels = torch.cat(label_list)
        for i, squeezer in enumerate(squeezers):
            outputs_diff = torch.cat(outputs_diff_list[i])
            print(outputs_diff.shape)
            self.classifiers[i].fit(self.device, x=outputs_diff, y=labels, mode='FPR', param=0.01)

    def test(self, data_loader, squeezers):
        for images, labels in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model_forward(images)
            predicts = torch.zeros_like(labels)
            for i, squeezer in enumerate(squeezers):
                outputs_diff = outputs - self.model_forward(squeezer.transform(images))
                predicts = predicts | self.classifiers[i].predict(self.device, x=outputs_diff, y=labels).to(self.device)
            self.stat.count(predicts, labels)

    def model_forward(self, images):
        if self.output_mode == 'logits':
            return self.model(images).detach()
        elif self.output_mode == 'probabilities':
            return torch.nn.functional.softmax(self.model(images).detach(),dim=1)
        else:
            warnings.warn("output_mode is not set properly. "
                          "Please read config.ini to check; (currently using logits as default)")
            return self.model(images).detach()
