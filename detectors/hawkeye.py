import torch
from .feature_squeezing import FeatureSqueezing

class Hawkeye(FeatureSqueezing):
    def __init__(self, model, classifiers, output_mode):
        super(Hawkeye, self).__init__(model, classifiers, output_mode)
        self.classifiers = []
        for classifier in classifiers:
            self.classifiers.append(classifier.to(self.device))

    def train(self, data_loader, squeezers, learning_rate=1e-4, num_epochs=10):
        """
        could write to multi processor for later update
        :param data_loader: combined with adversarial examples and natural examples; labels are 0 or 1 represent natural
        or adversarial examples.
        :param squeezers:
        :param learning_rate:
        :param num_epochs:
        :return:
        """
        for epoch in range(num_epochs):
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model_forward(images)
                for i, squeezer in enumerate(squeezers):
                    outputs_diff = outputs - self.model_forward(squeezer.transform(images))
                    loss = self.classifiers[i].fit(self.device, x=outputs_diff, y=labels, learning_rate=learning_rate)
                    if epoch % 10 == 0:
                        print("Epoch [{}/{}], Step [{}] Loss: {:.4f}".format(epoch + 1, num_epochs, i + 1, loss))

    def test(self, data_loader, squeezers):
        for images, labels in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model_forward(images)
            predicts = torch.ones_like(labels)
            for i, squeezer in enumerate(squeezers):
                logit_diff = outputs - self.model_forward(squeezer.transform(images))
                predicts = predicts & self.classifiers[i].predict(self.device, x=logit_diff, y=labels)
            self.stat.count(predicts, labels)
