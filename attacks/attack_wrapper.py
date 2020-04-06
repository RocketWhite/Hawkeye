from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch

class AttackWrapper():
    def __init__(self, adversarial_attack_instance):
        self.instance = adversarial_attack_instance
        self.transform = self.attack
        self.total = 0.0
        self.correct = 0.0

    def attack(self, x, y):
        ori_outputs = self.instance.model(x)
        imgs = self.instance(x, y)
        label = torch.ones_like(y)
        ae_outputs = self.instance.model(imgs)
        _, ori_predicted = torch.max(ori_outputs.data, 1)
        _, ad_predicted = torch.max(ae_outputs.data, 1)
        self.total += (ori_predicted == y).sum()
        self.correct += ((ori_predicted == y) * (ad_predicted == y)).sum()
        return imgs, label

    def transform(self, device, data_loader=None):

        x_for_detector = None
        y_for_detector = None

        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            ae_images, ae_label = self.attack(images, labels)
            label = torch.zeros_like(ae_label)
            if x_for_detector is not None:
                x_for_detector = torch.cat((x_for_detector, images), 0)
            else:
                x_for_detector = images
            if y_for_detector is not None:
                y_for_detector = torch.cat((y_for_detector, torch.zeros_like(labels).to(device)), 0)
            else:
                y_for_detector = torch.zeros_like(labels)

            x_for_detector = torch.cat((x_for_detector, ae_images), 0)
            y_for_detector = torch.cat((y_for_detector, torch.ones_like(labels, dtype=torch.long).to(device)), 0)
            dataset = TensorDataset(x_for_detector.detach(), y_for_detector.detach())
        return DataLoader(dataset, batch_size=data_loader.batch_size, shuffle=data_loader.shuffle)


    def save(self, path, dataset):
        a = torch.Tensor()


    def accuracy(self):
        if self.total == 0:
            raise ValueError("No successful attack yet.")
        return self.correct/self.total


