from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
import os
import math


class AttackWrapper():
    def __init__(self, adversarial_attack_instance):
        self.instance = adversarial_attack_instance
        self.total = 0.0
        self.correct = 0.0

    def attack(self, x, y):
        ori_outputs = self.instance.model(x)
        imgs = self.instance(x, y)
        ae_outputs = self.instance.model(imgs)
        _, ori_predicted = torch.max(ori_outputs.data, 1)
        _, ae_predicted = torch.max(ae_outputs.data, 1)
        self.total += (ori_predicted == y).sum()
        self.correct += ((ori_predicted == y) * (ae_predicted == y)).sum()
        return imgs

    def transform(self, device, data_loader):
        total_iteration = math.ceil(len(data_loader.dataset)/data_loader.batch_size)
        x = []
        y = []
        i = 0
        for images, labels in data_loader:
            i += 1
            images = images.to(device)
            labels = labels.to(device)
            ae_images = self.attack(images, labels)
            x.append(ae_images)
            y.append(labels)
            i = len(y)
            if i % 100 == 0:
                print("finish {}/{} batches".format(str(i), str(total_iteration)))

        x = torch.cat(x, 0)
        y = torch.cat(y, 0)
        return x, y

    def save(self, device, data_loader, path):
        device2 = torch.device('cpu')
        dataset = self.transform(device=device, data_loader=data_loader)
        dataset = (dataset[0].to(device2), dataset[1].to(device2))
        with open(path, 'wb') as f:
            torch.save(dataset, f)

    def accuracy(self):
        if self.total == 0:
            raise ValueError("No successful attack yet.")
        return self.correct/self.total


