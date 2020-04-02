from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch

class AttackWrapper():
    def __init__(self, adversarial_attack_instance, device):
        self.instance = adversarial_attack_instance
        self.device = device
        self.transform = self.attack
        self.total = 0.0
        self.correct = 0.0

    def attack(self, x, y):
        ori_outputs = self.instance.model(x)
        imgs = self.instance(x, y)
        ae_outputs = self.instance.model(imgs)
        _, ori_predicted = torch.max(ori_outputs.data, 1)
        _, ad_predicted = torch.max(ae_outputs.data, 1)
        self.total += (ori_predicted == y).sum()
        self.correct += ((ori_predicted == y) * (ad_predicted == y)).sum()
        return imgs

    def generate(self, dataloader):
        batch_size = dataloader.batch_size
        x_for_detector = torch.zeros(0,dtype=torch.float32).to(self.device)
        y_for_detector = torch.zeros(0,dtype=torch.float32).to(self.device)

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            ae_images = self.attack(images, labels)
            x_for_detector = torch.cat((x_for_detector, images), 0)
            y_for_detector = torch.cat((y_for_detector, torch.zeros_like(labels,dtype=torch.float32).to(self.device)), 0)
            x_for_detector = torch.cat((x_for_detector, ae_images), 0)
            y_for_detector = torch.cat((y_for_detector, torch.ones_like(labels, dtype=torch.float32).to(self.device)), 0)
        dataset = TensorDataset(x_for_detector.detach(), y_for_detector.detach())
        return DataLoader(dataset=dataset, batch_size=batch_size)

    def accuracy(self):
        if self.total == 0:
            raise ValueError("No successful attack yet.")
        return self.correct/self.total


