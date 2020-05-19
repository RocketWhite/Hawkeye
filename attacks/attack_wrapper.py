import torch
from utils import Counter
from torch.utils.data import TensorDataset, DataLoader
import math


class AttackWrapper():
    def __init__(self, attacker, model_wrapper):
        self.attacker = attacker
        self.model = model_wrapper.model
        self.device = next(self.model.parameters()).device
        self.stat = Counter()
        self.epsilon = []

    def attack(self, x, y):
        ori_outputs = self.model(x)
        imgs = self.attacker(x, y)
        self.epsilon.append(imgs-x)
        label = torch.ones_like(y)
        ae_outputs = self.model(imgs)
        _, ori_predicted = torch.max(ori_outputs.data, 1)
        _, ad_predicted = torch.max(ae_outputs.data, 1)
        self.stat.count(ad_predicted, ori_predicted)
        return imgs, label

    def transform(self, dataloader, num_batch, shuffle=True):
        total_iteration = math.ceil(min(num_batch, len(dataloader.dataset)) / dataloader.batch_size)
        x = []
        y = []
        for i, (images, labels) in enumerate(dataloader):
            if i >= total_iteration:
                break
            images = images.to(self.device)
            labels = labels.to(self.device)
            ae_images, _ = self.attack(images, labels)
            x.append(ae_images)
            y.append(labels)
            i = len(y)
            print("finish {}/{} batches".format(str(i), str(total_iteration)))

        x = torch.cat(x, dim=0).cpu()
        y = torch.cat(y, 0).cpu()
        dataset = TensorDataset(x.detach(), y.detach())
        return DataLoader(dataset, batch_size=dataloader.batch_size, shuffle=shuffle)
