import torch
from .attack_wrapper import AttackWrapper


class ImageNetAttackWrapper(AttackWrapper):
    def __init__(self, attacker, model_wrapper):
        super().__init__(attacker, model_wrapper)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape([1,3,1,1])
        self.std = torch.tensor([0.229, 0.224, 0.225]).reshape([1,3,1,1])

    def attack(self, x, y):
        print(x)
        x = x * self.std + self.mean
        print(x)
        ori_outputs = self.model(x)
        imgs = self.attacker(x, y)
        imgs = (imgs - self.mean) / self.std
        self.epsilon.append(imgs-x)
        label = torch.ones_like(y)
        ae_outputs = self.model(imgs)
        _, ori_predicted = torch.max(ori_outputs.data, 1)
        _, ad_predicted = torch.max(ae_outputs.data, 1)
        self.stat.count(ad_predicted, ori_predicted)
        return imgs, label

