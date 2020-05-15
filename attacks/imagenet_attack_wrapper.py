import torch
from .attack_wrapper import AttackWrapper


class ImageNetAttackWrapper(AttackWrapper):
    def __init__(self, attacker, model_wrapper):
        super().__init__(attacker, model_wrapper)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape([1,3,1,1]).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).reshape([1,3,1,1]).to(self.device)

    def attack(self, x, y):
        ori_outputs = self.model(x)
        x = x * self.std + self.mean
        imgs = self.attacker(x, y)
        self.epsilon.append(imgs-x)
        imgs = imgs * self.std + self.mean
        label = torch.ones_like(y)
        ae_outputs = self.model(imgs)
        _, ori_predicted = torch.max(ori_outputs.data, 1)
        _, ad_predicted = torch.max(ae_outputs.data, 1)
        self.stat.count(ad_predicted, ori_predicted)
        return imgs, label

