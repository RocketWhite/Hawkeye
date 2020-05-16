import importlib
import os
import torch
from attacks import AttackWrapper
from pathlib import Path


class Generator:
    def __init__(self, exp):
        self.exp = exp
        self.attackers = self.load_attacker()

    def run(self):
        dataloader = {}
        dataloader['train'] = self.exp.load_natural_data(train=True, transform=self.exp.model.transform)
        dataloader['test'] = self.exp.load_natural_data(train=False, transform=self.exp.model.transform)
        for key, file in self.exp.file.items():
            num = int(self.exp.cfg.get("generator", "num_" + key))
            if not os.path.exists(file):
                os.makedirs(self.exp.path[key], exist_ok=True)
                attacker = AttackWrapper(self.attackers[key], self.exp.model)
                outputs = attacker.transform(dataloader[key], num)
                print("L0 norm of the epsilon is {}".format(
                    torch.mean(torch.norm(torch.cat(attacker.epsilon), p=0, dim=(1,2,3)))))
                print("L1 norm of the epsilon is {}".format(
                    torch.mean(torch.norm(torch.cat(attacker.epsilon), p=1, dim=(1, 2, 3)))))
                print("L2 norm of the epsilon is {}".format(
                    torch.mean(torch.norm(torch.cat(attacker.epsilon), p=2, dim=(1,2,3)))))
                print("Li norm of the epsilon is {}".format(
                    torch.mean(torch.norm(torch.cat(attacker.epsilon), p=float('inf'), dim=(1, 2, 3)))))
                tensors = outputs.dataset.tensors
                with open(self.exp.file[key], 'wb') as f:
                    torch.save(tensors, f)

            else:
                print("Skip generating adversariral examples in {}.".format(self.exp.file[key]))

    def load_attacker(self):
        def load(name, params):
            torchattack_obj = importlib.import_module("torchattacks.attacks")
            model_obj = importlib.import_module("attacks")
            self.exp.model.load()
            try:
                # try load our own attack method first
                attack_instance = getattr(model_obj, name)(self.exp.model.model, **params)

            except AttributeError:
                # if it doesn't exist, load torchattack.
                attack_instance = getattr(torchattack_obj, name)(self.exp.model.model, **params)
            return attack_instance

        train_attack_name = self.exp.cfg.get('attack', 'train_attack_method')
        train_attack_params = dict([a, float(x)] for a, x in self.exp.cfg.items("train_attack_parameters"))
        test_attack_name = self.exp.cfg.get('attack', 'test_attack_method')
        test_attack_params = dict([a, float(x)] for a, x in self.exp.cfg.items("test_attack_parameters"))
        train_attacker = load(train_attack_name, train_attack_params)
        test_attacker = load(test_attack_name, test_attack_params)
        return {'train': train_attacker, 'test': test_attacker}
