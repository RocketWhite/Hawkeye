import torch
import torch.nn as nn
import torch.optim as optim
from torchattacks.attacks import CW


class CW_ImageNet(CW):
    def __init__(self, model, targeted=False, c=1e-4, kappa=0, iters=1000, lr=0.01):
        super().__init__(model, bool(targeted), float(c), float(kappa), int(iters), float(lr))
    
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape([1,3,1,1]).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape([1,3,1,1]).to(self.device)
        images = images.to(self.device)
        labels = labels.to(self.device)

        # f-function in the paper
        def f(x):
            outputs = self.model(x)
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

            i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())

            # If targeted, optimize for making the other class most likely
            if self.targeted:
                return torch.clamp(i - j, min=-self.kappa)

            # If untargeted, optimize for making the other class most likely
            else:
                return torch.clamp(j - i, min=-self.kappa)

        w = torch.zeros_like(images).to(self.device)
        w.detach_()
        w.requires_grad = True

        optimizer = optim.Adam([w], lr=self.lr)
        prev = 1e10

        for step in range(self.iters):

            a = (1 / 2 * (nn.Tanh()(w) + 1) - mean) / std
            loss1 = nn.MSELoss(reduction='sum')(a, images)
            loss2 = torch.sum(self.c * f(a))
            if step % 1000 == 0:
                print(step, loss1, loss2)
            cost = loss1 + loss2

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Early Stop when loss does not converge.
            if step % (self.iters // 10) == 0:
                if cost > prev:
                    print('CW Attack is stopped due to CONVERGENCE....')
                    return a
                prev = cost

            print('- CW Attack Progress : %2.2f %%        ' % ((step + 1) / self.iters * 100), end='\r')
        adv_images = ((1 / 2 * (nn.Tanh()(w) + 1)).detach() - mean) / std
        return adv_images
