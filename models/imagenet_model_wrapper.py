import torch
from utils import Counter
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import transforms
from models.model_wrapper import ModelWrapper

class ImageNetModelWrapper(ModelWrapper):
    def __init__(self, model):
        super(ImageNetModelWrapper, self).__init__(model)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        self.stat = Counter()

        def test(self, dataloader):
            for data, targets in dataloader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                outputs = self.predict(data)
                _, predicted = torch.max(outputs.data, 1)
                self.stat.count(predicted, targets)

            return self.stat

        def load(self):
            pass