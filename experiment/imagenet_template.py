import torch
import configparser
import scipy
import importlib
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from models.imagenet_model_wrapper import ImageNetModelWrapper
from .experiment_template import DetectorExperiment

class ImageNetExperiment(DetectorExperiment):
    def __init__(self, device):
        super(ImageNetExperiment, self).__init__(device)

    def load_model(self, dataset):
        model_name = self.cfg.get("model", "model")
        print("Loading Pretrained model...")
        obj = importlib.import_module("torchvision.models")
        model = getattr(obj, model_name)(pretrained=True).to(self.device)
        model = ImageNetModelWrapper(model)
        return model

    def load_natural_data(self, train, transform):
        dataset_name = self.cfg.get("dataset", "dataset")
        root = "./data/datasets/ImageNet"
        obj = importlib.import_module("torchvision.datasets")
        Dataset = getattr(obj, dataset_name)
        if train:
            string = 'train'
        else:
            string = 'val'
        dataset = Dataset(root, split=string, transform=transform)
        batch_size = int(self.cfg.get("dataset", "batch_size"))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader
