from experiment.experiment_template import DetectorExperiment
from experiment.imagenet_template import ImageNetExperiment
from experiment.adversarial_generator import Generator
import torch
import configparser
if __name__  == "__main__":
    cfg = configparser.ConfigParser()
    cfg.read("./config.ini")
    retrain = bool(int(cfg.get("model", "retrain")))
    dataset = cfg.get("dataset", "dataset")
    device = torch.device('cuda: 1' if torch.cuda.is_available() else 'cpu')
    if dataset == 'ImageNet':
        exp = ImageNetExperiment(device)
    else:
        exp = DetectorExperiment(device)

    # run model
    exp.run()
