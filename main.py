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
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    if dataset == 'ImageNet':
        exp = ImageNetExperiment(device)
    else:
        exp = DetectorExperiment(device)

    # retrain if model_dict doesn't exist
    if retrain:
        train_loader = exp.load_natural_data(train=True, transform=exp.model.transform)
        exp.model.train(train_loader)

    # generate adversarial examples if data is not exists.
    generator = Generator(exp)
    generator.run()

    # run model
    exp.run()
