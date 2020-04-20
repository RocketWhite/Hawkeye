from experiment.experiment_template import DetectorExperiment
from experiment.imagenet_template import ImageNetExperiment
from experiment.adversarial_generator import Generator
import os

if __name__  == "__main__":
    retrain = 1
    #  exp = ImageNetExperiment()
    exp = DetectorExperiment()
    device = exp.device
    # retrain if model_dict doesn't exist
    if retrain:
        train_loader = exp.load_natural_data(train=True, transform=exp.model.transform)
        exp.model.train(train_loader)

    # generate adversarial examples if data is not exists.
    generator = Generator(exp)
    generator.run()

    # run model
    exp.run()
