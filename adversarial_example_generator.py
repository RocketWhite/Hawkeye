import torch
import os
import importlib
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
from attacks import AttackWrapper
from pathlib import Path
import warnings


def generate(model_name, dataset_name, attack_name, attack_params):
    path = './adversarial_examples/{}{}_{}_{}/'.format(
        model_name, dataset_name, attack_name, "_".join([str(elem) for elem in attack_params.values()]))
    print(path)
    data_folder = Path(path)
    data_folder.mkdir(parents=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Select and Load a dataset
    print("Loading the dataset")
    try:  # first try to load the dataset of our own
        obj = importlib.import_module("datasets")
        dataset = getattr(obj, dataset_name)
    except AttributeError:  # if doesn't, try torchvision.datasets
        obj = importlib.import_module("torchvision.datasets")
        dataset = getattr(obj, dataset_name)
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
    training_set = dataset("datasets", train=True, transform=transform, download=True)
    test_set = dataset("datasets", train=False, transform=transform, download=True)
    training_loader = DataLoader(training_set, shuffle=True, batch_size = 1000)
    test_loader = DataLoader(test_set, shuffle=True, batch_size = 1000)

    # 2. Load the pretrained model
    # pretrained model_dict name specification: ClassnameDatasetname.ckpt eg: ResNetCIFAR10.ckpt
    print("Loading the pretrained model...")


    obj = importlib.import_module("models")
    model = getattr(obj, model_name + dataset_name)().to(device)
    # 2. loading the attacker

    torchattack_obj = importlib.import_module("torchattacks.torchattacks")
    model_obj = importlib.import_module("attacks")

    try:
        # try load our own attack method first
        attack_instance = getattr(model_obj, attack_name)(model, **attack_params)

    except AttributeError:
        # if it doesn't exist, load torchattack.
        attack_instance = getattr(torchattack_obj, attack_name)(model, **attack_params)

    attacker = AttackWrapper(attack_instance)

    # 3. generating adversarial examples
    print("Generating...")
    attacker.save(device, training_loader, path + 'train.pt')
    attacker.save(device, test_loader, path + 'test.pt')

def load_adversarial_examples(root_dir, model_name, dataset_name, attack_name, attack_params, train=True):
    path = './adversarial_examples/{}{}_{}_{}/'.format(
        model_name, dataset_name, attack_name, "_".join([str(elem) for elem in attack_params.values()]))
    if train == True:
        with open(Path(root_dir) / Path(path) / "train.pt", 'rb') as f:
            tensor_adversarial = torch.load(f)
    else:
        with open(Path(root_dir) / Path(path) / "test.pt", 'rb') as f:
            tensor_adversarial = torch.load(f)
    return tensor_adversarial


def constuct_adversarial_dataset(dataloader_natural, dataloader_adversarial, num_natural, num_adversarial):
    """
    natural examples and adversarial examples are different(before load to dataloader, the format is different)
    """
    x_natural = []
    y_natural = []
    x_adversarial = []
    y_adversarial = []
    for data, targets in dataloader_natural:
        x_natural.append(data)
        y_natural.append(torch.zeros_like(targets))
        if len(y_natural) >= num_natural:
            break
    for data, targets in dataloader_adversarial:
        x_adversarial.append(data)
        y_adversarial.append(torch.ones_like(targets))
        if len(y_adversarial) >= num_adversarial:
            break
    return TensorDataset(torch.cat(x_natural[0:num_natural] + x_adversarial[0:num_adversarial], dim=0),
                         torch.cat(y_natural[0:num_natural] + y_adversarial[0:num_adversarial], dim=0))


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    model_name = 'ResNet'
    dataset_name = 'CIFAR10'
    attack_name = 'CW'
    attack_params = {'c': 0.8}
    generate(model_name, dataset_name, attack_name, attack_params)