import torch
import os
import importlib
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
from attacks import AttackWrapper
from pathlib import Path
import warnings
import configparser

def generate(model_name, dataset_name, attack_name, attack_params):
    cfg = configparser.ConfigParser()
    cfg.read("config.ini")
    path = './adversarial_examples/{}{}_{}_{}/'.format(
        model_name, dataset_name, attack_name, "_".join([str(elem) for elem in attack_params.values()]))
    data_folder = Path(path)
    data_folder.mkdir(parents=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Select and Load a dataset
    print("Loading the dataset")
    try:
        root = cfg.get("dataset", "root")
        print(root)
    except:
        root = "datasets"    
    print(root)
    try:
        # first try to load the dataset of our own
        obj = importlib.import_module("datasets")
        dataset = getattr(obj, dataset_name)
    except AttributeError:
        # if doesn't, try torchvision.datasets
        obj = importlib.import_module("torchvision.datasets")
        dataset = getattr(obj, dataset_name)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])
    if dataset_name == 'ImageNet':
        training_set =  dataset(root, split='train', transform=transform)
        test_set = dataset(root, split='val', transform=transform)
    else:
        training_set = dataset(root, train=True, transform=transform, download=True)
        test_set = dataset(root, train=False, transform=transform, download=True)
    training_loader = DataLoader(training_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128)

    # 2. Load the pretrained model
    # pretrained model_dict name specification: ClassnameDatasetname.ckpt eg: ResNetCIFAR10.ckpt
    print("Loading the pretrained model...")
    try:
        # get own model first.
        obj = importlib.import_module("models")
        model = getattr(obj, model_name + dataset_name)().to(device)
        model_dict_path = "./models_dict/%s.ckpt" % model.__class__.__name__
        model.load_state_dict(torch.load(model_dict_path))

    except:
        obj = importlib.import_module("torchvision.models")
        model = getattr(obj, model_name)(pretrained=True)

    model = model.to(device)
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
    attacker.save(device, training_loader, 10000, path + 'train.pt')
    attacker.save(device, test_loader, 1000, path + 'test.pt')

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
    dataset_name = 'MNIST'
    attack_name = 'FGSM'
    attack_params = {'eps': 0.032}
    generate(model_name, dataset_name, attack_name, attack_params)
