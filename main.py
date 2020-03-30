import torch
import configparser
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, ImageNet
from models import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler


def make_weights_for_sampling(dataset, sampling):
    n_classes = len(dataset.classes)
    count = [0] * n_classes
    weight_per_class = [0.] * n_classes
    weight = [0] * len(dataset.targets)
    for item in dataset.targets:
        count[item] += 1
    if sampling == 0:
        # for this kind of sampling, weight_per_class becomes a switch.
        for idx, val in enumerate(dataset.targets):
            if sum(weight_per_class) == n_classes:
                break
            if weight_per_class[val] == 0:
                weight[idx] = 1
                weight_per_class[val] = 1

    elif sampling == 1:
        for i in range(n_classes):
            weight_per_class[i] = float(sum(count)) / float(count[i])
        for idx, val in enumerate(dataset.targets):
            weight[idx] = weight_per_class[val]
    else:
        raise(ValueError("Undefined value %s, check config.ini or documents" % sampling))

    return weight


def main():
    # 0. Load the config file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = configparser.ConfigParser()
    cfg.read("config.ini")

    # 1. Select and Load a dataset
    dataset_name = cfg.get("dataset", "dataset")
    if dataset_name == "MNIST":
        dataset = MNIST
    elif dataset_name == "FashionMNIST":
        dataset = FashionMNIST
    elif dataset_name == "Cifar":
        dataset = CIFAR10
    elif dataset_name == "ImageNet":
        dataset = ImageNet
    else:
        obj = __import__("datasets", dataset_name)
        dataset = getattr(obj, dataset_name)
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
    training_set = dataset("datasets", train=True, transform=transform, download=True)
    test_set = dataset("datasets", train=False, transform=transform, download=True)

    # 2. Load a pre-trained model or train a model
    test_loader = DataLoader(test_set, shuffle=True)
    model_name = cfg.get("model", "model")
    obj = __import__("models", model_name + dataset_name)
    AClass = getattr(obj, model_name + dataset_name)
    model = AClass().to(device)
    if bool(int(cfg.get("model","retrain"))):
        batch_size = int(cfg.get("model", "batch_size"))
        training_loader = DataLoader(training_set, batch_size, shuffle=True)
        model = train(device, model, training_loader)
    else:
        path = "./models_dict/%s.ckpt" % model.__class__.__name__
        model.load_state_dict(torch.load(path))

    # 3. Evaluate the trained model
    # test(device, model, test_loader)

    # 4. Select some examples to attack
    nb_train_samples = int(cfg.get("data_sampling", "nb_train_samples"))
    nb_test_samples = int(cfg.get("data_sampling", "nb_test_samples"))
    batch_size = int(cfg.get("data_sampling", "batch_size"))
    sampling = int(cfg.get("data_sampling", "sampling"))
    if sampling == 2:
        adversarial_examples_train_loader = DataLoader(dataset=training_set,
                                                       batch_size=batch_size,
                                                       sampler=RandomSampler(training_set,
                                                                             num_samples=nb_train_samples))
        adversarial_example_test_loader = DataLoader(dataset=test_set,
                                                     batch_size=batch_size,
                                                     sampler=RandomSampler(test_set, num_samples=nb_test_samples))
    else:
        train_weights = make_weights_for_sampling(training_set, sampling)
        test_weights = make_weights_for_sampling(test_set, sampling)
        train_sampler = WeightedRandomSampler(train_weights, nb_train_samples)
        test_sampler = WeightedRandomSampler(test_weights, nb_test_samples)
        adversarial_examples_train_loader = DataLoader(dataset=training_set,
                                                       batch_size=batch_size,
                                                       sampler=train_sampler)
        adversarial_example_test_loader = DataLoader(dataset=test_set,
                                                     batch_size=batch_size,
                                                     sampler=test_sampler)

    # 5. Generate adversarial examples
    obj = __import__("models", model_name + dataset_name)
    AClass = getattr(obj, model_name + dataset_name)
    model = AClass().to(device)

    # 6. Evaluate robust classification techniques

    # 7. Detection Experiment


if __name__ == "__main__":
    main()
