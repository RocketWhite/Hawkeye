import torch
import configparser
import importlib
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, ImageNet
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
from attacks import AttackWrapper


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
    print("Loading config...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = configparser.ConfigParser()
    cfg.read("config.ini")

    # 1. Select and Load a dataset
    dataset_name = cfg.get("dataset", "dataset")
    try:
        obj = importlib.import_module("torchvision.datasets")
        dataset = getattr(obj, dataset_name)
    except AttributeError:
        obj = importlib.import_module("torchvision.datasets")
        dataset = getattr(obj, dataset_name)
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
    training_set = dataset("datasets", train=True, transform=transform, download=True)
    test_set = dataset("datasets", train=False, transform=transform, download=True)

    # 2. Load a pre-trained model or train a model
    print("Loading Pretrained model...")
    test_loader = DataLoader(test_set, shuffle=True)
    model_name = cfg.get("model", "model")
    obj = importlib.import_module("models")
    model = getattr(obj, model_name + dataset_name)().to(device)

    if bool(int(cfg.get("model","retrain"))):
        batch_size = int(cfg.get("model", "batch_size"))
        training_loader = DataLoader(training_set, batch_size, shuffle=True)
        model.fit(device, data_loader=training_loader)
    else:
        path = "./models_dict/%s.ckpt" % model.__class__.__name__
        model.load_state_dict(torch.load(path))

    # 3. Evaluate the trained model
    # print("Evaluating the pretrained model...")
    # model.predict(device, data_loader=test_loader)
    # print('Accuracy of the model on the test images: {}/{} = {}%'.format(
    #     model.correct, model.total, 100 * model.correct / model.total))

    # 4. Select some examples to attack
    print("Sampling data to attack...")
    nb_train_samples = int(cfg.get("data_sampling", "nb_train_samples"))
    nb_test_samples = int(cfg.get("data_sampling", "nb_test_samples"))
    batch_size = int(cfg.get("data_sampling", "batch_size"))
    sampling = int(cfg.get("data_sampling", "sampling"))
    if sampling == 2:
        detector_train_loader = DataLoader(dataset=training_set,
                                           batch_size=batch_size,
                                           sampler=RandomSampler(training_set, num_samples=nb_train_samples),
                                           shuffle=True)
        detector_test_loader = DataLoader(dataset=test_set,
                                          batch_size=batch_size,
                                          sampler=RandomSampler(test_set, num_samples=nb_test_samples))
    else:
        train_weights = make_weights_for_sampling(training_set, sampling)
        test_weights = make_weights_for_sampling(test_set, sampling)
        train_sampler = WeightedRandomSampler(train_weights, nb_train_samples)
        test_sampler = WeightedRandomSampler(test_weights, nb_test_samples)
        detector_train_loader = DataLoader(dataset=training_set,
                                                       batch_size=batch_size,
                                                       sampler=train_sampler)
        detector_test_loader = DataLoader(dataset=test_set,
                                                     batch_size=batch_size,
                                                     sampler=test_sampler)

    # 5 Loading attack method and detection method
    # 5.1 Loading attack method
    print("Loading the attack methods...")
    obj = importlib.import_module("torchattacks.torchattacks")

    train_attack_name = cfg.get("attack", "train_attack_method")
    train_params = dict([a, float(x)] for a, x in cfg.items("train_attack_parameters"))
    train_attack_instance = getattr(obj, train_attack_name)(model, **train_params)
    train_attacker = AttackWrapper(train_attack_instance)

    test_attack_name = cfg.get("attack", "test_attack_method")
    test_params = dict([a, float(x)] for a, x in cfg.items("test_attack_parameters"))
    test_attack_instance = getattr(obj, test_attack_name)(model, **test_params)
    test_attacker = AttackWrapper(test_attack_instance)

    # 5.2 Loading detectors
    print("Loading the detector...")
    # 5.2.1 Loading squeezers
    squeezers = []
    for name, squeezer in cfg.items("squeezer"):
        obj = __import__("squeezers", squeezer)
        squeezer_parameter = dict(cfg.items("squeezer_parameters_" + name))
        squeezers.append(getattr(obj, squeezer)(**squeezer_parameter))

    # 5.2.2 Loading Classifiers
    classifiers = []
    for name, classifier in cfg.items("classifier"):
        obj = __import__("classifiers", classifier)
        classifier_parameter = dict(cfg.items("classifier_parameters_" + name))
        classifiers.append(getattr(obj, classifier)(**classifier_parameter).to(device))

    # 5.2.3 Loading Detector
    detector_name = cfg.get("detector", "name")
    obj = __import__("detectors", detector_name)
    detector = getattr(obj, detector_name)(model, classifiers)

    # 6. Training detectors
    print("Training Detectors...")
    detector.training(device, data_loader=detector_train_loader, squeezers=squeezers, attackers=train_attacker)
    # 7. Evaluating the attack method
    print("Evaluating the attack method...")

    print('Accuracy of training samples\' Adversarial images: %f %%' % (100 * train_attacker.accuracy()))

    # 8. test detector
    print("Defending...")
    detector.test(device, data_loader=detector_test_loader, squeezers=squeezers, attackers=test_attacker)

    print('Accuracy of test samples\' Adversarial images: %f %%' % (100 * test_attacker.accuracy()))
    print("Evaluating the detector")
    # 9. Evaluate robust classification techniques
    # train
    detector.test(device, data_loader=detector_train_loader, squeezers=squeezers, attackers=train_attacker)
    for classifier in classifiers:
        correct, total = (classifier.correct, classifier.total)
        tp, tn, fp, fn = (
            classifier.true_positive, classifier.true_negative, classifier.false_positive, classifier.false_negative)
        print(correct, total, tp, tn, fp, fn)


    correct, total = (detector.correct, detector.total)
    tp, tn, fp, fn = (detector.true_positive, detector.true_negative, detector.false_positive, detector.false_negative)
    print(correct, total, tp, tn, fp, fn)
    detector.clear()
    # test
    detector.test(device, data_loader=detector_test_loader, squeezers=squeezers, attackers=test_attacker)
    for classifier in classifiers:
        correct, total = (classifier.correct, classifier.total)
        tp, tn, fp, fn = (
        classifier.true_positive, classifier.true_negative, classifier.false_positive, classifier.false_negative)
        print(correct, total, tp, tn, fp, fn)
    correct, total = (detector.correct, detector.total)
    tp, tn, fp, fn = (detector.true_positive, detector.true_negative, detector.false_positive, detector.false_negative)
    print(correct, total, tp, tn, fp, fn)


if __name__ == "__main__":
    main()

