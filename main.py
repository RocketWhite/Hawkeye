import torch
import configparser
import importlib
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from adversarial_example_generator import constuct_adversarial_dataset

def main():
    # 0. Load the config file
    print("Loading config...")
    cfg = configparser.ConfigParser()
    cfg.read("config.ini")

    # 1. Select and Load a dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = cfg.get("dataset", "dataset")
    try:
        # first try to load the dataset of our own
        obj = importlib.import_module("datasets")
        dataset = getattr(obj, dataset_name)
    except AttributeError:
        # if doesn't, try torchvision.datasets
        obj = importlib.import_module("torchvision.datasets")
        dataset = getattr(obj, dataset_name)
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
    batch_size = int(cfg.get("model", "batch_size"))
    training_set = dataset("datasets", train=True, transform=transform, download=True)
    test_set = dataset("datasets", train=False, transform=transform, download=True)
    training_loader = DataLoader(training_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)
    # 2. Load a pre-trained model or train a model
    # pretrained model_dict name specification: ClassnameDataset.ckpt eg: ResNetCIFAR10.ckpt
    print("Loading Pretrained model...")
    model_name = cfg.get("model", "model")
    obj = importlib.import_module("models")
    model = getattr(obj, model_name + dataset_name)().to(device)

    if bool(int(cfg.get("model","retrain"))):
        model.fit(device, data_loader=training_loader)
    else:
        path = "./models_dict/%s.ckpt" % model.__class__.__name__
        model.load_state_dict(torch.load(path))

    # 3. Evaluate the trained model
    if int(cfg.get("model", "test")) != 0:
        print("Evaluating the pretrained model...")
        model.predict(device, data_loader=test_loader)
        print('Accuracy of the model on the test images: {}/{} = {}%'.format(
            model.correct, model.total, 100 * model.correct / model.total))
        model.clear_stat()
    else:
        print("Skip evaluating the model")

    # 4. Load pre-generated adversarial examples or Select some examples to attack
    num_train_natural = int(cfg.get('data_sampling', 'nb_train_natural_samples'))
    num_train_adversarial = int(cfg.get('data_sampling', 'nb_train_adversarial_samples'))
    num_test_natural = int(cfg.get('data_sampling', 'nb_test_natural_samples'))
    num_test_adversarial = int(cfg.get('data_sampling', 'nb_test_adversarial_samples'))
    # the attack method for training detector and test can be different, load separately.
    train_attack_name = cfg.get("attack", "train_attack_method")
    train_params = dict([a, float(x)] for a, x in cfg.items("train_attack_parameters"))
    path = './adversarial_examples/{}{}_{}_{}/'.format(
        model_name, dataset_name, train_attack_name, "_".join([str(elem) for elem in train_params.values()]))
    with open(Path(path) / "train.pt", 'rb') as f:
        tensor_adversarial_for_training = torch.load(f)
    dataset_adversarial = TensorDataset(*tensor_adversarial_for_training)
    adversarial_example_for_train_set = constuct_adversarial_dataset(training_loader,
                                                                     DataLoader(dataset_adversarial,
                                                                                batch_size=batch_size,
                                                                                shuffle=True),
                                                                     num_train_natural, num_train_adversarial)

    # load attack method for test
    test_attack_name = cfg.get("attack", "test_attack_method")
    test_params = dict([a, float(x)] for a, x in cfg.items("test_attack_parameters"))
    path = './adversarial_examples/{}{}_{}_{}/'.format(
            model_name, dataset_name, test_attack_name, "_".join([str(elem) for elem in test_params.values()]))
    with open(Path(path) / "test.pt", 'rb') as f:
        tensor_adversarial_for_test = torch.load(f)
    dataset_adversarial = TensorDataset(*tensor_adversarial_for_test)
    adversarial_example_for_test_set = constuct_adversarial_dataset(test_loader,
                                                                    DataLoader(dataset_adversarial,
                                                                               batch_size=batch_size,
                                                                               shuffle=True),
                                                                     num_test_natural, num_test_adversarial)


    # 5 Loading detection method
    # 5.1 Loading detectors
    print("Loading the detector...")
    # 5.1.1 Loading squeezers
    squeezers = []
    for name, squeezer in cfg.items("squeezer"):
        obj = __import__("squeezers", squeezer)
        squeezer_parameter = dict(cfg.items("squeezer_parameters_" + name))
        squeezers.append(getattr(obj, squeezer)(**squeezer_parameter))

    # 5.1.2 Loading Classifiers
    classifiers = []
    for name, classifier in cfg.items("classifier"):
        obj = __import__("classifiers", classifier)
        classifier_parameter = dict(cfg.items("classifier_parameters_" + name))
        classifiers.append(getattr(obj, classifier)(**classifier_parameter).to(device))

    # 5.1.3 Loading Detector
    detector_name = cfg.get("detector", "name")
    obj = __import__("detectors", detector_name)
    detector = getattr(obj, detector_name)(model, classifiers)

    # 6. Evaluating the attack method
    print("Evaluating the attack method...")
    model.predict(device, data_loader=DataLoader(dataset_adversarial, batch_size=batch_size))
    print('Attack Success rate before detection: {}/{} = {}%'
          .format(model.total - model.correct, model.total, model.correct/model.total*100))
    model.clear_stat()
    # 7 Train the detector
    print("Training...")
    detector.training(device, data_loader=DataLoader(adversarial_example_for_test_set, batch_size=batch_size,shuffle=True), squeezers=squeezers)
    # 8. test detector
    print("Defending...")
    detector.test(device, data_loader=DataLoader(adversarial_example_for_test_set, batch_size=batch_size), squeezers=squeezers)
    print("Evaluating the detector")
    # 9. Evaluate robust classification techniques

    for i, classifier in enumerate(classifiers):
        correct, total = (classifier.correct, classifier.total)
        tp, tn, fp, fn = (
        classifier.true_positive, classifier.true_negative, classifier.false_positive, classifier.false_negative)
        print("Classifier {}: Correct:{} Total:{} TP:{} TN:{} FP:{} FN:{}"
              .format(i, correct, total, tp, tn, fp, fn))
    correct, total = (detector.correct, detector.total)
    tp, tn, fp, fn = (detector.true_positive, detector.true_negative, detector.false_positive, detector.false_negative)
    print("Detector: Correct:{} Total:{} TP:{} TN:{} FP:{} FN:{}".format(correct, total, tp, tn, fp, fn))



if __name__ == "__main__":
    main()

