import torch
import configparser
import importlib
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from models.model_wrapper import ModelWrapper
from experiment.adversarial_generator import Generator


class DetectorExperiment(object):
    def __init__(self, device):
        self.cfg = configparser.ConfigParser()
        self.cfg.read("./config.ini")
        self.device = device
        self.dataset = self.cfg.get("dataset", "dataset")
        self.model = self.load_model(self.dataset)
        self.path, self.file = self.load_ae_file()

    def run(self):
        # load dataset
        train_loader = self.load_natural_data(train=True, transform=self.model.transform)
        test_loader = self.load_natural_data(train=False, transform=self.model.transform)

        # retrain or not
        retrain = bool(int(self.cfg.get("model", "retrain")))
        if retrain:
            self.model.train(train_loader)
        self.model.load()
        # generate adversarial examples
        generator = Generator(self)
        generator.run()

        # load adversarial examples
        ae_train_loader = self.load_adversarial_data(train=True)
        ae_test_loader = self.load_adversarial_data(train=False)

        # generate examples for detector
        mixed_train_loader = self.generate_mixed_dataloader(train_loader, ae_train_loader, train=True)
        # print(len(train_loader.dataset))
        # print(len(ae_train_loader.dataset))
        # print(len(mixed_train_loader.dataset))
        mixed_test_loader = self.generate_mixed_dataloader(test_loader, ae_test_loader, train=False)

        # evaluate model if needed
        if int(self.cfg.get("model", "test")) != 0:
            print("Evaluating the pretrained model...")
            stat = self.model.test(test_loader)
            print('Accuracy of the model on the test images: {}/{} = {}%'.format(
                stat.correct, stat.total, 100 * stat.accuracy()))
            stat.clear()

        # evaluate the adversarial attack strength
        print("Evaluating the attack...")
        stat = self.model.test(ae_test_loader)
        print('Accuracy of the model on the adversarial images: {}/{} = {}%'.format(
            stat.correct, stat.total, 100 * stat.accuracy()))
        stat.clear()

        # load detector and training
        squeezers = self.load_squeezer()
        detector = self.load_detector(self.model.model)
        detector.fit(mixed_train_loader, squeezers)

        # defending
        detector.predict(mixed_test_loader, squeezers)
        # evaluating
        self.evaluate(detector)

    def load_natural_data(self, train, transform):
        """
        read config from config.ini
        return dataloader
        :return:
        """
        dataset_name = self.cfg.get("dataset", "dataset")
        root = "./data/datasets"
        try:
            # first try to load the dataset of our own
            obj = importlib.import_module("data.datasets")
            Dataset = getattr(obj, dataset_name)
        except AttributeError:
            # if doesn't, try torchvision.datasets
            obj = importlib.import_module("torchvision.datasets")
            Dataset = getattr(obj, dataset_name)
        else:
            raise AttributeError("Both datasets module and torchvision.datasets module "
                                 "doesn't contain your dataset: {}".format(dataset_name))
        dataset = Dataset(root, train, transform, download=True)
        batch_size = int(self.cfg.get("dataset", "batch_size"))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader

    def load_model(self, dataset):
        model_name = self.cfg.get("model", "model")
        print("Loading Pretrained model...")
        obj = importlib.import_module("models")
        model = getattr(obj, model_name + dataset)().to(self.device)
        model = ModelWrapper(model)
        return model

    def load_ae_file(self):
        train_attack_name = self.cfg.get('attack', 'train_attack_method')
        train_attack_params = dict([a, float(x)] for a, x in self.cfg.items("train_attack_parameters"))

        path_train = Path('./data/adversarial_examples/{}_{}_{}'.format(
            self.model.model.__class__.__name__,
            train_attack_name,
            "_".join([str(elem) for elem in train_attack_params.values()])))

        test_attack_name = self.cfg.get('attack', 'test_attack_method')
        test_attack_params = dict([a, float(x)] for a, x in self.cfg.items("test_attack_parameters"))

        path_test = Path('./data/adversarial_examples/{}_{}_{}/'.format(
            self.model.model.__class__.__name__,
            test_attack_name,
            "_".join([str(elem) for elem in test_attack_params.values()])))
        path = {'train': path_train, 'test': path_test}
        file = {'train': path_train / 'train.pt', 'test': path_test / 'test.pt'}
        return path, file

    def load_adversarial_data(self, train=True):
        if train:
            file = self.file['train']
        else:
            file = self.file['test']
        batch_size = int(self.cfg.get("dataset", "batch_size"))
        with open(file, 'rb') as f:
            tensors = torch.load(f)
            dataset = TensorDataset(tensors[0], tensors[1])
        return DataLoader(dataset, batch_size=batch_size)

    def load_squeezer(self):
        squeezers = []
        for name, squeezer in self.cfg.items("squeezer"):
            obj = __import__("squeezers", squeezer)
            squeezer_parameter = dict(self.cfg.items("squeezer_parameters_" + name))
            squeezers.append(getattr(obj, squeezer)(**squeezer_parameter))
        return squeezers

    def load_detector(self, model):
        model.eval()
        # load classifier
        classifiers = []
        output_mode = self.cfg.get("detector", "output")
        for name, classifier in self.cfg.items("classifier"):
            obj = __import__("classifiers", classifier)
            classifier_parameter = dict(self.cfg.items("classifier_parameters_" + name))
            classifiers.append(getattr(obj, classifier)(self.device, **classifier_parameter))
        detector_name = self.cfg.get("detector", "name")
        obj = __import__("detectors", detector_name)
        detector = getattr(obj, detector_name)(model, classifiers, output_mode)
        return detector

    def generate_mixed_dataloader(self, natural_dataloader, ae_dataloader, train):
        def constuct_dataloader(dataloader_natural, dataloader_adversarial, batch_natural, batch_adversarial,
                                batch_size):
            """
            natural examples and adversarial examples are different(before load to dataloader, the format is different)
            """
            x_natural = []
            y_natural = []
            x_adversarial = []
            y_adversarial = []
            for i, (data, targets) in enumerate(dataloader_natural):
                if i >= batch_natural:
                    break
                x_natural.append(data)
                y_natural.append(torch.zeros_like(targets))

            for i, (data, targets) in enumerate(dataloader_adversarial):
                if i >= batch_adversarial:
                    break
                x_adversarial.append(data)
                y_adversarial.append(torch.ones_like(targets))

            dataset = TensorDataset(torch.cat(x_natural + x_adversarial, dim=0),
                                    torch.cat(y_natural + y_adversarial, dim=0))
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)
        batch_size = int(self.cfg.get("dataset", "batch_size"))
        if train:
            num_natural = int(self.cfg.get('data_sampling', 'batch_train_natural_samples'))
            num_ae = int(self.cfg.get('data_sampling', 'batch_train_adversarial_samples'))
        else:
            num_natural = int(self.cfg.get('data_sampling', 'batch_test_natural_samples'))
            num_ae = int(self.cfg.get('data_sampling', 'batch_test_adversarial_samples'))

        return constuct_dataloader(natural_dataloader, ae_dataloader,
                                              num_natural, num_ae, batch_size)

    def evaluate(self, detector):
        for i, classifier in enumerate(detector.classifiers):
            correct, total = (classifier.stat.correct, classifier.stat.total)
            tp, tn, fp, fn = (
                classifier.stat.true_positive, classifier.stat.true_negative, classifier.stat.false_positive,
                classifier.stat.false_negative)
            print("Classifier {}: Correct:{} Total:{} TP:{} TN:{} FP:{} FN:{}"
                  .format(i, correct, total, tp, tn, fp, fn))
        correct, total = (detector.stat.correct, detector.stat.total)
        tp, tn, fp, fn = (detector.stat.true_positive, detector.stat.true_negative,
                          detector.stat.false_positive, detector.stat.false_negative)
        print("Detector: Correct:{} Total:{} TP:{} TN:{} FP:{} FN:{}".format(correct, total, tp, tn, fp, fn))
