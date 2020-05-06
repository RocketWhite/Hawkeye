import torch
from utils import Counter
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import transforms

class ModelWrapper(object):
    """
    based on model, offering train, test, and statistic methods.
    This class if for self-constructed CIFAR10 and MNIST model.
    """
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
        self.stat = Counter()
        self.path = "./data/models_dict/%s.ckpt" % self.model.__class__.__name__

    def fit(self, x, y, batch_size=128):
        data_loader = DataLoader(TensorDataset(x, y), batch_size)
        self.train(data_loader)

        return self

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def train(self, data_loader, num_epochs=80, learning_rate=1e-3):
        def update_lr(optimizer, lr):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Train the model

        total_step = len(data_loader)
        curr_lr = learning_rate
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(data_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                # Forward pass
                outputs = self.model(images)

                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

            # Decay learning rate
            if (epoch + 1) % 20 == 0:
                curr_lr /= 3
                update_lr(optimizer, curr_lr)

        torch.save(self.model.state_dict(), self.path)

    def test(self, dataloader):
        for data, targets in dataloader:
            data = data.to(self.device)
            targets = targets.to(self.device)
            outputs = self.predict(data)
            _, predicted = torch.max(outputs.data, 1)
            self.stat.count(predicted, targets)
        return self.stat

    def load(self):
        self.model.load_state_dict(torch.load(self.path))
