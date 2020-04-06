import torch
import torch.nn
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class NNClassifier(nn.Module):
    def __init__(self):
        super(NNClassifier, self).__init__()
        self.fc1 = nn.Linear(10, 10, bias=True)
        self.fc2 = nn.Linear(10, 10,bias=True)
        self.fc3 = nn.Linear(10, 10, bias=True)
        self.fc4 = nn.Linear(10, 2)
        self.relu = nn.ReLU(inplace=True)

        self.total = 0.
        self.correct = 0.
        self.true_positive = 0.
        self.true_negative = 0.
        self.false_positive = 0.
        self.false_negative = 0.

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        return out

    def training(self, device, data_loader, learning_rate=2e-4, num_epochs=10):
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(data_loader):
                images = images.to(device)
                labels = labels.to(device)
                loss = self.fit(device, images, labels, learning_rate)

                if (i + 1) % 10 == 0:
                    print("Epoch [{}/{}], Step [{}] Loss: {:.4f}"
                          .format(epoch + 1, num_epochs, i + 1, loss))

    def fit(self, device, x, y, learning_rate=2e-4):
        self.train()
        # Loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # fit
        x = x.to(device)
        y = y.to(device)
        outputs = self(x)

        # Backward and optimize
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


    # Save the model checkpoint
    # path = "./models_dict/%s.ckpt" % model.__class__.__name__
    # torch.save(model.state_dict(), path)


    def predict(self, device, x, y):
        # Test the model
        self.eval()

        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            _, output = torch.max(self.forward(x).data, 1)

            true_positive = torch.sum((output == 1).float() * (y == 1).float())
            true_negative = torch.sum((output == 0).float() * (y == 0).float())
            false_positive = torch.sum((output == 1).float() * (y == 0).float())
            false_negative = torch.sum((output == 0).float() * (y == 1).float())
            self.true_positive += true_positive
            self.true_negative += true_negative
            self.false_positive += false_positive
            self.false_negative += false_negative
            self.correct += true_positive + true_negative
            self.total += true_positive + true_negative + false_positive + false_negative


        return output

        # Save the model checkpoint
        # torch.save(model.state_dict(), './resnet.ckpt')

    def clear(self):
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
        self.correct = 0
        self.total = 0