import torch
import torch.nn
import torch.nn as nn
from utils import BinaryCounter
from torch.utils.data import DataLoader, TensorDataset


class NNClassifier(nn.Module):
    def __init__(self, device, num_epochs=10, learning_rate=2e-4, batch_size=100):
        super(NNClassifier, self).__init__()
        self.fc1 = nn.Linear(10, 10, bias=True)
        self.fc2 = nn.Linear(10, 10, bias=True)
        self.fc3 = nn.Linear(10, 10, bias=True)
        self.fc4 = nn.Linear(10, 2)
        self.relu = nn.ReLU(inplace=True)
        self.stat = BinaryCounter()
        self.device = device
        self.num_epochs = int(num_epochs)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)

    def forward(self, X):
        out = self.relu(self.fc1(X))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        return out

    def fit(self, X, y):
        self.train()
        data_loader = DataLoader(TensorDataset(X, y), batch_size=self.batch_size)
        # Loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # fit
        for epoch in range(self.num_epochs):
            for data, labels in data_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                outputs = self(data)
                # Backward and optimize
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if epoch % 5 == 0:
                #     print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, self.num_epochs, loss))

    def predict(self, X, y):
        # Test the model
        self.eval()

        with torch.no_grad():
            X = X.to(self.device)
            y = y.to(self.device)
            _, output = torch.max(self(X).data, 1)
            self.stat.count(output, y)
        return output.detach().cpu()

    def save_model(path):
        torch.save(self.state_dict(), path)
