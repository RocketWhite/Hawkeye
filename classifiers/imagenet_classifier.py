import torch
import torch.nn
import torch.nn as nn
from classifiers  import NNClassifier 

class ImageNetClassifier(NNClassifier):
    def __init__(self):
        super(ImageNetClassifier, self).__init__()
        self.fc1 = nn.Linear(1000, 1000, bias=True)
        self.fc2 = nn.Linear(1000, 100,bias=True)
        self.fc3 = nn.Linear(100, 10, bias=True)
        self.fc4 = nn.Linear(10, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        return out

    # def training(self, device, data_loader, learning_rate=2e-4, num_epochs=10):
    #     for epoch in range(num_epochs):
    #         for i, (images, labels) in enumerate(data_loader):
    #             images = images.to(device)
    #             labels = labels.to(device)
    #             loss = self.fit(device, images, labels, learning_rate)
    #
    #             if (i + 1) % 10 == 0:
    #                 print("Epoch [{}/{}], Step [{}] Loss: {:.4f}"
    #                       .format(epoch + 1, num_epochs, i + 1, loss))

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
            _, output = torch.max(self(x).data, 1)
            self.stat.count(output, y)
        return output

