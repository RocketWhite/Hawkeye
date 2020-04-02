import torch
import torch.nn
import torch.nn as nn

from torch.nn import functional as F

class NNClassifier(nn.Module):
    def __init__(self):
        super(NNClassifier, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.sigmoid(self.fc4(out)).squeeze()
        return out

    def fit(self, dataloader, learning_rate=2e-4, num_epochs=10):
        self.train()
        # Loss and optimizer
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)


        # Train the model

        curr_lr = learning_rate
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(dataloader):
                # Forward pass
                outputs = self(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                if (i + 1) % 100 == 0:
                    print("Epoch [{}/{}], Step {}, Loss: {:.4f}"
                          .format(epoch + 1, num_epochs, i + 1, loss.item()))

    # Save the model checkpoint
    # path = "./models_dict/%s.ckpt" % model.__class__.__name__
    # torch.save(model.state_dict(), path)


    def predict(self, x):
        # Test the model
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            return ((outputs - 0.5 > 0)).float()


        # Save the model checkpoint
        # torch.save(model.state_dict(), './resnet.ckpt')

