import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(3, 16, 5, padding=1)
        self.conv_layer2 = nn.Conv2d(16, 32, 5, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, 2)
        self.maxpool2 = nn.MaxPool2d(3, 2)
        self.conv_layer3 = nn.Conv2d(32, 64, 5, padding=1)

        self.fc1 = nn.Linear(30976, 64)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.relu1(out)
        out = self.conv_layer2(out)
        out = self.relu2(out)
        out = self.maxpool1(out)

        out = self.conv_layer3(out)
        out = self.relu3(out)
        out = self.maxpool2(out)

        out = torch.flatten(out, 1)

        out = self.fc1(out)
        out = self.relu4(out)
        out = self.fc2(out)
        out = self.relu5(out)
        out = self.fc3(out)
        return out


