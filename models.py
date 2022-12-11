import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatt = nn.Flatten()
        self.fc1 = nn.Linear(784, 240)
        self.fc2 = nn.Linear(240, 128)
        self.fc3 = nn.Linear(128, 10)

        self.drop1 = nn.Dropout(0.50)
        self.drop2 = nn.Dropout(0.50)

    def forward(self, x):

        x = self.flatt(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.fc3(x)

        return x


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 48, 3)
        self.conv4 = nn.Conv2d(48, 48, 3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.drop1 = nn.Dropout(0.50)
        self.drop2 = nn.Dropout(0.50)

        self.flatt = nn.Flatten()

        # Linear ouput layer
        self.fc1 = nn.Linear(768, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.pool1(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.pool2(x)

        x = self.flatt(x)

        x = self.drop1(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop2(x)

        x = self.out(x)

        return x


class BranchCNNShort(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv2d(1, 16, 3)
        self.conv21 = nn.Conv2d(1, 16, 3)

        self.conv12 = nn.Conv2d(16, 16, 3)
        self.conv22 = nn.Conv2d(16, 16, 3)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv13 = nn.Conv2d(16, 32, 3)
        self.conv23 = nn.Conv2d(16, 32, 3)

        self.conv14 = nn.Conv2d(32, 32, 3)
        self.conv24 = nn.Conv2d(32, 32, 3)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.flatt = nn.Flatten()
        self.norm = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, 10)

        self.drop1 = nn.Dropout(0.50)
        self.drop2 = nn.Dropout(0.50)

    def forward(self, x):

        # First blcok

        x_1 = self.conv11(x)
        x_1 = F.relu(x_1)
        x_2 = self.conv21(x)
        x_2 = F.relu(x_2)

        x_1 = self.conv12(x_1)
        x_1 = F.relu(x_1)
        x_2 = self.conv22(x_2)
        x_2 = F.relu(x_2)

        x_1 = self.pool1(x_1)
        x_2 = self.pool1(x_2)

        # Second block

        x_1 = self.conv13(x_1)
        x_1 = F.relu(x_1)
        x_2 = self.conv23(x_2)
        x_2 = F.relu(x_2)

        x_1 = self.conv14(x_1)
        x_1 = F.relu(x_1)
        x_2 = self.conv24(x_2)
        x_2 = F.relu(x_2)

        x_1 = self.pool2(x_1)
        x_2 = self.pool2(x_2)

        x_1 = self.flatt(x_1)
        x_2 = self.flatt(x_2)

        # Concatenate both x_1 and x_2

        x = torch.cat((x_1, x_2), dim=1)
        x = self.norm(x)

        x = self.drop1(x)

        # Linear layers

        x = self.fc1(x)
        x = F.relu(x)

        x = self.drop2(x)

        x = self.out(x)
        return x


class BranchCNNLarge(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv2d(1, 32, 3)
        self.conv21 = nn.Conv2d(1, 32, 3)

        self.conv12 = nn.Conv2d(32, 32, 3)
        self.conv22 = nn.Conv2d(32, 32, 3)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv13 = nn.Conv2d(32, 48, 3)
        self.conv23 = nn.Conv2d(32, 48, 3)

        self.conv14 = nn.Conv2d(48, 48, 3)
        self.conv24 = nn.Conv2d(48, 48, 3)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.flatt = nn.Flatten()
        self.norm = nn.BatchNorm1d(1536)
        self.fc1 = nn.Linear(1536, 256)
        self.out = nn.Linear(256, 10)

        self.drop1 = nn.Dropout(0.50)
        self.drop2 = nn.Dropout(0.50)

    def forward(self, x):

        # First blcok

        x_1 = self.conv11(x)
        x_1 = F.relu(x_1)
        x_2 = self.conv21(x)
        x_2 = F.relu(x_2)

        x_1 = self.conv12(x_1)
        x_1 = F.relu(x_1)
        x_2 = self.conv22(x_2)
        x_2 = F.relu(x_2)

        x_1 = self.pool1(x_1)
        x_2 = self.pool1(x_2)

        # Second block

        x_1 = self.conv13(x_1)
        x_1 = F.relu(x_1)
        x_2 = self.conv23(x_2)
        x_2 = F.relu(x_2)

        x_1 = self.conv14(x_1)
        x_1 = F.relu(x_1)
        x_2 = self.conv24(x_2)
        x_2 = F.relu(x_2)

        x_1 = self.pool2(x_1)
        x_2 = self.pool2(x_2)

        x_1 = self.flatt(x_1)
        x_2 = self.flatt(x_2)

        # Concatenate both x_1 and x_2

        x = torch.cat((x_1, x_2), dim=1)
        x = self.norm(x)

        x = self.drop1(x)

        # Linear layers

        x = self.fc1(x)
        x = F.relu(x)

        x = self.drop2(x)

        x = self.out(x)
        return x

