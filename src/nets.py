"""Neural Net Modules."""

from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.batch = nn.BatchNorm2d(2)
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 18 * 18, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.drp = nn.Dropout(0.3)

    def forward(self, x):
        x = self.batch(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drp(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drp(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.batch = nn.BatchNorm2d(2)
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 2)
        self.drp = nn.Dropout(0.3)

    def forward(self, x):
        x = self.batch(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drp(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drp(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drp(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.drp(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
