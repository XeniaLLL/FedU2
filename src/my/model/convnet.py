import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super(CNNModel, self).__init__()
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=in_features,
                              out_channels=16,
                              kernel_size=5,
                              stride=1,
                              padding=2)
        self.relu1 = nn.ReLU()
        # Max Pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16,
                              out_channels=32,
                              kernel_size=5,
                              stride=1,
                              padding=2)
        self.relu2 = nn.ReLU()
        # Max Pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        # C1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Maxpool1
        out = self.maxpool1(out)

        # c1
        out = self.cnn2(out)
        out = self.relu2(out)

        # Maxpool1
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        # Linear Function
        out = self.fc(out)
        # Output
        return out  # F.log_softmax(out, dim=1)
