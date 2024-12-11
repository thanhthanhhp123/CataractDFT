import torch
import torch.nn as nn

def convblock(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU()
    )

class BaseModel(nn.Module):
    def __init__(self, in_channels, num_classes = 2):
        super(BaseModel, self).__init__()
        self._conv = nn.Sequential(
            convblock(3, 64),
            nn.MaxPool2d(2),
            convblock(64, 128),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(401408, num_classes)

    def forward(self, x):
        x = self._conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class DualCNN(nn.Module):
    def __init__(self, n_blocks=3, num_classes = 2):
        super(DualCNN, self).__init__()
        self.n_blocks = n_blocks

        self.spatial_conv = nn.Sequential(
            convblock(3, 64),
            nn.MaxPool2d(2),
            convblock(64, 128),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(401408, 128)
        )
        self.frequency_conv = nn.Sequential(
            convblock(1, 64),
            nn.MaxPool2d(2),
            convblock(64, 128),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(401408, 128)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 2)
        )

    def forward(self, x_spatial, x_frequency):
        x_spatial = self.spatial_conv(x_spatial)
        x_frequency = self.frequency_conv(x_frequency)



        x = torch.cat((x_spatial, x_frequency), dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x