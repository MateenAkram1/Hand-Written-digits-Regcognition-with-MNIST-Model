import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # Input: 1 channel (grayscale), Output: 32 channels
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  
        self.fc2 = nn.Linear(128, 10)  # 10 classes for digits 0-9

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Output shape: (batch_size, 32, 26, 26)
        x = F.max_pool2d(x, 2)      # Output shape: (batch_size, 32, 13, 13)
        x = F.relu(self.conv2(x))   # Output shape: (batch_size, 64, 11, 11)
        x = F.max_pool2d(x, 2)      # Output shape: (batch_size, 64, 5, 5)
        x = x.view(x.size(0), -1)   # Flatten to (batch_size, 64*5*5)
        x = F.relu(self.fc1(x))     # Output shape: (batch_size, 128)
        x = self.fc2(x)             # Output shape: (batch_size, 10)
        return x