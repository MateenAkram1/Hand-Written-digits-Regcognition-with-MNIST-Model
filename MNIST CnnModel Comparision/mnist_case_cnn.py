import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTCaseCNN(nn.Module):
    def __init__(self, case: int = 1):
        super().__init__()
        if not (1 <= case <= 6):
            raise ValueError("Case must be between 1 and 6")
        self.case = case

        # Shared layers
        self.conv1  = nn.Conv2d(1,  32, 3)   # -> 26×26
        self.conv2  = nn.Conv2d(32, 64, 3)   # -> 24×24
        self.pool   = nn.MaxPool2d(2, 2)     # halves spatial dims
        self.drop25 = nn.Dropout(0.25)
        self.drop50 = nn.Dropout(0.50)

        # Figure out flatten_size by running a dummy through the conv/pool part
        flatten_size = self._compute_flatten_size()

        # Fully connected layers
        self.fc1 = nn.Linear(flatten_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def _compute_flatten_size(self):
        x = torch.zeros(1, 1, 28, 28)
        if   self.case == 1:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)

        elif self.case == 2:
            x = F.relu(self.conv1(x)); x = self.pool(x)
            x = F.relu(self.conv2(x)); x = self.pool(x)

        elif self.case == 3:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)

        elif self.case == 4:
            x = F.relu(self.conv1(x)); x = self.pool(x)
            x = F.relu(self.conv2(x)); x = self.pool(x)

        elif self.case == 5:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)

        elif self.case == 6:
            x = F.relu(self.conv1(x)); x = self.pool(x)
            x = F.relu(self.conv2(x)); x = self.pool(x)

        num_features = x.numel() // x.shape[0]
        return num_features

    def forward(self, x):
        # Convolutional / pooling / dropout backbone
        if   self.case == 1:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = self.drop25(x)

        elif self.case == 2:
            x = F.relu(self.conv1(x)); x = self.pool(x); x = self.drop25(x)
            x = F.relu(self.conv2(x)); x = self.pool(x); x = self.drop25(x)

        elif self.case == 3:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)

        elif self.case == 4:
            x = F.relu(self.conv1(x)); x = self.pool(x)
            x = F.relu(self.conv2(x)); x = self.pool(x)

        elif self.case == 5:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = self.drop25(x)

        elif self.case == 6:
            x = F.relu(self.conv1(x)); x = self.pool(x)
            x = F.relu(self.conv2(x)); x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Dense layers + dropout as per case
        if   self.case in {1, 5}:
            x = F.relu(self.fc1(x))
            x = self.drop50(x)
            x = self.fc2(x)

        elif self.case == 2:
            x = F.relu(self.fc1(x))
            x = self.drop50(x)
            x = self.fc2(x)

        elif self.case in {3, 4}:
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

        elif self.case == 6:
            x = F.relu(self.fc1(x))
            x = self.drop50(x)
            x = self.fc2(x)

        return x


if __name__ == "__main__":
    # Sanity‐check that each case runs without shape errors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for c in range(1, 7):
        model = MNISTCaseCNN(case=c).to(device)
        dummy = torch.randn(2, 1, 28, 28, device=device) 
        out = model(dummy)
        print(f"Case {c} → output shape {out.shape}")
