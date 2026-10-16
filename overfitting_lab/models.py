import torch.nn as nn

class LargeNet(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(2, 128), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(128, 128), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(128, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)
