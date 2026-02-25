import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(QNetwork, self).__init__()
        # Architecture simple : 2 couches cachées de 128 neurones
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x) # Retourne les Q-values pour chaque action