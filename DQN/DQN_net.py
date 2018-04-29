import torch.nn as nn
import torch.nn.functional as F
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class DQN_net(nn.Module):
    def __init__(self, action_space):
        super(DQN_net, self).__init__()
        self.conv1 = (nn.Conv2d(4, 16, 8, stride=4))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = (nn.Conv2d(16, 32, 4, stride=2))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = (nn.Conv2d(32, 32, 3, stride=1))
        self.bn3 = nn.BatchNorm2d(32)
        self.fully_connected = (nn.Linear(1568, 216))
        self.output = nn.Linear(216, action_space)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 1568)
        x = F.relu(self.fully_connected(x))
        return self.output(x.view(x.size(0), -1))