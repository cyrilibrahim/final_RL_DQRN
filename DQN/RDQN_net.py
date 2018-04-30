import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class RDQN_net(nn.Module):
    def __init__(self, action_space, hidden_dim = 216):
        super(RDQN_net, self).__init__()

        self.hidden_dim = hidden_dim
        self.action_space = action_space
        self.cnn = nn.Sequential(nn.Conv2d(1, 16, kernel_size=8, stride=4),
                                 nn.BatchNorm2d(16),
                                 nn.ReLU(),
                                 nn.Conv2d(16, 32, kernel_size=4, stride=2),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(),
                                 nn.Conv2d(32, 32, kernel_size=3, stride=1),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(),
                                 Flatten(),
                                 nn.Linear(1568, 216),
                                 nn.ReLU()
                                 )

        self.lstm = nn.LSTM(input_size=216,hidden_size=self.hidden_dim, num_layers=1, batch_first=False)
        self.output = nn.Sequential(nn.Linear(self.hidden_dim, self.action_space))

    def forward(self, x, hx = None, cx = None, batch_size=1, seq_length=1):
        if hx is None or cx is None:
            hx, cx = self.init_hidden(batch_size)
        x = x.view(batch_size*seq_length, 1, 84, 84)
        x = self.cnn(x)
        x = x.view(seq_length, batch_size, self.hidden_dim)
        self.lstm.flatten_parameters()
        x, (hx, cx) = self.lstm(x, (hx, cx))
        x = x.contiguous()
        x = x.view(batch_size*seq_length, self.hidden_dim)
        x = self.output(x)

        return x, (hx, cx)

    def init_hidden(self, batch_size):
        h0 = Variable(torch.randn(1, batch_size, self.hidden_dim).cuda())
        c0 = Variable(torch.randn(1, batch_size, self.hidden_dim).cuda())
        return h0, c0