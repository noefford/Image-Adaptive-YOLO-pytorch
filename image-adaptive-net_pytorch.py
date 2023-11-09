import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class extract_parameters(nn.Module):
    def __init__(self):
        super(extract_parameters, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(2048, 64)
        self.fc2 = nn.Linear(64, 14)
    
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x),negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x),negative_slope=0.1)
        x = F.leaky_relu(self.conv3(x),negative_slope=0.1)
        x = F.leaky_relu(self.conv4(x),negative_slope=0.1)
        x = F.leaky_relu(self.conv5(x),negative_slope=0.1)
        x = x.view(-1, 2048)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

torch.manual_seed(0)
net = extract_parameters()
input = torch.ones(1,3,256,256)
out = net(input)
print(out)