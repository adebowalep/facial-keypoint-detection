## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## This network takes in a square (224x224), grayscale image as input
        ## and outputs 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # input: 1x224x224
        self.conv1 = nn.Conv2d(1, 32, 5)     # -> 32x220x220 -> pool -> 32x110x110
        self.conv2 = nn.Conv2d(32, 64, 3)    # -> 64x108x108 -> pool -> 64x54x54
        self.conv3 = nn.Conv2d(64, 128, 3)   # -> 128x52x52  -> pool -> 128x26x26
        self.conv4 = nn.Conv2d(128, 256, 3)  # -> 256x24x24  -> pool -> 256x12x12

        self.pool = nn.MaxPool2d(2, 2)

        # batch norm after each conv helps stabilize training
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        # increasing dropout probability deeper into the network, as in NaimishNet,
        # to progressively fight overfitting on the higher-capacity layers
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.6)

        self.fc1 = nn.Linear(256 * 12 * 12, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 136)

    def forward(self, x):
        x = self.drop1(self.pool(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool(F.relu(self.bn3(self.conv3(x)))))
        x = self.drop4(self.pool(F.relu(self.bn4(self.conv4(x)))))

        x = x.view(x.size(0), -1)

        x = self.drop5(F.relu(self.fc1(x)))
        x = self.drop6(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x
