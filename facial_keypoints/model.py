"""CNN architecture for facial keypoint regression.

The layer shapes here are load-bearing: they must match the trained
checkpoint in ``models/keypoints_model_1_fp16.pt`` exactly, or
``load_state_dict`` will fail. Do not change layer names or dimensions
without retraining.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

INPUT_SIZE = 224
NUM_KEYPOINTS = 68
OUTPUT_DIM = NUM_KEYPOINTS * 2


class Net(nn.Module):
    """4-conv-block CNN that regresses 68 (x, y) facial keypoints.

    Input:  (N, 1, 224, 224) grayscale, normalized to roughly [-1, 1]
    Output: (N, 136) flattened (x, y) keypoint pairs
    """

    def __init__(self) -> None:
        super().__init__()

        # input: 1x224x224
        self.conv1 = nn.Conv2d(1, 32, 5)     # -> 32x220x220 -> pool -> 32x110x110
        self.conv2 = nn.Conv2d(32, 64, 3)    # -> 64x108x108 -> pool -> 64x54x54
        self.conv3 = nn.Conv2d(64, 128, 3)   # -> 128x52x52  -> pool -> 128x26x26
        self.conv4 = nn.Conv2d(128, 256, 3)  # -> 256x24x24  -> pool -> 256x12x12

        self.pool = nn.MaxPool2d(2, 2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        # dropout probability increases with depth, as in NaimishNet, since the
        # deeper/wider layers carry the most capacity and overfit first
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.6)

        self.fc1 = nn.Linear(256 * 12 * 12, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, OUTPUT_DIM)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop1(self.pool(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool(F.relu(self.bn3(self.conv3(x)))))
        x = self.drop4(self.pool(F.relu(self.bn4(self.conv4(x)))))

        x = x.view(x.size(0), -1)

        x = self.drop5(F.relu(self.fc1(x)))
        x = self.drop6(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x
