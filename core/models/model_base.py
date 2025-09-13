""" Architectures of the neural network """

import torch
import torch.nn as nn

from core.utils.time_exec_utils import log_execution_time


class ModelBase(nn.Module):
    def __init__(self):
        super(ModelBase, self).__init__()
        self.left_path = nn.Sequential(
            nn.Linear(48, 2000), nn.ReLU(),
            nn.Linear(2000, 2000), nn.ReLU(),
            nn.Linear(2000, 2000), nn.ReLU(),
            nn.Linear(2000, 2000)
        )
        self.right_path = nn.Sequential(
            nn.Linear(48, 2000), nn.ReLU(),
            nn.Linear(2000, 2000), nn.ReLU(),
            nn.Linear(2000, 2000), nn.ReLU(),
            nn.Linear(2000, 2000)
        )
        self.classifier = nn.Sequential(
            nn.Linear(4000, 2),
            nn.LogSoftmax(dim=1)
        )

    @log_execution_time
    def forward(self, x):
        left = x[:, :48]
        right = x[:, 48:]
        l_out = self.left_path(left)
        r_out = self.right_path(right)
        combined = torch.cat([l_out, r_out], dim=1)
        return self.classifier(combined)
