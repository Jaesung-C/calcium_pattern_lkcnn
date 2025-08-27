import torch
import torch.nn as nn
import numpy as np

class Conv1DModel(nn.Module):
    def __init__(self, kernel_size, input_shape, num_classes):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=kernel_size, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()

        self._feature_size = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(self._feature_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _get_conv_output(self, input_shape):
        dummy_input = torch.zeros(1, 1, input_shape)
        output = self.conv1(dummy_input)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.pool(output)
        return int(np.prod(output.size()))
