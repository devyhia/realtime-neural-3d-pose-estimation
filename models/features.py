import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from helpers.flatten import Flatten

class Features(nn.Module):
    def __init__(self):
        super(Features, self).__init__()

        # num_classes = 5
        # input_dim = 64
        channels = 3
        hidden_size = 256
        descriptor_size = 16

        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, 8),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(16, 7, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(12*12*7, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, descriptor_size)
        )

        self.weight_init()
    
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.fill_(1)
                m.weight.data.normal_(0, 0.005)

    def forward(self, anchors, pullers, pushers):
        """Generate the features using the forward pass
        
        Arguments:
            anchors {array} -- [batch_size, width, height, channels]
            pullers {array} -- [batch_size, width, height, channels]
            pushers {array} -- [batch_size, width, height, channels]
        
        Returns:
            tuple -- A tuple of features: (
                [batch_size, number_of_features],
                [batch_size, number_of_features],
                [batch_size, number_of_features]
            )
        """

        x = self.features(anchors)
        y = self.features(pullers)
        z = self.features(pushers)

        return x, y, z