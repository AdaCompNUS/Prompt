import torch
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class FCGenerator(nn.Module):
    def __init__(self, init_var=0.1, num_p=2000):
        super(FCGenerator, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               uniform(x, -init_var / 2, init_var / 2), nn.init.calculate_gain('relu'))

        self.gen = nn.Sequential(
            init_(nn.Linear(1, num_p * 3)))

        self.input_data = torch.ones((1, 1)).cuda()

    def forward(self, input=None):
        x = self.gen(self.input_data)
        x = x.view(x.size(0), -1, 3)

        return x
