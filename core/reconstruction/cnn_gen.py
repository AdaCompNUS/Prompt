import torch
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNGenerator(nn.Module):
    def __init__(self, img_c=3, num_p=4000):
        super(CNNGenerator, self).__init__()
        self.img_c = img_c
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               normal(x, 0, 0.005), nn.init.calculate_gain('relu'))

        self.img_bn = nn.BatchNorm2d(num_features=img_c)

        self.cnn0 = nn.Sequential(
            init_(nn.Conv2d(img_c, 16, 3, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(16, 16, 3, stride=1)), nn.ReLU())

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(17, 32, 3, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, 3, stride=1)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, 3, stride=1)), nn.ReLU(),
            # x1
            init_(nn.Conv2d(32, 64, 3, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            # x2
            init_(nn.Conv2d(64, 128, 3, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(128, 128, 3, stride=1)), nn.ReLU(),
            init_(nn.Conv2d(128, 128, 3, stride=1)), nn.ReLU(),
            # x3
            init_(nn.Conv2d(128, 256, 3, stride=1)), nn.ReLU(),
            Flatten(),
            init_(nn.Linear(1280, 512)), nn.ReLU())

        self.lstm = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)
        self.gen = nn.Sequential(init_(nn.Linear(512, 1024)), nn.ReLU(),
                                 init_(nn.Linear(1024, num_p * 3)))

    def forward(self, inputs):
        inputs = inputs.permute(0, 3, 1, 2)
        inputs = self.img_bn(inputs)

        self.lstm.flatten_parameters()

        frame = self.cnn0(inputs)
        ran_frame = torch.rand((frame.size(0), 1, 93, 125)).cuda()
        frame = torch.cat((frame, ran_frame), 1)
        frame = self.cnn(frame).view((1, inputs.shape[0], 512))

        x, last_hidden = self.lstm(frame)

        latent = last_hidden[0].view((1, 512))
        x = self.gen(latent)
        x = x.view(x.size(0), -1, 3)

        return x