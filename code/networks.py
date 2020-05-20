import torch.nn as nn
import torch
from loaders import get_mean_sigma
from layers import Conv2d, Normalization, ReLU, Flatten, Linear, Sequential


class SeqNet(nn.Module):

    def __init__(self):
        super(SeqNet, self).__init__()
        self.is_double = False
        self.skip_norm = False

    def forward(self, x, init_lambda=False):
        if isinstance(x, torch.Tensor) and self.is_double:
            x = x.to(dtype=torch.float64)
        x = self.blocks(x, init_lambda, skip_norm=self.skip_norm)
        return x

    def reset_bounds(self):
        for block in self.blocks:
            block.bounds = None

    def to_double(self):
        self.is_double = True
        for param_name, param_value in self.named_parameters():
            param_value.data = param_value.data.to(dtype=torch.float64)

    def forward_until(self, i, x):
        """ Forward until layer i (inclusive) """
        x = self.blocks.forward_until(i, x)
        return x

    def forward_from(self, i, x):
        """ Forward from layer i (exclusive) """
        x = self.blocks.forward_from(i, x)
        return x


class FFNN(SeqNet):

    def __init__(self, device, dataset, sizes, n_class=10, input_size=32, input_channel=3):
        super(FFNN, self).__init__()

        mean, sigma = get_mean_sigma(device, dataset)
        self.normalizer = Normalization(mean, sigma)

        layers = [Flatten(), Linear(input_size*input_size*input_channel, sizes[0]), ReLU(sizes[0])]
        for i in range(1, len(sizes)):
            layers += [
                Linear(sizes[i-1], sizes[i]),
                ReLU(sizes[i]),
            ]
        layers += [Linear(sizes[-1], n_class)]
        self.blocks = Sequential(*layers)


class ConvMed(SeqNet):

    def __init__(self, device, dataset, n_class=10, input_size=32, input_channel=3, width1=1, width2=1, linear_size=100):
        super(ConvMed, self).__init__()

        mean, sigma = get_mean_sigma(device, dataset)

        layers = [
            Normalization(mean, sigma),
            Conv2d(input_channel, 16*width1, 5, stride=2, padding=2, dim=input_size),
            ReLU((16*width1, input_size//2, input_size//2)),
            Conv2d(16*width1, 32*width2, 4, stride=2, padding=1, dim=input_size//2),
            ReLU((32*width2, input_size//4, input_size//4)),
            Flatten(),
            Linear(32*width2*(input_size // 4)*(input_size // 4), linear_size),
            ReLU(linear_size),
            Linear(linear_size, n_class),
        ]
        self.blocks = Sequential(*layers)


class ConvMedBig(SeqNet):

    def __init__(self, device, dataset, n_class=10, input_size=32, input_channel=3, width1=1, width2=1, width3=1, linear_size=100):
        super(ConvMedBig, self).__init__()

        mean, sigma = get_mean_sigma(device, dataset)
        self.normalizer = Normalization(mean, sigma)

        layers = [
            Normalization(mean, sigma),
            Conv2d(input_channel, 16*width1, 3, stride=1, padding=1, dim=input_size),
            ReLU((16*width1, input_size, input_size)),
            Conv2d(16*width1, 16*width2, 4, stride=2, padding=1, dim=input_size//2),
            ReLU((16*width2, input_size//2, input_size//2)),
            Conv2d(16*width2, 32*width3, 4, stride=2, padding=1, dim=input_size//2),
            ReLU((32*width3, input_size//4, input_size//4)),
            Flatten(),
            Linear(32*width3*(input_size // 4)*(input_size // 4), linear_size),
            ReLU(linear_size),
            Linear(linear_size, n_class),
        ]
        self.blocks = Sequential(*layers)


class cnn_2layer(SeqNet):
    def __init__(
        self, device, dataset, input_channel, input_size, width, linear_size
    ):
        super(cnn_2layer, self).__init__()

        mean, sigma = get_mean_sigma(device, dataset, IBP=True)
        self.normalizer = Normalization(mean, sigma)

        self.layers = [
            Normalization(mean, sigma),
            Conv2d(
                input_channel, 4 * width, 4,
                stride=2, padding=1, dim=input_size
            ),
            ReLU((4 * width, input_size//2, input_size//2)),
            Conv2d(
                4 * width, 8 * width, 4,
                stride=2, padding=1, dim=input_size//2
            ),
            ReLU((8 * width, input_size//4, input_size//4)),
            Flatten(),
            Linear(
                8 * width * (input_size // 4) * (input_size // 4), linear_size
            ),
            ReLU(linear_size),
            Linear(linear_size, 10),
        ]

    def converter(self, net):
        if isinstance(net, nn.Sequential):
            seq_model = net
        else:
            seq_model = net.module
        for idx, l in enumerate(seq_model):
            if isinstance(l, nn.Linear):
                self.layers[idx + 1].linear.weight.data.copy_(l.weight.data)
                self.layers[idx + 1].linear.bias.data.copy_(l.bias.data)
            if isinstance(l, nn.Conv2d):
                self.layers[idx + 1].conv.weight.data.copy_(l.weight.data)
                self.layers[idx + 1].conv.bias.data.copy_(l.bias.data)
        self.blocks = Sequential(*self.layers)


def model_cnn_2layer(in_ch, in_dim, width, linear_size=128):
    """
    CNN, small 2-layer (default kernel size is 4 by 4)
    Parameter:
        in_ch: input image channel, 1 for MNIST and 3 for CIFAR
        in_dim: input dimension, 28 for MNIST and 32 for CIFAR
        width: width multiplier
    """
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4 * width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4 * width, 8 * width, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8 * width * (in_dim // 4) * (in_dim // 4), linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model


class cnn_4layer(SeqNet):
    def __init__(
        self, device, dataset, input_channel, input_size, width, linear_size
    ):
        super(cnn_4layer, self).__init__()

        mean, sigma = get_mean_sigma(device, dataset, IBP=True)
        self.normalizer = Normalization(mean, sigma)

        self.layers = [
            Normalization(mean, sigma),
            Conv2d(
                input_channel, 4 * width, 3,
                stride=1, padding=1, dim=input_size
            ),
            ReLU((4 * width, input_size, input_size)),
            Conv2d(
                4 * width, 4 * width, 4,
                stride=2, padding=1, dim=input_size//2
            ),
            ReLU((4 * width, input_size//2, input_size//2)),
            Conv2d(
                4 * width, 8 * width, 3,
                stride=1, padding=1, dim=input_size//2
            ),
            ReLU((8 * width, input_size//2, input_size//2)),
            Conv2d(
                8 * width, 8 * width, 4,
                stride=2, padding=1, dim=input_size//4
            ),
            ReLU((8 * width, input_size//4, input_size//4)),
            Flatten(),
            Linear(
                8 * width * (input_size // 4) * (input_size // 4), linear_size
            ),
            ReLU(linear_size),
            Linear(linear_size, linear_size),
            ReLU(linear_size),
            Linear(linear_size, 10),
        ]

    def converter(self, net):
        if isinstance(net, nn.Sequential):
            seq_model = net
        else:
            seq_model = net.module
        for idx, l in enumerate(seq_model):
            if isinstance(l, nn.Linear):
                self.layers[idx + 1].linear.weight.data.copy_(l.weight.data)
                self.layers[idx + 1].linear.bias.data.copy_(l.bias.data)
            if isinstance(l, nn.Conv2d):
                self.layers[idx + 1].conv.weight.data.copy_(l.weight.data)
                self.layers[idx + 1].conv.bias.data.copy_(l.bias.data)
        self.blocks = Sequential(*self.layers)


def model_cnn_4layer(in_ch, in_dim, width, linear_size):
    """
    CNN, relatively large 4-layer
    Parameter:
        in_ch: input image channel, 1 for MNIST and 3 for CIFAR
        in_dim: input dimension, 28 for MNIST and 32 for CIFAR
        width: width multiplier
    """
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(4 * width, 4 * width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4 * width, 8 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8 * width, 8 * width, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8 * width * (in_dim // 4) * (in_dim // 4), linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model


class cnn_IBP_large(SeqNet):
    def __init__(
        self, device, dataset, input_channel, input_size, linear_size
    ):
        super(cnn_IBP_large, self).__init__()

        mean, sigma = get_mean_sigma(device, dataset, IBP=True)
        self.normalizer = Normalization(mean, sigma)

        self.layers = [
            Normalization(mean, sigma),
            Conv2d(
                input_channel, 64, 3,
                stride=1, padding=1, dim=input_size
            ),
            ReLU((64, input_size, input_size)),
            Conv2d(
                64, 64, 3,
                stride=1, padding=1, dim=input_size
            ),
            ReLU((64, input_size, input_size)),
            Conv2d(
                64, 128, 3,
                stride=2, padding=1, dim=input_size//2
            ),
            ReLU((128, input_size//2, input_size//2)),
            Conv2d(
                128, 128, 3,
                stride=1, padding=1, dim=input_size//2
            ),
            ReLU((128, input_size//2, input_size//2)),
            Conv2d(
                128, 128, 3,
                stride=1, padding=1, dim=input_size//2
            ),
            ReLU((128, input_size//2, input_size//2)),
            Flatten(),
            Linear(
                128 * (input_size // 2) * (input_size // 2), linear_size
            ),
            ReLU(linear_size),
            Linear(linear_size, 10),
        ]

    def converter(self, net):
        if isinstance(net, nn.Sequential):
            seq_model = net
        else:
            seq_model = net.module
        for idx, l in enumerate(seq_model):
            if isinstance(l, nn.Linear):
                self.layers[idx + 1].linear.weight.data.copy_(l.weight.data)
                self.layers[idx + 1].linear.bias.data.copy_(l.bias.data)
            if isinstance(l, nn.Conv2d):
                self.layers[idx + 1].conv.weight.data.copy_(l.weight.data)
                self.layers[idx + 1].conv.bias.data.copy_(l.bias.data)
        self.blocks = Sequential(*self.layers)


def IBP_large(in_ch, in_dim, linear_size=512):
    model = nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear((in_dim // 2) * (in_dim // 2) * 128, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model
