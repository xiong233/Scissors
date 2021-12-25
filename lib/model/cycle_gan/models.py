#encoding=utf-8
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import pdb
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha = 0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output = grad_outputs.neg() * ctx.alpha
        return output


def grad_reverse(x):
    return GRLayer.apply(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, 1, stride=1, padding=0),
            nn.ReLU(False),
        )

    def forward(self, x):
        x=self.conv(x)#1*64*2*3
        return x


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        self.sigmoid = nn.Sigmoid()

        model = [   nn.Conv2d(input_nc, 64, kernel_size=5, stride=1, padding=2),
                            nn.LeakyReLU(0.2, inplace=True)]

        model += [  nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                            nn.InstanceNorm2d(128),
                            nn.LeakyReLU(0.2, inplace=True)]

        model += [  nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                            nn.InstanceNorm2d(256),
                            nn.LeakyReLU(0.2, inplace=True)]

        model += [  nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0),
                            nn.InstanceNorm2d(512),
                            nn.LeakyReLU(0.2, inplace=True)]

        model += [  nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = grad_reverse(x)
        x = self.sigmoid(self.model(x)).squeeze(0).squeeze(0)
        return x
