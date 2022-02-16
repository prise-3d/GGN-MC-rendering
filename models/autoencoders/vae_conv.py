import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import os
import random


class VAE(nn.Module,):
    def __init__(self, input_chanels, input_size, output_channels, output_size, nz):
        super(VAE, self).__init__()

        #self.have_cuda = False
        self.nz = nz
        self.n_channels = input_chanels

        self.encoder = nn.Sequential(
            # input is (input_chanels) x input_size x input_size
            nn.Conv2d(input_chanels, input_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(input_size, input_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(input_size * 2),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(input_size * 2, input_size * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_size * 4),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(input_size * 4, 1024, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=False),
            # nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(1024, output_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(output_size * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(output_size * 8, output_size * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_size * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(output_size * 4, output_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_size * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(output_size * 2, output_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_size),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(output_size, output_channels, 4, 2, 4, bias=False),
            # nn.Tanh()
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

        self.fc1 = nn.Linear(1024 * self.n_channels * self.n_channels, 512)
        self.fc21 = nn.Linear(512, nz)
        self.fc22 = nn.Linear(512, nz)

        self.fc3 = nn.Linear(nz, 512)
        self.fc4 = nn.Linear(512, 1024)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        conv = self.encoder(x)
        # print("encode conv", conv.size())
        h1 = self.fc1(conv.view(-1, 1024 * self.n_channels * self.n_channels))
        # print("encode h1", h1.size())
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        # print("deconv_input", deconv_input.size())
        deconv_input = deconv_input.view(-1, 1024, 1, 1)
        # print("deconv_input", deconv_input.size())
        return self.decoder(deconv_input)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        # print("x", x.size())
        mu, logvar = self.encode(x)
        # print("mu, logvar", mu.size(), logvar.size())
        z = self.reparametrize(mu, logvar)
        # print("z", z.size())
        decoded = self.decode(z)
        # print("decoded", decoded.size())
        return decoded, mu, logvar

    def params(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())