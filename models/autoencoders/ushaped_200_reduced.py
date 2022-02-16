import torch
from torch import nn, optim
from torchvision import transforms, datasets


class UShapedAutoencoder(torch.nn.Module):

    # no use of features_maps currently (set defautlt with K)
    def __init__(self, input_chanels, output_channels, image_size):
        super(UShapedAutoencoder, self).__init__()

        
        self.K = 16

        # Input depth:

        # Output depth:
        # Intensity RGB
        # 3 channels

        # In 3 channels
        self.encoder0 = nn.Sequential(
            nn.Conv2d(input_chanels, self.K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.K, self.K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        # Out 100x100

        # In 100x100
        self.encoder1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(self.K, 2*self.K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(2*self.K),
            nn.Conv2d(2*self.K, 2*self.K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        # Out 50x50

        # In 50x50
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(2*self.K, 4*self.K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(4*self.K),
            nn.Conv2d(4*self.K, 4*self.K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        # Out 25x25

        # In 25x25 -> 12x12
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(4*self.K, 8*self.K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(8*self.K),
            nn.Conv2d(8*self.K, 8*self.K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        # In 12x12 -> 25x25
        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(8*self.K, 16*self.K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16*self.K),
            nn.Conv2d(16*self.K, 8*self.K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.ZeroPad2d((1, 0, 1, 0)),
        )

        # In 25x25 -> 50x50
        self.decoder0 = nn.Sequential(
            nn.ConvTranspose2d(16*self.K, 8*self.K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(8*self.K),
            nn.ConvTranspose2d(8*self.K, 4*self.K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        # In 50x50 -> 100x100
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(8*self.K, 4*self.K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(4*self.K),
            nn.ConvTranspose2d(4*self.K, 2*self.K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        # In 100x100 -> 200x200
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(4*self.K, 2*self.K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(2*self.K),
            nn.ConvTranspose2d(2*self.K, self.K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        # In 200x200 -> 200x200 (with specific number of channels)
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(2*self.K, self.K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(self.K, self.K, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.K, output_channels, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def params(self):
        return list(self.encoder0.parameters()) + \
               list(self.encoder1.parameters()) + \
               list(self.encoder2.parameters()) + \
               list(self.encoder3.parameters()) + \
               list(self.encoder4.parameters()) + \
               list(self.decoder0.parameters()) + \
               list(self.decoder1.parameters()) + \
               list(self.decoder2.parameters()) + \
               list(self.decoder3.parameters())

    def forward(self, x):
        x0 = self.encoder0(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
    
        x5 = self.decoder0(torch.cat((x4, x3), 1))
        x6 = self.decoder1(torch.cat((x5, x2), 1))
        x7 = self.decoder2(torch.cat((x6, x1), 1))
        x8 = self.decoder3(torch.cat((x7, x0), 1))

        return x8
