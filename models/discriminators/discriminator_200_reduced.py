import torch

class Discriminator(torch.nn.Module):
  def __init__(self, n_channels, input_size):
    super(Discriminator, self).__init__()
    self.input_size = input_size
    self.n_channels = n_channels
    self.K = 64
    self.main = torch.nn.Sequential(torch.nn.Conv2d(in_channels=n_channels, out_channels=self.K, kernel_size=3, stride=3, padding=2),
                                    torch.nn.BatchNorm2d(self.K),
                                    torch.nn.LeakyReLU(0.2, inplace=False),
                                    torch.nn.MaxPool2d(3, stride=2, padding=1),

                                    torch.nn.Conv2d(in_channels=self.K, out_channels=self.K * 2, kernel_size=3, stride=3, padding=2),
                                    torch.nn.BatchNorm2d(self.K * 2),
                                    torch.nn.LeakyReLU(0.2, inplace=False),
                                    torch.nn.MaxPool2d(3, stride=2, padding=1),

                                    torch.nn.Conv2d(in_channels=self.K * 2, out_channels=self.K * 4, kernel_size=3, stride=3, padding=2),
                                    torch.nn.BatchNorm2d(self.K * 4),
                                    torch.nn.LeakyReLU(0.2, inplace=False),
                                    torch.nn.MaxPool2d(3, stride=2, padding=1),

                                    # torch.nn.Conv2d(in_channels=input_size * 8, out_channels=input_size * 16, kernel_size=3, stride=3, padding=1),
                                    # torch.nn.BatchNorm2d(input_size * 16),
                                    # torch.nn.LeakyReLU(0.2, inplace=False),

                                    torch.nn.Flatten(),
                                    torch.nn.Linear(self.K * 4 * 2 * 2, self.K * 2),
                                    torch.nn.BatchNorm1d(self.K * 2),
                                    torch.nn.LeakyReLU(0.2, inplace=False),
                                    torch.nn.Dropout(0.5),

                                    # torch.nn.Linear(self.K * 4, self.K * 2),
                                    # torch.nn.BatchNorm1d(self.K * 2),
                                    # torch.nn.LeakyReLU(0.2, inplace=False),
                                    # torch.nn.Dropout(0.5),

                                    torch.nn.Linear(self.K * 2, self.K),
                                    torch.nn.BatchNorm1d(self.K),
                                    torch.nn.LeakyReLU(0.2, inplace=False),
                                    torch.nn.Dropout(0.5),

                                    torch.nn.Linear(self.K, 1),
                                    torch.nn.LeakyReLU(0.2, inplace=False),
                                    torch.nn.Dropout(0.5),
                                    
                                    # torch.nn.Dropout(0.5),
                                    torch.nn.Sigmoid())

  def forward(self, input_image):
    conv_out = self.main(input_image)
    return conv_out.view(-1, 1).squeeze(dim=1) # squeeze remove all 1 dim