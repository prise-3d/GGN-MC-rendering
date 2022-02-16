import torch

class Discriminator(torch.nn.Module):
  def __init__(self, n_channels, input_size):
    super(Discriminator, self).__init__()
    self.input_size = input_size
    self.n_channels = n_channels
    self.main = torch.nn.Sequential(torch.nn.Conv2d(in_channels=n_channels, out_channels=input_size, kernel_size=3, stride=1, padding=1),
                                    torch.nn.BatchNorm2d(input_size),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(0.3),
                                    torch.nn.Conv2d(in_channels=input_size, out_channels=input_size*2, kernel_size=4, stride=2, padding=1),
                                    torch.nn.BatchNorm2d(input_size * 2),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(0.3),
                                    torch.nn.Conv2d(in_channels=input_size * 2, out_channels=input_size * 4, kernel_size=4, stride=2, padding=1),
                                    torch.nn.BatchNorm2d(input_size * 4),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(0.3),
                                    torch.nn.Conv2d(in_channels=input_size * 4, out_channels=input_size * 8, kernel_size=4, stride=2, padding=1),
                                    torch.nn.BatchNorm2d(input_size * 8),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(0.3),
                                    torch.nn.Conv2d(in_channels=input_size * 8, out_channels=input_size * 16, kernel_size=4, stride=1, padding=1),
                                    torch.nn.BatchNorm2d(input_size * 16),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(0.3),
                                    torch.nn.Conv2d(in_channels=input_size * 16, out_channels=1, kernel_size=4, stride=2, padding=0),
                                    torch.nn.Sigmoid())

  def forward(self, input_image):
    conv_out = self.main(input_image)
    return conv_out.view(-1, 1).squeeze(dim=1) # squeeze remove all 1 dim