import torch

class ExtractLastCell(torch.nn.Module):
  def forward(self,x):
      out, _ = x
      return out[:,-1,:]



class Discriminator(torch.nn.Module):
  def __init__(self, input_size, output_size, layers, sequence):
    super(Discriminator, self).__init__()
    self.input_size = input_size
  
    self.output_size = output_size
    self.sequence = sequence
    self.layers = layers
    self.main = torch.nn.Sequential(torch.nn.LSTM(input_size=self.input_size, hidden_size=output_size, dropout=0.5, num_layers=layers),
                                    ExtractLastCell(),
                                    # torch.nn.Conv2d(in_channels=n_channels, out_channels=input_size, kernel_size=3, stride=1, padding=1),
                                    # torch.nn.BatchNorm2d(input_size),
                                    # torch.nn.LeakyReLU(0.2, inplace=False),
                                    # # torch.nn.MaxPool2d(3, stride=2, padding=1),

                                    # torch.nn.Conv2d(in_channels=input_size, out_channels=input_size*2, kernel_size=3, stride=2, padding=1),
                                    # torch.nn.BatchNorm2d(input_size * 2),
                                    # torch.nn.LeakyReLU(0.2, inplace=False),
                                    # # torch.nn.MaxPool2d(3, stride=2, padding=1),

                                    # torch.nn.Conv2d(in_channels=input_size * 2, out_channels=input_size * 4, kernel_size=3, stride=2, padding=1),
                                    # torch.nn.BatchNorm2d(input_size * 4),
                                    # torch.nn.LeakyReLU(0.2, inplace=False),
                                    # # torch.nn.MaxPool2d(3, stride=2, padding=1),

                                    # torch.nn.Conv2d(in_channels=input_size * 4, out_channels=input_size * 8, kernel_size=3, stride=2, padding=1),
                                    # torch.nn.BatchNorm2d(input_size * 8),
                                    # torch.nn.LeakyReLU(0.2, inplace=False),
                                    # # torch.nn.MaxPool2d(3, stride=2, padding=1),

                                    # torch.nn.Conv2d(in_channels=input_size * 8, out_channels=input_size * 16, kernel_size=3, stride=2, padding=1),
                                    # torch.nn.BatchNorm2d(input_size * 16),
                                    # torch.nn.LeakyReLU(0.2, inplace=False),

                                    torch.nn.Flatten(),
                                    torch.nn.Linear(output_size, 256),
                                    torch.nn.BatchNorm1d(256),
                                    torch.nn.LeakyReLU(0.1, inplace=False),
                                    torch.nn.Dropout(0.5),

                                    torch.nn.Linear(256, 32),
                                    torch.nn.BatchNorm1d(32),
                                    torch.nn.LeakyReLU(0.1, inplace=False),
                                    torch.nn.Dropout(0.5),

                                    # torch.nn.Linear(input_size * 2, input_size),
                                    # torch.nn.BatchNorm1d(input_size),
                                    # torch.nn.LeakyReLU(0.2, inplace=False),
                                    # torch.nn.Dropout(0.5),

                                    torch.nn.Linear(32, 1),
                                    torch.nn.LeakyReLU(0.1, inplace=False),
                                    torch.nn.Dropout(0.5),
                                    torch.nn.Sigmoid())

  def forward(self, input_image):
    conv_out = self.main(input_image)
    return conv_out.view(-1, 1).squeeze(dim=1) # squeeze remove all 1 dim