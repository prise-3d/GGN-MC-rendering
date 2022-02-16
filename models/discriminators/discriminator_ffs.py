import torch

class Discriminator(torch.nn.Module):
  def __init__(self, input_size, sequence):
    super(Discriminator, self).__init__()
    self.input_size = input_size
    self.sequence = sequence 
    self.main = torch.nn.Sequential(torch.nn.Flatten(),
                                    torch.nn.Linear(input_size * sequence, input_size * 16),
                                    torch.nn.BatchNorm1d(input_size * 16),
                                    torch.nn.LeakyReLU(0.1, inplace=False),
                                    torch.nn.Dropout(0.4),

                                    torch.nn.Linear(input_size * 16, input_size * 8),
                                    torch.nn.BatchNorm1d(input_size * 8),
                                    torch.nn.LeakyReLU(0.1, inplace=False),
                                    torch.nn.Dropout(0.4),

                                    torch.nn.Linear(input_size * 8, input_size * 4),
                                    torch.nn.BatchNorm1d(input_size * 4),
                                    torch.nn.LeakyReLU(0.1, inplace=False),
                                    torch.nn.Dropout(0.4),

                                    torch.nn.Linear(input_size * 4, input_size * 2),
                                    torch.nn.BatchNorm1d(input_size * 2),
                                    torch.nn.LeakyReLU(0.1, inplace=False),
                                    torch.nn.Dropout(0.4),

                                    torch.nn.Linear(input_size * 2, 1),
                                    # torch.nn.LeakyReLU(0.2, inplace=False),
                                    # torch.nn.Dropout(0.5),
                                    torch.nn.Sigmoid())

  def forward(self, input_image):
    conv_out = self.main(input_image)
    return conv_out.view(-1, 1).squeeze(dim=1) # squeeze remove all 1 dim