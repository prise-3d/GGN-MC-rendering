import torch 
from .pytorch_ssim import SSIM

# from .lmse import LMSE
# from .ssim import SSIM
# loss list

class Loss():

    loss_choices = ['mse', 'ssim', 'bce', 'L1']

    def __init__(self, name):
        self.name = name
        self.loss = self._instanciate(name)

    def compute(self, predicted, expected):

        if self.name == 'ssim':
            return 1 - self.loss(predicted, expected)

        return self.loss(predicted, expected)

    def _instanciate(self, name):
            
        if name not in self.loss_choices:
            raise Exception('invalid loss function choice')

        if name == 'L1':
            return torch.nn.L1Loss()

        if name == 'mse':
            return torch.nn.MSELoss()

        if name == 'ssim':
            return SSIM()

        if name == 'bce':
            return torch.nn.BCELoss()