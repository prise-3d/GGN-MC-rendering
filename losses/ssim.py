import torch

from skimage.metrics import structural_similarity
from ipfml.processing.transform import get_LAB_L

class SSIM(torch.nn.Module):

    def __init__(self):
        super(SSIM, self).__init__()

    def forward(self, inputs, targets):

        losses = []

        # for each element in batch compute score using Koncept Model
        for i, item in enumerate(inputs):
            img_input_array = item.detach().numpy()
            img_target_array = targets[i].detach().numpy()
            
            # get shape and then reshape using channels as 3 dimension
            c, h, w = img_input_array.shape

            img_input_array = get_LAB_L(img_input_array.reshape(h, w, c))
            img_target_array = get_LAB_L(img_target_array.reshape(h, w, c))

            # multichanel is applied on last dimension
            score = structural_similarity(img_input_array, img_target_array, data_range=img_target_array.max() - img_target_array.min())

            losses.append(score)

        loss_mean = torch.sum(torch.FloatTensor(losses)) / len(losses)
        loss_mean.requires_grad_()

        print(1. - loss_mean)

        return 1. - loss_mean