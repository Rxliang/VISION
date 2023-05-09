import torch
import torch.nn as nn
from mlp_mixer_pytorch import MLPMixer
torch.manual_seed(3407)
from siren_pytorch import SirenNet


device = 'cuda'

class MLP_model:
    def __init__(self, channels, patch_size, dim, depth, num_classes):
        self.image_size = (32,24)
        self.channels = channels
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_classes = num_classes

    def init_MLP_Mixer(self):

        # model = MLPMixer(
        #     image_size = self.image_size,
        #     channels = self.channels,
        #     patch_size = self.patch_size,
        #     dim = self.dim,
        #     depth = self.depth,
        #     num_classes = self.num_classes
        # )
        # return model
        net = SirenNet(
            dim_in = 768,                        # input dimension, ex. 2d coor
            dim_hidden = 320,                  # hidden dimension
            dim_out = 19004,                       # output dimension, ex. rgb value
            num_layers = 2,                    # number of layers
            final_activation = nn.Identity(),   # activation of final layer (nn.Identity() for direct output)
            w0_initial = 30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
        ).to(device)
        return net