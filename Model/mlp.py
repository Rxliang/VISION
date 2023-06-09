import torch
import torch.nn as nn
from mlp_mixer_pytorch import MLPMixer
torch.manual_seed(3407)
from siren_pytorch import SirenNet
from torchvision.ops import MLP

device = 'cuda'

class Siren(nn.Module):
    def __init__(self, output_size, hidden_size1, hidden_size2, num_layers,w0):
        super(MLP_basic, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size1
        self.inter_hidden = hidden_size2
        self.num_layers = num_layers
        self.w0 = w0
        
        self.net1 = SirenNet(
            dim_in = 768,                        # input dimension, ex. 2d coor
            dim_hidden = self.inter_hidden,                  # hidden dimension
            dim_out = self.hidden_size,                       # output dimension, ex. rgb value
            num_layers = self.num_layers,                    # number of layers
            final_activation = nn.Identity(),   # activation of final layer (nn.Identity() for direct output)
            w0_initial = self.w0                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
        )
        self.net2 = SirenNet(
            dim_in = self.hidden_size,                        # input dimension, ex. 2d coor
            dim_hidden = self.inter_hidden,                  # hidden dimension
            dim_out = self.output_size,                       # output dimension, ex. rgb value
            num_layers = 2,                    # number of layers
            final_activation = nn.Identity(),   # activation of final layer (nn.Identity() for direct output)
            w0_initial = self.w0                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
        )
        
    def forward(self, x):
        x = self.net1(x)
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = self.net2(x)

        return x
class MLP_basic(nn.Module):
    def __init__(self, output_size, hidden_size1):
        super(MLP_basic, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size1
        # self.inter_hidden = hidden_size2
        # self.num_layers = num_layers
        
        self.net1 = MLP(768,[512,256,128,hidden_size1])
        self.net2 = MLP(197*hidden_size1,[output_size])
        
    def forward(self, x):
        x = self.net1(x)
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = self.net2(x)

        return x
    
class MLP_model:
    def __init__(self, channels, patch_size, dim, depth, num_classes):
        self.image_size = (32,24)
        self.channels = channels
        # self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_classes = num_classes

    def init_Siren(self):

        model = MLPMixer(
            image_size = self.image_size,
            channels = self.channels,
            patch_size = self.patch_size,
            dim = self.dim,
            depth = self.depth,
            num_classes = self.num_classes
        )
        return model
        # net = SirenNet(
        #     dim_in = 24576,                        # input dimension, ex. 2d coor
        #     dim_hidden = 2048,                  # hidden dimension
        #     dim_out = 19004,                       # output dimension, ex. rgb value
        #     num_layers = 2,                    # number of layers
        #     final_activation = nn.Identity(),   # activation of final layer (nn.Identity() for direct output)
        #     w0_initial = 30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
        # ).to(device)
        # return net