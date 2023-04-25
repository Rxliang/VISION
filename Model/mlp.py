import torch
import torch.nn as nn
torch.manual_seed(3407)

class MLP_basic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(MLP_basic, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.layers = []
        self.layers.append(nn.Linear(input_size * 32, hidden_size))

        for i in range(num_layers-1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        self.layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.ModuleList(self.layers)
        
    def forward(self, x):
        x = x.view(-1, src.shape[1] * src.shape[2])

        for i in range(self.num_layers):
            x = torch.relu(self.layers[i](x))

        return x