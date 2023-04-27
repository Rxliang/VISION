import torch
from torch import nn

class Transformer_predictor(nn.Module):
    def __init__(self, layer_scale=[8, 8, 4], output_dim=19004, nhead=4, num_layers=8, feature_dim=768):
        super(Transformer_predictor, self).__init__()

        self.input_dim = 32 * feature_dim

        self.input_net = nn.ModuleList()
        temp_dim = self.input_dim
        for num in layer_scale:
            self.input_net.append(nn.Linear(temp_dim, temp_dim // num))
            temp_dim =  temp_dim // num


        self.encoder_layer = nn.TransformerEncoderLayer(d_model=temp_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.output_net = nn.ModuleList() 
        for index in range(len(self.input_net) - 1, -1, -1):
            self.output_net.append(torch.nn.Linear(self.input_net[index].out_features, self.input_net[index].in_features))
        
        self.output_net[-1] = torch.nn.Linear(self.output_net[-1].in_features, output_dim)

    def forward(self, src):
        src = src.view(-1, src.shape[1] * src.shape[2])

        for net in self.input_net:
            src = net(src)
            
        src = self.transformer_encoder(src)

        for net in self.output_net:
            src = net(src)

        return src
