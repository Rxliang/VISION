import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNTransformer(nn.Module):
    def __init__(self, feature_dim, output_dim, num_cnn_layers, num_transformer_layers,
     num_heads, hidden_dim, dropout_prob):
        super(CNNTransformer, self).__init__()

        # Define CNN layers
        self.cnn_layers = nn.ModuleList()
        in_channels = feature_dim
        for i in range(num_cnn_layers):
            out_channels = 2 * in_channels if i == 0 else in_channels // 2
            self.cnn_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            self.cnn_layers.append(nn.ReLU())
            self.cnn_layers.append(nn.BatchNorm1d(out_channels))
            self.cnn_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_channels = out_channels

        # Define Transformer layers
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=in_channels, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout_prob)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers) 

        # Define output layer
        self.output_layer = nn.Linear(in_channels, output_dim)

    def forward(self, x):
        # Pass input through CNN layers
        for layer in self.cnn_layers:
            x = layer(x)

        # Reshape to fit Transformer input shape
        x = x.permute(2, 0, 1)

        # Pass input through Transformer layer
        x = self.transformer(x)

        # Reshape back to fit output shape
        x = x.permute(1, 2, 0)

        # Pass through output layer
        x = self.output_layer(x[:, :, -1])

        return x
