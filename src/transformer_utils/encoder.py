import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (batch, features) → (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        
        z = self.encoder(x)  # (batch, 1, d_model)
        return z.squeeze(1)  # embedding