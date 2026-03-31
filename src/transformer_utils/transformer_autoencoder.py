import torch.nn as nn

class TransformerAutoencoder(nn.Module):
    def __init__(self, n_features, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        
        self.input_proj = nn.Linear(n_features, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder = nn.Linear(d_model, n_features)
        
    def forward(self, x):
        # x: (batch, features) → treat features as sequence length 1
        x = x.unsqueeze(1)  # (batch, seq_len=1, features)
        
        x = self.input_proj(x)          # (batch, 1, d_model)
        z = self.encoder(x)             # latent representation
        out = self.decoder(z)           # reconstruction
        
        return out.squeeze(1), z.squeeze(1)  # return reconstruction + embedding
      