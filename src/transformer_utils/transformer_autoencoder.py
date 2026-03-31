import torch as nn
from torch.nn import Module, Linear, TransformerEncoderLayer, TransformerEncoder, MSELoss

class TransformerAutoencoder(Module):
    """
    A simple transformer-based autoencoder for tabular data. The model consists of an input projection layer, 
    a transformer encoder, and a linear decoder. This Transformer is useful if you use sequences, 
    but for tabular data, you might want to consider using a simpler architecture.
    """
    def __init__(self, n_features, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        
        self.input_proj = Linear(n_features, d_model)
        
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder = Linear(d_model, n_features)
        
    def forward(self, x):
        # x: (batch, features) → treat features as sequence length 1
        x = x.unsqueeze(1)  # (batch, seq_len=1, features)
        
        x = self.input_proj(x)          # (batch, 1, d_model)
        z = self.encoder(x)             # latent representation
        out = self.decoder(z)           # reconstruction
        
        return out.squeeze(1), z.squeeze(1)  # return reconstruction + embedding
      
def train(model, dataloader, epochs=10, lr=1e-3):
    optimizer = nn.optim.Adam(model.parameters(), lr=lr)
    criterion = MSELoss()
        
    model.train()
        
    for epoch in range(epochs):
        total_loss = 0
            
        for x, _ in dataloader:  # ignore labels
            optimizer.zero_grad()
                
            x_hat, _ = model(x)
            loss = criterion(x_hat, x)
                
            loss.backward()
            optimizer.step()
                
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: loss = {total_loss:.4f}")

def extract_embeddings(model, dataloader):
    model.eval()
    embeddings = []
    
    with nn.no_grad():
        for x, _ in dataloader:
            _, z = model(x)
            embeddings.append(z)
    
    return nn.cat(embeddings, dim=0)