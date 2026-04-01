import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
def create_pairs(batch):
    """
    This function searches for couples of samples with the same MAC addresses and creates positive pairs.
    It only creates one positive pair per sample. According to contrastive learning, 
    positive pairs are the same samples but with different augmentations (e.g., different views of the same image), 
    in our case, we consider samples with the same MAC address as positive pairs,
    and samples with different MAC addresses as negative pairs.
    """
    xs, macs = batch
    x1, x2 = [], []
    
    for i in range(len(xs)):
        # positive: same MAC if possible
        for j in range(i+1, len(xs)):
            if macs[i] == macs[j]:
                x1.append(xs[i])
                x2.append(xs[j])
                break
    
    if len(x1) == 0:
        return x1, x2
    
    return torch.stack(x1), torch.stack(x2)


def contrastive_loss(z1, z2, temperature=0.1):
    """
    Compute the NT-Xent loss for contrastive learning. Assumes z1 and z2 are normalized embeddings of shape (batch, d_model).
    The loss encourages z1[i] to be close to z2[i] (positive pair) and far from z2[j] for j != i (negative pairs).
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    
    return F.cross_entropy(logits, labels)
    
def train_contrastive(model, dataloader, epochs=10):
    """
    Train the encoder using contrastive learning. 
    For each batch, create positive pairs through the MAC addresses and compute the contrastive loss.
    """
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        
    for epoch in range(epochs):
        # in each epoch, go through the dataloader and create pairs of samples with the same mac address
        total_loss = 0
        
        for xs, labels, macs in dataloader:
            # for each batch, we create pairs of samples with the same mac address, 
            # and compute the contrastive loss, which encourages the model to learn similar 
            # embeddings for samples with the same mac address, and different embeddings for 
            # samples with different mac addresses.    
            x1, x2 = create_pairs((xs, macs))
                
            if len(x1) == 0:
                continue
            
            # compute embeddings for the pairs    
            z1 = model(x1)
            z2 = model(x2)
            
            # compute the contrastive loss for the pairs that, according to contrastive learning, 
            # should have similar embeddings (positive pairs) and different embeddings (negative pairs).
            loss = contrastive_loss(z1, z2)
                
            opt.zero_grad()
            loss.backward()
            opt.step()
                
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: loss = {total_loss:.4f}")


def extract_embeddings(model, dataloader):
    """
    Extract embeddings from the trained encoder for all samples in the dataloader. 
    Returns a tensor of shape (num_samples, d_model).
    """
    model.eval()
    embeddings = []
        
    with torch.no_grad():
        for x, labels, macs in dataloader:
            z = model(x)
            embeddings.append(z)
    
    return torch.cat(embeddings)