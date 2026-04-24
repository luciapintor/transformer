import torch
import torch.nn as nn
import torch.nn.functional as F

class MatrixAutoencoder(nn.Module):
    """
    A simple autoencoder for tabular data. The model consists of an encoder and a decoder, 
    both implemented as feedforward neural networks.
    The encoder maps the input features to a lower-dimensional latent space (emb_size), 
    while the decoder reconstructs the original features from the latent representation.
    """
    def __init__(self, n_features, emb_size=64, hidden_dim=128):
        super().__init__()
        
        # Store the input parameters as instance variables for later use
        self.n_features = n_features
        self.emb_size = emb_size
        self.hidden_dim = hidden_dim
        
        # Encoder: maps input features to latent space
        # This encoder is a sequential container that allows us to stack layers together.
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim), # first layer maps input features to hidden dimension
            nn.ReLU(),                         # activation function to introduce non-linearity
            nn.Linear(hidden_dim, emb_size),     # second layer maps hidden dimension to latent space (emb_size)
        )
        
        # Decoder: reconstructs input from latent space
        # This decoder is also a sequential container that maps the latent representation back 
        # to the original feature space.
        self.decoder = nn.Sequential(
            nn.Linear(emb_size, hidden_dim),     # first layer maps latent space back to hidden dimension
            nn.ReLU(),                          # activation function to introduce non-linearity
            nn.Linear(hidden_dim, n_features)   # second layer maps hidden dimension back to original space
        )

    def forward(self, x):
        """
        This method defines the forward pass of the autoencoder. 
        It takes an input tensor x, passes it through the encoder to get a latent representation z,
        and then decodes z back to the original feature space to get the reconstruction x_hat.
        This method is called when we pass an input through the model (e.g., model(x) or self(x)).
        """
        
        z = self.encoder(x)        # (batch, emb_size)
        x_hat = self.decoder(z)    # (batch, n_features)
        
        return x_hat, z
    
    def fit(self, dataloader, epochs=10, lr=1e-3, device=None):
        """
        This method trains the autoencoder in an unsupervised way,
        since we want to extract embeddings without using the labels.
        It uses a loss function to measure the difference between the input 
        and the reconstruction.
        """
        
        # Move the model to the specified device (CPU or GPU)
        if device is None:
            device = next(self.parameters()).device
        self.to(device)
        
        # The optimizer is responsible for updating the model parameters based on the computed gradients.
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # his training loop iterates over epochs and batches of data
        for epoch in range(epochs):
            
            # Set the model to training mode
            self.train()
            # Initialize a variable to accumulate the total loss for the epoch
            total_loss = 0.0
            
            for batch in dataloader:
                
                x = batch[0]    #takes only the data 
                x = x.to(device).float()
                
                y = batch[1]    # takes labels 
                
                # Zero the gradients (reset the gradients before training this epoch)
                optimizer.zero_grad()
                
                """
                once the gradient is reset in the epoch, the training loop consists of the following steps:
                1. Forward Pass (feeding the model)
                2. Loss Calculation (computing the loss based on the model's output and the original input)
                3. Backward Pass (how much each parameter contributed to the error)
                4. Weight Update (updating the model parameters based on the computed gradients)
                """
                
                # Forward Pass: compute the model output and the latent representation
                x_hat, z = self(x)                
                    
                # Loss Calculation: compute the reconstruction loss and the contrastive loss
                rec_loss = reconstruction_loss_calc(x, x_hat)   # compute the reconstruction loss
                contr_loss = contrastive_loss_calc(z, y)        # compute the contrastive loss
                loss = rec_loss + contr_loss                    # total loss is the sum of reconstruction and contrastive losses
                
                # Backward pass: cumulate the gradients for all model parameters based on the computed loss
                loss.backward() 
                
                # Weight Update: update the model parameters based on the computed gradients
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}: loss = {avg_loss:.6f}")
            
    def encode(self, x):
        """
        This function encodes the input data into the latent space using the trained encoder.
        It sets the model to evaluation mode and passes the input through the encoder to get the latent
        representation z, which is then returned.
        """
        self.eval()
        with torch.no_grad():
            z = self.encoder(x.float())
        return z
 
    def encode_dataloader(self, dataloader, device=None):
        """
        This method extracts embeddings from the test set using the trained model.
         It sets the model to evaluation mode and iterates through the dataloader,
        passing the input through the model to get the latent representation z,
        which is then collected in a list and returned as a single tensor.
        """
        
        # Move the model to the specified device (CPU or GPU)
        if device is None:
            device = next(self.parameters()).device
        
        # Set the model to evaluation mode and iterate through the dataloader,   
        # passing the input through the model to get the latent representation z,   
        # which is then collected in a list and returned as a single tensor.
        self.eval()
        embeddings = []
        
        # Use torch.no_grad() to disable gradient computation, 
        # since we are only interested in the embeddings and not in updating the model parameters.
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0] 
                x = x.to(device).float()
                
                z = self.encoder(x)
                embeddings.append(z)
        
        return torch.cat(embeddings, dim=0)
    
def contrastive_loss_calc(z, y):
    """
    This function computes a simple contrastive loss considering 
    other samples with the same label as modified versions of the same sample.
    The temperature parameter is used to control the sharpness of the distribution of similarities 
    (lower values will result in a sharper distribution).
    The loss encourages the model to maximize the similarity between samples with the same label
    """
    # Normalize the embeddings to have unit length, which is important for computing similarity.
    # The normalization is done along the feature dimension (dim=1), so that each embedding vector has a magnitude of 1.
    z = F.normalize(z, dim=1)

    # Compute the similarity matrix between all pairs of embeddings in the batch,
    # scaled by the temperature parameter to control the sharpness of the distribution.
    sim = torch.matmul(z, z.T) 

    # Create a mask to identify which samples belong to the same class (label).
    y = y.view(-1, 1)
    pos_mask = (y == y.T).float().to(z.device)

    # remove self-comparisons from the similarity matrix by creating an identity matrix (self_mask) 
    # which has 1s on the diagonal (indicating self-comparisons) and 0s elsewhere.
    self_mask = torch.eye(z.size(0), device=z.device)
    
    # the positive mask will have 1s for pairs of samples with the same label (positive pairs) and 
    # 0s for pairs with different labels (negative pairs)
    pos_mask = pos_mask - self_mask
    
    # the negative mask will have 1s for pairs with different labels (negative pairs) and 
    # 0s for pairs with the same label (positive pairs)
    neg_mask = 1.0 - pos_mask - self_mask
    
    # positive similarity
    pos_sum = (sim * pos_mask).sum(dim=1)           # sum of similarities for positive pairs for each sample
    pos_count = pos_mask.sum(dim=1).clamp(min=1.0)  # count of positive pairs for each sample, clamped to avoid division by zero
    pos_mean = pos_sum / pos_count                  # mean similarity for positive pairs for each sample
    
    # negative similarity
    neg_sum = (sim * neg_mask).sum(dim=1)           # sum of similarities for negative pairs for each sample
    neg_count = neg_mask.sum(dim=1).clamp(min=1.0)  # count of negative pairs for each sample, clamped to avoid division by zero
    neg_mean = neg_sum / neg_count                  # mean similarity for negative pairs for each sample
    
    # Loss: increase similarity for positive pairs and decrease it for negative pairs
    loss = -(pos_mean - neg_mean)
    
    return loss.mean()

def reconstruction_loss_calc(x, x_hat):
    """
    This function computes the mean squared error (MSE) loss between the input x and the reconstruction x_hat.
    The MSE loss measures the average squared difference between the original input and its reconstruction, 
    encouraging the model to produce reconstructions that are as close as possible to the original inputs.
    """
    return F.mse_loss(x_hat, x)