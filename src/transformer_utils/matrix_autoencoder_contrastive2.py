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

        # Training loop: iterate over epochs and batches of data
        for epoch in range(epochs):
            
            # Set the model to training mode
            self.train()
            # Initialize a variable to accumulate the total loss for the epoch
            total_loss = 0.0
            
            for batch in dataloader:
                # The dataloader provides batches of data, which can be in different formats 
                # (e.g., list, tuple, or tensor).
                
                x = batch[0]    #takes only the data 
                x = x.to(device).float()
                
                y = batch[1]    # takes labels 
                
                # Zero the gradients (reset the gradients of all model parameters to zero)
                optimizer.zero_grad()
                
                # Forward pass: compute the model output and the latent representation
                x_hat, z = self(x)                
                    
                # compute contrastive loss between embeddings of samples with the same label, 
                # treating them as modified versions of the same sample
                loss = contrastive_loss_calc(z, y)
                
                # Backward pass: compute gradients and update model parameters
                loss.backward()
                # Update the model parameters based on the computed gradients
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
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device).float()
                
                z = self.encoder(x)
                embeddings.append(z)
        
        return torch.cat(embeddings, dim=0)
    
def contrastive_loss_calc(z, y, temperature=0.05):
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
    sim = torch.matmul(z, z.T) / temperature  # (B, B)

    # Create a mask to identify which samples belong to the same class (label).
    y = y.view(-1, 1)
    mask = (y == y.T).float()

    # remove self-similarity from the similarity matrix by creating an identity matrix (self_mask) and subtracting it from the mask.
    self_mask = torch.eye(z.size(0), device=z.device)
    mask = mask - self_mask

    # Compute the contrastive loss using the similarity matrix and the mask.
    exp_sim = torch.exp(sim) * (1 - self_mask)
    
    # To compute the log probabilities, we take the similarity matrix and apply the log-softmax function. 
    # This is done by subtracting the log of the sum of exponentiated similarities for each sample 
    # from the similarity matrix itself.
    # This gives us the log probability of the positive samples (those with the same label) for each sample in the batch.
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Finally, we compute the mean log probability of the positive samples for each sample in the batch 
    # by multiplying the log probabilities with the mask and summing over the positive samples,
    # and then dividing by the number of positive samples (the sum of the mask) 
    # to get the average log probability for each sample.
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

    # The loss is then computed as the negative mean of the log probabilities of the positive samples,
    # which encourages the model to maximize the similarity between samples with the same label.
    loss = -mean_log_prob_pos.mean()
    return loss