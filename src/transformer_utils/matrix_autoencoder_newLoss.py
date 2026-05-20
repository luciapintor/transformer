import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
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
            nn.Linear(hidden_dim, emb_size)     # second layer maps hidden dimension to latent space (emb_size)
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
        
        # The criterion is the loss function that measures the difference between the input and the reconstruction.
        criterion = nn.MSELoss()
        
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
                
                # Zero the gradients (reset the gradients of all model parameters to zero)
                optimizer.zero_grad()
                
                # Forward pass: compute the model output and the latent representation
                x_hat, _ = self(x)
                
                # ensure same shape
                if x_hat.shape != x.shape:
                    x = x.view_as(x_hat)
                
                # Compute the loss between the input and the reconstruction
                loss = criterion(x_hat, x)
                
                loss.backward()
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
        labels = []
        
        # Use torch.no_grad() to disable gradient computation, 
        # since we are only interested in the embeddings and not in updating the model parameters.
        with torch.no_grad():
            for batch in dataloader:
                # allow dataloaders that yield (x,) or (x,y)
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                    if len(batch) > 1:
                        labels.append(batch[1])
                else:
                    x = batch

                x = x.to(device).float()

                z = self.encoder(x)
                embeddings.append(z.cpu())
                if labels and isinstance(labels[-1], torch.Tensor):
                    # ensure labels stored on CPU
                    labels[-1] = labels[-1].cpu()

        embeddings = torch.cat(embeddings, dim=0)
        if len(labels) > 0:
            labels = torch.cat(labels, dim=0)
            return embeddings, labels

        return embeddings

    def fit_clustering(self, dataloader, epochs=10, lr=1e-3, device=None, eps=0.5, min_samples=5, recon_weight=0.0, surrogate='prototype', temperature=0.5):
        """
        Train the autoencoder to improve clustering quality.
        This method uses DBSCAN on the embeddings to compute a clustering score
        (Adjusted Rand Index vs ground truth). Because the clustering metric is
        non-differentiable, we optimize a differentiable surrogate using the
        true labels.

        Parameters:
        - dataloader: yields (x, y) tuples (y = ground-truth labels)
        - recon_weight: optional weight for reconstruction loss (0 = only surrogate)
        - surrogate: 'prototype', 'contrastive', or 'fisher'
        - temperature: temperature for contrastive / prototype logits
        """

        if device is None:
            device = next(self.parameters()).device
        self.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        recon_criterion = nn.MSELoss()

        for epoch in range(epochs):
            # --- collect all data once (on CPU) ---
            xs = []
            ys = []
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                    if len(batch) > 1:
                        y = batch[1]
                    else:
                        y = None
                else:
                    x = batch
                    y = None

                xs.append(x.cpu())
                if y is not None:
                    ys.append(y.cpu())

            X_cpu = torch.cat(xs, dim=0)
            if len(ys) == 0:
                raise ValueError("Dataloader must yield (x, y) tuples with ground-truth labels for clustering loss.")
            Y_cpu = torch.cat(ys, dim=0)

            # --- compute embeddings (no grad) for clustering evaluation ---
            self.eval()
            with torch.no_grad():
                Z_cpu = self.encoder(X_cpu.float().to(device)).cpu().numpy()

            # normalize embeddings before DBSCAN (consistent scaling)
            self.scaler = StandardScaler().fit(Z_cpu)
            Z_scaled = self.scaler.transform(Z_cpu)

            # run DBSCAN and compute ARI vs ground truth
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(Z_scaled)
            pred = clustering.labels_
            ari = adjusted_rand_score(Y_cpu.numpy(), pred)
            clustering_loss = 1.0 - ari

            # --- differentiable surrogate ---
            # recompute embeddings with grad on device
            self.train()
            X = X_cpu.to(device).float()
            Y = Y_cpu.to(device)
            Z = self.encoder(X)
            if surrogate == 'contrastive':
                surrogate_loss = self.nt_xent_loss(Z, Y, temperature=temperature)
            elif surrogate == 'prototype':
                surrogate_loss = self.prototype_loss(Z, Y, temperature=temperature)
            else:
                unique_labels = torch.unique(Y)
                within = torch.tensor(0.0, device=device)
                between = torch.tensor(0.0, device=device)

                overall_mean = Z.mean(dim=0, keepdim=True)
                n_classes = 0
                for lab in unique_labels:
                    mask = (Y == lab)
                    cnt = mask.sum()
                    if cnt <= 0:
                        continue
                    n_classes += 1
                    z_lab = Z[mask]
                    mu_k = z_lab.mean(dim=0, keepdim=True)
                    # within-class scatter (sum of squared distances)
                    within = within + ((z_lab - mu_k).pow(2).sum()) / (cnt.float() + 1e-8)
                    # between-class scatter (weighted by class size)
                    between = between + (cnt.float() * (mu_k - overall_mean).pow(2).sum())

                if n_classes == 0:
                    surrogate_loss = torch.tensor(0.0, device=device)
                else:
                    # normalize by number of classes to keep scale stable
                    between = between / float(n_classes)
                    surrogate_loss = within / (between + 1e-8)

            # optional reconstruction loss (reconstruct all X)
            if recon_weight > 0.0:
                X_hat, _ = self(X)
                recon_loss = recon_criterion(X_hat, X)
            else:
                recon_loss = torch.tensor(0.0, device=device)

            total_loss = surrogate_loss * (1.0 - recon_weight) + recon_loss * recon_weight

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}: ARI={ari:.4f} clustering_loss={clustering_loss:.4f} surrogate_loss={surrogate_loss.item():.6f}")

    def nt_xent_loss(self, z, labels, temperature=0.5):
        """
        NT-Xent loss for batches with possibly multiple positives per anchor.
        z: (N, D) embeddings (not necessarily normalized)
        labels: (N,) long tensor with class labels
        """
        device = z.device
        z_norm = F.normalize(z, dim=1)
        sim = torch.matmul(z_norm, z_norm.t()) / (temperature + 1e-8)

        labels = labels.view(-1)
        N = z.size(0)

        # mask to exclude self-comparisons
        self_mask = torch.eye(N, dtype=torch.bool, device=device)

        exp_sim= torch.exp(sim)

        loss_sum = torch.tensor(0.0, device=device)
        valid = 0
        for i in range(N):
            pos_mask = (labels == labels[i])
            pos_mask[i] = False
            if pos_mask.sum() == 0:
                continue
            numerator = exp_sim[i][pos_mask].sum()
            denominator = exp_sim[i][~self_mask[i]].sum()
            loss_i = -torch.log((numerator / (denominator + 1e-12)) + 1e-12)
            loss_sum += loss_i
            valid += 1

        if valid == 0:
            return torch.tensor(0.0, device=device)
        return loss_sum / float(valid)

    def prototype_loss(self, z, labels, temperature=0.5):
        """
        Prototype-centered cross-entropy loss.
        Each class is represented by the mean of its embeddings in the batch.
        """
        labels = labels.view(-1)
        classes = torch.unique(labels)

        if len(classes) <= 1:
            return torch.tensor(0.0, device=z.device)

        prototypes = []
        class_indices = torch.empty_like(labels)
        for idx, cls in enumerate(classes):
            mask = labels == cls
            class_indices[mask] = idx
            prototypes.append(z[mask].mean(dim=0))

        prototypes = torch.stack(prototypes, dim=0)
        z_norm = F.normalize(z, dim=1)
        p_norm = F.normalize(prototypes, dim=1)
        logits = torch.matmul(z_norm, p_norm.t()) / (temperature + 1e-8)
        return F.cross_entropy(logits, class_indices)

