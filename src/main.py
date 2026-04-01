from torch.utils.data import DataLoader

from prepare_dataset.probe_dataset import ProbeDataset
from transformer_utils.encoder import Encoder, train_contrastive, extract_embeddings
from clustering.kmeans_embeddings import kmeans_embeddings


if __name__ == '__main__':
    
    # Define hyperparameters and configurations
    n_samples = 1000 # number of samples in the dataset
    n_features = 10 # number of features in the dataset
    n_classes = 15 # number of classes for classification (set to 1 for regression)
    
    # Generate a random dataset # TODO replace with ProbeDataset when it's ready
    dataset = ProbeDataset(data=None, labels=None) # create an instance of the dataset

    # Use a DataLoader to handle batching and shuffling
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # TODO: select a good encoder architecture
    model = Encoder(n_features)
    
    # Train the model using contrastive learning
    train_contrastive(model, dataloader, epochs=10)
    
    # Extract embeddings for clustering
    embeddings = extract_embeddings(model, dataloader)
    
    # TODO: Since KMeans is a supervised clustering algorithm, it requires the number of clusters (n_clusters) 
    # to be specified in advance.
    cluster_labels = kmeans_embeddings(embeddings, n_clusters=n_classes)
    
    print(cluster_labels[:20])
    
    pass
