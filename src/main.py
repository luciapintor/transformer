from torch.utils.data import DataLoader

from transformer_utils.transformer_autoencoder import TransformerAutoencoder, train, extract_embeddings
from prepare_dataset.my_dataset import MyDataset
from clustering.kmeans_embeddings import kmeans_embeddings


if __name__ == '__main__':
    
    # Define hyperparameters and configurations
    n_samples = 1000 # number of samples in the dataset
    n_features = 10 # number of features in the dataset
    n_classes = 2 # number of classes for classification (set to 1 for regression)
    
    # Generate a random dataset # TODO replace with ProbeDataset when it's ready
    full_dataset = MyDataset(n_samples=n_samples, n_features=n_features, n_classes=n_classes) # create an instance of the dataset
    dataset_train, dataset_val, dataset_test = full_dataset.separate_train_val_test()
  
    # The Dataloader id a torch utility that divides the dataset in batches 
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=32)
    
    # TODO: change the model to an unsupervised one, and change the training loop accordingly. 
    # Transformer is useful if you use sequences, but for tabular data, you might want to consider using 
    # a simpler architecture, such as an encoder.
    model = TransformerAutoencoder(n_features)
    
    train(model, train_loader, epochs=10)
    
    embeddings = extract_embeddings(model, test_loader)
    
    # TODO: Since KMeans is a supervised clustering algorithm, it requires the number of clusters (n_clusters) 
    # to be specified in advance.
    cluster_labels = kmeans_embeddings(embeddings, n_clusters=n_classes)
    
    print(cluster_labels[:20])
    
    
    pass
