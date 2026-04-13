from torch.utils.data import DataLoader
import torch

from transformer_utils.matrix_autoencoder import MatrixAutoencoder
from prepare_dataset.probe_dataset import ProbeDataset
from clustering.kmeans_embeddings import kmeans_embeddings


if __name__ == '__main__':
    
    json_path = "Dataset/dataset_burst_json/scenario_0_burst_features.json"
    
    # Load the dataset from a JSON file and preprocess it
    full_dataset = ProbeDataset(path_json=json_path, preprocess=True)
    # Separate the dataset into training, validation, and test sets
    dataset_train, dataset_val, dataset_test = full_dataset.separate_train_val_test()

    # Ricava automaticamente le informazioni reali dal dataset scenario_0
    n_samples = len(full_dataset)                     # number of samples in the dataset
    n_features = len(full_dataset.data[0])            # number of features in the dataset 
    n_classes = full_dataset.count_distinct_labels()  # number of classes for classification (set to 1 for regression)

    print(f"Total samples: {n_samples}")
    print(f"Number of features: {n_features}")
    print(f"Distinct labels: {full_dataset.get_distinct_labels()}")
    print(f"Number of classes: {n_classes}")

    # The Dataloader id a torch utility that divides the dataset in batches 
    train_loader = DataLoader(
        dataset_train,
        batch_size=32,
        shuffle=True, # shuffle the training data at every epoch to improve generalization
        collate_fn=ProbeDataset.collate_probe_batch # this function converts a batch of samples from the dataset into tensors
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=32, 
        collate_fn=ProbeDataset.collate_probe_batch # this function converts a batch of samples from the dataset into tensors
    )
    
    emb_size = 64        # dimension of the latent space (embedding size)
    hidden_dim = 128    # dimension of the hidden layer in the autoencoder 
    
    # Initialize the autoencoder model with the number of features in the dataset
    # The autoencoder will learn to compress the input data into a lower-dimensional latent space 
    # and then reconstruct it back to the original space.
    model = MatrixAutoencoder(n_features, emb_size=emb_size, hidden_dim=hidden_dim)
    
    n_epochs = 100
    #TODO??? implementare il gradient descent nel fit? anche se mi pare ci sia
    learning_rate = 0.5
    
    # training the model in an unsupervised way, since we want to extract embeddings without using the labels.
    model.fit(dataloader=train_loader, epochs=n_epochs, lr = learning_rate)
    
    # Extract embeddings from the test set using the trained model
    embeddings = model.encode_dataloader(dataloader=test_loader)
    
    # TODO: Since KMeans is a supervised clustering algorithm, it requires the number of clusters (n_clusters) 
    # to be specified in advance. We can substitute it with DBSCAN
    cluster_labels = kmeans_embeddings(embeddings, n_clusters=n_classes)
    
    print(cluster_labels)
    
    pass