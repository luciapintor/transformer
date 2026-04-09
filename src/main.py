from torch.utils.data import DataLoader
import torch

from transformer_utils.matrix_autoencoder import MatrixAutoencoder
from prepare_dataset.probe_dataset import ProbeDataset
from clustering.kmeans_embeddings import kmeans_embeddings

def collate_probe_batch(batch):
    """TODO: spostare in prepare_dataset.probe_dataset.py"""
    """
    Converte un batch di campioni del ProbeDataset in tensori PyTorch.

    Supporta due formati:
    - (record_dict, label)
    - (record_dict, label, mac_address)

    Returns:
        X: tensore float32 di shape [batch_size, n_features]
        y: tensore long di shape [batch_size]
    """
    first_item = batch[0]

    if len(first_item) == 2:
        records, labels = zip(*batch)
    elif len(first_item) == 3:
        records, labels, _ = zip(*batch)  # ignoriamo i MAC address
    else:
        raise ValueError(
            f"Formato batch non supportato: attesi 2 o 3 elementi per sample, trovati {len(first_item)}"
        )

    feature_names = sorted(records[0].keys())

    X = torch.tensor(
        [[record[name] for name in feature_names] for record in records],
        dtype=torch.float32
    )

    y = torch.tensor(labels, dtype=torch.long)

    return X, y

if __name__ == '__main__':
    
    json_path = "Dataset/dataset_burst_json/scenario_0_burst_features.json"
    
    # Load the dataset from a JSON file
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
        collate_fn=collate_probe_batch # this function converts a batch of samples from the dataset into tensors
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=32, 
        collate_fn=collate_probe_batch # this function converts a batch of samples from the dataset into tensors
    )
    
    # TODO: change the model to an unsupervised one, and change the training loop accordingly. 
    # Transformer is useful if you use sequences, but for tabular data, you might want to consider using 
    # a simpler architecture, such as an encoder.
    model = MatrixAutoencoder(n_features)
    
    n_epochs = 100
    learning_rate = 0.1
    
    # training the model in an unsupervised way, since we want to extract embeddings without using the labels.
    model.fit(dataloader=train_loader, epochs=n_epochs, lr = learning_rate)
    
    # Extract embeddings from the test set using the trained model
    embeddings = model.encode_dataloader(dataloader=test_loader)
    
    # TODO: Since KMeans is a supervised clustering algorithm, it requires the number of clusters (n_clusters) 
    # to be specified in advance. We can substitute it with DBSCAN
    cluster_labels = kmeans_embeddings(embeddings, n_clusters=n_classes)
    
    print(cluster_labels[:20])
    
    pass