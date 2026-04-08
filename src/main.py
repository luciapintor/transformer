from torch.utils.data import DataLoader
import torch

from transformer_utils.transformer_autoencoder import TransformerAutoencoder, train, extract_embeddings
from prepare_dataset.probe_dataset import ProbeDataset
from clustering.kmeans_embeddings import kmeans_embeddings

def collate_probe_batch(batch):
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
    
    # Define hyperparameters and configurations
    n_samples = None # number of samples in the dataset
    n_features = None # number of features in the dataset
    n_classes = None # number of classes for classification (set to 1 for regression)
    
    # Generate a random dataset # TODO replace with ProbeDataset when it's ready
    #full_dataset = MyDataset(n_samples=n_samples, n_features=n_features, n_classes=n_classes) # create an instance of the dataset
    #dataset_train, dataset_val, dataset_test = full_dataset.separate_train_val_test()

    full_dataset = ProbeDataset(
        "Dataset/dataset_burst_json/scenario_0_burst_features.json",
        preprocess=True
    )
    dataset_train, dataset_val, dataset_test = full_dataset.separate_train_val_test()

    # Ricava automaticamente le informazioni reali dal dataset scenario_0
    n_samples = len(full_dataset)
    n_features = len(full_dataset.data[0])
    n_classes = full_dataset.count_distinct_labels()

    print(f"Total samples: {n_samples}")
    print(f"Number of features: {n_features}")
    print(f"Distinct labels: {full_dataset.get_distinct_labels()}")
    print(f"Number of classes: {n_classes}")

  
    # The Dataloader id a torch utility that divides the dataset in batches 
    train_loader = DataLoader(
        dataset_train,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_probe_batch
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=32,
        collate_fn=collate_probe_batch
    )
    
    # TODO: change the model to an unsupervised one, and change the training loop accordingly. 
    # Transformer is useful if you use sequences, but for tabular data, you might want to consider using 
    # a simpler architecture, such as an encoder.
    model = TransformerAutoencoder(n_features)
    
    train(model, train_loader, epochs=100, lr = 0.1)
    
    embeddings = extract_embeddings(model, test_loader)
    
    # TODO: Since KMeans is a supervised clustering algorithm, it requires the number of clusters (n_clusters) 
    # to be specified in advance.
    cluster_labels = kmeans_embeddings(embeddings, n_clusters=n_classes)
    
    print(cluster_labels[:20])
    
    
    pass