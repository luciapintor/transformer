from torch.utils.data import DataLoader
import torch
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)

from transformer_utils.matrix_autoencoder import MatrixAutoencoder
from prepare_dataset.probe_dataset import ProbeDataset


if __name__ == '__main__':

    #definiamo i dataset di train e test, con i rispettivi path ai file json
    train_json_path = "Dataset/dataset_burst_json_veri/scenario_0_burst_features.json"
    test_json_path  = "Dataset/dataset_burst_json_veri/scenario_1_burst_features.json"

    #TODO: definire un batch size adeguato, considerando la dimensione del dataset 
    batch_size = 64

    #creo i 2 dataset
    dataset_train = ProbeDataset(
        path_json=train_json_path,
        preprocess=True,
        include_mac_features=False
    )

    dataset_test = ProbeDataset(
        path_json=test_json_path,
        preprocess=True,
        include_mac_features=False
    )

    #anche se è un numero fissato, è meglio definirlo dinamicamente 
    n_features = len(dataset_train.data[0])

    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ProbeDataset.collate_probe_batch
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ProbeDataset.collate_probe_batch
    )

    model = MatrixAutoencoder(n_features, emb_size=64, hidden_dim=128)

    # train SOLO su scenario 0
    model.fit(dataloader=train_loader, epochs=100, lr=1e-3)

    # encoding SOLO di scenario 1
    embeddings = model.encode_dataloader(dataloader=test_loader)

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    dbscan = DBSCAN(eps=0.1, min_samples=1)
    cluster_labels = dbscan.fit_predict(embeddings)

    # True label del test set
    # Servono solo per valutare i cluster trovati           
    true_labels = dataset_test.labels           
                
    # ----------------------------          
    # Metriche di valutazione           
    # ----------------------------          
                
    # ARI: concordanza tra cluster trovati e gruppi veri            
    ari = adjusted_rand_score(true_labels, cluster_labels)          
                
    # NMI: informazione condivisa tra labels vere e cluster         
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)         
                
    # Homogeneity: ogni cluster contiene quasi una sola label?          
    homogeneity = homogeneity_score(true_labels, cluster_labels)            
                
    # Completeness: ogni label vera finisce quasi tutta in un cluster?          
    completeness = completeness_score(true_labels, cluster_labels)          
                
    # V-measure: sintesi di homogeneity e completeness          
    v_measure = v_measure_score(true_labels, cluster_labels)               
                
    print(f"ARI: {ari:.4f}")            
    print(f"NMI: {nmi:.4f}")            
    print(f"Homogeneity: {homogeneity:.4f}")            
    print(f"Completeness: {completeness:.4f}")          
    print(f"V-measure: {v_measure:.4f}")            
    print(f"Numero cluster: {n_clusters}")          
    print(f"Punti rumore: {n_noise}")           
                
    output_values = []          
    for i, (features, label, mac_address) in enumerate(dataset_test):           
        output_values.append({          
            "sample_index": i,          
            "mac_address": mac_address,         
            "true_label": label,            
            "cluster": cluster_labels[i],           
        })          
                
    df = pd.DataFrame(output_values)            
    df = df.sort_values("true_label")           
    print(df)           
                
    df.to_csv("transformer/clustering_output/output_s0_train_s1_test.csv", index=False)         

    output_values = []
    for i, (features, label, mac_address) in enumerate(dataset_test):
        output_values.append({
            "sample_index": i,
            "mac_address": mac_address,
            "true_label": label,
            "cluster": cluster_labels[i],
        })

    df = pd.DataFrame(output_values)
    df = df.sort_values("true_label")
    print(df)

    df.to_csv("transformer/clustering_output/output_s0_train_s1_test.csv", index=False)