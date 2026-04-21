from torch.utils.data import DataLoader
import torch
import pandas as pd

from sklearn.cluster import DBSCAN

from transformer_utils.matrix_autoencoder import MatrixAutoencoder
from transformer_utils.evaluation_metric_calc import calc_evaluation_metrics
from prepare_dataset.probe_dataset import ProbeDataset

if __name__ == '__main__':

    #definiamo i dataset di train e test, con i rispettivi path ai file json
    # Invece, usa il nuovo metodo per specificare liste di scenari
    train_scenarios = [0,1,2,3,4,5,6,7]  # Lista di scenari per il training
    test_scenarios = [8,9,10]   # Lista di scenari per il test
    base_path = "Dataset/dataset_burst_json_veri/"

    #TODO: definire un batch size adeguato, considerando la dimensione del dataset 
    batch_size = 128

    #creo i 2 dataset usando il nuovo metodo
    dataset_train = ProbeDataset.from_scenario_list(
        scenario_list=train_scenarios,
        base_path=base_path,
        preprocess=True,
        include_mac_features=False
    )

    dataset_test = ProbeDataset.from_scenario_list(
        scenario_list=test_scenarios,
        base_path=base_path,
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

    dbscan = DBSCAN(eps=0.1, min_samples=4)
    cluster_labels = dbscan.fit_predict(embeddings)

    # True label del test set
    # Servono solo per valutare i cluster trovati           
    true_labels = dataset_test.labels           

    # scarto i campioni considerati rumore da DBSCAN per valutare le metriche del clustering        

    true_labels_filtered = []
    cluster_labels_filtered = []

    for t, c in zip(true_labels, cluster_labels):
        if c != -1:
            true_labels_filtered.append(t)
            cluster_labels_filtered.append(c)

    metrics_undiscarded = calc_evaluation_metrics(true_labels_filtered, cluster_labels_filtered)
    print("CALCOLO SENZA RUMORE")
    print(f"ARI: {metrics_undiscarded['ari']:.4f}")            
    print(f"NMI: {metrics_undiscarded['nmi']:.4f}")            
    print(f"Homogeneity: {metrics_undiscarded['homogeneity']:.4f}")            
    print(f"Completeness: {metrics_undiscarded['completeness']:.4f}")          
    print(f"V-measure: {metrics_undiscarded['v_measure']:.4f}")    

    print(f"--------------------------------------------------------------")
    print("Numero di classi:", len(set(dataset_test.labels)))
    print(f"Numero di cluster trovati senza rumore: {len(set(cluster_labels_filtered))}")  
    print(f"Cluster labels: {set(cluster_labels_filtered)}")     

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