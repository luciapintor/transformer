from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np
import os

from sklearn.cluster import DBSCAN

from transformer_utils.evaluation_metric_calc import calc_evaluation_metrics
from prepare_dataset.probe_dataset import ProbeDataset

if __name__ == '__main__':

# ====================================================================
#                   PARAMETRI DATASET TRAIN E TEST
# ====================================================================
    
    train_scenarios = [0,1,2,3,4,5,6,7]  # Lista di scenari per il training
    test_scenarios = [8,9,10]            # Lista di scenari per il test
    base_path = "Dataset/dataset_burst_json_veri"   # Percorso base dei file JSON
    batch_size = 256                #TODO: definire un batch size adeguato, considerando la dimensione del dataset
    is_bursts = True               # Se True, tratta i file come file di bursts di PR, altrimenti come file di PR individuali
    preprocess = True               # Se True, applica preprocessamento ai dati
    include_mac_features = False    # Se True, include gli indirizzi MAC nel dataset

# ====================================================================
#                   PARAMETRI CLUSTERING
# ====================================================================
    eps = 0.1               #raggio massimo per considerare due campioni come vicini in DBSCAN
    min_samples = 4         #numero minimo di campioni per diventare cluster

    # creo i 2 dataset usando il nuovo metodo
    print("[INFO] Loading training dataset...")
    dataset_train = ProbeDataset.from_scenario_list(
        scenario_list=train_scenarios,
        base_path=base_path,
        is_bursts=is_bursts,
        preprocess=preprocess,
        include_mac_features=include_mac_features
    )

    print("[INFO] Loading test dataset...")
    dataset_test = ProbeDataset.from_scenario_list(
        scenario_list=test_scenarios,
        base_path=base_path,
        is_bursts=is_bursts,
        preprocess=preprocess,
        include_mac_features=include_mac_features
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ProbeDataset.collate_probe_batch
    )

    # Estrai le feature preprocessate dal test dataset
    print("[INFO] Extracting features from test dataset...")
    embeddings_list = []
    for batch in test_loader:
        features_batch = batch[0]  # Prendi solo le features
        if isinstance(features_batch, torch.Tensor):
            features_batch = features_batch.detach().cpu().numpy()
        embeddings_list.append(features_batch)
    
    embeddings = np.vstack(embeddings_list)

    print(f"[INFO] Running DBSCAN with eps={eps}, min_samples={min_samples}...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
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

    print("\n" + "="*60)
    print("CLUSTERING RESULTS (NOISE EXCLUDED)")
    print("="*60)
    metrics_undiscarded = calc_evaluation_metrics(true_labels_filtered, cluster_labels_filtered)
    print(f"ARI: {metrics_undiscarded['ari']:.4f}")
    print(f"NMI: {metrics_undiscarded['nmi']:.4f}")
    print(f"Homogeneity: {metrics_undiscarded['homogeneity']:.4f}")
    print(f"Completeness: {metrics_undiscarded['completeness']:.4f}")
    print(f"V-measure: {metrics_undiscarded['v_measure']:.4f}")

    print("-" * 60)
    print(f"Number of classes: {len(set(dataset_test.labels))}")
    print(f"Number of clusters found (noise excluded): {len(set(cluster_labels_filtered))}")
    print(f"Cluster labels: {set(cluster_labels_filtered)}")
    print(f"Number of noise points: {list(cluster_labels).count(-1)}")
    print("="*60 + "\n")

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

    # Create output directory if it doesn't exist
    output_path = "transformer/clustering_output/output_noEncode.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[INFO] Results saved to {output_path}")