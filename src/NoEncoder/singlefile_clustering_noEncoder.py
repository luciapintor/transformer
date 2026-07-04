import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from torch.utils.data import DataLoader
import torch
import pandas as pd
import json
import numpy as np
import os

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from transformer_utils.matrix_autoencoder import MatrixAutoencoder
from transformer_utils.evaluation_metric_calc import calc_evaluation_metrics
from prepare_dataset.probe_dataset import ProbeDataset
from converting_pcap.extract_features import extract_from_pcap

def pcap_to_json(pcap_file, output_json):
    dataset = extract_from_pcap(pcap_file=pcap_file)

    #aggiungo label fittizzia, solo per avere un formato standard
    for record in dataset:
        record["label"] = -1

    with open(output_json, "w") as f:
        json.dump(dataset, f, indent=4)

if __name__ == '__main__':

# ====================================================================
#                   PARAMETRI DATASET TRAIN E TEST
# ====================================================================
    
    probe_file = "/home/giuff/Tesi/TransformerTry/Dataset/Bonn_Dataset/json/samsung_a51_not_associated_screen_off_powersave_off_macrand_on_f5e5a79d.json"  # Percorso del file pcap da cui estrarre i dati
    output_json = ""  # Percorso del file JSON di output
    isPcap = False                 # Se True, tratta i file come file pcap, altrimenti come file di bursts di PR
    batch_size = 64                #TODO: definire un batch size adeguato, considerando la dimensione del dataset
    preprocess = True               # Se True, applica preprocessamento ai dati
    include_mac_features = False    # Se True, include gli indirizzi MAC nel dataset

# ====================================================================
#                   PARAMETRI CLUSTERING
# ====================================================================
    eps = 0.1               #raggio massimo per considerare due campioni come vicini in DBSCAN
    min_samples = 4         #numero minimo di campioni per diventare cluster

    #converto il pcap in json se isPcap è True, altrimenti uso direttamente il json già presente
    if isPcap:
        pcap_to_json(probe_file, output_json)
        json_path = output_json
    else:
        json_path = probe_file
        
    full_dataset = ProbeDataset(path_json=json_path, preprocess=preprocess, include_mac_features=include_mac_features)
    
    test_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ProbeDataset.collate_probe_batch
    )

    n_probe_test = len(full_dataset.data)    #numero di campioni nel dataset di test

    # Estrai le feature preprocessate dal test dataset
    print("[INFO] Extracting features from test dataset...")
    embeddings_list = []
    for batch in test_loader:
        features_batch = batch[0]  # Prendi solo le features
        embeddings_list.append(features_batch)

    embeddings = np.vstack(embeddings_list)

    # Applica MinMaxScaler per normalizzare le feature
    scaler = MinMaxScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(embeddings_scaled) 

    # scarto i campioni considerati rumore da DBSCAN per valutare le metriche del clustering        

    true_labels_filtered = []
    cluster_labels_filtered = []
    discarded_pr = 0

    for c in cluster_labels:
        if c != -1:
            cluster_labels_filtered.append(c)
        else:
            discarded_pr += 1

    print("CALCOLO SENZA RUMORE")
    print(f"Probe considerate: {n_probe_test}")
    print(f"Probe considerate rumore (cluster -1): {discarded_pr} --> {100*(discarded_pr/n_probe_test):.2f}%")

    print(f"--------------------------------------------------------------")
    if not isPcap:
        print("Numero di dispositivi:", len(set(full_dataset.labels)))
    print(f"Numero di cluster trovati senza rumore: {len(set(cluster_labels_filtered))}")  
    print(f"Cluster labels: {set(cluster_labels_filtered)}")     

    output_values = []          
    for i, (features, label, mac_address) in enumerate(full_dataset):           
        output_values.append({          
            "sample_index": i,          
            "mac_address": mac_address,       
            "cluster": cluster_labels[i],           
        })
                
    df = pd.DataFrame(output_values)
    df = df.sort_values("cluster")
    print(df)

    output_csv = "/home/giuff/Tesi/TransformerTry/transformer/clustering_output/outputClustering.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df.to_csv(output_csv, index=False)     