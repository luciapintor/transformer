from torch.utils.data import DataLoader
import torch
import pandas as pd
import json

from sklearn.cluster import DBSCAN

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
    
    pcap_file = "/example.pcap"  # Percorso del file pcap da cui estrarre i dati
    output_json = "Dataset/dataset_json_from_pcap/dataset_from_pcap.json"  # Percorso del file JSON di output
    isPcap = True                # Se True, tratta i file come file pcap, altrimenti come file di bursts di PR
    batch_size = 64                #TODO: definire un batch size adeguato, considerando la dimensione del dataset
    preprocess = True               # Se True, applica preprocessamento ai dati
    include_mac_features = False    # Se True, include gli indirizzi MAC nel dataset

# ====================================================================
#                   PARAMETRI MODELLO
# ====================================================================

    emb_size = 64           #dimensione dell'embedding finale prodotto dall'encoder
    hidden_dim = 128        #dimensione del layer nascosto dell'autoencoder
    epochs = 100             #numero di sessioni di training del modello
    learning_rate = 1e-3    #tasso di apprendimento per l'ottimizzazione del modello

# ====================================================================
#                   PARAMETRI CLUSTERING
# ====================================================================
    eps = 0.1               #raggio massimo per considerare due campioni come vicini in DBSCAN
    min_samples = 4         #numero minimo di campioni per diventare cluster

    #converto il pcap in json se isPcap è True, altrimenti uso direttamente il json già presente
    if isPcap:
        pcap_to_json(pcap_file, output_json)
    json_path = output_json
    
    full_dataset = ProbeDataset(path_json=json_path, preprocess=True)

    #divido il json in train e test
    dataset_train, dataset_val, dataset_test = full_dataset.separate_train_val_test()

    #dimensioni dei vari dataset
    n_features = len(full_dataset.data[0])
    n_probe_train = len(dataset_train)
    n_probe_test = len(dataset_test)

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

    model = MatrixAutoencoder(n_features, emb_size=emb_size, hidden_dim=hidden_dim)

    # train SOLO sugli scenari di training
    model.fit(dataloader=train_loader, epochs=epochs, lr=learning_rate)

    # encoding SOLO degli scenari di test
    embeddings = model.encode_dataloader(dataloader=test_loader)

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(embeddings) 

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
    print(f"Numero di cluster trovati senza rumore: {len(set(cluster_labels_filtered))}")  
    print(f"Cluster labels: {set(cluster_labels_filtered)}")     

    output_values = []          
    for i, (features, label, mac_address) in enumerate(dataset_test):           
        output_values.append({          
            "sample_index": i,          
            "mac_address": mac_address,       
            "cluster": cluster_labels[i],           
        })
                
    df = pd.DataFrame(output_values)            
    df = df.sort_values("cluster")           
    print(df)           
                
    df.to_csv("transformer/clustering_output/output_s0_train_s1_test.csv", index=False)         