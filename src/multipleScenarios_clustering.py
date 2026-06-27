from torch.utils.data import DataLoader
import torch
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from transformer_utils.matrix_autoencoder import MatrixAutoencoder
from transformer_utils.evaluation_metric_calc import calc_evaluation_metrics
from prepare_dataset.probe_dataset import ProbeDataset

if __name__ == '__main__':

# ====================================================================
#                   PARAMETRI DATASET TRAIN E TEST
# ====================================================================
    
    train_scenarios = [0]  # Lista di scenari per il training
    test_scenarios = [5]            # Lista di scenari per il test
    base_path = "Dataset/dataset_burst_json_veri"   # Percorso base dei file JSON
    batch_size = 64                #TODO: definire un batch size adeguato, considerando la dimensione del dataset
    is_bursts = True               # Se True, tratta i file come file di bursts di PR, altrimenti come file di PR individuali
    preprocess = True               # Se True, applica preprocessamento ai dati
    include_mac_features = False    # Se True, include gli indirizzi MAC nel dataset
    normalize = False                 # Se True, normalizza i dati prima di usare DBSCAN

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

    #creo i 2 dataset usando il nuovo metodo
    dataset_train = ProbeDataset.from_scenario_list(
        scenario_list=train_scenarios,
        base_path=base_path,
        is_bursts=is_bursts,
        preprocess=preprocess,
        include_mac_features=include_mac_features
    )

    dataset_test = ProbeDataset.from_scenario_list(
        scenario_list=test_scenarios,
        base_path=base_path,
        is_bursts=is_bursts,
        preprocess=preprocess,
        include_mac_features=include_mac_features
    )

    #dimensioni dei vari dataset
    n_features = len(dataset_train.data[0])
    n_probe_train = len(dataset_train.data)  #numero di campioni nel dataset di training
    n_probe_test = len(dataset_test.data)    #numero di campioni nel dataset di test

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

    # Applica MinMaxScaler per normalizzare le feature
    if normalize:
        scaler = MinMaxScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        embeddings = embeddings_scaled  # Aggiorna embeddings con le versioni scalate

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = dbscan.fit_predict(embeddings)

    # True label del test set
    # Servono solo per valutare i cluster trovati           
    true_labels = dataset_test.labels           

    # scarto i campioni considerati rumore da DBSCAN per valutare le metriche del clustering        

    true_labels_filtered = []
    cluster_labels_filtered = []
    discarded_pr = 0

    for t, c in zip(true_labels, cluster_labels):
        if c != -1:
            true_labels_filtered.append(t)
            cluster_labels_filtered.append(c)
        else:
            discarded_pr += 1

    metrics_undiscarded = calc_evaluation_metrics(true_labels_filtered, cluster_labels_filtered)
    print("CALCOLO SENZA RUMORE")
    print(f"Probe considerate rumore (cluster -1): {discarded_pr} --> {100*(discarded_pr/n_probe_test):.2f}%")
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