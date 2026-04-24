import torch

from clustering.DBSCAN_results import dbscan_results
from transformer_utils.matrix_autoencoder import MatrixAutoencoder
from prepare_dataset.probe_dataset2 import ProbeDataset
from prepare_dataset.probe_dataloader2 import ProbeDataLoader

if __name__ == '__main__':

# ====================================================================
#                   PARAMETRI DATASET TRAIN E TEST
# ====================================================================
    
    train_scenario_ids = [0,1,2,3,4,5,6,7]  # Lista di scenari per il training
    test_scenario_ids = [8,9,10]            # Lista di scenari per il test
    base_path = "Dataset/dataset_merged_probes_json/data_with_labels"   # Percorso base dei file JSON
    batch_size = 256                #definire un batch size adeguato, considerando la dimensione del dataset
    is_bursts = True                # Se True, tratta i file come file di bursts di PR, altrimenti come file di PR individuali
    preprocess = True               # Se True, applica preprocessamento ai dati
    include_mac_features = False    # Se True, include gli indirizzi MAC nel dataset

# ====================================================================
#                   PARAMETRI MODELLO
# ====================================================================

    emb_size = 64           #dimensione dell'embedding finale prodotto dall'encoder
    hidden_dim = 128        #dimensione del layer nascosto dell'autoencoder
    epochs = 1              #numero di sessioni di training del modello
    learning_rate = 1e-3    #tasso di apprendimento per l'ottimizzazione del modello

# ====================================================================
#                   PARAMETRI CLUSTERING
# ====================================================================
    eps = 0.1               #raggio massimo per considerare due campioni come vicini in DBSCAN
    min_samples = 4         #numero minimo di campioni per diventare cluster
    
# ====================================================================
#                  CREAZIONE PATH SCENARI TRAIN E TEST
# ====================================================================
    train_scenarios = [f"{base_path}/scenario_{i}_full.json" for i in train_scenario_ids]
    test_scenarios = [f"{base_path}/scenario_{i}_full.json" for i in test_scenario_ids]

    #creo i 2 dataset usando il nuovo metodo
    dataset_train = ProbeDataset(path_json=train_scenarios, preprocess=preprocess, include_mac_features=include_mac_features)

    dataset_test = ProbeDataset(path_json=test_scenarios, preprocess=preprocess, include_mac_features=include_mac_features)
    
    #anche se è un numero fissato, è meglio definirlo dinamicamente 
    n_features = len(dataset_train.data.keys()) 

    # creo i dataloader per train e test, usando il nuovo metodo
    train_loader = ProbeDataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = ProbeDataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # Autoencoder to extract embeddings from the data
    model = MatrixAutoencoder(n_features, emb_size=emb_size, hidden_dim=hidden_dim)
    model.fit(dataloader=train_loader, epochs=epochs, lr=learning_rate) # training only on training scenarios
    embeddings = model.encode_dataloader(dataloader=test_loader)        # extracting embeddings 
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    print("CLUSTERING DBSCAN OF RAW DATA")
    dbscan_results(eps=eps, min_samples=min_samples, my_data=dataset_test.data, true_labels=dataset_test.labels) 
    
    print("CLUSTERING DBSCAN OF EMBEDDINGS")
    dbscan_results(eps=eps, min_samples=min_samples, my_data=embeddings, true_labels=dataset_test.labels)