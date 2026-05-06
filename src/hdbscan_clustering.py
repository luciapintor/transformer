from torch.utils.data import DataLoader
import torch
import pandas as pd

import hdbscan

from transformer_utils.matrix_autoencoder import MatrixAutoencoder
from transformer_utils.evaluation_metric_calc import calc_evaluation_metrics
from prepare_dataset.probe_dataset import ProbeDataset

if __name__ == '__main__':

# ====================================================================
#                   PARAMETRI DATASET TRAIN E TEST
# ====================================================================
    
    train_scenarios = [0,1]  # Lista di scenari per il training
    test_scenarios = [2,3]            # Lista di scenari per il test
    base_path = "Dataset/dataset_burst_json_veri"   # Percorso base dei file JSON
    batch_size = 64                #TODO: definire un batch size adeguato, considerando la dimensione del dataset
    is_bursts = True               # Se True, tratta i file come file di bursts di PR, altrimenti come file di PR individuali
    preprocess = True               # Se True, applica preprocessamento ai dati
    include_mac_features = False    # Se True, include gli indirizzi MAC nel dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.mixture import BayesianGaussianMixture

from transformer_utils.matrix_autoencoder import MatrixAutoencoder
from transformer_utils.evaluation_metric_calc import calc_evaluation_metrics
from prepare_dataset.probe_dataset import ProbeDataset


def print_metrics(title, metrics):
    print(title)
    print(f"ARI: {metrics['ari']:.4f}")
    print(f"NMI: {metrics['nmi']:.4f}")
    print(f"Homogeneity: {metrics['homogeneity']:.4f}")
    print(f"Completeness: {metrics['completeness']:.4f}")
    print(f"V-measure: {metrics['v_measure']:.4f}")


def evaluate_clustering(true_labels, cluster_labels, discard_noise=False):
    """
    Calcola le metriche di clustering.

    Se discard_noise=True, scarta i campioni con cluster -1.
    Utile per DBSCAN, dove -1 indica rumore.
    Per BayesianGaussianMixture normalmente non serve, perché non produce rumore.
    """

    true_labels_filtered = []
    cluster_labels_filtered = []
    discarded_pr = 0

    for t, c in zip(true_labels, cluster_labels):
        if discard_noise and c == -1:
            discarded_pr += 1
        else:
            true_labels_filtered.append(t)
            cluster_labels_filtered.append(c)

    metrics = calc_evaluation_metrics(true_labels_filtered, cluster_labels_filtered)

    return metrics, true_labels_filtered, cluster_labels_filtered, discarded_pr


def run_dbscan_clustering(embeddings, eps, min_samples):
    """
    Esegue DBSCAN sugli embedding.
    """

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(embeddings)

    return cluster_labels, dbscan


def run_bayesian_gmm_clustering(
    embeddings,
    max_components=20,
    weight_concentration_prior=0.01,
    covariance_type="full",
    random_state=42,
):
    """
    Esegue clustering bayesiano sugli embedding usando BayesianGaussianMixture.

    max_components indica il numero massimo di componenti gaussiane.
    Con il prior dirichlet_process, il modello puo' assegnare peso quasi nullo
    alle componenti che non servono.
    """

    bayes_gmm = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type=covariance_type,
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=weight_concentration_prior,
        max_iter=1000,
        n_init=5,
        random_state=random_state,
    )

    cluster_labels = bayes_gmm.fit_predict(embeddings)

    weights = bayes_gmm.weights_
    used_components = np.sum(weights > 1e-3)

    print("\n=== INFO BAYESIAN GMM ===")
    print(f"Numero massimo componenti: {max_components}")
    print(f"Componenti usate, peso > 1e-3: {used_components}")
    print("Pesi componenti:")
    print(np.round(weights, 4))

    return cluster_labels, bayes_gmm


def save_clustering_output(dataset_test, cluster_labels, output_csv_path):
    """
    Salva su CSV il risultato del clustering per ogni campione del test set.
    """

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
    df.to_csv(output_csv_path, index=False)

    return df


if __name__ == '__main__':

# ====================================================================
#                   PARAMETRI DATASET TRAIN E TEST
# ====================================================================

    train_scenarios = [0, 1]       # Lista di scenari per il training
    test_scenarios = [2, 3]        # Lista di scenari per il test
    base_path = "Dataset/dataset_burst_json_veri"   # Percorso base dei file JSON
    batch_size = 64                # Numero di campioni per batch
    is_bursts = True               # True: file di burst PR; False: PR individuali
    preprocess = True              # True: applica preprocessamento ai dati
    include_mac_features = False   # True: include gli indirizzi MAC nelle feature

# ====================================================================
#                   PARAMETRI MODELLO
# ====================================================================

    emb_size = 64           # Dimensione dell'embedding prodotto dall'encoder
    hidden_dim = 128        # Dimensione del layer nascosto dell'autoencoder
    epochs = 10             # Numero di epoche di training
    learning_rate = 1e-3    # Learning rate

# ====================================================================
#                   PARAMETRI CLUSTERING DBSCAN
# ====================================================================

    eps = 0.1               # Raggio massimo per considerare due campioni vicini
    min_samples = 4         # Numero minimo di campioni per formare un cluster

# ====================================================================
#                   PARAMETRI CLUSTERING BAYESIAN GMM
# ====================================================================

    max_components = 20                 # Numero massimo di componenti gaussiane
    weight_concentration_prior = 0.01   # Valori piccoli tendono a usare meno componenti
    covariance_type = "full"            # Prova anche "diag" se full e' instabile
    random_state = 42

# ====================================================================
#                   CREAZIONE DATASET
# ====================================================================

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

    n_features = len(dataset_train.data[0])
    n_probe_train = len(dataset_train.data)
    n_probe_test = len(dataset_test.data)

    print("\n=== INFO DATASET ===")
    print(f"Scenari training: {train_scenarios}")
    print(f"Scenari test: {test_scenarios}")
    print(f"Numero feature: {n_features}")
    print(f"Campioni training: {n_probe_train}")
    print(f"Campioni test: {n_probe_test}")
    print(f"Numero classi test: {len(set(dataset_test.labels))}")

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

# ====================================================================
#                   TRAIN AUTOENCODER
# ====================================================================

    model = MatrixAutoencoder(n_features, emb_size=emb_size, hidden_dim=hidden_dim)

    # Train solo sugli scenari di training
    model.fit(dataloader=train_loader, epochs=epochs, lr=learning_rate)

    # Encoding solo degli scenari di test
    embeddings = model.encode_dataloader(dataloader=test_loader)

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    else:
        embeddings = np.asarray(embeddings)

    true_labels = dataset_test.labels

# ====================================================================
#                   CLUSTERING DBSCAN
# ====================================================================

    dbscan_labels, dbscan_model = run_dbscan_clustering(
        embeddings=embeddings,
        eps=eps,
        min_samples=min_samples
    )

    dbscan_metrics, true_labels_dbscan_filtered, dbscan_labels_filtered, discarded_pr = evaluate_clustering(
        true_labels=true_labels,
        cluster_labels=dbscan_labels,
        discard_noise=True
    )

    print("\n==============================================================")
    print("RISULTATI DBSCAN - CALCOLO SENZA RUMORE")
    print("==============================================================")
    print(f"Probe considerate rumore cluster -1: {discarded_pr} --> {100 * (discarded_pr / n_probe_test):.2f}%")
    print_metrics("Metriche DBSCAN:", dbscan_metrics)
    print(f"Numero di cluster trovati senza rumore: {len(set(dbscan_labels_filtered))}")
    print(f"Cluster labels DBSCAN senza rumore: {set(dbscan_labels_filtered)}")

    save_clustering_output(
        dataset_test=dataset_test,
        cluster_labels=dbscan_labels,
        output_csv_path="transformer/clustering_output/output_dbscan_multiple_scenarios.csv"
    )

# ====================================================================
#                   CLUSTERING BAYESIAN GMM
# ====================================================================

    bayes_labels, bayes_model = run_bayesian_gmm_clustering(
        embeddings=embeddings,
        max_components=max_components,
        weight_concentration_prior=weight_concentration_prior,
        covariance_type=covariance_type,
        random_state=random_state,
    )

    bayes_metrics, true_labels_bayes_filtered, bayes_labels_filtered, discarded_pr_bayes = evaluate_clustering(
        true_labels=true_labels,
        cluster_labels=bayes_labels,
        discard_noise=False
    )

    print("\n==============================================================")
    print("RISULTATI BAYESIAN GAUSSIAN MIXTURE")
    print("==============================================================")
    print_metrics("Metriche Bayesian GMM:", bayes_metrics)
    print(f"Numero di cluster trovati: {len(set(bayes_labels))}")
    print(f"Cluster labels Bayesian GMM: {set(bayes_labels)}")

    # Probabilita' di appartenenza dei primi campioni
    bayes_probs = bayes_model.predict_proba(embeddings)
    print("\nProbabilita' di appartenenza dei primi 5 campioni:")
    print(np.round(bayes_probs[:5], 3))

    save_clustering_output(
        dataset_test=dataset_test,
        cluster_labels=bayes_labels,
        output_csv_path="transformer/clustering_output/output_bayesian_gmm_multiple_scenarios.csv"
    )
# ====================================================================
#                   PARAMETRI MODELLO
# ====================================================================

    emb_size = 64           #dimensione dell'embedding finale prodotto dall'encoder
    hidden_dim = 128        #dimensione del layer nascosto dell'autoencoder
    epochs = 10             #numero di sessioni di training del modello
    learning_rate = 1e-3    #tasso di apprendimento per l'ottimizzazione del modello

# ====================================================================
#                   PARAMETRI CLUSTERING
# ====================================================================
    min_cluster_size = 10   #dimensione minima di un cluster in HDBSCAN
    min_samples = 4         #numero minimo di campioni per considerare una zona densa

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

    # dataset_train.labels = torch.zeros(len(dataset_train.labels), dtype=torch.long)
    # dataset_test.labels = torch.zeros(len(dataset_test.labels), dtype=torch.long)

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

    # train_loader.dataset.labels = torch.zeros(len(train_loader.dataset.labels), dtype=torch.long)
    # test_loader.dataset.labels = torch.zeros(len(test_loader.dataset.labels), dtype=torch.long)

    # train SOLO sugli scenari di training
    model.fit(dataloader=train_loader, epochs=epochs, lr=learning_rate)

    # encoding SOLO degli scenari di test
    embeddings = model.encode_dataloader(dataloader=test_loader)

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    cluster_labels = hdbscan_model.fit_predict(embeddings)

    # True label del test set
    # Servono solo per valutare i cluster trovati           
    true_labels = dataset_test.labels           

    # scarto i campioni considerati rumore da HDBSCAN per valutare le metriche del clustering        

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
    print(f"Probe considerate rumore da HDBSCAN (cluster -1): {discarded_pr} --> {100*(discarded_pr/n_probe_test):.2f}%")
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
            "cluster_probability": hdbscan_model.probabilities_[i],           
        })
                
    df = pd.DataFrame(output_values)            
    df = df.sort_values("true_label")           
    print(df)           
                
    df.to_csv("transformer/clustering_output/output_hdbscan_s0_train_s1_test.csv", index=False)         

    output_values = []
    for i, (features, label, mac_address) in enumerate(dataset_test):
        output_values.append({
            "sample_index": i,
            "mac_address": mac_address,
            "true_label": label,
            "cluster": cluster_labels[i],
            "cluster_probability": hdbscan_model.probabilities_[i],
        })

    df = pd.DataFrame(output_values)
    df = df.sort_values("true_label")
    print(df)

    df.to_csv("transformer/clustering_output/output_hdbscan_s0_train_s1_test.csv", index=False)