from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from transformer_utils.matrix_autoencoder import MatrixAutoencoder
from transformer_utils.evaluation_metric_calc import calc_evaluation_metrics
from prepare_dataset.probe_dataset import ProbeDataset


# ====================================================================
#                   FUNZIONI DI SUPPORTO
# ====================================================================

def print_metrics(title, metrics):
    print(title)
    print(f"ARI: {metrics['ari']:.4f}")
    print(f"NMI: {metrics['nmi']:.4f}")
    print(f"Homogeneity: {metrics['homogeneity']:.4f}")
    print(f"Completeness: {metrics['completeness']:.4f}")
    print(f"V-measure: {metrics['v_measure']:.4f}")


def evaluate_without_noise(true_labels, cluster_labels, n_probe_test):
    """
    Valuta il clustering scartando i campioni con label -1.
    Serve soprattutto per DBSCAN, dove -1 indica rumore.
    """
    true_labels_filtered = []
    cluster_labels_filtered = []
    discarded_pr = 0

    for t, c in zip(true_labels, cluster_labels):
        if c != -1:
            true_labels_filtered.append(t)
            cluster_labels_filtered.append(c)
        else:
            discarded_pr += 1

    metrics = calc_evaluation_metrics(true_labels_filtered, cluster_labels_filtered)

    return metrics, true_labels_filtered, cluster_labels_filtered, discarded_pr


def save_clustering_output(dataset_test, cluster_labels, output_csv_path):
    """
    Salva un CSV con sample_index, mac_address, true_label e cluster predetto.
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

    print(f"\nCSV salvato in: {output_csv_path}")

    return df


def prepare_embeddings_for_bayesian_gmm(
    embeddings,
    pca_components=20,
    random_state=42,
):
    """
    Prepara gli embedding per il Bayesian GMM.

    Passaggi:
    1. conversione a numpy
    2. StandardScaler
    3. PCA opzionale

    La PCA serve a rendere il Bayesian GMM piu' stabile, specialmente se
    gli embedding hanno molte dimensioni o dimensioni quasi ridondanti.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    else:
        embeddings = np.asarray(embeddings)

    print("\n=== PREPROCESSING EMBEDDING PER BAYESIAN GMM ===")
    print(f"Shape embedding originali: {embeddings.shape}")

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    n_samples, n_features = embeddings_scaled.shape

    # PCA non puo' avere piu' componenti di min(n_samples, n_features)
    max_pca_components = min(n_samples, n_features)
    pca_components = min(pca_components, max_pca_components)

    if pca_components is not None and pca_components < n_features:
        pca = PCA(n_components=pca_components, random_state=random_state)
        embeddings_bayes = pca.fit_transform(embeddings_scaled)

        print(f"Shape embedding dopo PCA: {embeddings_bayes.shape}")
        print(f"Varianza spiegata PCA: {np.sum(pca.explained_variance_ratio_):.4f}")

        return embeddings_bayes, scaler, pca

    print("PCA non applicata: uso embedding scalati.")
    return embeddings_scaled, scaler, None


def run_bayesian_gmm_clustering(
    embeddings,
    max_components=60,
    weight_concentration_prior=10.0,
    covariance_type="diag",
    reg_covar=1e-3,
    random_state=42,
):
    """
    Clustering bayesiano con BayesianGaussianMixture.

    Nota importante:
    - n_components e' il numero MASSIMO di componenti.
    - con dirichlet_distribution il modello resta bayesiano, ma tende meno
      a spegnere quasi tutto rispetto a dirichlet_process con prior piccolo.
    """
    bayes_gmm = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type=covariance_type,
        weight_concentration_prior_type="dirichlet_distribution",
        weight_concentration_prior=weight_concentration_prior,
        reg_covar=reg_covar,
        max_iter=1000,
        n_init=5,
        random_state=random_state,
    )

    cluster_labels = bayes_gmm.fit_predict(embeddings)

    weights = bayes_gmm.weights_
    components_with_weight = int(np.sum(weights > 1e-3))
    assigned_clusters = len(set(cluster_labels))

    print("\n=== INFO BAYESIAN GMM ===")
    print(f"Numero massimo componenti: {max_components}")
    print(f"Covariance type: {covariance_type}")
    print(f"reg_covar: {reg_covar}")
    print("weight_concentration_prior_type: dirichlet_distribution")
    print(f"weight_concentration_prior: {weight_concentration_prior}")
    print(f"Componenti con peso > 1e-3: {components_with_weight}")
    print(f"Cluster effettivamente assegnati: {assigned_clusters}")
    print("Pesi componenti:")
    print(np.round(weights, 4))

    return cluster_labels, bayes_gmm


if __name__ == '__main__':

    # ====================================================================
    #                   PARAMETRI DATASET TRAIN E TEST
    # ====================================================================

    train_scenarios = [0, 1]
    test_scenarios = [2, 3]
    base_path = "Dataset/dataset_burst_json_veri"
    batch_size = 64
    is_bursts = True
    preprocess = True
    include_mac_features = False

    # ====================================================================
    #                   PARAMETRI MODELLO
    # ====================================================================

    emb_size = 64
    hidden_dim = 128
    epochs = 10
    learning_rate = 1e-3
    random_state = 42

    # ====================================================================
    #                   PARAMETRI DBSCAN
    # ====================================================================

    eps = 0.1
    min_samples = 4

    # ====================================================================
    #                   PARAMETRI BAYESIAN GMM
    # ====================================================================

    max_components = 80
    weight_concentration_prior = 100.0
    covariance_type = "spherical"
    reg_covar = 1e-4
    
    pca_components_bayes = 30

    # ====================================================================
    #                   CARICAMENTO DATASET
    # ====================================================================

    dataset_train = ProbeDataset.from_scenario_list(
        scenario_list=train_scenarios,
        base_path=base_path,
        is_bursts=is_bursts,
        preprocess=preprocess,
        include_mac_features=include_mac_features,
    )

    dataset_test = ProbeDataset.from_scenario_list(
        scenario_list=test_scenarios,
        base_path=base_path,
        is_bursts=is_bursts,
        preprocess=preprocess,
        include_mac_features=include_mac_features,
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
        collate_fn=ProbeDataset.collate_probe_batch,
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ProbeDataset.collate_probe_batch,
    )

    # ====================================================================
    #                   TRAINING AUTOENCODER
    # ====================================================================

    model = MatrixAutoencoder(
        n_features,
        emb_size=emb_size,
        hidden_dim=hidden_dim,
    )

    # Train solo sugli scenari di training
    model.fit(
        dataloader=train_loader,
        epochs=epochs,
        lr=learning_rate,
    )

    # Encoding solo degli scenari di test
    embeddings = model.encode_dataloader(dataloader=test_loader)

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    else:
        embeddings = np.asarray(embeddings)

    true_labels = dataset_test.labels

    # ====================================================================
    #                   DBSCAN SU EMBEDDING ORIGINALI
    # ====================================================================

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(embeddings)

    dbscan_metrics, true_labels_filtered, dbscan_labels_filtered, discarded_pr = evaluate_without_noise(
        true_labels=true_labels,
        cluster_labels=dbscan_labels,
        n_probe_test=n_probe_test,
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
        output_csv_path="transformer/clustering_output/output_dbscan.csv",
    )

    # ====================================================================
    #                   BAYESIAN GMM SU EMBEDDING SCALATI + PCA
    # ====================================================================

    embeddings_bayes, bayes_scaler, bayes_pca = prepare_embeddings_for_bayesian_gmm(
        embeddings=embeddings,
        pca_components=pca_components_bayes,
        random_state=random_state,
    )

    bayes_labels, bayes_model = run_bayesian_gmm_clustering(
        embeddings=embeddings_bayes,
        max_components=max_components,
        weight_concentration_prior=weight_concentration_prior,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        random_state=random_state,
    )

    bayes_metrics = calc_evaluation_metrics(true_labels, bayes_labels)

    print("\n==============================================================")
    print("RISULTATI BAYESIAN GAUSSIAN MIXTURE")
    print("==============================================================")
    print_metrics("Metriche Bayesian GMM:", bayes_metrics)
    print(f"Numero di cluster trovati: {len(set(bayes_labels))}")
    print(f"Cluster labels Bayesian GMM: {set(bayes_labels)}")

    probs = bayes_model.predict_proba(embeddings_bayes)
    print("\nProbabilita' di appartenenza dei primi 5 campioni:")
    print(np.round(probs[:5], 3))

    save_clustering_output(
        dataset_test=dataset_test,
        cluster_labels=bayes_labels,
        output_csv_path="transformer/clustering_output/output_bayesian_gmm.csv",
    )

    # ====================================================================
    #                   RIEPILOGO FINALE
    # ====================================================================

    print("\n==============================================================")
    print("RIEPILOGO")
    print("==============================================================")
    print(f"Classi reali nel test set: {len(set(true_labels))}")
    print(f"Cluster DBSCAN senza rumore: {len(set(dbscan_labels_filtered))}")
    print(f"Cluster Bayesian GMM: {len(set(bayes_labels))}")
    print(f"ARI DBSCAN: {dbscan_metrics['ari']:.4f}")
    print(f"ARI Bayesian GMM: {bayes_metrics['ari']:.4f}")