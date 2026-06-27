from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from transformer_utils.evaluation_metric_calc import calc_evaluation_metrics
from prepare_dataset.probe_dataset import ProbeDataset


def extract_features_from_loader(loader):
    """
    Estrae le feature preprocessate dal DataLoader.

    In questo script NON viene usato nessun encoder:
    le feature vengono date direttamente a DBSCAN.
    """
    features_list = []

    for batch in loader:
        features_batch = batch[0]  # batch[0] contiene le feature

        if isinstance(features_batch, torch.Tensor):
            features_batch = features_batch.detach().cpu().numpy()

        features_list.append(features_batch)

    return np.vstack(features_list)


def print_clustering_results(true_labels, cluster_labels):
    """
    Stampa le metriche di clustering ignorando i punti classificati
    come rumore da DBSCAN, cioè quelli con cluster = -1.
    """
    true_labels_filtered = []
    cluster_labels_filtered = []

    for true_label, cluster_label in zip(true_labels, cluster_labels):
        if cluster_label != -1:
            true_labels_filtered.append(true_label)
            cluster_labels_filtered.append(cluster_label)

    num_classes = len(set(true_labels))
    num_clusters = len(set(cluster_labels_filtered))
    num_noise_points = int(np.sum(cluster_labels == -1))

    print("\n" + "=" * 60)
    print("CLUSTERING RESULTS - NO ENCODER")
    print("=" * 60)

    print(f"Number of samples: {len(true_labels)}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of clusters found, noise excluded: {num_clusters}")
    print(f"Number of noise points: {num_noise_points}")

    if len(cluster_labels_filtered) == 0:
        print("\n[WARNING] DBSCAN ha classificato tutti i punti come rumore.")
        print("          Prova ad aumentare eps oppure a diminuire min_samples.")
        print("=" * 60 + "\n")
        return

    metrics = calc_evaluation_metrics(true_labels_filtered, cluster_labels_filtered)

    print("-" * 60)
    print("Metrics, noise excluded:")
    print(f"ARI: {metrics['ari']:.4f}")
    print(f"NMI: {metrics['nmi']:.4f}")
    print(f"Homogeneity: {metrics['homogeneity']:.4f}")
    print(f"Completeness: {metrics['completeness']:.4f}")
    print(f"V-measure: {metrics['v_measure']:.4f}")

    print("-" * 60)
    print(f"Cluster labels: {set(cluster_labels_filtered)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":

    # ====================================================================
    #                   PERCORSI
    # ====================================================================

    # Cartella principale del progetto transformer/
    project_root = Path(__file__).resolve().parents[1]

    # Dataset usato per il clustering
    base_path = project_root / "Dataset" / "dataset_burst_json_veri"

    # File CSV di output
    output_path = project_root / "clustering_output" / "output_noEncode.csv"

    # ====================================================================
    #                   PARAMETRI DATASET
    # ====================================================================

    # In questa versione NON esistono train e test.
    # Non c'è nessun encoder da addestrare.
    # Si scelgono semplicemente gli scenari su cui fare clustering.
    scenarios_to_cluster = [1,2,3,4]

    batch_size = 256
    is_bursts = True
    preprocess = True

    # False: il MAC non viene usato come feature.
    include_mac_features = False

    # ====================================================================
    #                   PARAMETRI CLUSTERING
    # ====================================================================

    eps = 0.1
    min_samples = 4

    # Se True, applica MinMaxScaler alle feature prima di DBSCAN.
    # Consigliato perché DBSCAN è sensibile alla scala delle feature.
    use_scaler = True

    # ====================================================================
    #                   CARICAMENTO DATASET
    # ====================================================================

    print("[INFO] Loading dataset...")
    print(f"[INFO] Scenarios: {scenarios_to_cluster}")
    print(f"[INFO] Base path: {base_path}")

    dataset = ProbeDataset.from_scenario_list(
        scenario_list=scenarios_to_cluster,
        base_path=str(base_path),
        is_bursts=is_bursts,
        preprocess=preprocess,
        include_mac_features=include_mac_features
    )

    print(f"[INFO] Dataset loaded: {len(dataset)} samples")
    print(f"[INFO] Number of true labels: {len(set(dataset.labels))}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ProbeDataset.collate_probe_batch
    )

    # ====================================================================
    #                   ESTRAZIONE FEATURE
    # ====================================================================

    print("[INFO] Extracting preprocessed features...")
    features = extract_features_from_loader(loader)

    # Protezione contro eventuali NaN o infiniti
    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    print(f"[INFO] Feature matrix shape: {features.shape}")

    # ====================================================================
    #                   NORMALIZZAZIONE
    # ====================================================================

    if use_scaler:
        print("[INFO] Applying MinMaxScaler...")
        scaler = MinMaxScaler()
        features_for_clustering = scaler.fit_transform(features)
    else:
        features_for_clustering = features

    # ====================================================================
    #                   DBSCAN
    # ====================================================================

    print(f"[INFO] Running DBSCAN with eps={eps}, min_samples={min_samples}...")

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='euclidean',
    )

    cluster_labels = dbscan.fit_predict(features_for_clustering)

    # Le label vere servono solo per valutare il clustering.
    # Non vengono usate da DBSCAN.
    true_labels = dataset.labels

    print_clustering_results(true_labels, cluster_labels)

    # ====================================================================
    #                   SALVATAGGIO RISULTATI
    # ====================================================================

    output_values = []

    for sample_index, (_, true_label, mac_address) in enumerate(dataset):
        output_values.append({
            "sample_index": sample_index,
            "mac_address": mac_address,
            "true_label": true_label,
            "cluster": cluster_labels[sample_index],
        })

    df = pd.DataFrame(output_values)
    df = df.sort_values("true_label")

    print(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n[INFO] Results saved to {output_path}")