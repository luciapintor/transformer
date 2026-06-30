from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from transformer_utils.evaluation_metric_calc import calc_evaluation_metrics
from prepare_dataset.probe_dataset import ProbeDataset


def extract_features_from_loader(loader):
    features_list = []
    for batch in loader:
        features_batch = batch[0]
        if isinstance(features_batch, torch.Tensor):
            features_batch = features_batch.detach().cpu().numpy()
        features_list.append(features_batch)
    return np.vstack(features_list)


def print_clustering_results(true_labels, cluster_labels):
    true_labels_filtered    = []
    cluster_labels_filtered = []

    for true_label, cluster_label in zip(true_labels, cluster_labels):
        if cluster_label != -1:
            true_labels_filtered.append(true_label)
            cluster_labels_filtered.append(cluster_label)

    n_total           = len(true_labels)
    num_classes        = len(set(true_labels))
    num_clusters        = len(set(cluster_labels_filtered))
    num_noise_points     = int(np.sum(cluster_labels == -1))
    noise_percentage     = 100 * num_noise_points / n_total if n_total > 0 else 0.0

    print("\n" + "=" * 60)
    print("CLUSTERING RESULTS - NO ENCODER")
    print("=" * 60)
    print(f"Number of samples:                        {n_total}")
    print(f"Number of classes:                        {num_classes}")
    print(f"Number of clusters found (noise excluded):{num_clusters}")
    print(f"Error (classes - clusters):               {num_classes - num_clusters}")
    print(f"Number of noise points:                   {num_noise_points}")
    print(f"Percentage of probes as noise:             {noise_percentage:.2f}%")

    if len(cluster_labels_filtered) == 0:
        print("\n[WARNING] DBSCAN ha classificato tutti i punti come rumore.")
        print("          Prova ad aumentare eps oppure a diminuire min_samples.")
        print("=" * 60 + "\n")
        return

    metrics = calc_evaluation_metrics(true_labels_filtered, cluster_labels_filtered)

    print("-" * 60)
    print("Metrics, noise excluded:")
    print(f"ARI:          {metrics['ari']:.4f}")
    print(f"NMI:          {metrics['nmi']:.4f}")
    print(f"Homogeneity:  {metrics['homogeneity']:.4f}")
    print(f"Completeness: {metrics['completeness']:.4f}")
    print(f"V-measure:    {metrics['v_measure']:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":

# ====================================================================
#   SCENARIO_TEMPLATE
#
#   Metti il path completo con {N} come placeholder per il numero
#   di scenario. Esempio:
#     "/home/giuff/Tesi/Dataset/.../scenario_{N}_full.json"
# ====================================================================

    SCENARIO_TEMPLATE = (
        "/home/giuff/Tesi/TransformerTry/Dataset/dataset_merged_probes_json/data with labels"
        "/scenario_{N}_full.json"
    )

    def load_scenarios(scenario_list):
        datasets = []
        for n in scenario_list:
            path = SCENARIO_TEMPLATE.replace("{N}", str(n))
            if not Path(path).exists():
                raise FileNotFoundError(f"Scenario {n} non trovato: {path}")
            print(f"  Carico scenario {n}: {Path(path).name}")
            ds = ProbeDataset(
                path_json=path,
                preprocess=preprocess,
                include_mac_features=include_mac_features,
            )
            datasets.append(ds)
        return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

# ====================================================================
#   PARAMETRI DATASET
# ====================================================================

    scenarios_to_cluster     = [0,1,2,3]
    batch_size               = 256
    preprocess                = True
    include_mac_features      = False
    remove_constant_features  = True

# ====================================================================
#   PARAMETRI CLUSTERING
# ====================================================================

    use_scaler  = True
    eps = 0.001
    # -------------------------------------------------------------------
    # MIN_SAMPLES DINAMICO
    # -------------------------------------------------------------------
    MIN_SAMPLES_COEF = 0.28
    MIN_SAMPLES_COEF = 0
    MIN_SAMPLES_FLOOR = 4   # non scendere mai sotto questo valore
    

# ====================================================================
#   OUTPUT
# ====================================================================

    output_path = SRC_ROOT / "clustering_output" / "output_noEncode.csv"

# ====================================================================
#   CARICAMENTO DATASET
# ====================================================================

    print("[INFO] Loading dataset...")
    print(f"[INFO] Scenarios: {scenarios_to_cluster}")

    dataset = load_scenarios(scenarios_to_cluster)

    print(f"[INFO] Dataset loaded: {len(dataset)} samples")
    true_labels = dataset.labels if hasattr(dataset, 'labels') else \
                  [dataset[i][1] for i in range(len(dataset))]
    print(f"[INFO] Number of true labels: {len(set(true_labels))}")

    # Calcolo dinamico di min_samples in base alla dimensione del dataset
    import math
    n_samples   = len(dataset)
    min_samples = max(MIN_SAMPLES_FLOOR,
                      int(MIN_SAMPLES_COEF * math.sqrt(n_samples)))
    print(f"[INFO] min_samples dinamico: {MIN_SAMPLES_COEF} * sqrt({n_samples}) "
          f"= {MIN_SAMPLES_COEF * math.sqrt(n_samples):.1f} → {min_samples}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ProbeDataset.collate_probe_batch
    )

# ====================================================================
#   ESTRAZIONE FEATURE E NORMALIZZAZIONE
# ====================================================================

    print("[INFO] Extracting preprocessed features...")
    features = extract_features_from_loader(loader)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"[INFO] Feature matrix shape before selection: {features.shape}")

# ====================================================================
#   RIMOZIONE FEATURE COSTANTI
# ====================================================================

    if remove_constant_features:
        feature_variance = np.var(features, axis=0)
        feature_mask     = feature_variance > 0

        n_original_features = features.shape[1]
        n_selected_features = int(np.sum(feature_mask))
        n_removed_features  = n_original_features - n_selected_features

        features = features[:, feature_mask]

        print("[INFO] Removing constant features...")
        print(f"[INFO] Original features: {n_original_features}")
        print(f"[INFO] Selected features: {n_selected_features}")
        print(f"[INFO] Removed constant features: {n_removed_features}")
        print(f"[INFO] Feature matrix shape after selection: {features.shape}")

    if use_scaler:
        print("[INFO] Applying MinMaxScaler...")
        features = MinMaxScaler().fit_transform(features)

# ====================================================================
#   DBSCAN
# ====================================================================

    print(f"[INFO] Running DBSCAN (eps={eps}, min_samples={min_samples})...")
    cluster_labels = DBSCAN(eps=eps, min_samples=min_samples,
                             metric='euclidean').fit_predict(features)

    print_clustering_results(true_labels, cluster_labels)

# ====================================================================
#   SALVATAGGIO RISULTATI
# ====================================================================

    output_values = []
    for i, (_, true_label, mac_address) in enumerate(dataset):
        output_values.append({
            "sample_index": i,
            "mac_address":  mac_address,
            "true_label":   true_label,
            "cluster":      cluster_labels[i],
        })

    df = pd.DataFrame(output_values).sort_values("true_label")
    print(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[INFO] Results saved to {output_path}")