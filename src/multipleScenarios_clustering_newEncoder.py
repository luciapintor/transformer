from torch.utils.data import DataLoader, ConcatDataset, Dataset
from pathlib import Path
import torch
import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN

from transformer_utils.newEncoder import MatrixAutoencoder
from transformer_utils.evaluation_metric_calc import calc_evaluation_metrics
from prepare_dataset.probe_dataset import ProbeDataset


class ConstantFeatureFilteredDataset(Dataset):
    """Wrapper che applica una maschera di feature senza duplicare tutto il dataset in RAM."""

    def __init__(self, base_dataset, selected_feature_names):
        self.base_dataset = base_dataset
        self.selected_feature_names = list(selected_feature_names)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        record, label, mac_address = self.base_dataset[idx]

        filtered_record = {
            name: record.get(name, 0.0)
            for name in self.selected_feature_names
        }

        return filtered_record, label, mac_address


def compute_non_constant_feature_names(dataset, batch_size, variance_threshold=1e-12):
    """Calcola sul TRAIN le feature non costanti, senza materializzare tutto il dataset."""

    first_record = dataset[0][0]
    feature_names = sorted(first_record.keys())

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ProbeDataset.collate_probe_batch
    )

    n_samples = 0
    feature_sum = None
    feature_sum_sq = None

    for batch in loader:
        features_batch = batch[0]

        if isinstance(features_batch, torch.Tensor):
            features_batch = features_batch.detach().cpu().numpy()

        features_batch = np.asarray(features_batch, dtype=np.float64)
        features_batch = np.nan_to_num(features_batch, nan=0.0, posinf=0.0, neginf=0.0)

        if feature_sum is None:
            n_features = features_batch.shape[1]
            feature_sum = np.zeros(n_features, dtype=np.float64)
            feature_sum_sq = np.zeros(n_features, dtype=np.float64)

        feature_sum += np.sum(features_batch, axis=0)
        feature_sum_sq += np.sum(features_batch ** 2, axis=0)
        n_samples += features_batch.shape[0]

    feature_mean = feature_sum / n_samples
    feature_variance = (feature_sum_sq / n_samples) - (feature_mean ** 2)
    feature_mask = feature_variance > variance_threshold

    selected_feature_names = [
        name for name, keep in zip(feature_names, feature_mask)
        if keep
    ]

    n_original_features = len(feature_names)
    n_selected_features = len(selected_feature_names)
    n_removed_features = n_original_features - n_selected_features

    print("[INFO] Removing constant features using TRAIN set...")
    print(f"[INFO] Original features: {n_original_features}")
    print(f"[INFO] Selected features: {n_selected_features}")
    print(f"[INFO] Removed constant features: {n_removed_features}")

    return selected_feature_names


if __name__ == '__main__':

# ====================================================================
#   SCENARIO_TEMPLATE
#
#   Metti il path completo con {N} come placeholder per il numero
#   di scenario. Esempio:
#     "/home/giuff/Tesi/Dataset/.../scenario_{N}_full.json"
# ====================================================================

    SCENARIO_TEMPLATE = (
        "/home/giuff/Tesi/Dataset/dataset_merged_probes_json/data with labels"
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

    train_scenarios      = [0, 1]
    test_scenarios       = [2, 3]
    batch_size           = 64
    preprocess           = True
    include_mac_features = False
    remove_constant_features = True

# ====================================================================
#   PARAMETRI MODELLO
# ====================================================================

    emb_size      = 64
    hidden_dim    = 128
    epochs        = 30
    learning_rate = 1e-3

# ====================================================================
#   PARAMETRI CLUSTERING
# ====================================================================

    min_samples = 4

# ====================================================================
#   CARICAMENTO DATASET
# ====================================================================

    dataset_train = load_scenarios(train_scenarios)
    dataset_test  = load_scenarios(test_scenarios)

    n_probe_train = len(dataset_train)
    n_probe_test  = len(dataset_test)

    if remove_constant_features:
        selected_feature_names = compute_non_constant_feature_names(
            dataset_train,
            batch_size=batch_size
        )
        dataset_train = ConstantFeatureFilteredDataset(dataset_train, selected_feature_names)
        dataset_test = ConstantFeatureFilteredDataset(dataset_test, selected_feature_names)
        n_features = len(selected_feature_names)
    else:
        first_record = dataset_train[0][0]
        n_features = len(first_record)

    print(f"[INFO] Number of train samples: {n_probe_train}")
    print(f"[INFO] Number of test samples:  {n_probe_test}")
    print(f"[INFO] Number of input features: {n_features}")

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
#   TRAINING
# ====================================================================

    model = MatrixAutoencoder(n_features, emb_size=emb_size, hidden_dim=hidden_dim)

    model.fit_clustering(
        dataloader=train_loader,
        epochs=epochs,
        lr=learning_rate,
        min_samples=min_samples,
        recon_weight=0.3,
        surrogate='prototype',
        temperature=0.1
    )

# ====================================================================
#   ENCODING DEL TEST SET
# ====================================================================

    print("[INFO] Encoding test set...")
    enc_out = model.encode_dataloader(dataloader=test_loader)
    print("[INFO] Encoding completed.")

    if isinstance(enc_out, tuple):
        embeddings, returned_labels = enc_out
        true_labels = returned_labels.detach().cpu().numpy() \
                      if isinstance(returned_labels, torch.Tensor) \
                      else np.array(returned_labels)
    else:
        embeddings  = enc_out
        true_labels = np.array([dataset_test[i][1] for i in range(len(dataset_test))])

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    embeddings = embeddings.astype(np.float32)
    print(f"[INFO] Embeddings shape: {embeddings.shape}")

    eps_test, k_distances = MatrixAutoencoder.estimate_eps(
        embeddings, k=min_samples, percentile=90
    )
    print(f"\nEps stimato sul test set: {eps_test:.4f} "
          f"(k-dist min={k_distances.min():.4f}, "
          f"max={k_distances.max():.4f}, "
          f"median={np.median(k_distances):.4f})")

    print(f"[INFO] Running DBSCAN (eps={eps_test:.4f}, min_samples={min_samples})...")
    dbscan = DBSCAN(
        eps=eps_test,
        min_samples=min_samples,
        n_jobs=1
    )
    cluster_labels = dbscan.fit_predict(embeddings)
    print("[INFO] DBSCAN completed.")

# ====================================================================
#   VALUTAZIONE
# ====================================================================

    true_labels_filtered    = []
    cluster_labels_filtered = []
    discarded_pr            = 0

    for t, c in zip(true_labels, cluster_labels):
        if c != -1:
            true_labels_filtered.append(t)
            cluster_labels_filtered.append(c)
        else:
            discarded_pr += 1

    metrics = calc_evaluation_metrics(true_labels_filtered, cluster_labels_filtered)

    print("\n========== RISULTATI CLUSTERING (senza rumore) ==========")
    print(f"Probe scartate come rumore: {discarded_pr} / {n_probe_test} "
          f"({100 * discarded_pr / n_probe_test:.2f}%)")
    print(f"ARI:          {metrics['ari']:.4f}")
    print(f"NMI:          {metrics['nmi']:.4f}")
    print(f"Homogeneity:  {metrics['homogeneity']:.4f}")
    print(f"Completeness: {metrics['completeness']:.4f}")
    print(f"V-measure:    {metrics['v_measure']:.4f}")
    print(f"Classi vere:  {len(set(true_labels))}")
    print(f"Cluster trovati: {len(set(cluster_labels_filtered))}")
    print("==========================================================\n")

# ====================================================================
#   SALVATAGGIO OUTPUT
# ====================================================================

    output_values = []
    for i, (_, label, mac_address) in enumerate(dataset_test):
        output_values.append({
            "sample_index": i,
            "mac_address":  mac_address,
            "true_label":   label,
            "cluster":      cluster_labels[i],
        })

    df = pd.DataFrame(output_values).sort_values("true_label")
    print(df)

    output_path = Path("transformer/clustering_output/output_newEncoder.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Results saved to {output_path}")
