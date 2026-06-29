from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
import torch
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from transformer_utils.matrix_autoencoder import MatrixAutoencoder
from transformer_utils.evaluation_metric_calc import calc_evaluation_metrics
from prepare_dataset.probe_dataset import ProbeDataset

if __name__ == '__main__':

# ====================================================================
#   SCENARIO_TEMPLATE
#
#   Metti il path completo con {N} come placeholder per il numero
#   di scenario. Esempio:
#     "/home/giuff/Tesi/Dataset/.../scenario_{N}_full.json"
# ====================================================================

    SCENARIO_TEMPLATE = (
        "/home/giuff/Tesi/TransformerTry/Dataset/dataset_burst_json_veri"
        "/scenario_{N}_burst_features.json"
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
                is_bursts=is_bursts,
                preprocess=preprocess,
                include_mac_features=include_mac_features,
            )
            datasets.append(ds)
        return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

# ====================================================================
#   PARAMETRI DATASET TRAIN E TEST
# ====================================================================

    train_scenarios      = [0, 1]
    test_scenarios       = [2, 3]
    batch_size           = 64
    is_bursts            = True    #se è impostato a False si consiglia di impostare normalize a True, viceversa il contrario
    preprocess           = True
    include_mac_features = False
    normalize            = False

# ====================================================================
#   PARAMETRI MODELLO
# ====================================================================

    emb_size      = 64
    hidden_dim    = 128
    epochs        = 50
    learning_rate = 1e-3

# ====================================================================
#   PARAMETRI CLUSTERING
# ====================================================================

    eps         = 0.01
    min_samples = 4

# ====================================================================
#   CARICAMENTO DATASET
# ====================================================================

    dataset_train = load_scenarios(train_scenarios)
    dataset_test  = load_scenarios(test_scenarios)

    n_features    = len(dataset_train.data[0]) if hasattr(dataset_train, 'data') else len(dataset_train[0][0])
    n_probe_train = len(dataset_train)
    n_probe_test  = len(dataset_test)

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
    model.fit(dataloader=train_loader, epochs=epochs, lr=learning_rate)

    embeddings = model.encode_dataloader(dataloader=test_loader)

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    if normalize:
        scaler = MinMaxScaler()
        embeddings = scaler.fit_transform(embeddings)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = dbscan.fit_predict(embeddings)

    true_labels = dataset_test.labels if hasattr(dataset_test, 'labels') else \
                  [dataset_test[i][1] for i in range(len(dataset_test))]

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
    print("CALCOLO SENZA RUMORE")
    print(f"Probe considerate rumore: {discarded_pr} --> {100*(discarded_pr/n_probe_test):.2f}%")
    print(f"ARI:          {metrics['ari']:.4f}")
    print(f"NMI:          {metrics['nmi']:.4f}")
    print(f"Homogeneity:  {metrics['homogeneity']:.4f}")
    print(f"Completeness: {metrics['completeness']:.4f}")
    print(f"V-measure:    {metrics['v_measure']:.4f}")
    print(f"Numero di classi: {len(set(true_labels))}")
    print(f"Cluster trovati:  {len(set(cluster_labels_filtered))}")

    output_values = []
    for i, (features, label, mac_address) in enumerate(dataset_test):
        output_values.append({
            "sample_index": i,
            "mac_address":  mac_address,
            "true_label":   label,
            "cluster":      cluster_labels[i],
        })

    df = pd.DataFrame(output_values).sort_values("true_label")
    print(df)
    df.to_csv("transformer/clustering_output/output_s0_train_s1_test.csv", index=False)