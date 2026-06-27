from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN

from transformer_utils.newEncoder import MatrixAutoencoder
from transformer_utils.evaluation_metric_calc import calc_evaluation_metrics
from prepare_dataset.probe_dataset import ProbeDataset


if __name__ == '__main__':

# ====================================================================
#   PARAMETRI DATASET
#
#   Si usano scenari diversi per training e test, in modo che il modello
#   venga valutato su device mai visti durante il training.
# ====================================================================

    train_scenarios      = [0, 1]
    test_scenarios       = [2, 3]
    base_path            = "Dataset/dataset_burst_json_veri"
    batch_size           = 64
    is_bursts            = True    # ogni record è un burst di PR (non una singola PR)
    preprocess           = True    # applica ie_to_transformerIE → vettore 140 feature
    include_mac_features = False   # il MAC non entra nelle feature (è la label)

# ====================================================================
#   PARAMETRI MODELLO
# ====================================================================

    emb_size      = 64     # dimensione dello spazio latente dell'encoder
    hidden_dim    = 128    # dimensione del layer nascosto intermedio
    epochs        = 30
    learning_rate = 1e-3

# ====================================================================
#   PARAMETRI CLUSTERING
#
#   eps viene stimato automaticamente ad ogni epoch dentro fit_clustering
#   tramite il metodo delle k-distanze, quindi il valore qui sotto viene
#   usato solo da DBSCAN sul test set finale. Viene sovrascritto con
#   l'eps stimato sugli embedding del test set.
#   min_samples: numero minimo di PR in un intorno per formare un cluster.
# ====================================================================

    min_samples = 4

# ====================================================================
#   CARICAMENTO DATASET
#
#   ProbeDataset.from_scenario_list carica e concatena i JSON di tutti
#   gli scenari indicati. Con preprocess=True, ogni record viene passato
#   attraverso ie_to_transformerIE.preprocess_burst, che converte le IE
#   grezze in un vettore numerico di 140 feature.
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

    # NOTA IMPORTANTE: le label NON vanno azzerate.
    # Servono a fit_clustering per la surrogate loss (prototype/contrastive).
    # Nel vecchio codice venivano azzerate con torch.zeros(...), rendendo
    # il training completamente cieco: la prototype_loss riceveva una sola
    # classe e restituiva sempre 0. → BUG RIMOSSO.

    n_features    = len(dataset_train.data[0])  # 140 con preprocess=True
    n_probe_train = len(dataset_train.data)
    n_probe_test  = len(dataset_test.data)

    # I DataLoader gestiscono il batching. shuffle=True in training
    # aiuta la surrogate loss a vedere campioni di classi diverse
    # nello stesso batch. In test shuffle=False per mantenere l'ordine.
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
#
#   fit_clustering ottimizza la loss combinata:
#     total = surrogate_loss * 0.7  +  recon_loss * 0.3
#
#   Ad ogni epoch stima automaticamente eps sugli embedding correnti
#   e lo usa per valutare DBSCAN (solo per monitoraggio).
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
#
#   encode_dataloader passa tutte le PR del test set attraverso l'encoder
#   addestrato e restituisce gli embedding (n_test, emb_size).
#   Nessuno scaling viene applicato: DBSCAN opera direttamente nello
#   spazio euclideo dell'encoder, dove le distanze sono significative.
# ====================================================================

    enc_out = model.encode_dataloader(dataloader=test_loader)

    if isinstance(enc_out, tuple):
        embeddings, returned_labels = enc_out
        true_labels = returned_labels.detach().cpu().numpy() \
                      if isinstance(returned_labels, torch.Tensor) \
                      else np.array(returned_labels)
    else:
        embeddings  = enc_out
        true_labels = np.array(dataset_test.labels)

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # Stima eps sugli embedding del test set con lo stesso metodo usato
    # durante il training: k-distanze con k = min_samples, percentile 90.
    # Questo garantisce che eps sia calibrato sulla geometria attuale degli
    # embedding, senza bisogno di trovarlo a mano.
    eps_test, k_distances = MatrixAutoencoder.estimate_eps(
        embeddings, k=min_samples, percentile=90
    )
    print(f"\nEps stimato sul test set: {eps_test:.4f} "
          f"(k-dist min={k_distances.min():.4f}, "
          f"max={k_distances.max():.4f}, "
          f"median={np.median(k_distances):.4f})")

    # DBSCAN sugli embedding grezzi (senza scaling) con eps stimato
    dbscan         = DBSCAN(eps=eps_test, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(embeddings)

    # Le true label vengono rilette dal dataset (non sono state azzerate)
    true_labels = dataset_test.labels

# ====================================================================
#   VALUTAZIONE
#
#   DBSCAN assegna label -1 ai punti considerati rumore (outlier).
#   Si filtrano prima di calcolare le metriche, così si valuta solo
#   la qualità del clustering sui punti effettivamente assegnati.
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
    print(f"Classi vere:  {len(set(dataset_test.labels))}")
    print(f"Cluster trovati: {len(set(cluster_labels_filtered))}")
    print("==========================================================\n")

# ====================================================================
#   SALVATAGGIO OUTPUT
# ====================================================================

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
    df.to_csv("transformer/clustering_output/output_newEncoder.csv", index=False)