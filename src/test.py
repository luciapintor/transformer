from torch.utils.data import DataLoader
import torch
import numpy as np

from prepare_dataset.probe_dataset import ProbeDataset


if __name__ == "__main__":
    # ==========================================================
    # PARAMETRI
    # ==========================================================
    test_scenarios = [8, 9, 10]
    base_path = "Dataset/dataset_merged_probes_json/data with labels"
    batch_size = 256
    is_bursts = False
    preprocess = True
    include_mac_features = False

    # ==========================================================
    # LOAD DATASET
    # ==========================================================
    print("[INFO] Loading test dataset...")
    dataset_test = ProbeDataset.from_scenario_list(
        scenario_list=test_scenarios,
        base_path=base_path,
        is_bursts=is_bursts,
        preprocess=preprocess,
        include_mac_features=include_mac_features
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ProbeDataset.collate_probe_batch
    )

    print("\n" + "=" * 60)
    print("SANITY CHECK DATASET / DATALOADER")
    print("=" * 60)

    # ==========================================================
    # CONTROLLO 1: PRIMO SAMPLE DEL DATASET
    # ==========================================================
    first_features, first_label, first_mac = dataset_test[0]

    print("[DATASET] First sample label:", first_label)
    print("[DATASET] First sample mac:", first_mac)
    print("[DATASET] Number of feature keys:", len(first_features))

    print("\n[DATASET] Feature keys:")
    for k in sorted(first_features.keys()):
        print(" -", k)

    print("\n[CHECK] MAC feature values of first sample:")
    for i in range(6):
        print(f"mac_{i} =", first_features[f"mac_{i}"])

    print("\n[CHECK] Unique values in MAC columns over first 20 samples:")
    for i in range(6):
        vals = {dataset_test[j][0][f'mac_{i}'] for j in range(min(20, len(dataset_test)))}
        print(f"mac_{i}: {vals}")

    suspicious_keys = []
    for k in sorted(first_features.keys()):
        kl = k.lower()
        if "label" in kl or "cluster" in kl or "true" in kl or "mac" in kl:
            suspicious_keys.append(k)

    print("\n[CHECK] Suspicious keys found in features:", suspicious_keys)

    if suspicious_keys:
        print("[WARNING] Ci sono chiavi sospette dentro le feature!")
    else:
        print("[OK] Nessuna chiave sospetta trovata nelle feature.")

    # ==========================================================
    # CONTROLLO 2: PRIMO BATCH DEL DATALOADER
    # ==========================================================
    for batch in test_loader:
        X, y, Z = batch

        print("\n[DATALOADER] X shape:", X.shape)
        print("[DATALOADER] y shape:", y.shape)
        print("[DATALOADER] First 10 y:", y[:10].tolist())

        print("\n[CHECK] dataset_test[0][1] =", dataset_test[0][1])
        print("[CHECK] y[0] =", y[0].item())

        print("[CHECK] len(first_features.keys()) =", len(first_features))
        print("[CHECK] X.shape[1] =", X.shape[1])

        feature_names = sorted(first_features.keys())

        print("\n[DATALOADER] First sample reconstructed (prime 20 feature):")
        for i, name in enumerate(feature_names[:20]):
            print(f"{name}: {X[0][i].item()}")

        if Z is not None:
            print("\n[DATALOADER] First MAC bytes from Z:", Z[0])
        else:
            print("\n[DATALOADER] Z is None")

        break

    # ==========================================================
    # CONTROLLO 3: CONFRONTO X DIRETTO vs X DA DATALOADER
    # ==========================================================
    print("\n" + "=" * 60)
    print("CHECK X DIRECT VS X LOADER")
    print("=" * 60)

    records = [dataset_test[i][0] for i in range(len(dataset_test))]
    feature_names = sorted(records[0].keys())

    X_direct = np.array(
        [[record[name] for name in feature_names] for record in records],
        dtype=np.float32
    )

    print("[DIRECT] X_direct shape:", X_direct.shape)

    loader_batches = []
    for batch in test_loader:
        X_batch = batch[0]
        if isinstance(X_batch, torch.Tensor):
            X_batch = X_batch.detach().cpu().numpy()
        loader_batches.append(X_batch)

    X_loader = np.vstack(loader_batches)

    print("[LOADER] X_loader shape:", X_loader.shape)
    print("[CHECK] X_direct == X_loader ?", np.allclose(X_direct, X_loader))

    if X_direct.shape == X_loader.shape:
        diff = np.abs(X_direct - X_loader).max()
        print("[CHECK] Max absolute difference:", diff)

    print("=" * 60)