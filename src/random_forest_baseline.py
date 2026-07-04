from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from prepare_dataset.probe_dataset import ProbeDataset


# ============================================================================
#   RANDOM FOREST - IMPORTANZA FEATURE / INFORMATION ELEMENT
#
#   Questo script NON serve per fare clustering e NON serve per valutare il
#   conteggio dei dispositivi.
#
#   Serve solo a rispondere alla domanda:
#   "Quali feature / IE sono più importanti per distinguere le label note?"
#
#   La Random Forest viene addestrata sugli scenari etichettati indicati in
#   train_scenarios. Dopo il training vengono salvati:
#
#   1) feature_importance_random_forest.csv
#      Importanza delle singole feature preprocessate.
#
#   2) ie_importance_random_forest.csv
#      Importanza aggregata per IE/gruppo logico.
# ============================================================================


def dataset_to_numpy(dataset, feature_names=None):
    """Converte ProbeDataset/ConcatDataset in X, y."""
    if feature_names is None:
        feature_names = sorted(dataset[0][0].keys())

    X = []
    y = []

    for i in range(len(dataset)):
        record, label, _ = dataset[i]
        X.append([record.get(name, 0.0) for name in feature_names])
        y.append(label)

    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y)

    return X, y


def load_scenarios(scenario_list, scenario_template, preprocess, include_mac_features):
    datasets = []

    for n in scenario_list:
        path = Path(scenario_template.replace("{N}", str(n)))
        if not path.exists():
            raise FileNotFoundError(f"Scenario {n} non trovato: {path}")

        print(f"  Carico scenario {n}: {path.name}")
        ds = ProbeDataset(
            path_json=path,
            preprocess=preprocess,
            include_mac_features=include_mac_features,
        )
        datasets.append(ds)

    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def feature_to_ie_group(feature_name):
    """
    Mappa una feature preprocessata al relativo IE/gruppo logico.

    Modifica questa funzione se nel tuo preprocessing usi nomi diversi.
    """
    name = feature_name.lower()

    # IE 0 - SSID
    if name.startswith("ie0") or "ssid" in name:
        return "IE 0 - SSID"

    # IE 1 - Supported Rates
    if name.startswith("ie1") or "supported_rates" in name or "support_rates" in name:
        return "IE 1 - Supported Rates"

    # IE 45 - HT Capabilities
    if name.startswith("ht_") or "ht_cap" in name or "rx_mcs" in name:
        return "IE 45 - HT Capabilities"

    # IE 50 - Extended Supported Rates
    if name.startswith("ie50") or "extended_supported_rates" in name or "ext_supported_rates" in name:
        return "IE 50 - Extended Supported Rates"

    # IE 107 - Interworking
    if name.startswith("interworking") or name.startswith("ie107"):
        return "IE 107 - Interworking"

    # IE 127 - Extended Capabilities
    if name.startswith("extcap") or name.startswith("extended_cap") or name.startswith("ie127"):
        return "IE 127 - Extended Capabilities"

    # IE 191 - VHT Capabilities
    if name.startswith("vht") or name.startswith("ie191"):
        return "IE 191 - VHT Capabilities"

    # IE 221 - Vendor Specific
    if name.startswith("vendor") or name.startswith("ie221") or "oui" in name:
        return "IE 221 - Vendor Specific"

    # Feature generiche di presenza/lunghezza IE
    if "present" in name or "length" in name or name.startswith("ie"):
        return "Presenza/lunghezza IE"

    return "Altro"


if __name__ == "__main__":

# ============================================================================
#   PARAMETRI DATASET
# ============================================================================

    SCENARIO_TEMPLATE = (
        "/home/giuff/Tesi/TransformerTry/Dataset/dataset_merged_probes_json/data with labels"
        "/scenario_{N}_full.json"
    )

    # Usa solo scenari etichettati per addestrare la RF e stimare le importanze.
    # Qui non stiamo valutando il modello su dispositivi nuovi.
    train_scenarios = [0, 1, 2]

    preprocess = True
    include_mac_features = False
    remove_constant_features = True
    use_scaler = True

# ============================================================================
#   PARAMETRI RANDOM FOREST
# ============================================================================

    n_estimators = 500
    max_depth = None
    min_samples_leaf = 1
    class_weight = "balanced"
    random_state = 42
    n_jobs = -1

# ============================================================================
#   OUTPUT
# ============================================================================

    output_dir = SRC_ROOT / "clustering_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_importance_path = output_dir / "feature_importance_random_forest.csv"
    ie_importance_path = output_dir / "ie_importance_random_forest.csv"

# ============================================================================
#   CARICAMENTO DATASET
# ============================================================================

    print("[INFO] Loading dataset for feature/IE importance...")
    print(f"[INFO] Train scenarios: {train_scenarios}")
    dataset_train = load_scenarios(
        train_scenarios,
        SCENARIO_TEMPLATE,
        preprocess,
        include_mac_features,
    )

# ============================================================================
#   ESTRAZIONE FEATURE
# ============================================================================

    feature_names = sorted(dataset_train[0][0].keys())
    X_train, y_train = dataset_to_numpy(dataset_train, feature_names)

    print(f"[INFO] X_train shape before selection: {X_train.shape}")
    print(f"[INFO] Number of labels/devices in train: {len(set(y_train))}")

# ============================================================================
#   RIMOZIONE FEATURE COSTANTI FIT SOLO SUL TRAIN
# ============================================================================

    if remove_constant_features:
        feature_variance = np.var(X_train, axis=0)
        feature_mask = feature_variance > 0

        selected_feature_names = [
            name for name, keep in zip(feature_names, feature_mask)
            if keep
        ]

        X_train = X_train[:, feature_mask]

        print("[INFO] Removing constant features using TRAIN set...")
        print(f"[INFO] Original features: {len(feature_names)}")
        print(f"[INFO] Selected features: {len(selected_feature_names)}")
        print(f"[INFO] Removed constant features: {len(feature_names) - len(selected_feature_names)}")
    else:
        selected_feature_names = feature_names

# ============================================================================
#   NORMALIZZAZIONE
# ============================================================================

    if use_scaler:
        print("[INFO] Applying MinMaxScaler fitted on TRAIN set...")
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)

# ============================================================================
#   TRAINING RANDOM FOREST
# ============================================================================

    print("[INFO] Training Random Forest for feature importance...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    print("[INFO] Training completed.")

# ============================================================================
#   IMPORTANZA SINGOLE FEATURE
# ============================================================================

    df_feature_importance = pd.DataFrame({
        "feature": selected_feature_names,
        "ie_group": [feature_to_ie_group(name) for name in selected_feature_names],
        "importance": model.feature_importances_,
    })

    df_feature_importance = df_feature_importance.sort_values(
        "importance",
        ascending=False,
    )

# ============================================================================
#   IMPORTANZA AGGREGATA PER IE / GRUPPO LOGICO
# ============================================================================

    df_ie_importance = (
        df_feature_importance
        .groupby("ie_group", as_index=False)
        .agg(
            total_importance=("importance", "sum"),
            mean_importance=("importance", "mean"),
            n_features=("feature", "count"),
        )
        .sort_values("total_importance", ascending=False)
    )

    # Percentuale rispetto all'importanza totale, che per Random Forest vale circa 1.
    total = df_ie_importance["total_importance"].sum()
    df_ie_importance["importance_percent"] = 100.0 * df_ie_importance["total_importance"] / total

    df_ie_importance = df_ie_importance[
        ["ie_group", "total_importance", "importance_percent", "mean_importance", "n_features"]
    ]

# ============================================================================
#   STAMPA RISULTATI
# ============================================================================

    print("\n" + "=" * 80)
    print("IMPORTANZA AGGREGATA PER IE / GRUPPO LOGICO")
    print("=" * 80)
    print(df_ie_importance.to_string(index=False))

    print("\n" + "=" * 80)
    print("TOP 30 FEATURE SINGOLE")
    print("=" * 80)
    print(df_feature_importance.head(30).to_string(index=False))

# ============================================================================
#   SALVATAGGIO CSV
# ============================================================================

    df_feature_importance.to_csv(feature_importance_path, index=False)
    df_ie_importance.to_csv(ie_importance_path, index=False)

    print(f"\n[INFO] Feature importance saved to: {feature_importance_path}")
    print(f"[INFO] IE importance saved to:      {ie_importance_path}")