"""
Attributi classe:
path: percorso del JSON a cui l'istanza fa riferimento
data: lista di dizionari, ciascuno rappresentante un record del JSON
labels: lista di etichette corrispondenti ai record in data
mac_addresses: lista di MAC address corrispondenti ai record in data
"""

import json
import torch
from pathlib import Path

from torch.utils.data import Dataset

try:
    from .ie_to_transformerIE import preprocess_list
except ImportError:
    from ie_to_transformerIE import preprocess_list


class ProbeDataset(Dataset):
    def __init__(
        self,
        path_json,
        preprocess: bool = False,
        include_mac_features: bool = False
    ):
        """
        Carica data/label da JSON per dataset.

        Args:
            path_json: percorso file JSON o directory JSON.
            preprocess: se True, preprocessa subito tutti i record caricati.
            include_mac_features:
                - False -> il MAC NON entra in self.data, ma resta in self.mac_addresses
                - True  -> il MAC entra in self.data, così il preprocessing può trasformarlo
        """
        if include_mac_features and not preprocess:
            raise ValueError(
                "include_mac_features=True richiede preprocess=True, "
                "altrimenti il MAC resterebbe una stringa dentro data"
            )

        self.path = Path(path_json)
        self.include_mac_features = include_mac_features

        self.data, self.labels, self.mac_addresses = self.load_json(
            self.path,
            keep_mac_in_data=include_mac_features
        )

        if preprocess:
            self.preprocess_data()

    def load_json_dict(self, obj: dict, keep_mac_in_data: bool = False):
        """Converte un JSON con root dict nel formato interno (data, labels, mac_addresses)."""
        data = []
        labels = []
        mac_addresses = []

        for sample_id, record in obj.items():
            if not isinstance(record, dict):
                raise ValueError(f"Record {sample_id} non valido: atteso dict")

            if "label" not in record:
                raise ValueError(f"Record {sample_id} senza label")

            burst_record = dict(record)
            label = burst_record.pop("label")
            mac_address = sample_id

            if keep_mac_in_data:
                burst_record["mac"] = mac_address
            else:
                burst_record.pop("mac", None)

            data.append(burst_record)
            labels.append(label)
            mac_addresses.append(mac_address)

        return data, labels, mac_addresses

    def load_json_list(self, obj: list, keep_mac_in_data: bool = False):
        """Converte un JSON con root list nel formato interno (data, labels, mac_addresses)."""
        data = []
        labels = []
        mac_addresses = []

        for i, record in enumerate(obj):
            if not isinstance(record, dict):
                raise ValueError(f"Record in posizione {i} non valido: atteso dict")

            if "label" not in record:
                raise ValueError(f"Record in posizione {i} senza label")

            if "mac" not in record:
                raise ValueError(f"Record in posizione {i} senza mac")

            burst_record = dict(record)
            label = burst_record.pop("label")
            mac_address = burst_record.get("mac")

            if not keep_mac_in_data:
                burst_record.pop("mac", None)

            data.append(burst_record)
            labels.append(label)
            mac_addresses.append(mac_address)

        return data, labels, mac_addresses

    def load_json(self, path_json, keep_mac_in_data: bool = False):
        """
        Carica un file JSON o una directory di JSON e ritorna (data, labels, mac_addresses).

        Supporta due strutture JSON:
        1) dict: {mac_address: record}
        2) list: [{..., "mac": ..., "label": ...}, ...]
        """
        p = Path(path_json)
        if not p.exists():
            raise FileNotFoundError(f"Percorso non trovato: {path_json}")

        data = []
        labels = []
        mac_addresses = []

        def extend_dataset(partial_data, partial_labels, partial_mac_addresses):
            data.extend(partial_data)
            labels.extend(partial_labels)
            mac_addresses.extend(partial_mac_addresses)

        def load_single_json(json_file: Path):
            with open(json_file, "r", encoding="utf-8") as f:
                obj = json.load(f)

            if isinstance(obj, dict):
                return self.load_json_dict(obj, keep_mac_in_data=keep_mac_in_data)

            if isinstance(obj, list):
                return self.load_json_list(obj, keep_mac_in_data=keep_mac_in_data)

            raise ValueError(
                f"File JSON non valido: {json_file}. "
                f"Root atteso dict o list, trovato {type(obj).__name__}"
            )

        if p.is_file():
            partial_data, partial_labels, partial_mac_addresses = load_single_json(p)
            extend_dataset(partial_data, partial_labels, partial_mac_addresses)
        else:
            for json_file in sorted(p.glob("*.json")):
                partial_data, partial_labels, partial_mac_addresses = load_single_json(json_file)
                extend_dataset(partial_data, partial_labels, partial_mac_addresses)

        return data, labels, mac_addresses

    def preprocess_data(self):
        """
        Preprocessa tutti i record del dataset usando ie_to_transformerIE.
        """
        if not isinstance(self.data, list):
            raise TypeError(f"self.data deve essere una lista, trovato {type(self.data).__name__}")

        self.data = preprocess_list(self.data)

        for record in self.data:
            record.pop("label", None)  # rimuove la label se presente

    def separate_train_val_test(self, train_ratio=0.7, val_ratio=0.0, seed: int = 42):
        """
        Suddivide il dataset in train, validation e test.

        Args:
            train_ratio: frazione di campioni da usare per il training.
            val_ratio: frazione di campioni da usare per la validation.
            seed: seed per rendere lo split riproducibile.

        Returns:
            Una tripla (dataset_train, dataset_val, dataset_test).
        """
        if not (0 < train_ratio < 1):
            raise ValueError("train_ratio deve essere compreso tra 0 e 1")

        if not (0 <= val_ratio < 1):
            raise ValueError("val_ratio deve essere compreso tra 0 e 1")

        if train_ratio + val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio deve essere < 1")

        total_samples = len(self)
        train_size = int(train_ratio * total_samples)
        val_size = int(val_ratio * total_samples)
        test_size = total_samples - train_size - val_size

        generator = torch.Generator().manual_seed(seed)

        dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(
            self,
            [train_size, val_size, test_size],
            generator=generator
        )

        return dataset_train, dataset_val, dataset_test

    def count_distinct_labels(self):
        """Conta il numero di etichette distinte nel dataset."""
        return len(set(self.labels))

    def get_distinct_labels(self) -> list:
        """
        Restituisce la lista ordinata dei valori distinti di label presenti nel dataset.
        """
        return sorted(set(self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.mac_addresses[idx]

    @staticmethod
    def collate_probe_batch(batch):
        """TODO: spostare in prepare_dataset.probe_dataset.py"""
        """
        Converte un batch di campioni del ProbeDataset in tensori PyTorch.

        Supporta due formati:
        - (record_dict, label)
        - (record_dict, label, mac_address)

        Returns:
            X: tensore float32 di shape [batch_size, n_features]
            y: tensore long di shape [batch_size]
            Z: lista dei MAC convertiti in 6 interi, oppure None
        """
        first_item = batch[0]

        if len(first_item) == 2:
            records, labels = zip(*batch)
            mac_addresses = None
        elif len(first_item) == 3:
            records, labels, mac_addresses = zip(*batch)
        else:
            raise ValueError(
                f"Formato batch non supportato: attesi 2 o 3 elementi per sample, trovati {len(first_item)}"
            )

        feature_names = sorted(records[0].keys())

        X = torch.tensor(
            [[record[name] for name in feature_names] for record in records],
            dtype=torch.float32
        )

        y = torch.tensor(labels, dtype=torch.long)

        if mac_addresses is not None:
            # ["AA:BB:CC:DD:EE:FF"] -> [[170, 187, 204, 221, 238, 255]]
            Z = [[int(x, 16) for x in mac.split(":")] for mac in mac_addresses]
        else:
            Z = None

        return X, y, Z