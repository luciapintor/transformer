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
from .ie_to_transformerIE import preprocess_list


class ProbeDataset(Dataset):
    def __init__(self, path_json, preprocess: bool = False):
        """Carica data/label da JSON per dataset.

        Args:
            path_json: percorso file JSON o directory JSON.
            preprocess: se True, preprocessa subito tutti i record caricati.
        """
        self.path = Path(path_json)
        self.data, self.labels, self.mac_addresses = self.load_json(self.path)

        if preprocess:
            self.preprocess_data()

    def load_json(self, path_json):
        """Carica un file JSON o una directory di JSON e ritorna (data, labels, mac_addresses).

        Se path_json è un file .json, lo divide in data e label e restituisce le liste.
        Se path_json è una directory, itera su tutti i file .json e li unisce.
        """
        p = Path(path_json)
        if not p.exists():
            raise FileNotFoundError(f"Percorso non trovato: {path_json}")

        merged = {}
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                merged = json.load(f)
        else:
            for json_file in sorted(p.glob("*.json")):
                with open(json_file, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if not isinstance(d, dict):
                        raise ValueError(f"File JSON non valido: {json_file}")
                    merged.update(d)

        data = []
        labels = []
        mac_addresses = []

        for sample_id, record in merged.items():
            if not isinstance(record, dict):
                raise ValueError(f"Record {sample_id} non valido: atteso dict")

            if "label" not in record:
                raise ValueError(f"Record {sample_id} senza label")

            burst_without_label = dict(record)
            label = burst_without_label.pop("label")
            mac_address = sample_id

            data.append(burst_without_label)
            labels.append(label)
            mac_addresses.append(mac_address)

        return data, labels, mac_addresses

    def preprocess_data(self):
        """
        Preprocessa tutti i record del dataset usando ie_to_transformerIE.
        """
        if not isinstance(self.data, list):
            raise TypeError(f"self.data deve essere una lista, trovato {type(self.data).__name__}")

        self.data = preprocess_list(self.data)

        for record in self.data:
            record.pop("label", None) #rimuove la label se presente

    def separate_train_val_test(self, train_ratio=0.7, val_ratio=0.15, seed: int = 42):
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

        Supporta due formati: sarebbero i formati di un sample del ProbeDataset:
        - (record_dict, label)
        - (record_dict, label, mac_address)

        Returns:
            X: tensore float32 di shape [batch_size, n_features]
            y: tensore long di shape [batch_size]
        """
        first_item = batch[0]

        if len(first_item) == 2:
            records, labels = zip(*batch)
        elif len(first_item) == 3:
            records, labels, mac_addresses = zip(*batch)  # ignoriamo i MAC address
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

        # mac_addresses = ["AA:BB:CC:DD:EE:FF", "11:22:33:44:55:66"] --> Z = [170, 187, 204, 221, 238, 255],    [ 17,  34,  51,  68,  85, 102]
        Z = [[int(x, 16) for x in mac.split(":")] for mac in mac_addresses] 
        
        #with open("chissacosacontieneX.txt", "w", encoding="utf-8") as f:
        #    f.write(str((X.data[0])))
        return X, y, Z

    

if __name__ == "__main__":
    dataset = ProbeDataset(
        "Dataset/dataset_burst_json/scenario_0_burst_features.json",
        preprocess=True
    )

    print(f"Numero di record: {len(dataset)}")
    print(dataset.data[0])
    print(dataset.labels[0])