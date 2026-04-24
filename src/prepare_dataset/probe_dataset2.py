import json
import pandas as pd
import torch

from torch.utils.data import Dataset
from prepare_dataset.ie_to_transformerIE import preprocess_list

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

        self.path = path_json
        self.include_mac_features = include_mac_features
        
        json_data = self.get_json_data(json_files=self.path)
        
        self.data, self.labels, self.mac_addresses = self.load_json_list(json_data, keep_mac_in_data=include_mac_features)

        if preprocess:
            self.preprocess_data()
            
        # Converti la lista di dati in un DataFrame pandas      
        self.data = pd.DataFrame(self.data)  
            
    def get_json_data(self, json_files):
        """Carica i dati da JSON e restituisce un dizionario."""
        
        if isinstance(json_files, list):
            # Se è una lista di file, carica e combina i dati da tutti i file
            json_data = []
            for json_file in json_files:
                with open(json_file, "r", encoding="utf-8") as f:
                    json_data.extend(json.load(f))
        else:
            # Se è un singolo file, carica i dati da quel file
            with open(json_files, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                    
        return json_data

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

    def preprocess_data(self):
        """
        Preprocessa tutti i record del dataset usando ie_to_transformerIE.
        """
        if not isinstance(self.data, list):
            raise TypeError(f"self.data deve essere una lista, trovato {type(self.data).__name__}")

        self.data = preprocess_list(self.data)

        for record in self.data:
            record.pop("label", None)  # rimuove la label se presente

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
        return self.data.iloc[idx, :], self.labels[idx], self.mac_addresses[idx]


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