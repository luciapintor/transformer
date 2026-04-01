#ha dato problemi il primo commit

import json
from pathlib import Path

import torch 
from torch.utils.data import Dataset
from ie_to_transformerIE import preprocess_json_file

class ProbeDataset(Dataset):
    def __init__(self, path_json):
        """Carica data/label da JSON per dataset.

        path_json: percorso file JSON o directory JSON.
        """
        self.path = Path(path_json)
        self.data, self.labels = self.load_json(self.path)

    def load_json(self, path_json):
        """Carica un file JSON o una directory di JSON e ritorna (data, labels)."""
        p = Path(path_json)
        if not p.exists():
            raise FileNotFoundError(f"Percorso non trovato: {path_json}")

        merged = {}
        if p.is_file():
            with open(p, 'r', encoding='utf-8') as f:
                merged = json.load(f)
        else:
            for json_file in sorted(p.glob('*.json')):
                with open(json_file, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                    if not isinstance(d, dict):
                        raise ValueError(f"File JSON non valido: {json_file}")
                    merged.update(d)

        data = []
        labels = []

        for burst_id, record in merged.items():
            if not isinstance(record, dict):
                continue
            labels.append(record.get('label', None))
            data.append(record)

        return data, labels


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
