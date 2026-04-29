

# HOW TO USE - singlefile_clustering.py

Questo script prende un dataset di Probe Request, addestra un autoencoder, estrae gli embedding e applica DBSCAN per fare clustering.

Alla fine salva un file CSV con:

- indice del campione;
- MAC address;
- cluster assegnato da DBSCAN.

---

## 1. Dove modificare i parametri

Tutti i parametri principali si trovano nella parte iniziale del `main`, sotto questi blocchi:

```python
# PARAMETRI DATASET TRAIN E TEST
# PARAMETRI MODELLO
# PARAMETRI CLUSTERING
````

Le variabili da modificare sono principalmente:

```python
probe_file = "Dataset/dataset_merged_probes_json/data with labels/scenario_0_full.json"
output_json = "Dataset/dataset_json_from_pcap/dataset_from_pcap.json"
isPcap = False
batch_size = 64
preprocess = True
include_mac_features = False
```

---

## 2. Usare un file JSON già pronto

Se hai già un file JSON con le Probe Request, devi impostare:

```python
isPcap = False
```

Poi devi mettere in `probe_file` il percorso del JSON, ad esempio:

```python
probe_file = "Dataset/dataset_merged_probes_json/data with labels/scenario_0_full.json"
```

Esempio completo:

```python
probe_file = "Dataset/dataset_merged_probes_json/data with labels/scenario_0_full.json"
output_json = "Dataset/dataset_json_from_pcap/dataset_from_pcap.json"
isPcap = False
```

In questo caso `output_json` non viene usato, perché il dataset è già in formato JSON.

---

## 3. Usare un file PCAP

Se parti da un file `.pcap`, devi impostare:

```python
isPcap = True
```

Poi devi mettere in `probe_file` il percorso del PCAP:

```python
probe_file = "Dataset/pcap/scenario_0.pcap"
```

E devi indicare in `output_json` dove salvare il JSON generato:

```python
output_json = "Dataset/dataset_json_from_pcap/dataset_from_pcap.json"
```

Esempio completo:

```python
probe_file = "Dataset/pcap/scenario_0.pcap"
output_json = "Dataset/dataset_json_from_pcap/dataset_from_pcap.json"
isPcap = True
```

In questo caso lo script prima converte il PCAP in JSON e poi usa quel JSON per il clustering.

---

Sostituiscila con questa:

## 4. Parametri del dataset

### Batch size

```python
batch_size = 64
````

Indica quanti campioni vengono elaborati insieme dal modello in ogni batch.

Viene usato sia durante il training dell’autoencoder, sia durante l’encoding del test set.

Un valore più alto può rendere l’esecuzione più veloce, ma richiede più memoria.

Un valore più basso usa meno memoria, ma può rallentare l’esecuzione.

### Preprocessing

```python
preprocess = True
```

Se è `True`, viene applicato il preprocessing definito in `ProbeDataset`.

Nel codice viene usato qui:

```python
full_dataset = ProbeDataset(path_json=json_path, preprocess=preprocess, include_mac_features=include_mac_features)
```

---

### MAC address come feature

```python
include_mac_features = False
```

Questa variabile indica se usare o no il MAC address come feature.

Consigliato:

```python
include_mac_features = False
```

Così il modello non impara direttamente dai MAC address.


---

## 7. Parametri del modello

```python
emb_size = 64
hidden_dim = 128
epochs = 50
learning_rate = 1e-3
```

Significato:

| Parametro       | Significato                                     |
| --------------- | ----------------------------------------------- |
| `emb_size`      | dimensione dell'embedding prodotto dall'encoder |
| `hidden_dim`    | dimensione interna dell'autoencoder             |
| `epochs`        | numero di epoche di training                    |
| `learning_rate` | tasso di apprendimento                          |

Per fare una prova veloce puoi usare:

```python
epochs = 5
```

Per un training più completo puoi aumentare:

```python
epochs = 50
```

---

## 8. Parametri di DBSCAN

```python
eps = 0.1
min_samples = 4
```

Significato:

| Parametro     | Significato                                                |
| ------------- | ---------------------------------------------------------- |
| `eps`         | distanza massima tra due campioni per considerarli vicini  |
| `min_samples` | numero minimo di campioni necessari per formare un cluster |

Un valore più basso di `eps` rende DBSCAN più restrittivo, quindi potrebbe trovare meno cluster e più rumore.

## 9. Divisione train, validation e test

Il dataset viene diviso con:

```python
dataset_train, dataset_val, dataset_test = full_dataset.separate_train_val_test()
```

Il modello viene addestrato solo su `dataset_train`.

Il clustering viene fatto solo su `dataset_test`.

---

## 10. Output dello script

Lo script stampa a terminale:

```text
CALCOLO SENZA RUMORE
Probe considerate: ...
Probe considerate rumore (cluster -1): ...
Numero di cluster trovati senza rumore: ...
Cluster labels: ...
```

Poi stampa una tabella con:

```text
sample_index
mac_address
cluster
```

---

## 11. File CSV generato

Alla fine viene salvato un CSV qui:

```python
df.to_csv("transformer/clustering_output/outputClustering.csv", index=False)
```

Quindi il file di output sarà:

```bash
transformer/clustering_output/outputClustering.csv
```

Il CSV contiene:

| Colonna        | Significato                       |
| -------------- | --------------------------------- |
| `sample_index` | indice del campione nel test set  |
| `mac_address`  | MAC address associato al campione |
| `cluster`      | cluster assegnato da DBSCAN       |

Il cluster `-1` indica rumore, cioè un campione non assegnato a nessun cluster.

---

## 13. Esempio: uso con un solo file JSON

Configurazione consigliata:

```python
probe_file = "Dataset/dataset_merged_probes_json/data with labels/scenario_0_full.json"
output_json = "Dataset/dataset_json_from_pcap/dataset_from_pcap.json"
isPcap = False

batch_size = 64
preprocess = True
include_mac_features = False

emb_size = 64
hidden_dim = 128
epochs = 50
learning_rate = 1e-3

eps = 0.1
min_samples = 4
```

Questa configurazione usa direttamente il JSON indicato in `probe_file`.

---

## 14. Esempio: uso con un file PCAP

Configurazione consigliata:

```python
probe_file = "Dataset/pcap/scenario_0.pcap"
output_json = "Dataset/dataset_json_from_pcap/dataset_from_pcap.json"
isPcap = True

batch_size = 64
preprocess = True
include_mac_features = False

emb_size = 64
hidden_dim = 128
epochs = 50
learning_rate = 1e-3

eps = 0.1
min_samples = 4
```

Questa configurazione converte il PCAP in JSON e poi usa quel JSON per il clustering.

---
