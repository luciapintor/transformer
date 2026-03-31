import torch 
from torch.utils.data import Dataset

class ProbeDataset(Dataset):
    def __init__(self, data, labels):
        """ TODO: modify the __init__ method to get the data and labels from the json file. """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
