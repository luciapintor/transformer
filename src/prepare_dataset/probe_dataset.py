import torch 
from torch.utils.data import Dataset

class ProbeDataset(Dataset):
    def __init__(self, data, labels):
        """ TODO: modify the __init__ method to get the data and labels from the json file. """
        super().__init__()
        n_samples = 1000    # number of samples in the dataset
        n_features = 10     # number of features in the dataset
        n_classes = 15      # number of classes for classification (set to 1 for regression)
        n_macs = 15         # number of unique MAC addresses
        macs = [f"MAC_{i%n_macs}" for i in range(n_samples)]  
        
        self.values = torch.randn(n_samples, n_features)  # random features
        self.labels = torch.randint(0, n_classes, (n_samples,))  # random labels for classification
        
        # self.macs is a concateation of macs, until it reaches the length of n_samples, 
        # then it will be truncated to n_samples
        self.macs = (macs * (n_samples // len(macs) + 1))[:n_samples]  # repeat and truncate to n_samples
        
    def __len__(self):
        return len(self.values)  # number of samples in the dataset

    def __getitem__(self, index):
        return self.values[index], self.labels[index], self.macs[index] 
