import torch as nn
from torch.utils.data import Dataset

class MyDataset(Dataset):
  """A simple dataset class that generates random data for demonstration purposes."""  
  def __init__(self, n_samples=1000, n_features=10, n_classes=2):
    super(MyDataset, self).__init__()
    self.values = nn.randn(n_samples, n_features)  # random features
    self.labels = nn.randint(0, n_classes, (n_samples,))  # random labels for classification

  def __len__(self):
    return len(self.values)  # number of samples in the dataset

  def __getitem__(self, index):
    return self.values[index], self.labels[index]

  def separate_train_val_test(self, train_ratio=0.7, val_ratio=0.15):
      """Splits the dataset into training, validation, and test sets based on the provided ratios."""
      total_samples = len(self)
      train_size = int(train_ratio * total_samples)
      val_size = int(val_ratio * total_samples)
      test_size = total_samples - train_size - val_size
      
      # Use random_split to create the subsets
      dataset_train, dataset_val, dataset_test = nn.utils.data.random_split(self, [train_size, val_size, test_size])
      
      return dataset_train, dataset_val, dataset_test
    