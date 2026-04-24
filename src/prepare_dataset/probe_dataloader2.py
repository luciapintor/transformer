
import torch
from torch.utils.data import DataLoader

class ProbeDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, collate_fn=None):
        """
        This class is a custom DataLoader for the ProbeDataset. 
        It inherits from PyTorch's DataLoader and allows us to specify a custom collate function 
        to convert batches of data into tensors.

        Args:
            dataset (_type_): _description_
            batch_size (int, optional): _description_. Defaults to 32.
            shuffle (bool, optional): _description_. Defaults to True.
            collate_fn (_type_, optional): _description_. Defaults to None.
        """
        # Call the parent constructor to initialize the DataLoader with the provided parameters
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        
        # If no custom collate function is provided, use the default one defined in this class
        if collate_fn is None:
            self.collate_fn = self.probe_collate_fn
        
    def probe_collate_fn (self, batch):
        """
        Convert a batch of ProbeDataset samples into PyTorch tensors.

        Supports:
        - (record_dict, label)
        - (record_dict, label, mac_address)

        Returns:
            X: FloatTensor [batch_size, n_features]
            y: LongTensor [batch_size]
            Z: list of MAC addresses as integer lists, or None
        """       
        first_item = batch[0]

        if len(first_item) == 2:
            records, labels = zip(*batch)
            mac_addresses = None
        elif len(first_item) == 3:
            records, labels, mac_addresses = zip(*batch)
        else:
            raise ValueError(
                f"Batch format not supported: expected 2 or 3 elements per sample, found {len(first_item)}"
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