import torch
from torch.utils.data import Dataset


# Tensor Dataset for tracking data
class TrackingDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Dataset for LSTM windowed sequences — stores (seq_len, features) tensors
class LSTMWindowDataset(Dataset):
    def __init__(self, data):
        # data: np.ndarray of shape (N, seq_len, features)
        self.data = torch.FloatTensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # (seq_len, features)
