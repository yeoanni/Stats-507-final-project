import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def create_dataloaders(dataset_name, batch_size=32):
    try:
        data = np.load(f'datasets/{dataset_name}_data.npy')
        labels = np.load(f'datasets/{dataset_name}_labels.npy')
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset files for {dataset_name} not found in 'datasets/' directory.")

    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    dataset = TimeSeriesDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader, labels
