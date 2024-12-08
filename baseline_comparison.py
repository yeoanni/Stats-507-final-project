import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMAnomalyDetector, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        if len(out.shape) == 3:  # Ensure 3D tensor
            out = out[:, -1, :]  # Take the last time step's output
        return self.fc(out)


def create_synthetic_data(num_samples=1000, num_features=12, anomaly_rate=0.05):
    data = np.random.rand(num_samples, num_features)
    labels = (np.random.rand(num_samples) < anomaly_rate).astype(int)  # 5% anomalies
    return data, labels


class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_dataloaders(data, labels, batch_size=32):
    # Convert to PyTorch tensors
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    # Create Dataset and DataLoader
    dataset = TimeSeriesDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_scores = [], []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model(inputs).squeeze()
        y_true.extend(targets.cpu().numpy())
        y_scores.extend(outputs.cpu().numpy())

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    return {'PR-AUC': pr_auc, 'ROC-AUC': roc_auc}


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, labels = create_synthetic_data()
    dataloader = create_dataloaders(data, labels, batch_size=16)

    lstm_model = LSTMAnomalyDetector(input_dim=12, hidden_dim=64, num_layers=2, output_dim=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

    lstm_model.train()
    for epoch in range(10):  # Short training loop for demo purposes
        epoch_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = lstm_model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/10, Loss: {epoch_loss:.4f}")

    lstm_results = evaluate_model(lstm_model, dataloader, device)
    print(f"LSTM Results: PR-AUC = {lstm_results['PR-AUC']:.3f}, ROC-AUC = {lstm_results['ROC-AUC']:.3f}")

    data_flat = data[:, -1]  # Use last feature as a simple example
    moving_avg = np.convolve(data_flat, np.ones(5)/5, mode='same')
    anomaly_scores = np.abs(data_flat - moving_avg)
    precision, recall, _ = precision_recall_curve(labels, anomaly_scores)
    pr_auc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(labels, anomaly_scores)
    roc_auc = auc(fpr, tpr)
    print(f"Moving Average Results: PR-AUC = {pr_auc:.3f}, ROC-AUC = {roc_auc:.3f}")
