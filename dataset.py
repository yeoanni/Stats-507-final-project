


"""




"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import logging
import json

class NABDataset:
    """Handler for the Numenta Anomaly Benchmark dataset"""
    
    def __init__(self, base_path: str, data_dir: str = "data"):
        """
        Initialize NAB dataset handler
        
        Args:
            base_path: Base path to NAB dataset
            data_dir: Directory containing the data files (default: "data")
        """
        self.base_path = Path(base_path)
        self.data_path = self.base_path / data_dir
        self.labels_path = self.base_path / "labels"
        
        # Dictionary to store loaded data
        self.data = {}
        self.labels = {}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load labels
        self._load_labels()
    
    def _load_labels(self):
        """Load anomaly labels from combined_windows.json"""
        try:
            with open(self.labels_path / "combined_windows.json", 'r') as f:
                self.labels = json.load(f)
            self.logger.info("Successfully loaded labels")
        except FileNotFoundError:
            self.logger.error("Labels file not found")
            raise
    
    def load_data(self, category: str):
        """
        Load data from a specific category
        
        Args:
            category: Category name (e.g., 'realAWSCloudwatch', 'realAdExchange')
        """
        category_path = self.data_path / category
        
        if not category_path.exists():
            self.logger.error(f"Category {category} not found")
            raise FileNotFoundError(f"Category {category} not found")
        
        # Load all CSV files in the category
        for file_path in category_path.glob("*.csv"):
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
            file_name = file_path.stem
            self.data[file_name] = df
            self.logger.info(f"Loaded {file_name}")
    
    def preprocess_data(self, window_size: int = 100, stride: int = 1):
        """
        Preprocess the loaded data for the transformer model
        
        Args:
            window_size: Size of the sliding window
            stride: Stride for sliding window
        
        Returns:
            dict: Preprocessed data with sequences and labels
        """
        processed_data = {}
        
        for file_name, df in self.data.items():
            # Normalize the value column
            scaler = MinMaxScaler()
            values = scaler.fit_transform(df['value'].values.reshape(-1, 1))
            
            # Create sequences
            sequences = []
            labels = []
            
            for i in range(0, len(df) - window_size + 1, stride):
                seq = values[i:i + window_size]
                sequences.append(seq)
                
                # Check if any timestamp in this window is labeled as anomaly
                window_times = df['timestamp'].iloc[i:i + window_size]
                is_anomaly = any(
                    any(start <= t <= end 
                        for start, end in self.labels.get(file_name, []))
                    for t in window_times
                )
                labels.append(1 if is_anomaly else 0)
            
            processed_data[file_name] = {
                'sequences': np.array(sequences),
                'labels': np.array(labels),
                'scaler': scaler
            }
            
            self.logger.info(f"Preprocessed {file_name}: {len(sequences)} sequences")
        
        return processed_data

class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series data"""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def create_dataloaders(processed_data, batch_size=32, train_ratio=0.8):
    """
    Create train and validation dataloaders
    
    Args:
        processed_data: Dictionary containing preprocessed sequences and labels
        batch_size: Batch size for dataloaders
        train_ratio: Ratio of data to use for training
    
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    # Combine all sequences and labels
    all_sequences = []
    all_labels = []
    
    for data_dict in processed_data.values():
        all_sequences.append(data_dict['sequences'])
        all_labels.append(data_dict['labels'])
    
    all_sequences = np.concatenate(all_sequences)
    all_labels = np.concatenate(all_labels)
    
    # Split into train and validation
    n_train = int(len(all_sequences) * train_ratio)
    
    train_sequences = all_sequences[:n_train]
    train_labels = all_labels[:n_train]
    val_sequences = all_sequences[n_train:]
    val_labels = all_labels[n_train:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_sequences, train_labels)
    val_dataset = TimeSeriesDataset(val_sequences, val_labels)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader