import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.labels[idx]])

class TimeSeriesPreprocessor:
    def __init__(self, window_size=288, stride=12):
        self.window_size = window_size
        self.stride = stride
        self.scaler = MinMaxScaler()
        
    def add_time_features(self, df):
        """Enhanced time features"""
        df = df.copy()
        
        # Basic time features
        df['hour'] = df.timestamp.dt.hour
        df['day_of_week'] = df.timestamp.dt.dayofweek
        df['day_of_month'] = df.timestamp.dt.day
        
        # Cyclical encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
        
        # Add rolling statistics
        df['rolling_mean'] = df['value'].rolling(window=12, min_periods=1).mean()
        df['rolling_std'] = df['value'].rolling(window=12, min_periods=1).std()
        df['rolling_max'] = df['value'].rolling(window=12, min_periods=1).max()
        df['rolling_min'] = df['value'].rolling(window=12, min_periods=1).min()
        
        # Add lag features
        df['lag_1'] = df['value'].shift(1)
        df['lag_6'] = df['value'].shift(6)
        df['lag_12'] = df['value'].shift(12)
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df

    def create_sequences(self, df):
        """Create sequences with enhanced features"""
        # Scale all numerical features
        feature_columns = ['value', 'rolling_mean', 'rolling_std', 'rolling_max', 
                         'rolling_min', 'lag_1', 'lag_6', 'lag_12',
                         'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        
        scaled_features = self.scaler.fit_transform(df[feature_columns])
        
        sequences = []
        labels = []
        
        # Calculate statistical thresholds
        mean = df['value'].mean()
        std = df['value'].std()
        upper_threshold = mean + 2*std
        lower_threshold = mean - 2*std
        
        for i in range(0, len(df) - self.window_size, self.stride):
            seq = scaled_features[i:i + self.window_size]
            next_val = df['value'].iloc[i + self.window_size]
            
            # Label is 1 if next value is anomalous
            is_anomaly = (next_val > upper_threshold) or (next_val < lower_threshold)
            
            sequences.append(seq)
            labels.append(float(is_anomaly))
            
            # Add augmented sequences for anomalies
            if is_anomaly:
                # Add slight random variations to create additional anomaly examples
                noise = np.random.normal(0, 0.1, seq.shape)
                augmented_seq = seq + noise
                sequences.append(augmented_seq)
                labels.append(1.0)
        
        return np.array(sequences), np.array(labels)
    
    def create_dataloaders(self, df=None, batch_size=16, train_split=0.8):
        if df is None:
            base_url = "https://raw.githubusercontent.com/numenta/NAB/master/data/"
            file_path = f"{base_url}realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv"
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df = self.add_time_features(df)
        sequences, labels = self.create_sequences(df)
        
        # Shuffle data
        indices = np.random.permutation(len(sequences))
        sequences = sequences[indices]
        labels = labels[indices]
        
        train_size = int(len(sequences) * train_split)
        
        train_sequences = sequences[:train_size]
        train_labels = labels[:train_size]
        val_sequences = sequences[train_size:]
        val_labels = labels[train_size:]
        
        train_dataset = TimeSeriesDataset(train_sequences, train_labels)
        val_dataset = TimeSeriesDataset(val_sequences, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        return train_loader, val_loader