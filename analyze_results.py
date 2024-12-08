import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.preprocess import TimeSeriesPreprocessor
from models.transformer import TransformerAnomaly
from pathlib import Path

class ModelAnalyzer:
    def __init__(self, model_path='models/saved/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = TimeSeriesPreprocessor()
        self.model = TransformerAnomaly().to(self.device)
        
        # Load only the model state dict from the checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    def analyze_performance(self):
        """Analyze model performance"""
        print("Loading data...")
        base_url = "https://raw.githubusercontent.com/numenta/NAB/master/data/"
        file_path = f"{base_url}realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv"
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Get predictions
        print("Making predictions...")
        df_eval = self.preprocessor.add_time_features(df)
        sequences, labels = self.preprocessor.create_sequences(df_eval)
        
        predictions = []
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i + batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                outputs = self.model(batch_tensor)
                predictions.extend(outputs.cpu().numpy())
        
        predictions = np.array(predictions).flatten()
        
        # Create visualizations
        print("Generating visualizations...")
        self.generate_report(predictions, labels, df)
        
    def generate_report(self, predictions, labels, df):
        """Generate comprehensive analysis report"""
        save_path = 'results/analysis_report'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # 1. Time Series Plot
        plt.figure(figsize=(15, 10))
        
        # Original data with predictions
        plt.subplot(3, 1, 1)
        window_size = self.preprocessor.window_size
        valid_timestamps = df['timestamp'][window_size:len(predictions)+window_size]
        plt.plot(df['timestamp'], df['value'], label='Original', alpha=0.7)
        
        # Mark anomalies
        threshold = 0.3
        anomaly_mask = predictions > threshold
        anomaly_times = valid_timestamps[anomaly_mask]
        anomaly_values = df['value'][window_size:len(predictions)+window_size][anomaly_mask]
        plt.scatter(anomaly_times, anomaly_values, color='red', label=f'Detected Anomalies (threshold={threshold})')
        plt.title('CPU Utilization with Detected Anomalies')
        plt.legend()
        
        # 2. Prediction Scores
        plt.subplot(3, 1, 2)
        plt.plot(valid_timestamps, predictions, label='Anomaly Score')
        plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        plt.title('Anomaly Scores Over Time')
        plt.legend()
        
        # 3. Score Distribution
        plt.subplot(3, 1, 3)
        plt.hist(predictions, bins=50, density=True, alpha=0.7)
        plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
        plt.title('Distribution of Anomaly Scores')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/anomaly_detection_results.png')
        plt.close()
        
        # Save metrics report
        with open(f'{save_path}/detection_report.txt', 'w') as f:
            f.write("Anomaly Detection Report\n")
            f.write("======================\n\n")
            f.write(f"Total windows analyzed: {len(predictions)}\n")
            f.write(f"Number of anomalies detected: {np.sum(anomaly_mask)}\n")
            f.write(f"Percentage of anomalies: {100 * np.sum(anomaly_mask) / len(predictions):.2f}%\n")
            f.write(f"\nThreshold used: {threshold}\n")
            
            # Calculate statistics
            mean_score = np.mean(predictions)
            std_score = np.std(predictions)
            f.write(f"\nScore Statistics:\n")
            f.write(f"Mean score: {mean_score:.4f}\n")
            f.write(f"Standard deviation: {std_score:.4f}\n")
            f.write(f"Min score: {np.min(predictions):.4f}\n")
            f.write(f"Max score: {np.max(predictions):.4f}\n")

def main():
    analyzer = ModelAnalyzer()
    analyzer.analyze_performance()
    print("Analysis complete! Check results/analysis_report/ for detailed findings.")

if __name__ == "__main__":
    main()