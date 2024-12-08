import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from utils.preprocess import TimeSeriesPreprocessor
from models.transformer import TransformerAnomaly
from pathlib import Path

def evaluate_model(model_path='models/saved/best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading and preprocessing data...")
    preprocessor = TimeSeriesPreprocessor()
    base_url = "https://raw.githubusercontent.com/numenta/NAB/master/data/"
    file_path = f"{base_url}realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv"
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("Preparing evaluation data...")
    df_eval = preprocessor.add_time_features(df)
    sequences, labels = preprocessor.create_sequences(df_eval)
    
    print("Loading model...")
    model = TransformerAnomaly().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    print("Making predictions...")
    predictions = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)
            outputs = model(batch_tensor)
            predictions.extend(outputs.cpu().numpy())
    
    predictions = np.array(predictions).flatten()
    
    precision, recall, _ = precision_recall_curve(labels, predictions)
    fpr, tpr, _ = roc_curve(labels, predictions)
    pr_auc = auc(recall, precision)
    roc_auc = auc(fpr, tpr)
    
    print("Generating visualizations...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(recall, precision)
    ax1.set_title(f'Precision-Recall Curve (AUC = {pr_auc:.3f})')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.grid(True)
    
    ax2.plot(fpr, tpr)
    ax2.set_title(f'ROC Curve (AUC = {roc_auc:.3f})')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.grid(True)
    
    plt.tight_layout()
    
    Path('results').mkdir(exist_ok=True)
    
    plt.savefig('results/performance_curves.png')
    print("Results saved to results/performance_curves.png")
    
    print("\nEvaluation Metrics:")
    print(f"PR-AUC: {pr_auc:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")


if __name__ == "__main__":
    evaluate_model()
