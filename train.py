import torch
import torch.nn as nn
import logging
from utils.preprocess import TimeSeriesPreprocessor
from models.transformer import TransformerAnomaly
import numpy as np
from pathlib import Path

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, pred, target):
        weight = torch.where(target == 1, self.pos_weight, torch.tensor(1.0))
        loss = -(weight * (target * torch.log(pred + 1e-10) + 
                (1 - target) * torch.log(1 - pred + 1e-10)))
        return loss.mean()

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    preprocessor = TimeSeriesPreprocessor()
    train_loader, val_loader = preprocessor.create_dataloaders(batch_size=16)
    
    # Calculate positive class weight
    pos_weight = torch.tensor((1 - 0.0327) / 0.0327)
    logger.info(f"Positive class weight: {pos_weight:.2f}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = TransformerAnomaly(
        input_dim=12,
        d_model=32,
        nhead=4,
        num_layers=1
    ).to(device)
    
    criterion = WeightedBCELoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )
    
    # Training parameters
    num_epochs = 150  # Increased from 100
    best_val_loss = float('inf')
    patience = 15     # Increased from 10
    patience_counter = 0
    min_improvement = 1e-5  # Minimum improvement threshold
    train_losses = []
    val_losses = []
    
    # Create save directory
    save_dir = Path('models/saved')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for batch, labels in train_loader:
            batch = batch.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch, labels in val_loader:
                batch = batch.to(device)
                labels = labels.to(device)
                
                output = model(batch)
                loss = criterion(output, labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Learning Rate: {current_lr:.6f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping with minimum improvement threshold
        if avg_val_loss < best_val_loss - min_improvement:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, save_dir / 'best_model.pth')
            logger.info("Saved new best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered. Best val loss: {best_val_loss:.6f}")
                break
    
    # Plot training history
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/training_history.png')
    plt.close()

if __name__ == "__main__":
    main()