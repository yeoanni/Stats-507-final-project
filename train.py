


import torch
import torch.nn as nn
import torch.optim as optim
from model import TimeSeriesTransformer
from dataset import NABDataset, create_dataloaders
import logging
import time

def train_model(train_loader, val_loader, device='cpu'):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize model
    model = TimeSeriesTransformer().to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 10
    best_val_loss = float('inf')
    
    logger.info("Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        start_time = time.time()
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.float().to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(device)
                labels = labels.float().to(device)
                
                outputs = model(data)
                loss = criterion(outputs.squeeze(), labels)
                
                val_loss += loss.item()
                
                predicted = (outputs.squeeze() > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        epoch_time = time.time() - start_time
        
        logger.info(f'Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s')
        logger.info(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        logger.info(f'Validation Accuracy: {accuracy:.2f}%')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info('Saved new best model')
        
        logger.info('-' * 60)

def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Load and preprocess data
    nab = NABDataset("NAB")
    nab.load_data("realAWSCloudwatch")
    processed_data = nab.preprocess_data(window_size=100, stride=1)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(processed_data, batch_size=32)
    
    # Train the model
    train_model(train_loader, val_loader, device)

if __name__ == "__main__":
    main()