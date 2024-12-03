


from dataset import NABDataset, create_dataloaders
import logging

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Path to NAB dataset - update this to your NAB directory path
    nab_path = "NAB"  # Change this to where you cloned the NAB repository
    
    try:
        # Initialize dataset handler
        logger.info("Initializing NAB dataset handler...")
        nab = NABDataset(nab_path)

        # Load a specific category
        logger.info("Loading AWS Cloudwatch data...")
        nab.load_data("realAWSCloudwatch")

        # Preprocess the data
        logger.info("Preprocessing data...")
        processed_data = nab.preprocess_data(window_size=100, stride=1)

        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_loader, val_loader = create_dataloaders(processed_data, batch_size=32)
        
        logger.info(f"Number of training batches: {len(train_loader)}")
        logger.info(f"Number of validation batches: {len(val_loader)}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()