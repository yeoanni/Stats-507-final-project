# Transformer-based Anomaly Detection in Time Series Data

## Overview
This project implements a Transformer-based approach for detecting anomalies in time series data, specifically focused on cloud infrastructure metrics from the Numenta Anomaly Benchmark (NAB) dataset. The implementation achieves strong performance with PR-AUC of 0.857 and ROC-AUC of 0.976.

## Project Structure
```
├── data/
│   ├── raw/                  # Raw data files from NAB
│   └── processed/            # Preprocessed data files
├── models/
│   ├── __init__.py
│   ├── transformer.py        # Transformer model implementation
│   └── saved/               # Saved model checkpoints
├── utils/
│   ├── __init__.py
│   └── preprocessing.py     # Data preprocessing utilities
├── evaluate.py              # Model evaluation script
├── train.py                # Training script
└── README.md
```

## Installation
```bash
# Clone the repository
git clone https://github.com/yeoanni/Stats-507-final-project.git
cd Stats-507-final-project

# Install required packages
pip install torch pandas numpy sklearn matplotlib
```

## Usage

### Training
```python
python train.py
```
This will:
- Load and preprocess the NAB dataset
- Train the Transformer model
- Save the best model based on validation performance

### Evaluation
```python
python evaluate.py
```
This will:
- Load the trained model
- Generate performance metrics
- Create visualization plots

## Model Architecture
- Input dimension: 12 features
- Transformer encoder with 4 attention heads
- Binary classification output
- Window size: 288 points (24 hours)
- Stride: 12 points

## Results
- PR-AUC: 0.857
- ROC-AUC: 0.976
- Successfully detects anomalies in CPU utilization patterns
- Effective handling of temporal dependencies

## Data
The project uses the Numenta Anomaly Benchmark (NAB) dataset, focusing on AWS EC2 CPU utilization metrics. Data preprocessing includes:
- Time-based feature engineering
- Sequence windowing
- Min-max scaling
- Temporal encoding

## Contributing
This project is part of STATS 507 coursework. Contributions welcome through pull requests.

## License
[MIT License](LICENSE)

## Contact
Yuhan Ye - yuhanye@umich.edu

## Acknowledgments
- NAB dataset providers
- PyTorch team
- STATS 507 course staff
