# Transformer-based Anomaly Detection in Time Series Data

## Project Structure
```
├── models/                     # Model-related code and saved models
├── proposal/                   # Project proposal documents
├── results/                    # Experimental results and analysis
├── utils/                      # Utility functions and helpers
├── README.md                   # Project documentation
├── analyze_results.py          # Results analysis script
├── baseline_comparison.py      # Baseline model comparison
├── data_loader.py             # Data loading utilities
├── evaluate.py                # Model evaluation script
├── train.py                   # Training script
└── visualize_results.py       # Visualization utilities
```

## Overview
This project implements a Transformer-based approach for anomaly detection in time series data, focusing on cloud infrastructure metrics. Using the NAB (Numenta Anomaly Benchmark) dataset, we demonstrate the effectiveness of Transformer architectures for identifying anomalous patterns in CPU utilization data.

## Key Features
- Custom Transformer architecture for time series data
- Comprehensive data preprocessing pipeline
- Comparative analysis with baseline methods
- Visualization tools for result analysis
- Real-world application on cloud infrastructure metrics

## Performance
- PR-AUC: 0.857
- ROC-AUC: 0.976
- Training converged in 96 epochs
- Effective anomaly detection on AWS EC2 CPU metrics

## Prerequisites
```bash
python>=3.8
torch
pandas
numpy
matplotlib
scikit-learn
```

## Usage
1. Training the model:
```bash
python train.py
```

2. Evaluating performance:
```bash
python evaluate.py
```

3. Analyzing results:
```bash
python analyze_results.py
```

4. Visualizing results:
```bash
python visualize_results.py
```

5. Comparing with baselines:
```bash
python baseline_comparison.py
```

## Results
Results are stored in the `results/` directory, including:
- Performance metrics
- Visualization plots
- Analysis reports
- Comparative studies

## Author
Yuhan Ye (yuhanye@umich.edu)

## Acknowledgments
- STATS 507 course staff
- Numenta Anomaly Benchmark (NAB) dataset
