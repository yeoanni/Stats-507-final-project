# Transformer-based Anomaly Detection in Time Series Data

## Project Structure
```
├── models/                               # Model architecture and saved states
│   ├── __pycache__/                     # Python cache files
│   │   ├── __init__.cpython-312.pyc
│   │   └── transformer.cpython-312.pyc
│   ├── saved/                           # Saved model checkpoints
│   │   └── best_model.pth              # Best performing model state
│   ├── __init__.py                     # Package initializer
│   └── transformer.py                   # Transformer model implementation
├── proposal/                            # Project documentation
│   ├── 507 Final Proposal.pdf          # Original project proposal
│   └── references.bib                  # Bibliography references
├── results/                            # Analysis outputs and visualizations
│   ├── analysis_report/                # Detailed performance analysis
│   │   ├── anomaly_detection_results.png  # Visual results of detection
│   │   └── detection_report.txt        # Detailed metrics and findings
│   ├── performance_curves.png          # PR and ROC curves
│   └── training_history.png            # Training and validation loss plots
├── utils/                              # Utility functions and helpers
│   ├── __pycache__/                    # Python cache files
│   │   ├── __init__.cpython-312.pyc
│   │   └── preprocess.cpython-312.pyc
│   ├── __init__.py                    # Package initializer
│   └── preprocess.py                  # Data preprocessing utilities
├── analyze_results.py                  # Results analysis script
├── baseline_comparison.py              # Comparison with baseline methods
├── data_loader.py                     # Data loading and management
├── evaluate.py                        # Model evaluation script
├── train.py                          # Model training script
├── visualize_results.py              # Results visualization utilities
└── README.md                         # Project documentation
```

## Project Overview
This project implements a Transformer-based approach for detecting anomalies in time series data, focusing on cloud infrastructure metrics from the Numenta Anomaly Benchmark (NAB) dataset. The implementation achieves strong performance with PR-AUC of 0.857 and ROC-AUC of 0.976.

## Component Details

### Models
- `transformer.py`: Implements the custom Transformer architecture with self-attention mechanisms for time series processing
- `saved/best_model.pth`: Contains trained model weights achieving optimal validation performance

### Utils
- `preprocess.py`: Contains data preprocessing pipeline including:
  - Time-based feature engineering
  - Sequence windowing (288 points with stride 12)
  - Min-max scaling
  - Temporal encoding

### Results
- `analysis_report/`: Contains detailed performance analysis including:
  - Visual representations of detected anomalies
  - Comprehensive performance metrics
- Performance visualizations showing model training progress and evaluation metrics

## Installation and Setup
```bash
# Clone the repository
git clone https://github.com/yeoanni/Stats-507-final-project.git
cd Stats-507-final-project

# Install required packages
pip install -r requirements.txt
```

## Usage
- Python 3.8 or later
- Recommended: Create a virtual environment to manage dependencies.

### Training
```python
python train.py
```
This script:
- Loads and preprocesses the NAB dataset
- Trains the Transformer model
- Implements early stopping
- Saves the best model checkpoint

### Evaluation
```python
python evaluate.py
```
This script:
- Evaluates model performance
- Generates performance metrics
- Creates visualization plots

### Analysis
```python
python analyze_results.py
```
This script:
- Performs detailed analysis of model performance
- Generates visual results
- Creates comprehensive performance report

## Performance
- PR-AUC: 0.857
- ROC-AUC: 0.976
- Successfully detects anomalies in CPU utilization patterns
- Effective at identifying both sudden spikes and gradual anomalies

## Dataset
Uses the Numenta Anomaly Benchmark (NAB) dataset:
- Focus on AWS EC2 CPU utilization metrics
- 5-minute sampling intervals
- Labeled anomalies for supervised learning

## Future Work
- Real-time streaming implementation
- Multi-variate time series support
- Domain-specific fine-tuning capabilities
- Unsupervised learning approaches

## Author
Yuhan Ye (yuhanye@umich.edu)

## License
MIT License

## Acknowledgments
- NAB dataset providers
- STATS 507 course staff
