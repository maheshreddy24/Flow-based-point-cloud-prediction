
# Flow-Based Point Cloud Prediction

## Overview

Flow-Based Point Cloud Prediction is a deep learning-based framework that predicts future point clouds given an initial point cloud. The model leverages **flow matching**, which learns the transformation (flow) from one point cloud to another, enabling accurate predictions of future states.

## Features
- **Flow Matching Learning**: The model learns the flow between two consecutive point clouds.
- **Future State Prediction**: Given an initial point cloud, the model predicts how it evolves over time.
- **Pretrained Weights**: The `experiments/` section includes pretrained weights for quick evaluation.
- **Modular Implementation**: Well-structured scripts for training, evaluation, and optimization.

## Codebase Structure
```
Flow-based-point-cloud-prediction/
│── experiments/        # Contains pretrained weights and experiment logs
│── datasets.py         # Data loading and preprocessing functions
│── main.py             # Entry point to train and evaluate the model
│── models.py           # Model architecture for flow matching
│── optimisation.py     # Optimization algorithms and loss functions
```

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/maheshreddy24/Flow-based-point-cloud-prediction.git
cd Flow-based-point-cloud-prediction
pip install -r requirements.txt
```

## Usage
### Running the Model
To start the training or prediction process, simply run:
```bash
python main.py
```
This script handles data loading, training, and evaluation.

### Using Pretrained Weights
To use pretrained weights for evaluation, modify the configuration in `experiments/` and load the corresponding checkpoint.

## Future Work
- Enhancing flow estimation accuracy.
- Extending support for more point cloud datasets.
- Improving efficiency in large-scale point cloud processing.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

