# pydis_nn

A lightweight Python package for training and using neural networks to interpolate 5-dimensional numerical datasets. Built with TensorFlow/Keras and optimized for fast training and inference.

## Features

- **Data Handling**: Load, validate, preprocess, and split 5D datasets (`.pkl` format)
- **Neural Network**: Configurable feedforward neural network with customizable architecture
- **Training**: Train models with optional validation data and early stopping
- **Evaluation**: Compute R² scores and MSE metrics on test sets
- **Utilities**: Generate synthetic 5D datasets for testing

## Quick Start

### Installation

```bash
pip install pydis_nn
```

### Basic Usage

```python
from pydis_nn import NeuralNetwork, load_and_preprocess

# Load and preprocess your dataset
data = load_and_preprocess('your_dataset.pkl', random_state=42)

# Create and train a neural network
model = NeuralNetwork(
    hidden_sizes=[64, 32, 16],  # 3 hidden layers
    learning_rate=0.001,
    max_iter=300,
    random_state=42
)

model.fit(
    data['X_train'],
    data['y_train'],
    X_val=data['X_val'],
    y_val=data['y_val']
)

# Make predictions
predictions = model.predict(data['X_test'])

# Evaluate performance
r2_score = model.score(data['X_test'], data['y_test'])
print(f"R² Score: {r2_score:.4f}")
```

## Documentation

📚 **Full Documentation**: [Read the Docs](https://pydis-nn.readthedocs.io/)

The documentation includes:
- API Reference
- User Guides (Installation, Usage, Dataset Format)
- Performance Profiling
- Testing Information

## Requirements

- Python >= 3.10
- NumPy >= 1.24.0
- TensorFlow >= 2.13.0
- scikit-learn >= 1.3.0
- SciPy >= 1.10.0

## Dataset Format

Your dataset should be a `.pkl` (pickle) file containing a dictionary:

```python
{
    'X': numpy.ndarray,  # Shape: (n_samples, 5) - exactly 5 features
    'y': numpy.ndarray   # Shape: (n_samples,) - target values
}
```

## Features in Detail

### Data Module (`pydis_nn.data`)

- `load_dataset()`: Load and validate 5D datasets from `.pkl` files
- `split_data()`: Split data into train/validation/test sets
- `standardize_features()`: Standardize features using training statistics
- `load_and_preprocess()`: Complete preprocessing pipeline

### Neural Network Module (`pydis_nn.neuralnetwork`)

- `NeuralNetwork`: Configurable neural network class
  - Customizable hidden layer sizes
  - Adam optimizer with configurable learning rate
  - Early stopping support
  - Returns training history for visualization

### Utilities (`pydis_nn.utils`)

- `generate_sample_dataset()`: Generate synthetic 5D datasets for testing

## Performance

The package is optimized for fast training:
- 10,000 samples train in under 32 seconds on a MacBook Air M2
- Peak memory usage: ~5 MB during training
- Efficient sub-linear scaling with dataset size

See the [Performance Profiling](https://pydis-nn.readthedocs.io/en/latest/performance.html) documentation for detailed benchmarks.

## License

MIT License - See LICENSE file for details.

## Author

Harvey Bermingham

## Links

- **Documentation**: https://pydis-nn.readthedocs.io/
- **Source Code**: https://github.com/harveydgb/full-stack-nn (or your repository URL)



