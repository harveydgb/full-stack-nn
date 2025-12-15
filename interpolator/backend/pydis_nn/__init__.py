"""
pydis_nn: Neural network package for 5D data interpolation.
"""

__version__ = "1.0.0"

from .data import (
    load_dataset,
    split_data,
    standardize_features,
    load_and_preprocess
)
from .neuralnetwork import NeuralNetwork
from .utils import generate_sample_dataset

__all__ = [
    'load_dataset',
    'split_data',
    'standardize_features',
    'load_and_preprocess',
    'NeuralNetwork',
    'generate_sample_dataset',
    '__version__'
]

