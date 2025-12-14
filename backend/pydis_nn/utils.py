"""
Utility functions for generating sample datasets and other helpers.
"""

import numpy as np
import math
from typing import Dict


def generate_sample_dataset(n: int = 1000, seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Generate a synthetic 5D dataset for testing and demos.
    
    Creates a dataset with 5 features (x1-x5) and a target variable y
    that is a non-linear combination of the features:
    - sin/cos terms from x1, x2
    - Exponential (Gaussian-like) term from x3, x4
    - Quadratic term from x5
    - Cross-product term from x1, x4
    - Plus Gaussian noise
    
    Args:
        n: Number of samples to generate (default: 1000)
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Dictionary with keys 'X' and 'y':
        - 'X': numpy array with shape (n, 5) containing features
        - 'y': numpy array with shape (n,) containing targets
    """
    rng = np.random.default_rng(seed)
    
    # Generate feature matrix with 5 features
    X = rng.random((n, 5))
    
    # Extract features
    x1, x2, x3, x4, x5 = X.T
    
    # Calculate target using non-linear combination
    y = (
        np.sin(2 * math.pi * x1) * np.cos(2 * math.pi * x2)
        + 0.3 * np.exp(-((x3 - 0.5) ** 2 + (x4 - 0.5) ** 2) / 0.02)
        + 0.5 * x5**2
        - 0.2 * x1 * x4
    )
    
    # Add Gaussian noise
    y += rng.normal(0, 0.01, size=n)
    
    # Return as float32 arrays, y as 1D (not column vector)
    return {
        'X': X.astype(np.float32),
        'y': y.astype(np.float32)
    }

