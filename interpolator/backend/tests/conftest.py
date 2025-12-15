"""
Pytest configuration and shared fixtures for tests.
"""

import pytest
import numpy as np
import pickle
import tempfile
import os
from pathlib import Path

from pydis_nn.utils import generate_sample_dataset
from pydis_nn.neuralnetwork import NeuralNetwork
from fastapi.testclient import TestClient
import sys

# Add backend directory to path for importing main
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import app


@pytest.fixture
def sample_dataset():
    """Generate a sample 5D dataset for testing."""
    return generate_sample_dataset(n=1000, seed=42)


@pytest.fixture
def sample_dataset_small():
    """Generate a small sample dataset for faster tests."""
    return generate_sample_dataset(n=100, seed=123)


@pytest.fixture
def temp_pkl_file(sample_dataset):
    """Create a temporary .pkl file with sample dataset."""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
        pickle.dump(sample_dataset, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_pkl_file_small(sample_dataset_small):
    """Create a temporary .pkl file with small sample dataset."""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
        pickle.dump(sample_dataset_small, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_dataset_with_missing():
    """Generate a dataset with some missing values for testing."""
    data = generate_sample_dataset(n=100, seed=456)
    X, y = data['X'], data['y']
    
    # Introduce some NaN values
    X[5:10, 0] = np.nan
    X[15:20, 2] = np.nan
    y[25:30] = np.nan
    
    return {'X': X, 'y': y}


@pytest.fixture
def sample_dataset_wrong_features():
    """Generate a dataset with wrong number of features (not 5)."""
    rng = np.random.default_rng(789)
    X = rng.random((100, 4))  # 4 features instead of 5
    y = rng.random(100)
    return {'X': X.astype(np.float32), 'y': y.astype(np.float32)}


@pytest.fixture
def trained_model(sample_dataset_small):
    """Create a trained neural network model for testing."""
    from pydis_nn.data import split_data, standardize_features
    
    X = sample_dataset_small['X']
    y = sample_dataset_small['y']
    
    splits = split_data(X, y, random_state=42)
    X_train_scaled, X_val_scaled, _, _ = standardize_features(
        splits['X_train'], splits['X_val']
    )
    
    model = NeuralNetwork(
        hidden_sizes=[16, 8],
        learning_rate=0.01,
        max_iter=10,
        random_state=42
    )
    
    model.fit(X_train_scaled, splits['y_train'], X_val_scaled, splits['y_val'])
    
    return model


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def app_state_reset():
    """Reset app.state before and after tests."""
    # Reset before test
    app.state.dataset_path = None
    app.state.model = None
    app.state.scaler = None
    
    yield
    
    # Reset after test
    app.state.dataset_path = None
    app.state.model = None
    app.state.scaler = None

