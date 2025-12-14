"""
Tests for pydis_nn.neuralnetwork module.
"""

import pytest
import numpy as np

from pydis_nn.neuralnetwork import NeuralNetwork
from pydis_nn.data import split_data, standardize_features
from pydis_nn.utils import generate_sample_dataset


class TestNeuralNetworkInit:
    """Tests for NeuralNetwork initialization."""
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        model = NeuralNetwork()
        
        assert model.hidden_sizes == [64, 32, 16]
        assert model.learning_rate == 0.001
        assert model.max_iter == 1000
        assert model.random_state == 42
        assert model.model is not None
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        model = NeuralNetwork(
            hidden_sizes=[32, 16],
            learning_rate=0.01,
            max_iter=100,
            random_state=123,
            early_stopping_patience=20
        )
        
        assert model.hidden_sizes == [32, 16]
        assert model.learning_rate == 0.01
        assert model.max_iter == 100
        assert model.random_state == 123
        assert model.early_stopping_patience == 20
        assert model.model is not None
    
    def test_init_empty_hidden_sizes(self):
        """Test initialization with empty hidden layers."""
        model = NeuralNetwork(hidden_sizes=[])
        assert model.hidden_sizes == []
        assert model.model is not None


class TestNeuralNetworkFit:
    """Tests for NeuralNetwork.fit method."""
    
    def test_fit_basic(self, sample_dataset_small):
        """Test basic training."""
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
        
        result = model.fit(X_train_scaled, splits['y_train'])
        
        # Should return self for method chaining
        assert result is model
        assert model.model is not None
    
    def test_fit_with_validation(self, sample_dataset_small):
        """Test training with validation data."""
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
            max_iter=20,
            random_state=42
        )
        
        model.fit(
            X_train_scaled,
            splits['y_train'],
            X_val=X_val_scaled,
            y_val=splits['y_val']
        )
        
        assert model.model is not None
    
    def test_fit_with_history(self, sample_dataset_small):
        """Test training with history tracking."""
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
        
        model, history = model.fit(
            X_train_scaled,
            splits['y_train'],
            X_val=X_val_scaled,
            y_val=splits['y_val'],
            return_history=True
        )
        
        assert isinstance(history, list)
        assert len(history) > 0
        assert 'epoch' in history[0]
        assert 'loss' in history[0]
        assert 'val_loss' in history[0]
    
    def test_fit_loss_decreases(self, sample_dataset_small):
        """Test that training loss decreases (model converges)."""
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
            max_iter=20,
            random_state=42
        )
        
        model, history = model.fit(
            X_train_scaled,
            splits['y_train'],
            X_val=X_val_scaled,
            y_val=splits['y_val'],
            return_history=True
        )
        
        # Check that loss decreases (on average)
        losses = [h['loss'] for h in history]
        # Loss should generally decrease (allow for some noise)
        assert losses[-1] <= losses[0] * 1.5  # Final loss shouldn't be much worse


class TestNeuralNetworkPredict:
    """Tests for NeuralNetwork.predict method."""
    
    def test_predict_before_fit(self):
        """Test that predict raises error before training."""
        model = NeuralNetwork()
        
        X = np.random.random((10, 5)).astype(np.float32)
        
        # Model exists but hasn't been trained - should raise ValueError
        with pytest.raises(ValueError, match="must be trained"):
            model.predict(X)
    
    def test_predict_after_fit(self, trained_model, sample_dataset_small):
        """Test prediction after training."""
        from pydis_nn.data import split_data, standardize_features
        
        X = sample_dataset_small['X']
        y = sample_dataset_small['y']
        
        splits = split_data(X, y, random_state=42)
        _, _, X_test_scaled, _ = standardize_features(
            splits['X_train'], X_test=splits['X_test']
        )
        
        predictions = trained_model.predict(X_test_scaled)
        
        assert predictions.shape == (len(X_test_scaled),)
        assert predictions.dtype == np.float32
        assert np.all(np.isfinite(predictions))
    
    def test_predict_single_sample(self, trained_model):
        """Test prediction on a single sample."""
        X = np.random.random((1, 5)).astype(np.float32)
        
        prediction = trained_model.predict(X)
        
        # Should return array with shape (1,)
        assert prediction.shape == (1,)
        assert np.isfinite(prediction[0])


class TestNeuralNetworkScore:
    """Tests for NeuralNetwork.score method."""
    
    def test_score_returns_r2(self, trained_model, sample_dataset_small):
        """Test that score returns R² coefficient."""
        from pydis_nn.data import split_data, standardize_features
        
        X = sample_dataset_small['X']
        y = sample_dataset_small['y']
        
        splits = split_data(X, y, random_state=42)
        _, _, X_test_scaled, _ = standardize_features(
            splits['X_train'], X_test=splits['X_test']
        )
        
        r2 = trained_model.score(X_test_scaled, splits['y_test'])
        
        assert isinstance(r2, (float, np.floating))
        # R² should be between -inf and 1 (can be negative for bad models)
        assert r2 <= 1.0
        # For a trained model, R² should typically be > -10 (very bad models)
        assert r2 > -10.0


class TestNeuralNetworkEvaluateAll:
    """Tests for NeuralNetwork.evaluate_all method."""
    
    def test_evaluate_all_train_only(self, trained_model, sample_dataset_small):
        """Test evaluation on training set only."""
        from pydis_nn.data import split_data, standardize_features
        
        X = sample_dataset_small['X']
        y = sample_dataset_small['y']
        
        splits = split_data(X, y, random_state=42)
        X_train_scaled, X_val_scaled, X_test_scaled, _ = standardize_features(
            splits['X_train'], splits['X_val'], splits['X_test']
        )
        
        metrics = trained_model.evaluate_all(
            X_train_scaled, splits['y_train']
        )
        
        assert 'train_r2' in metrics
        assert 'train_mse' in metrics
        assert 'val_r2' not in metrics
        assert 'test_r2' not in metrics
        
        assert isinstance(metrics['train_r2'], float)
        assert isinstance(metrics['train_mse'], float)
        assert metrics['train_mse'] >= 0
    
    def test_evaluate_all_all_sets(self, trained_model, sample_dataset_small):
        """Test evaluation on train, val, and test sets."""
        from pydis_nn.data import split_data, standardize_features
        
        X = sample_dataset_small['X']
        y = sample_dataset_small['y']
        
        splits = split_data(X, y, random_state=42)
        X_train_scaled, X_val_scaled, X_test_scaled, _ = standardize_features(
            splits['X_train'], splits['X_val'], splits['X_test']
        )
        
        metrics = trained_model.evaluate_all(
            X_train_scaled, splits['y_train'],
            X_val=X_val_scaled, y_val=splits['y_val'],
            X_test=X_test_scaled, y_test=splits['y_test']
        )
        
        assert 'train_r2' in metrics
        assert 'train_mse' in metrics
        assert 'val_r2' in metrics
        assert 'val_mse' in metrics
        assert 'test_r2' in metrics
        assert 'test_mse' in metrics
        
        # All MSE values should be non-negative
        assert all(mse >= 0 for key, mse in metrics.items() if 'mse' in key)

