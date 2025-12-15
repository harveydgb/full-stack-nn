"""
Tests for pydis_nn.data module.
"""

import pytest
import numpy as np
import pickle
import tempfile
import os
from pathlib import Path

from pydis_nn.data import (
    load_raw_dataset,
    load_dataset,
    split_data,
    standardize_features,
    load_and_preprocess
)
from pydis_nn.utils import generate_sample_dataset


class TestLoadRawDataset:
    """Tests for load_raw_dataset function."""
    
    def test_load_valid_dataset(self, temp_pkl_file, sample_dataset):
        """Test loading a valid dataset."""
        result = load_raw_dataset(temp_pkl_file)
        
        assert 'X' in result
        assert 'y' in result
        assert result['X'].shape == sample_dataset['X'].shape
        assert result['y'].shape == sample_dataset['y'].shape
        np.testing.assert_array_equal(result['X'], sample_dataset['X'])
        np.testing.assert_array_equal(result['y'], sample_dataset['y'])
    
    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_raw_dataset("nonexistent_file.pkl")
    
    def test_load_invalid_format(self):
        """Test loading a file with invalid format."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump([1, 2, 3], f)  # Not a dict
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Expected dictionary format"):
                load_raw_dataset(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_load_missing_keys(self):
        """Test loading a dict without 'X' or 'y' keys."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump({'wrong_key': np.array([1, 2, 3])}, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="must contain 'X' and 'y' keys"):
                load_raw_dataset(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestLoadDataset:
    """Tests for load_dataset function."""
    
    def test_load_valid_5d_dataset(self, temp_pkl_file):
        """Test loading a valid 5D dataset."""
        result = load_dataset(temp_pkl_file)
        
        assert 'X' in result
        assert 'y' in result
        assert result['X'].shape[1] == 5  # Exactly 5 features
        assert result['X'].ndim == 2
        assert result['y'].ndim == 1
        assert result['X'].shape[0] == result['y'].shape[0]
    
    def test_load_dataset_wrong_features(self):
        """Test loading a dataset with wrong number of features."""
        data = generate_sample_dataset(n=100, seed=999)
        # Create dataset with 4 features instead of 5
        data['X'] = data['X'][:, :4]
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="must have exactly 5 features"):
                load_dataset(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_load_dataset_with_missing_values(self, sample_dataset_with_missing):
        """Test loading a dataset with missing values (should handle them)."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(sample_dataset_with_missing, f)
            temp_path = f.name
        
        try:
            result = load_dataset(temp_path)
            
            # Should handle missing values (remove NaN from y, impute NaN in X)
            assert np.all(np.isfinite(result['y']))
            assert np.all(np.isfinite(result['X']))
            # Should have fewer samples than original (some y values were NaN)
            assert result['X'].shape[0] < sample_dataset_with_missing['X'].shape[0]
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_load_dataset_mismatched_shapes(self):
        """Test loading a dataset where X and y have different sample counts."""
        data = generate_sample_dataset(n=100, seed=888)
        data['y'] = data['y'][:50]  # Make y shorter
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="same number of samples"):
                load_dataset(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestSplitData:
    """Tests for split_data function."""
    
    def test_split_default_ratios(self, sample_dataset):
        """Test splitting with default ratios."""
        X = sample_dataset['X']
        y = sample_dataset['y']
        
        splits = split_data(X, y, random_state=42)
        
        assert 'X_train' in splits
        assert 'X_val' in splits
        assert 'X_test' in splits
        assert 'y_train' in splits
        assert 'y_val' in splits
        assert 'y_test' in splits
        
        # Check shapes
        assert splits['X_train'].shape[0] == splits['y_train'].shape[0]
        assert splits['X_val'].shape[0] == splits['y_val'].shape[0]
        assert splits['X_test'].shape[0] == splits['y_test'].shape[0]
        
        # Check proportions (approximately)
        total_samples = X.shape[0]
        assert abs(len(splits['X_train']) / total_samples - 0.7) < 0.02
        assert abs(len(splits['X_val']) / total_samples - 0.15) < 0.02
        assert abs(len(splits['X_test']) / total_samples - 0.15) < 0.02
    
    def test_split_custom_ratios(self, sample_dataset):
        """Test splitting with custom ratios."""
        X = sample_dataset['X']
        y = sample_dataset['y']
        
        splits = split_data(X, y, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42)
        
        total_samples = X.shape[0]
        assert abs(len(splits['X_train']) / total_samples - 0.8) < 0.02
        assert abs(len(splits['X_val']) / total_samples - 0.1) < 0.02
        assert abs(len(splits['X_test']) / total_samples - 0.1) < 0.02
    
    def test_split_ratios_dont_sum_to_one(self, sample_dataset):
        """Test that invalid ratios raise ValueError."""
        X = sample_dataset['X']
        y = sample_dataset['y']
        
        with pytest.raises(ValueError, match="must equal 1.0"):
            split_data(X, y, train_size=0.5, val_size=0.3, test_size=0.3)
    
    def test_split_reproducibility(self, sample_dataset):
        """Test that split is reproducible with same random_state."""
        X = sample_dataset['X']
        y = sample_dataset['y']
        
        splits1 = split_data(X, y, random_state=42)
        splits2 = split_data(X, y, random_state=42)
        
        np.testing.assert_array_equal(splits1['X_train'], splits2['X_train'])
        np.testing.assert_array_equal(splits1['X_val'], splits2['X_val'])
        np.testing.assert_array_equal(splits1['X_test'], splits2['X_test'])


class TestStandardizeFeatures:
    """Tests for standardize_features function."""
    
    def test_standardize_train_only(self, sample_dataset):
        """Test standardizing training data only."""
        from pydis_nn.data import split_data
        
        splits = split_data(sample_dataset['X'], sample_dataset['y'], random_state=42)
        
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = standardize_features(
            splits['X_train']
        )
        
        assert X_train_scaled is not None
        assert X_val_scaled is None
        assert X_test_scaled is None
        assert scaler is not None
        
        # Check that training data is approximately standardized (mean~0, std~1)
        # Use more lenient tolerance for float32 precision
        assert np.abs(X_train_scaled.mean(axis=0)).max() < 1e-6
        assert np.abs(X_train_scaled.std(axis=0) - 1.0).max() < 1e-6
    
    def test_standardize_train_val_test(self, sample_dataset):
        """Test standardizing train, val, and test sets."""
        from pydis_nn.data import split_data
        
        splits = split_data(sample_dataset['X'], sample_dataset['y'], random_state=42)
        
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = standardize_features(
            splits['X_train'],
            X_val=splits['X_val'],
            X_test=splits['X_test']
        )
        
        assert X_train_scaled is not None
        assert X_val_scaled is not None
        assert X_test_scaled is not None
        assert scaler is not None
        
        # Check that scaler was fit on training data only
        # Use more lenient tolerance for float32 precision
        assert np.abs(X_train_scaled.mean(axis=0)).max() < 1e-6
        assert np.abs(X_train_scaled.std(axis=0) - 1.0).max() < 1e-6
        
        # Val and test should use training statistics, so they won't have mean=0, std=1
        # but should be transformed correctly
        assert X_val_scaled.shape == splits['X_val'].shape
        assert X_test_scaled.shape == splits['X_test'].shape


class TestLoadAndPreprocess:
    """Tests for load_and_preprocess function."""
    
    def test_load_and_preprocess_with_standardization(self, temp_pkl_file):
        """Test complete pipeline with standardization."""
        result = load_and_preprocess(temp_pkl_file, random_state=42)
        
        assert 'X_train' in result
        assert 'X_val' in result
        assert 'X_test' in result
        assert 'y_train' in result
        assert 'y_val' in result
        assert 'y_test' in result
        assert 'scaler' in result
        
        # Check that data is standardized
        # Use more lenient tolerance for float32 precision
        assert np.abs(result['X_train'].mean(axis=0)).max() < 1e-6
        assert np.abs(result['X_train'].std(axis=0) - 1.0).max() < 1e-6
    
    def test_load_and_preprocess_without_standardization(self, temp_pkl_file):
        """Test complete pipeline without standardization."""
        result = load_and_preprocess(temp_pkl_file, standardize=False, random_state=42)
        
        assert 'scaler' not in result
        assert 'X_train' in result
        assert 'X_val' in result
        assert 'X_test' in result
    
    def test_load_and_preprocess_custom_splits(self, temp_pkl_file):
        """Test complete pipeline with custom split ratios."""
        result = load_and_preprocess(
            temp_pkl_file,
            train_size=0.8,
            val_size=0.1,
            test_size=0.1,
            random_state=42
        )
        
        total = (len(result['X_train']) + len(result['X_val']) + len(result['X_test']))
        assert abs(len(result['X_train']) / total - 0.8) < 0.02
        assert abs(len(result['X_val']) / total - 0.1) < 0.02
        assert abs(len(result['X_test']) / total - 0.1) < 0.02

