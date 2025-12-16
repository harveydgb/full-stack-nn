"""
Data handling module for loading and preprocessing 5D datasets.

This module provides functionality to load datasets from pickle files,
validate dimensions, handle missing values, split into train/val/test sets,
and standardize features.
"""

import pickle
import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from logger import Logger

logger = Logger()


def _load_pickle_file(filepath: str) -> Dict:
    """
    Internal helper to load pickle file and perform basic validation.
    
    Loads the file and validates it's a dictionary with 'X' and 'y' keys.
    Does not validate dimensions or handle missing values.
    
    Args:
        filepath: Path to the .pkl file
        
    Returns:
        Dictionary with keys 'X' and 'y' containing raw numpy arrays
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid (not a dict, missing keys)
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Error loading pickle file: {str(e)}")
    
    # Expect dictionary format with 'X' and 'y' keys
    if not isinstance(data, dict):
        raise ValueError(f"Expected dictionary format, got {type(data)}")
    if 'X' not in data or 'y' not in data:
        raise ValueError("Dictionary must contain 'X' and 'y' keys")
    
    return data


def load_raw_dataset(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load raw dataset from .pkl file without preprocessing.
    
    Args:
        filepath: Path to the .pkl file
        
    Returns:
        Dictionary with keys 'X' and 'y' containing numpy arrays
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the data format is invalid
    """
    data = _load_pickle_file(filepath)
    
    X = np.array(data['X'])
    y = np.array(data['y'])
    
    # Convert to numpy arrays and ensure 2D for X, 1D for y
    X = np.atleast_2d(X) if X.ndim == 1 else X
    y = np.squeeze(y)
    
    return {'X': X, 'y': y}


def load_dataset(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load a 5D dataset from a pickle file.
    
    Expected format: dictionary with 'X' and 'y' keys.
    - 'X': numpy array with shape (n_samples, 5) containing features
    - 'y': numpy array with shape (n_samples,) containing targets
    
    Args:
        filepath: Path to the .pkl file
        
    Returns:
        Dictionary with keys 'X' and 'y' containing numpy arrays
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the data format is invalid or dimensions are wrong
    """
    logger.info(f"Loading dataset from {filepath}")
    data = _load_pickle_file(filepath)
    
    X = np.array(data['X'])
    y = np.array(data['y'])
    
    # Convert to numpy arrays and ensure 2D for X, 1D for y
    X = np.atleast_2d(X) if X.ndim == 1 else X
    y = np.squeeze(y)
    
    # Validate dimensions
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D array, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same number of samples. X: {X.shape[0]}, y: {y.shape[0]}")
    if X.shape[1] != 5:
        raise ValueError(f"X must have exactly 5 features, got {X.shape[1]}")
    
    # Handle missing values
    logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    X, y = _handle_missing_values(X, y)
    logger.info(f"After handling missing values: {X.shape[0]} samples")
    
    return {'X': X, 'y': y}


def _handle_missing_values(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle missing values in X and y.
    
    Strategy:
    - For X: replace NaN/inf with column mean
    - For y: remove samples with missing target values
    
    Args:
        X: Feature array (n_samples, 5)
        y: Target array (n_samples,)
        
    Returns:
        Cleaned X and y arrays
    """
    # Check for missing values in y - remove those samples entirely
    valid_mask = np.isfinite(y)
    if not np.all(valid_mask):
        n_removed = np.sum(~valid_mask)
        logger.warning(f"Removing {n_removed} samples with missing/invalid target values")
        X = X[valid_mask]
        y = y[valid_mask]
    
    # Handle missing values in X - replace with column mean
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        if np.any(~np.isfinite(col)):
            # Compute mean ignoring NaN/inf
            valid_values = col[np.isfinite(col)]
            if len(valid_values) == 0:
                # If entire column is invalid, use 0
                replacement = 0.0
            else:
                replacement = np.mean(valid_values)
            
            X[~np.isfinite(col), col_idx] = replacement
    
    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples,)
        train_size: Proportion for training set (default: 0.7)
        val_size: Proportion for validation set (default: 0.15)
        test_size: Proportion for test set (default: 0.15)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with keys 'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test'
        
    Raises:
        ValueError: If proportions don't sum to approximately 1.0
    """
    # Validate proportions sum to ~1.0
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0, atol=0.01):
        raise ValueError(f"train_size + val_size + test_size must equal 1.0, got {total}")
    
    logger.info(f"Splitting data: train={train_size:.2f}, val={val_size:.2f}, test={test_size:.2f} (n_samples={X.shape[0]})")
    
    # First split: separate test set
    test_prop = test_size / (train_size + val_size + test_size)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_prop, random_state=random_state
    )
    
    # Second split: separate train and validation from remaining data
    val_prop = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_prop, random_state=random_state
    )
    
    logger.info(f"Data split complete: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }


def standardize_features(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], StandardScaler]:
    """
    Standardize features using training set statistics.
    
    Fits scaler on training data, then transforms train/val/test sets.
    
    Args:
        X_train: Training features (n_samples, n_features)
        X_val: Validation features (optional)
        X_test: Test features (optional)
        
    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
        If X_val or X_test are None, corresponding output will be None
    """
    logger.info("Standardizing features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    logger.info("Feature standardization complete")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def load_and_preprocess(
    filepath: str,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    standardize: bool = True,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Complete pipeline: load dataset, split, and optionally standardize.
    
    Convenience function that combines all data handling steps.
    
    Args:
        filepath: Path to the .pkl file
        train_size: Proportion for training set
        val_size: Proportion for validation set
        test_size: Proportion for test set
        standardize: Whether to standardize features (default: True)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with all data arrays and optionally the scaler:
        - 'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test'
        - 'scaler' (if standardize=True)
        - 'X_train' etc. will be standardized if standardize=True
    """
    # Load and validate
    data = load_dataset(filepath)
    X, y = data['X'], data['y']
    
    # Split
    splits = split_data(X, y, train_size, val_size, test_size, random_state)
    
    # Standardize if requested
    if standardize:
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = standardize_features(
            splits['X_train'], splits['X_val'], splits['X_test']
        )
        splits['X_train'] = X_train_scaled
        splits['X_val'] = X_val_scaled
        splits['X_test'] = X_test_scaled
        splits['scaler'] = scaler
    
    return splits


def calculate_dataset_statistics(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    X_processed: np.ndarray,
    y_processed: np.ndarray
) -> Dict:
    """
    Calculate comprehensive statistics for a dataset.
    
    Computes statistics from both raw (unprocessed) and processed data:
    - Raw stats: missing values count, feature ranges (pre-standardization)
    - Processed stats: duplicate rows, memory usage, feature statistics
    
    Args:
        X_raw: Raw feature array before preprocessing (n_samples, 5)
        y_raw: Raw target array before preprocessing (n_samples,)
        X_processed: Processed feature array after handling missing values (n_samples, 5)
        y_processed: Processed target array after handling missing values (n_samples,)
        
    Returns:
        Dictionary containing:
        - 'missing_values': int - Count of missing/invalid values in raw data
        - 'feature_ranges': List[Dict] - List of {'min': float, 'max': float} for each feature (raw)
        - 'duplicate_rows': int - Number of duplicate rows in processed data
        - 'memory_usage_mb': float - Memory usage in MB for processed data
        - 'feature_stats': Dict with min_avg, max_avg, target_mean, target_std, target_min, target_max
    """
    # Calculate missing values from raw data (before any preprocessing)
    missing_values = int(np.sum(~np.isfinite(X_raw)) + np.sum(~np.isfinite(y_raw)))
    
    # Calculate feature ranges from raw data (pre-standardization, pre-missing value handling)
    feature_ranges = [
        {
            'min': float(np.nanmin(X_raw[:, i])),
            'max': float(np.nanmax(X_raw[:, i]))
        }
        for i in range(X_raw.shape[1])
    ]
    
    # Calculate duplicate rows from processed data (after missing values handled)
    duplicate_rows = int(len(X_processed) - len(np.unique(X_processed, axis=0)))
    
    # Calculate memory usage from processed data
    memory_usage_mb = float((X_processed.nbytes + y_processed.nbytes) / (1024 * 1024))
    
    # Calculate feature stats from processed data
    feature_stats = {
        'min_avg': float(X_processed.min(axis=0).mean()),
        'max_avg': float(X_processed.max(axis=0).mean()),
        'target_mean': float(y_processed.mean()),
        'target_std': float(y_processed.std()),
        'target_min': float(y_processed.min()),
        'target_max': float(y_processed.max())
    }
    
    return {
        'missing_values': missing_values,
        'feature_ranges': feature_ranges,
        'duplicate_rows': duplicate_rows,
        'memory_usage_mb': memory_usage_mb,
        'feature_stats': feature_stats
    }

