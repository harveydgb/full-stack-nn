"""
Benchmarking script for training time and accuracy metrics.

Measures performance across different dataset sizes (1K to 10K in 1K increments).
Early stopping is disabled for comparable results across all tests.
"""

import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np

from pydis_nn.utils import generate_sample_dataset
from pydis_nn.data import load_and_preprocess
from pydis_nn.neuralnetwork import NeuralNetwork


# Configuration
DATASET_SIZES = list(range(1000, 11000, 1000))  # 1K to 10K in 1K increments
HYPERPARAMETERS = {
    'hidden_sizes': [64, 32, 16],
    'learning_rate': 0.001,
    'max_iter': 300,  # Fixed epochs for comparability
    'random_state': 42,
    'early_stopping_patience': 10000  # Disable early stopping (never triggers)
}
SPLIT_RATIOS = {
    'train_size': 0.7,
    'val_size': 0.15,
    'test_size': 0.15
}


def benchmark_dataset_size(n_samples: int) -> Dict:
    """
    Benchmark training time and accuracy for given dataset size.
    
    Args:
        n_samples: Number of samples in the dataset
        
    Returns:
        Dictionary containing benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking dataset size: {n_samples:,} samples")
    print(f"{'='*60}")
    
    # Generate dataset
    print("Generating dataset...")
    data = generate_sample_dataset(n=n_samples, seed=42)
    
    # Preprocess (load_and_preprocess handles loading from dict in memory)
    # We need to save to temp file for load_and_preprocess, or use split_data directly
    # Let's use the direct approach for efficiency
    from pydis_nn.data import split_data, standardize_features
    
    X, y = data['X'], data['y']
    
    print("Splitting data...")
    splits = split_data(
        X, y,
        train_size=SPLIT_RATIOS['train_size'],
        val_size=SPLIT_RATIOS['val_size'],
        test_size=SPLIT_RATIOS['test_size'],
        random_state=42
    )
    
    print("Standardizing features...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = standardize_features(
        splits['X_train'], splits['X_val'], splits['X_test']
    )
    
    # Create model
    print("Creating model...")
    model = NeuralNetwork(**HYPERPARAMETERS)
    
    # Train model and measure time
    print(f"Training model (max_iter={HYPERPARAMETERS['max_iter']} epochs)...")
    start_time = time.perf_counter()
    
    model.fit(
        X_train_scaled,
        splits['y_train'],
        X_val=X_val_scaled,
        y_val=splits['y_val'],
        return_history=False
    )
    
    training_time = time.perf_counter() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    print("Evaluating model...")
    metrics = model.evaluate_all(
        X_train_scaled, splits['y_train'],
        X_val=X_val_scaled, y_val=splits['y_val'],
        X_test=X_test_scaled, y_test=splits['y_test']
    )
    
    result = {
        'dataset_size': n_samples,
        'training_time_seconds': round(training_time, 4),
        'train_r2': round(metrics['train_r2'], 6),
        'val_r2': round(metrics['val_r2'], 6),
        'test_r2': round(metrics['test_r2'], 6),
        'train_mse': round(metrics['train_mse'], 6),
        'val_mse': round(metrics['val_mse'], 6),
        'test_mse': round(metrics['test_mse'], 6),
        'epochs_used': HYPERPARAMETERS['max_iter']  # Fixed epochs
    }
    
    print(f"Results:")
    print(f"  Training time: {result['training_time_seconds']:.2f}s")
    print(f"  Test R²: {result['test_r2']:.4f}")
    print(f"  Test MSE: {result['test_mse']:.6f}")
    
    return result


def main():
    """Run benchmarks for all dataset sizes."""
    print(f"\n{'#'*60}")
    print(f"# Performance Benchmark Suite")
    print(f"# Dataset sizes: {min(DATASET_SIZES):,} to {max(DATASET_SIZES):,} (1K increments)")
    print(f"# Epochs: {HYPERPARAMETERS['max_iter']} (early stopping disabled)")
    print(f"{'#'*60}\n")
    
    results = []
    total_start_time = time.perf_counter()
    
    for size in DATASET_SIZES:
        try:
            result = benchmark_dataset_size(size)
            results.append(result)
        except Exception as e:
            print(f"ERROR: Benchmark failed for {size} samples: {e}")
            raise
    
    total_time = time.perf_counter() - total_start_time
    
    # Prepare output
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'hyperparameters': HYPERPARAMETERS,
        'split_ratios': SPLIT_RATIOS,
        'total_benchmark_time_seconds': round(total_time, 2),
        'results': results
    }
    
    # Save results to JSON
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'benchmark_results.json'
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Benchmarking complete!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")
    
    # Print summary table
    print("\nSummary:")
    print(f"{'Size':<8} {'Time (s)':<12} {'Test R²':<10} {'Test MSE':<12}")
    print("-" * 45)
    for result in results:
        print(f"{result['dataset_size']:<8} "
              f"{result['training_time_seconds']:<12.2f} "
              f"{result['test_r2']:<10.4f} "
              f"{result['test_mse']:<12.6f}")


if __name__ == '__main__':
    main()

