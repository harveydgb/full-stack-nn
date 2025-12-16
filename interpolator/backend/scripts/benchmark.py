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
N_RUNS = 5  # Number of runs per dataset size for statistical robustness
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


def calculate_statistics(run_results: List[Dict]) -> Dict:
    """
    Calculate statistics (mean, std, min, max) across multiple runs.
    
    Args:
        run_results: List of result dictionaries from individual runs
        
    Returns:
        Dictionary containing statistics for each metric
    """
    metrics = [
        'training_time_seconds', 'train_r2', 'val_r2', 'test_r2',
        'train_mse', 'val_mse', 'test_mse'
    ]
    
    stats = {}
    for metric in metrics:
        values = [r[metric] for r in run_results]
        stats[f'{metric}_mean'] = round(float(np.mean(values)), 6)
        stats[f'{metric}_std'] = round(float(np.std(values)), 6)
        stats[f'{metric}_min'] = round(float(np.min(values)), 6)
        stats[f'{metric}_max'] = round(float(np.max(values)), 6)
    
    return stats


def benchmark_dataset_size(n_samples: int, run_number: int = None) -> Dict:
    """
    Benchmark training time and accuracy for given dataset size (single run).
    
    Args:
        n_samples: Number of samples in the dataset
        run_number: Optional run number for logging (1-indexed)
        
    Returns:
        Dictionary containing benchmark results for a single run
    """
    run_label = f" (Run {run_number}/{N_RUNS})" if run_number else ""
    print(f"\n{'='*60}")
    print(f"Benchmarking dataset size: {n_samples:,} samples{run_label}")
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
    """Run benchmarks for all dataset sizes with multiple runs for statistical robustness."""
    print(f"\n{'#'*60}")
    print(f"# Performance Benchmark Suite")
    print(f"# Dataset sizes: {min(DATASET_SIZES):,} to {max(DATASET_SIZES):,} (1K increments)")
    print(f"# Runs per size: {N_RUNS} (for statistical robustness)")
    print(f"# Epochs: {HYPERPARAMETERS['max_iter']} (early stopping disabled)")
    print(f"{'#'*60}\n")
    
    results = []
    total_start_time = time.perf_counter()
    
    for size in DATASET_SIZES:
        try:
            print(f"\n{'#'*60}")
            print(f"# Dataset Size: {size:,} samples ({N_RUNS} runs)")
            print(f"{'#'*60}")
            
            # Run multiple times
            run_results = []
            for run_idx in range(1, N_RUNS + 1):
                result = benchmark_dataset_size(size, run_number=run_idx)
                run_results.append(result)
            
            # Calculate statistics across runs
            stats = calculate_statistics(run_results)
            
            # Create aggregated result with statistics
            aggregated_result = {
                'dataset_size': size,
                'n_runs': N_RUNS,
                'epochs_used': HYPERPARAMETERS['max_iter'],
                **stats,
                'individual_runs': run_results  # Keep individual runs for detailed analysis
            }
            
            results.append(aggregated_result)
            
            # Print summary for this dataset size
            print(f"\nSummary for {size:,} samples (across {N_RUNS} runs):")
            print(f"  Training time: {stats['training_time_seconds_mean']:.2f}s "
                  f"(±{stats['training_time_seconds_std']:.2f}s)")
            print(f"  Test R²: {stats['test_r2_mean']:.4f} "
                  f"(±{stats['test_r2_std']:.4f}, range: [{stats['test_r2_min']:.4f}, {stats['test_r2_max']:.4f}])")
            print(f"  Test MSE: {stats['test_mse_mean']:.6f} "
                  f"(±{stats['test_mse_std']:.6f})")
            
        except Exception as e:
            print(f"ERROR: Benchmark failed for {size} samples: {e}")
            raise
    
    total_time = time.perf_counter() - total_start_time
    
    # Prepare output
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'hyperparameters': HYPERPARAMETERS,
        'split_ratios': SPLIT_RATIOS,
        'n_runs_per_size': N_RUNS,
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
    print("\n" + "="*60)
    print("SUMMARY TABLE (averaged across runs)")
    print("="*60)
    print(f"{'Size':<8} {'Time (s)':<15} {'Test R²':<15} {'Test MSE':<15}")
    print(f"{'':<8} {'Mean ± Std':<15} {'Mean ± Std':<15} {'Mean ± Std':<15}")
    print("-" * 60)
    for result in results:
        print(f"{result['dataset_size']:<8} "
              f"{result['training_time_seconds_mean']:.2f}±{result['training_time_seconds_std']:.2f}  "
              f"{result['test_r2_mean']:.4f}±{result['test_r2_std']:.4f}    "
              f"{result['test_mse_mean']:.6f}±{result['test_mse_std']:.6f}")


if __name__ == '__main__':
    main()

