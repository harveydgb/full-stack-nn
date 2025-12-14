"""
Memory profiling script for training and prediction phases.

Uses tracemalloc (built-in) to profile memory usage at key points.
"""

import tracemalloc
import json
from pathlib import Path
from datetime import datetime
from typing import Dict

from pydis_nn.utils import generate_sample_dataset
from pydis_nn.data import split_data, standardize_features
from pydis_nn.neuralnetwork import NeuralNetwork


# Configuration
DATASET_SIZE = 5000  # Standard size for memory profiling
HYPERPARAMETERS = {
    'hidden_sizes': [64, 32, 16],
    'learning_rate': 0.001,
    'max_iter': 300,
    'random_state': 42,
    'early_stopping_patience': 10000  # Disable early stopping
}
SPLIT_RATIOS = {
    'train_size': 0.7,
    'val_size': 0.15,
    'test_size': 0.15
}


def get_memory_mb() -> float:
    """
    Get current peak memory usage in MB.
    
    Returns:
        Peak memory usage in megabytes
    """
    current, peak = tracemalloc.get_traced_memory()
    return peak / (1024 * 1024)


def profile_memory() -> Dict:
    """
    Profile memory usage during training and prediction phases.
    
    Returns:
        Dictionary containing memory usage at key points
    """
    print(f"\n{'='*60}")
    print(f"Memory Profiling")
    print(f"Dataset size: {DATASET_SIZE:,} samples")
    print(f"{'='*60}\n")
    
    tracemalloc.start()
    memory_profile = {}
    
    # Reset peak tracking
    tracemalloc.stop()
    tracemalloc.start()
    
    # 1. Dataset loading
    print("1. Generating dataset...")
    data = generate_sample_dataset(n=DATASET_SIZE, seed=42)
    memory_profile['after_dataset_load_mb'] = round(get_memory_mb(), 2)
    print(f"   Memory: {memory_profile['after_dataset_load_mb']:.2f} MB")
    
    # 2. Data preprocessing - splitting
    print("2. Splitting data...")
    X, y = data['X'], data['y']
    splits = split_data(
        X, y,
        train_size=SPLIT_RATIOS['train_size'],
        val_size=SPLIT_RATIOS['val_size'],
        test_size=SPLIT_RATIOS['test_size'],
        random_state=42
    )
    memory_profile['after_splitting_mb'] = round(get_memory_mb(), 2)
    print(f"   Memory: {memory_profile['after_splitting_mb']:.2f} MB")
    
    # 3. Data preprocessing - standardization
    print("3. Standardizing features...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = standardize_features(
        splits['X_train'], splits['X_val'], splits['X_test']
    )
    memory_profile['after_preprocessing_mb'] = round(get_memory_mb(), 2)
    print(f"   Memory: {memory_profile['after_preprocessing_mb']:.2f} MB")
    
    # 4. Model creation
    print("4. Creating model...")
    model = NeuralNetwork(**HYPERPARAMETERS)
    memory_profile['after_model_creation_mb'] = round(get_memory_mb(), 2)
    print(f"   Memory: {memory_profile['after_model_creation_mb']:.2f} MB")
    
    # 5. Before training
    print("5. Before training...")
    memory_profile['before_training_mb'] = round(get_memory_mb(), 2)
    print(f"   Memory: {memory_profile['before_training_mb']:.2f} MB")
    
    # 6. During training (peak)
    print("6. Training model (monitoring peak memory)...")
    model.fit(
        X_train_scaled,
        splits['y_train'],
        X_val=X_val_scaled,
        y_val=splits['y_val'],
        return_history=False
    )
    memory_profile['peak_during_training_mb'] = round(get_memory_mb(), 2)
    print(f"   Peak memory: {memory_profile['peak_during_training_mb']:.2f} MB")
    
    # 7. After training
    print("7. After training...")
    memory_profile['after_training_mb'] = round(get_memory_mb(), 2)
    print(f"   Memory: {memory_profile['after_training_mb']:.2f} MB")
    
    # 8. Before prediction
    print("8. Before prediction...")
    memory_profile['before_prediction_mb'] = round(get_memory_mb(), 2)
    print(f"   Memory: {memory_profile['before_prediction_mb']:.2f} MB")
    
    # 9. During prediction (batch of 100 samples)
    print("9. Making predictions (100 samples)...")
    test_sample = X_test_scaled[:100]
    predictions = model.predict(test_sample)
    memory_profile['peak_during_prediction_mb'] = round(get_memory_mb(), 2)
    print(f"   Peak memory: {memory_profile['peak_during_prediction_mb']:.2f} MB")
    
    # 10. After prediction
    print("10. After prediction...")
    memory_profile['after_prediction_mb'] = round(get_memory_mb(), 2)
    print(f"   Memory: {memory_profile['after_prediction_mb']:.2f} MB")
    
    tracemalloc.stop()
    
    # Calculate memory increases
    memory_profile['training_memory_increase_mb'] = round(
        memory_profile['peak_during_training_mb'] - memory_profile['before_training_mb'], 2
    )
    memory_profile['prediction_memory_increase_mb'] = round(
        memory_profile['peak_during_prediction_mb'] - memory_profile['before_prediction_mb'], 2
    )
    
    return memory_profile


def main():
    """Run memory profiling."""
    print(f"\n{'#'*60}")
    print(f"# Memory Profiling Suite")
    print(f"# Dataset size: {DATASET_SIZE:,} samples")
    print(f"# Epochs: {HYPERPARAMETERS['max_iter']} (early stopping disabled)")
    print(f"{'#'*60}")
    
    try:
        profile = profile_memory()
        
        # Prepare output
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': DATASET_SIZE,
            'hyperparameters': HYPERPARAMETERS,
            'memory_usage_mb': profile
        }
        
        # Save results to JSON
        output_dir = Path(__file__).parent.parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'memory_profile.json'
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Memory profiling complete!")
        print(f"Results saved to: {output_path}")
        print(f"{'='*60}\n")
        
        # Print summary
        print("\nMemory Usage Summary:")
        print(f"  Peak during training: {profile['peak_during_training_mb']:.2f} MB")
        print(f"  Peak during prediction: {profile['peak_during_prediction_mb']:.2f} MB")
        print(f"  Training memory increase: {profile['training_memory_increase_mb']:.2f} MB")
        print(f"  Prediction memory increase: {profile['prediction_memory_increase_mb']:.2f} MB")
        
    except Exception as e:
        tracemalloc.stop()
        print(f"\nERROR: Memory profiling failed: {e}")
        raise


if __name__ == '__main__':
    main()

