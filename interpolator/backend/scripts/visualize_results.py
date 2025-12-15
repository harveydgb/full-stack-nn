"""
Visualization script for benchmark and profiling results.

Generates plots from JSON results files.
"""

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def load_benchmark_results(results_path: Optional[Path] = None) -> dict:
    """Load benchmark results from JSON file."""
    if results_path is None:
        results_path = Path(__file__).parent.parent / 'outputs' / 'benchmark_results.json'
    
    if not results_path.exists():
        raise FileNotFoundError(f"Benchmark results not found: {results_path}")
    
    with open(results_path, 'r') as f:
        return json.load(f)


def load_memory_profile(profile_path: Optional[Path] = None) -> dict:
    """Load memory profile from JSON file."""
    if profile_path is None:
        profile_path = Path(__file__).parent.parent / 'outputs' / 'memory_profile.json'
    
    if not profile_path.exists():
        raise FileNotFoundError(f"Memory profile not found: {profile_path}")
    
    with open(profile_path, 'r') as f:
        return json.load(f)


def plot_training_time_vs_dataset_size(results: dict, output_dir: Path):
    """Plot training time vs dataset size."""
    dataset_sizes = [r['dataset_size'] for r in results['results']]
    training_times = [r['training_time_seconds'] for r in results['results']]
    
    plt.figure(figsize=(10, 6))
    plt.plot(dataset_sizes, training_times, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Dataset Size (samples)', fontsize=12)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.title('Training Time vs Dataset Size', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / 'training_time_vs_dataset_size.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_vs_dataset_size(results: dict, output_dir: Path):
    """Plot R² and MSE vs dataset size."""
    dataset_sizes = [r['dataset_size'] for r in results['results']]
    test_r2 = [r['test_r2'] for r in results['results']]
    test_mse = [r['test_mse'] for r in results['results']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # R² plot
    ax1.plot(dataset_sizes, test_r2, marker='o', linewidth=2, markersize=8, color='green')
    ax1.set_xlabel('Dataset Size (samples)', fontsize=12)
    ax1.set_ylabel('Test R² Score', fontsize=12)
    ax1.set_title('Test R² vs Dataset Size', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(test_r2) * 0.99, max(test_r2) * 1.01])
    
    # MSE plot
    ax2.plot(dataset_sizes, test_mse, marker='o', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('Dataset Size (samples)', fontsize=12)
    ax2.set_ylabel('Test MSE', fontsize=12)
    ax2.set_title('Test MSE vs Dataset Size', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'accuracy_vs_dataset_size.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_memory_usage(profile: dict, output_dir: Path):
    """Plot memory usage at different phases."""
    memory_data = profile['memory_usage_mb']
    
    phases = [
        'after_dataset_load',
        'after_preprocessing',
        'before_training',
        'peak_during_training',
        'after_training',
        'before_prediction',
        'peak_during_prediction',
        'after_prediction'
    ]
    
    values = [memory_data.get(f'{phase}_mb', 0) for phase in phases]
    labels = [phase.replace('_', ' ').title() for phase in phases]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(phases)), values, color='steelblue', alpha=0.7)
    
    # Highlight peak phases
    peak_indices = [3, 6]  # peak_during_training, peak_during_prediction
    for idx in peak_indices:
        bars[idx].set_color('crimson')
        bars[idx].set_alpha(0.9)
    
    plt.xlabel('Phase', fontsize=12)
    plt.ylabel('Memory Usage (MB)', fontsize=12)
    plt.title('Memory Usage Across Different Phases', fontsize=14, fontweight='bold')
    plt.xticks(range(len(phases)), labels, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / 'memory_usage_by_phase.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_scaling_analysis(results: dict, output_dir: Path):
    """Plot scaling analysis (time per sample)."""
    dataset_sizes = np.array([r['dataset_size'] for r in results['results']])
    training_times = np.array([r['training_time_seconds'] for r in results['results']])
    time_per_sample = training_times / dataset_sizes
    
    plt.figure(figsize=(10, 6))
    plt.plot(dataset_sizes, time_per_sample, marker='o', linewidth=2, markersize=8, color='purple')
    plt.xlabel('Dataset Size (samples)', fontsize=12)
    plt.ylabel('Training Time per Sample (seconds)', fontsize=12)
    plt.title('Scaling Analysis: Training Time per Sample', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / 'scaling_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all visualizations from benchmark and profiling results."""
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print(f"{'='*60}\n")
    
    # Load benchmark results
    try:
        benchmark_results = load_benchmark_results()
        print(f"✓ Loaded benchmark results ({len(benchmark_results['results'])} dataset sizes)")
        
        # Generate benchmark plots
        plot_training_time_vs_dataset_size(benchmark_results, output_dir)
        plot_accuracy_vs_dataset_size(benchmark_results, output_dir)
        plot_scaling_analysis(benchmark_results, output_dir)
        
    except FileNotFoundError as e:
        print(f"⚠ Skipping benchmark visualizations: {e}")
    
    # Load memory profile
    try:
        memory_profile = load_memory_profile()
        print(f"✓ Loaded memory profile")
        
        # Generate memory plot
        plot_memory_usage(memory_profile, output_dir)
        
    except FileNotFoundError as e:
        print(f"⚠ Skipping memory visualizations: {e}")
    
    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

