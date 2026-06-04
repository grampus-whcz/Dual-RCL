#!/usr/bin/env python3
"""
Step 5: Train PatternMatcher CNN model using synthetic data.

Generates synthetic time series data for 11 anomaly pattern types,
then trains a 1-D CNN classifier and saves it as patterncla.pt.

This is a standalone script that does not require GAIA data.

The 11 anomaly pattern types (matching metric_anomaly.py an_type list):
    0: Level shift up
    1: Level shift down
    2: Steady increase
    3: Steady decrease
    4: Single spike
    5: Single dip
    6: Transient level shift up
    7: Transient level shift down
    8: Multiple spikes
    9: Multiple dips
    10: Fluctuations

Usage:
    python scripts/step5_train_patternmatcher.py
    python scripts/step5_train_patternmatcher.py --samples-per-class 1000 --epochs 200
"""

import argparse
import os
import sys

import numpy as np
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from metric_anomaly import CNNClassifier


def generate_level_shift_up(length, baseline=1.0, shift=None, noise_std=0.1):
    """Generate a level shift up pattern: baseline then jump up."""
    if shift is None:
        shift = np.random.uniform(1.0, 3.0)
    shift_point = np.random.randint(length // 3, 2 * length // 3)
    signal = np.full(length, baseline)
    signal[shift_point:] += shift
    signal += np.random.randn(length) * noise_std
    return signal


def generate_level_shift_down(length, baseline=5.0, shift=None, noise_std=0.1):
    """Generate a level shift down pattern: baseline then drop."""
    if shift is None:
        shift = np.random.uniform(1.0, 3.0)
    shift_point = np.random.randint(length // 3, 2 * length // 3)
    signal = np.full(length, baseline)
    signal[shift_point:] -= shift
    signal += np.random.randn(length) * noise_std
    return signal


def generate_steady_increase(length, baseline=1.0, slope=None, noise_std=0.1):
    """Generate a steady increase pattern: linear upward trend."""
    if slope is None:
        slope = np.random.uniform(0.05, 0.3)
    signal = baseline + slope * np.arange(length)
    signal += np.random.randn(length) * noise_std
    return signal


def generate_steady_decrease(length, baseline=8.0, slope=None, noise_std=0.1):
    """Generate a steady decrease pattern: linear downward trend."""
    if slope is None:
        slope = np.random.uniform(0.05, 0.3)
    signal = baseline - slope * np.arange(length)
    signal += np.random.randn(length) * noise_std
    return signal


def generate_single_spike(length, baseline=1.0, spike_height=None, noise_std=0.1):
    """Generate a single spike pattern: baseline with one sharp peak."""
    if spike_height is None:
        spike_height = np.random.uniform(3.0, 8.0)
    spike_pos = np.random.randint(5, length - 5)
    signal = np.full(length, baseline)
    signal[spike_pos] += spike_height
    signal += np.random.randn(length) * noise_std
    return signal


def generate_single_dip(length, baseline=5.0, dip_depth=None, noise_std=0.1):
    """Generate a single dip pattern: baseline with one sharp valley."""
    if dip_depth is None:
        dip_depth = np.random.uniform(3.0, 8.0)
    dip_pos = np.random.randint(5, length - 5)
    signal = np.full(length, baseline)
    signal[dip_pos] -= dip_depth
    signal += np.random.randn(length) * noise_std
    return signal


def generate_transient_level_shift_up(length, baseline=1.0, shift=None, noise_std=0.1):
    """Generate a transient level shift up: shift up then back to normal."""
    if shift is None:
        shift = np.random.uniform(1.0, 3.0)
    start = np.random.randint(3, length // 3)
    end = start + np.random.randint(3, length // 3)
    end = min(end, length - 1)
    signal = np.full(length, baseline)
    signal[start:end] += shift
    signal += np.random.randn(length) * noise_std
    return signal


def generate_transient_level_shift_down(length, baseline=5.0, shift=None, noise_std=0.1):
    """Generate a transient level shift down: shift down then back to normal."""
    if shift is None:
        shift = np.random.uniform(1.0, 3.0)
    start = np.random.randint(3, length // 3)
    end = start + np.random.randint(3, length // 3)
    end = min(end, length - 1)
    signal = np.full(length, baseline)
    signal[start:end] -= shift
    signal += np.random.randn(length) * noise_std
    return signal


def generate_multiple_spikes(length, baseline=1.0, n_spikes=None, noise_std=0.1):
    """Generate multiple spikes: baseline with several peaks."""
    if n_spikes is None:
        n_spikes = np.random.randint(2, 5)
    signal = np.full(length, baseline)
    positions = np.random.choice(range(2, length - 2), size=min(n_spikes, length - 4), replace=False)
    for pos in positions:
        signal[pos] += np.random.uniform(2.0, 6.0)
    signal += np.random.randn(length) * noise_std
    return signal


def generate_multiple_dips(length, baseline=5.0, n_dips=None, noise_std=0.1):
    """Generate multiple dips: baseline with several valleys."""
    if n_dips is None:
        n_dips = np.random.randint(2, 5)
    signal = np.full(length, baseline)
    positions = np.random.choice(range(2, length - 2), size=min(n_dips, length - 4), replace=False)
    for pos in positions:
        signal[pos] -= np.random.uniform(2.0, 6.0)
    signal += np.random.randn(length) * noise_std
    return signal


def generate_fluctuations(length, baseline=3.0, amplitude=None, noise_std=0.1):
    """Generate fluctuations: high-frequency oscillation."""
    if amplitude is None:
        amplitude = np.random.uniform(0.5, 2.0)
    freq = np.random.uniform(0.3, 1.0)
    phase = np.random.uniform(0, 2 * np.pi)
    t = np.arange(length)
    signal = baseline + amplitude * np.sin(freq * t + phase)
    signal += np.random.randn(length) * noise_std
    return signal


# Generator functions indexed by pattern type (0-10)
PATTERN_GENERATORS = [
    generate_level_shift_up,          # 0: Level shift up
    generate_level_shift_down,        # 1: Level shift down
    generate_steady_increase,         # 2: Steady increase
    generate_steady_decrease,         # 3: Steady decrease
    generate_single_spike,            # 4: Single spike
    generate_single_dip,              # 5: Single dip
    generate_transient_level_shift_up,# 6: Transient level shift up
    generate_transient_level_shift_down,  # 7: Transient level shift down
    generate_multiple_spikes,         # 8: Multiple spikes
    generate_multiple_dips,           # 9: Multiple dips
    generate_fluctuations,            # 10: Fluctuations
]

PATTERN_NAMES = [
    'Level shift up', 'Level shift down', 'Steady increase', 'Steady decrease',
    'Single spike', 'Single dip', 'Transient level shift up', 'Transient level shift down',
    'Multiple spikes', 'Multiple dips', 'Fluctuations'
]


def generate_synthetic_data(samples_per_class=500, length=30):
    """Generate synthetic training data for all 11 pattern types.

    Args:
        samples_per_class: Number of samples per pattern type
        length: Length of each time series (default 30)

    Returns:
        X: np.ndarray of shape (N*11, 30) - time series data
        y: np.ndarray of shape (N*11,) - integer labels 0-10
    """
    X_list = []
    y_list = []

    for label, generator in enumerate(PATTERN_GENERATORS):
        for _ in range(samples_per_class):
            # Randomize noise level for diversity
            noise_std = np.random.uniform(0.05, 0.3)
            sample = generator(length, noise_std=noise_std)
            X_list.append(sample)
            y_list.append(label)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)

    # Shuffle
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]

    logger.info(f"Generated {len(y)} synthetic samples "
                f"({samples_per_class} per class, {len(PATTERN_GENERATORS)} classes)")
    return X, y


def parse_args():
    parser = argparse.ArgumentParser(
        description='Step 5: Train PatternMatcher CNN model using synthetic data'
    )
    parser.add_argument(
        '--output-path', type=str, default='./patterncla.pt',
        help='Output path for the trained model'
    )
    parser.add_argument(
        '--samples-per-class', type=int, default=500,
        help='Number of synthetic samples per pattern type'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Maximum training epochs'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)

    logger.info(f"Generating synthetic data: {args.samples_per_class} samples/class, "
                f"{len(PATTERN_GENERATORS)} classes")

    # Generate training data
    X, y = generate_synthetic_data(samples_per_class=args.samples_per_class, length=30)

    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Train CNN classifier
    logger.info("Training CNN classifier...")
    clf = CNNClassifier(class_num=11)
    clf.fit(X, y, max_epoch=args.epochs, stop_epoch=10)

    # Save model
    clf.save_model(args.output_path)
    logger.info(f"Model saved to {args.output_path}")

    # Verify load
    clf2 = CNNClassifier(class_num=11)
    clf2.load_model(args.output_path)
    logger.info("Model loaded successfully for verification")

    logger.info("Step 5 complete.")


if __name__ == '__main__':
    main()
