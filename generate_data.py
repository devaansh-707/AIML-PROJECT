"""Generate synthetic training data for the disease prediction model."""

import numpy as np
import pandas as pd
from pathlib import Path


def create_synthetic_dataset(n_samples: int = 300) -> pd.DataFrame:
    """Create synthetic patient data for training."""
    rng = np.random.default_rng(42)
    ages = rng.integers(20, 80, size=n_samples)
    bmi = rng.normal(28, 4, size=n_samples)
    heart_rate = rng.normal(75, 8, size=n_samples)
    gender = rng.choice(["M", "F"], size=n_samples)
    noise = rng.normal(0, 1, size=n_samples)
    outcome_prob = 1 / (1 + np.exp(-(0.04 * (ages - 50) + 0.1 * (bmi - 25) + noise)))
    outcome = (outcome_prob > 0.5).astype(int)

    df = pd.DataFrame(
        {
            "age": ages,
            "gender": gender,
            "bmi": bmi,
            "heart_rate": heart_rate,
            "discharge_notes": rng.choice(["none", "short", "long"], size=n_samples),
            "bill_amount": rng.normal(1000, 200, size=n_samples),
            "outcome": outcome,
        }
    )
    return df


if __name__ == "__main__":
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Generate and save training data
    train_df = create_synthetic_dataset(500)
    train_path = data_dir / "train.csv"
    train_df.to_csv(train_path, index=False)
    print(f"✓ Generated training data: {train_path} ({len(train_df)} samples)")
    
    # Generate and save test data (without outcome column)
    test_df = create_synthetic_dataset(100).drop(columns=["outcome"])
    test_path = data_dir / "test.csv"
    test_df.to_csv(test_path, index=False)
    print(f"✓ Generated test data: {test_path} ({len(test_df)} samples)")
    
    print("\nData generation complete! You can now run training with:")
    print("  make train")
    print("  or")
    print("  python train.py")
