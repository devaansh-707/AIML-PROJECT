"""Generate synthetic multi-disease dataset for demonstration."""

import numpy as np
import pandas as pd
from pathlib import Path


def create_multi_disease_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """Create synthetic patient data with multiple disease types."""
    rng = np.random.default_rng(42)
    
    # Feature columns
    ages = rng.integers(18, 85, size=n_samples)
    gender = rng.choice(["M", "F"], size=n_samples)
    bmi = rng.normal(27, 5, size=n_samples).clip(15, 50)
    heart_rate = rng.normal(75, 12, size=n_samples).clip(50, 120)
    blood_pressure = rng.normal(120, 15, size=n_samples).clip(80, 180)
    cholesterol = rng.normal(200, 40, size=n_samples).clip(120, 350)
    
    # Define diseases with different risk factors
    diseases = []
    
    for i in range(n_samples):
        age = ages[i]
        bmi_val = bmi[i]
        hr = heart_rate[i]
        bp = blood_pressure[i]
        chol = cholesterol[i]
        
        # Risk scores for different diseases
        diabetes_risk = 0.05 * (bmi_val - 25) + 0.03 * (age - 40) + rng.normal(0, 2)
        hypertension_risk = 0.08 * (bp - 120) + 0.04 * (age - 50) + rng.normal(0, 2)
        heart_disease_risk = 0.06 * (chol - 200) + 0.05 * (age - 55) + 0.03 * (bmi_val - 25) + rng.normal(0, 2)
        asthma_risk = -0.02 * age + rng.normal(0, 3)
        arthritis_risk = 0.08 * (age - 45) + rng.normal(0, 2)
        
        # Assign disease based on highest risk
        risks = {
            "Diabetes": diabetes_risk,
            "Hypertension": hypertension_risk,
            "Heart_Disease": heart_disease_risk,
            "Asthma": asthma_risk,
            "Arthritis": arthritis_risk,
            "Healthy": rng.normal(-3, 1),  # Baseline for healthy
        }
        
        diagnosed_disease = max(risks, key=risks.get)
        diseases.append(diagnosed_disease)
    
    df = pd.DataFrame({
        "age": ages,
        "gender": gender,
        "bmi": bmi,
        "heart_rate": heart_rate,
        "blood_pressure": blood_pressure,
        "cholesterol": cholesterol,
        "prognosis": diseases,  # Target column with disease names
    })
    
    return df


if __name__ == "__main__":
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Generate and save training data
    train_df = create_multi_disease_dataset(1000)
    train_path = data_dir / "train.csv"
    train_df.to_csv(train_path, index=False)
    print(f"✓ Generated training data: {train_path} ({len(train_df)} samples)")
    print(f"  Diseases: {train_df['prognosis'].value_counts().to_dict()}")
    
    # Generate and save test data (without prognosis column)
    test_df = create_multi_disease_dataset(200).drop(columns=["prognosis"])
    test_path = data_dir / "test.csv"
    test_df.to_csv(test_path, index=False)
    print(f"✓ Generated test data: {test_path} ({len(test_df)} samples)")
    
    print("\nData generation complete! You can now run:")
    print("  make train")
    print("  make infer")
