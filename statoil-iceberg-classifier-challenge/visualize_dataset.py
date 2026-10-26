import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "data", "processed", "train.json")
    
    if not os.path.exists(train_path):
        print(f"Dataset not found at: {train_path}")
        return
        
    print("Loading train.json...")
    train = pd.read_json(train_path)
    print(f"Train dataset shape: {train.shape}")
    print(f"Columns: {list(train.columns)}")
    
    # Check data types and nulls
    print("\nDataset Info:")
    print(train.info())
    
    # Class distribution
    print("\nClass distribution (is_iceberg):")
    counts = train["is_iceberg"].value_counts()
    percentages = train["is_iceberg"].value_counts(normalize=True) * 100
    for val, count in counts.items():
        print(f"  Class {val}: {count} ({percentages[val]:.2f}%)")
        
    # Analyze inc_angle
    print("\nIncidence angle info:")
    # Replace string "na" with NaN for analysis
    train["inc_angle_clean"] = pd.to_numeric(train["inc_angle"], errors="coerce")
    missing_inc = train["inc_angle_clean"].isna().sum()
    print(f"  Total missing/invalid incidence angles: {missing_inc} (out of {len(train)})")
    
    # Save a bar chart of class balance
    plt.figure(figsize=(6, 4))
    plt.bar(["Ship (0)", "Iceberg (1)"], [counts[0], counts[1]], color=["#1f77b4", "#ff7f0e"])
    plt.title("Class Balance (is_iceberg)")
    plt.ylabel("Number of Samples")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("class_balance.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Let's find one ship and one iceberg to visualize
    ship_idx = train[train["is_iceberg"] == 0].index[0]
    iceberg_idx = train[train["is_iceberg"] == 1].index[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Plot Ship (HH and HV bands)
    b1_ship = np.array(train.loc[ship_idx, "band_1"], dtype=np.float32).reshape(75, 75)
    b2_ship = np.array(train.loc[ship_idx, "band_2"], dtype=np.float32).reshape(75, 75)
    axes[0, 0].imshow(b1_ship, cmap="gray")
    axes[0, 0].set_title(f"Ship (ID: {train.loc[ship_idx, 'id']}) - Band 1 (HH)")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(b2_ship, cmap="gray")
    axes[0, 1].set_title(f"Ship (ID: {train.loc[ship_idx, 'id']}) - Band 2 (HV)")
    axes[0, 1].axis("off")
    
    # Plot Iceberg (HH and HV bands)
    b1_ice = np.array(train.loc[iceberg_idx, "band_1"], dtype=np.float32).reshape(75, 75)
    b2_ice = np.array(train.loc[iceberg_idx, "band_2"], dtype=np.float32).reshape(75, 75)
    axes[1, 0].imshow(b1_ice, cmap="gray")
    axes[1, 0].set_title(f"Iceberg (ID: {train.loc[iceberg_idx, 'id']}) - Band 1 (HH)")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(b2_ice, cmap="gray")
    axes[1, 1].set_title(f"Iceberg (ID: {train.loc[iceberg_idx, 'id']}) - Band 2 (HV)")
    axes[1, 1].axis("off")
    
    plt.tight_layout()
    plt.savefig("sar_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nSaved class_balance.png and sar_visualization.png successfully.")

if __name__ == "__main__":
    main()
