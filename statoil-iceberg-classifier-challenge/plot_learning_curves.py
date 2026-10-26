import os
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    sns.set_theme(style="whitegrid")
    
    # Extract training and validation histories from logs
    histories = {
        "Custom CNN": {
            "train_loss": [0.6502, 0.6163, 0.6058, 0.6005, 0.5829, 0.5007, 0.4739, 0.4622],
            "val_loss":   [0.6537, 0.7014, 0.6821, 0.6321, 0.5331, 0.5466, 0.4567, 0.4794],
            "train_acc":  [0.6337, 0.6719, 0.6711, 0.6820, 0.6703, 0.7443, 0.7568, 0.7381],
            "val_acc":    [0.5826, 0.5888, 0.6044, 0.6324, 0.7383, 0.7352, 0.7539, 0.7165]
        },
        "ResNet50": {
            "train_loss": [0.6521, 0.5926, 0.5614, 0.5328, 0.4966, 0.4755, 0.4442, 0.4255],
            "val_loss":   [0.5868, 0.5718, 0.5618, 0.5051, 0.4946, 0.4578, 0.4543, 0.4405],
            "train_acc":  [0.6165, 0.6797, 0.7140, 0.7436, 0.7662, 0.7677, 0.7903, 0.7935],
            "val_acc":    [0.7072, 0.7134, 0.7259, 0.7508, 0.7383, 0.7757, 0.7570, 0.7788]
        },
        "EfficientNetB0": {
            "train_loss": [0.6574, 0.6069, 0.5812, 0.5687, 0.5475, 0.5576, 0.5215, 0.5213],
            "val_loss":   [0.6270, 0.6057, 0.6009, 0.5853, 0.5722, 0.5681, 0.5436, 0.5400],
            "train_acc":  [0.6095, 0.6828, 0.6952, 0.7030, 0.7327, 0.7085, 0.7560, 0.7482],
            "val_acc":    [0.6760, 0.6729, 0.6791, 0.6885, 0.7040, 0.6916, 0.7414, 0.7321]
        },
        "MobileNetV2": {
            "train_loss": [0.6672, 0.6170, 0.6142, 0.5853, 0.5824, 0.5813, 0.5722, 0.5633],
            "val_loss":   [0.6296, 0.6225, 0.6058, 0.5932, 0.5908, 0.5955, 0.5712, 0.5797],
            "train_acc":  [0.5924, 0.6586, 0.6547, 0.6758, 0.6843, 0.7015, 0.6906, 0.7046],
            "val_acc":    [0.6542, 0.6542, 0.6604, 0.6760, 0.6854, 0.6511, 0.7040, 0.6791]
        },
        "DenseNet121": {
            "train_loss": [0.7366, 0.6168, 0.5682, 0.5481, 0.5114, 0.4695, 0.4374, 0.4367],
            "val_loss":   [0.6049, 0.5401, 0.5285, 0.5028, 0.4778, 0.4649, 0.4668, 0.4481],
            "train_acc":  [0.5425, 0.6508, 0.6890, 0.7116, 0.7319, 0.7849, 0.7958, 0.7872],
            "val_acc":    [0.6760, 0.7103, 0.7196, 0.7321, 0.7601, 0.7757, 0.7757, 0.7913]
        }
    }

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    epochs = range(1, 9)
    
    for col_idx, (model_name, history) in enumerate(histories.items()):
        # Loss curves
        axes[0, col_idx].plot(epochs, history["train_loss"], 'o-', label="Train Loss", color="#1f77b4", linewidth=2)
        axes[0, col_idx].plot(epochs, history["val_loss"], 'o--', label="Val Loss", color="#ff7f0e", linewidth=2)
        axes[0, col_idx].set_title(f"{model_name} Loss", fontsize=13, fontweight='bold')
        axes[0, col_idx].set_xlabel("Epoch", fontsize=11)
        axes[0, col_idx].set_ylabel("Loss", fontsize=11)
        axes[0, col_idx].legend(fontsize=9)
        axes[0, col_idx].grid(True, linestyle='--', alpha=0.6)
        axes[0, col_idx].set_xticks(epochs)
        
        # Accuracy curves
        axes[1, col_idx].plot(epochs, history["train_acc"], 's-', label="Train Acc", color="#2ca02c", linewidth=2)
        axes[1, col_idx].plot(epochs, history["val_acc"], 's--', label="Val Acc", color="#d62728", linewidth=2)
        axes[1, col_idx].set_title(f"{model_name} Accuracy", fontsize=13, fontweight='bold')
        axes[1, col_idx].set_xlabel("Epoch", fontsize=11)
        axes[1, col_idx].set_ylabel("Accuracy", fontsize=11)
        axes[1, col_idx].legend(fontsize=9)
        axes[1, col_idx].grid(True, linestyle='--', alpha=0.6)
        axes[1, col_idx].set_xticks(epochs)
        axes[1, col_idx].set_ylim(0.5, 0.85)

    plt.suptitle("Training vs. Validation Performance Curves Across 8 Epochs", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    out_path = "learning_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated and saved: {out_path}")

if __name__ == "__main__":
    main()
