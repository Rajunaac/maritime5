import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_logs(log_path):
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found at {log_path}")
        
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    histories = {}
    current_model = None
    
    model_regex = re.compile(r"--- Training ([\w\s\+]+) ---")
    epoch_regex = re.compile(r"Epoch (\d+): Train Loss = ([\d\.]+), Train Acc = ([\d\.]+) \| Val Loss = ([\d\.]+), Val Acc = ([\d\.]+)")
    
    for line in lines:
        model_match = model_regex.search(line)
        if model_match:
            current_model = model_match.group(1).strip()
            histories[current_model] = []
            continue
            
        if current_model:
            epoch_match = epoch_regex.search(line)
            if epoch_match:
                histories[current_model].append(float(epoch_match.group(4))) # Index 4 is Val Loss
                
    return histories

def main():
    log_path = r"C:\Users\SRMAP\.gemini\antigravity-ide\brain\a980460b-79d6-472d-b33b-be9522a465cb\.system_generated\tasks\task-262.log"
    
    try:
        histories = parse_logs(log_path)
    except Exception as e:
        print(f"Error reading log: {e}")
        return
        
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6.5))
    
    # Custom color palette for the 6 models
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    for idx, (model_name, val_losses) in enumerate(histories.items()):
        epochs = range(1, len(val_losses) + 1)
        plt.plot(epochs, val_losses, 'o-', label=f"{model_name} (Best: {min(val_losses):.4f})", 
                 color=colors[idx % len(colors)], linewidth=2, markersize=5)
                 
    plt.title("Validation Log Loss Convergence Comparison", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Log Loss", fontsize=12)
    plt.xlim(0.5, 38)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="upper right", fontsize=10, frameon=True, shadow=True)
    
    out_path = "val_log_loss_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated and saved: {out_path}")

if __name__ == "__main__":
    main()
