import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "results_comparison.csv")
    
    if not os.path.exists(csv_path):
        print(f"Results CSV not found at: {csv_path}")
        return
        
    print("Loading results...")
    df = pd.read_json(csv_path) if csv_path.endswith('.json') else pd.read_csv(csv_path)
    
    # Preprocess strings with commas (like parameters)
    df["Params_numeric"] = df["Params"].astype(str).str.replace(",", "").astype(float)
        
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Define custom color palette
    colors = sns.color_palette("viridis", len(df))
    
    # 1. Validation Log Loss (Lower is better)
    df_sorted_loss = df.sort_values(by="Log Loss")
    sns.barplot(x="Model", y="Log Loss", data=df_sorted_loss, ax=axes[0, 0], palette="Blues_r")
    axes[0, 0].set_title("Validation Log Loss (Lower is Better)", fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel("Model", fontsize=12)
    axes[0, 0].set_ylabel("Log Loss", fontsize=12)
    for p in axes[0, 0].patches:
        axes[0, 0].annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10, fontweight='bold')
                            
    # 2. Validation Accuracy & ROC-AUC (Higher is better)
    df_melted = df.melt(id_vars=["Model"], value_vars=["Accuracy", "ROC-AUC"], var_name="Metric", value_name="Score")
    sns.barplot(x="Model", y="Score", hue="Metric", data=df_melted, ax=axes[0, 1], palette="Set2")
    axes[0, 1].set_title("Validation Accuracy & ROC-AUC Comparison", fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel("Model", fontsize=12)
    axes[0, 1].set_ylabel("Score", fontsize=12)
    axes[0, 1].set_ylim(0.6, 1.0)
    axes[0, 1].legend(loc="lower right")
    for p in axes[0, 1].patches:
        if p.get_height() > 0: # Avoid annotations for empty bars
            axes[0, 1].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)

    # 3. Model Efficiency: Parameters vs. Log Loss
    sns.scatterplot(x="Params_numeric", y="Log Loss", hue="Model", style="Model", data=df, ax=axes[1, 0], s=250, palette="deep")
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_title("Model Complexity vs. Log Loss", fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel("Number of Parameters (Log Scale)", fontsize=12)
    axes[1, 0].set_ylabel("Validation Log Loss", fontsize=12)
    axes[1, 0].grid(True, which="both", ls="--")
    
    # Annotate scatter points
    for idx, row in df.iterrows():
        axes[1, 0].annotate(row["Model"], (row["Params_numeric"], row["Log Loss"]),
                            textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, fontweight='bold')

    # 4. Training Time
    df_sorted_time = df.sort_values(by="Training Time (s)")
    sns.barplot(x="Model", y="Training Time (s)", data=df_sorted_time, ax=axes[1, 1], palette="rocket")
    axes[1, 1].set_title("Training Time Comparison (seconds)", fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel("Model", fontsize=12)
    axes[1, 1].set_ylabel("Training Time (s)", fontsize=12)
    for p in axes[1, 1].patches:
        axes[1, 1].annotate(f"{p.get_height():.1f}s", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig("results_comparison_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Saved results_comparison_dashboard.png successfully.")

if __name__ == "__main__":
    main()
