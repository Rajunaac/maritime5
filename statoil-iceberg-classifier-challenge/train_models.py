import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix, roc_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------------------------------
# 1. Dataset and Data Loader Definition
# ----------------------------------------------------
class IcebergDataset(Dataset):
    def __init__(self, X, y=None, transform=None):
        # PyTorch expects (Channels, Height, Width) format, i.e., shape (N, C, H, W)
        # X shape is (N, 75, 75, 3), we need to transpose to (N, 3, 75, 75)
        X_transposed = np.transpose(X, (0, 3, 1, 2))
        self.X = torch.tensor(X_transposed, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        if self.transform:
            # Apply transform to PyTorch tensor
            img = self.transform(img)
        if self.y is not None:
            return img, self.y[idx]
        return img

# ----------------------------------------------------
# 2. Model Definitions
# ----------------------------------------------------
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Conv block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Conv block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Conv block 3
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling & classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

def get_resnet50():
    # Load pretrained resnet50
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Replace final linear layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 1),
        nn.Sigmoid()
    )
    return model

def get_efficientnet_b0():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 1),
        nn.Sigmoid()
    )
    return model

def get_mobilenet_v2():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 1),
        nn.Sigmoid()
    )
    return model

def get_densenet121():
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 1),
        nn.Sigmoid()
    )
    return model

class ResNetDenseNetAttention(nn.Module):
    def __init__(self):
        super(ResNetDenseNetAttention, self).__init__()
        # Pretrained backbones
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet_features = nn.Sequential(*list(self.resnet.children())[:-1]) # GAP output, shape (N, 2048, 1, 1)
        
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.densenet_features = nn.Sequential(
            self.densenet.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # GAP output, shape (N, 1024, 1, 1)
        )
        
        # Feature projections
        self.resnet_proj = nn.Linear(2048, 512)
        self.densenet_proj = nn.Linear(1024, 512)
        
        # Multi-head Self-Attention
        self.attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extract features
        r_feat = self.resnet_features(x).view(x.size(0), -1) # (N, 2048)
        d_feat = self.densenet_features(x).view(x.size(0), -1) # (N, 1024)
        
        # Project to same dimension
        r_proj = self.resnet_proj(r_feat) # (N, 512)
        d_proj = self.densenet_proj(d_feat) # (N, 512)
        
        # Stack features: shape (N, 2, 512)
        stacked = torch.stack([r_proj, d_proj], dim=1)
        
        # Apply Self-Attention
        attn_out, _ = self.attn(stacked, stacked, stacked) # (N, 2, 512)
        
        # Average pooling over the two models' features
        pooled = attn_out.mean(dim=1) # (N, 512)
        
        # Final classification
        out = self.classifier(pooled)
        return out

# ----------------------------------------------------
# 3. Model Training Loop
# ----------------------------------------------------
def train_model(model, train_loader, val_loader, model_name, epochs, lr, is_pretrained=False, fine_tune_epochs=0, fine_tune_lr=1e-5):
    print(f"\n--- Training {model_name} ---")
    model = model.to(device)
    criterion = nn.BCELoss()
    
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }
    
    best_val_loss = float("inf")
    checkpoint_path = f"best_{model_name.lower().replace(' ', '_')}.pth"
    
    # Phase 1: Train Head Only (if pre-trained) or Train Full Model (if custom)
    if is_pretrained:
        print("Phase 1: Training classifier head only...")
        if "Attention" in model_name:
            # Freeze backbones specifically
            for param in model.resnet.parameters():
                param.requires_grad = False
            for param in model.densenet.parameters():
                param.requires_grad = False
            # Rest of the parameters (projections, attention, classifier) remain trainable
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(trainable_params, lr=lr)
        else:
            # Freeze backbone
            for param in model.parameters():
                param.requires_grad = False
            
            # Unfreeze classifier parameters
            if hasattr(model, "fc"):
                for param in model.fc.parameters():
                    param.requires_grad = True
                optimizer = optim.Adam(model.fc.parameters(), lr=lr)
            elif hasattr(model, "classifier"):
                for param in model.classifier.parameters():
                    param.requires_grad = True
                optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
            else:
                optimizer = optim.Adam(model.parameters(), lr=lr)
            
        total_epochs = epochs
    else:
        print("Training custom model from scratch...")
        optimizer = optim.Adam(model.parameters(), lr=lr)
        total_epochs = epochs

    def run_epochs(num_eps, opt, start_ep=0, patience=8):
        nonlocal best_val_loss
        epochs_no_improve = 0
        for epoch in range(num_eps):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                opt.step()
                
                train_loss += loss.item() * imgs.size(0)
                preds = (outputs >= 0.5).float()
                train_correct += (preds == labels).sum().item()
                train_total += imgs.size(0)
                
            epoch_train_loss = train_loss / train_total
            epoch_train_acc = train_correct / train_total
            
            # Validation
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * imgs.size(0)
                    preds = (outputs >= 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += imgs.size(0)
                    
            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            
            history["train_loss"].append(epoch_train_loss)
            history["val_loss"].append(epoch_val_loss)
            history["train_acc"].append(epoch_train_acc)
            history["val_acc"].append(epoch_val_acc)
            
            print(f"Epoch {start_ep + epoch + 1:02d}: "
                  f"Train Loss = {epoch_train_loss:.4f}, Train Acc = {epoch_train_acc:.4f} | "
                  f"Val Loss = {epoch_val_loss:.4f}, Val Acc = {epoch_val_acc:.4f}")
            
            # Checkpoint saving based on lowest val log loss
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  --> Saved new best checkpoint to {checkpoint_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"  --> Early stopping triggered at epoch {start_ep + epoch + 1} (no val loss improvement for {patience} epochs).")
                    break

    # Run head training
    run_epochs(total_epochs, optimizer, start_ep=0)
    
    # Phase 2: Fine-Tuning (if pre-trained and fine_tune_epochs > 0)
    if is_pretrained and fine_tune_epochs > 0:
        print("\nPhase 2: Fine-tuning full network with small learning rate...")
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
        
        # Load best weights from head training
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            
        ft_optimizer = optim.Adam(model.parameters(), lr=fine_tune_lr)
        run_epochs(fine_tune_epochs, ft_optimizer, start_ep=total_epochs)
        
    # Load best overall weights before returning
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        
    return model, history

# ----------------------------------------------------
# 4. Evaluation Helper
# ----------------------------------------------------
def evaluate_model(model, loader):
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            all_probs.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(labels.numpy().flatten())
            
    probs = np.array(all_probs)
    targets = np.array(all_targets)
    preds = (probs >= 0.5).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(targets, preds),
        "precision": precision_score(targets, preds, zero_division=0),
        "recall": recall_score(targets, preds, zero_division=0),
        "f1": f1_score(targets, preds, zero_division=0),
        "roc_auc": roc_auc_score(targets, probs),
        "log_loss": log_loss(targets, probs),
        "probs": probs,
        "targets": targets,
        "preds": preds
    }
    return metrics

# ----------------------------------------------------
# Main Execution Pipeline
# ----------------------------------------------------
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "data", "processed", "train.json")
    test_path = os.path.join(base_dir, "data", "processed", "test.json")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Datasets train.json or test.json are missing in data/processed/. Please run extract_dataset.py first.")
        return
        
    # Load JSON files
    print("Loading train.json dataset (this might take a few moments)...")
    train = pd.read_json(train_path)
    print("Loading test.json dataset...")
    test = pd.read_json(test_path)
    
    print(f"Train samples: {len(train)}, Test samples: {len(test)}")
    
    # ----------------------------------------------------
    # Data Reshaping & Band Stacking (HH, HV, Average)
    # ----------------------------------------------------
    print("Reshaping flattened bands and creating 3-channel input...")
    
    def process_bands(df):
        band_1 = np.array([np.array(band, dtype=np.float32).reshape(75, 75) for band in df["band_1"]])
        band_2 = np.array([np.array(band, dtype=np.float32).reshape(75, 75) for band in df["band_2"]])
        band_3 = (band_1 + band_2) / 2.0
        # Stack channels to get shape (N, 75, 75, 3)
        X = np.stack([band_1, band_2, band_3], axis=-1)
        return X

    X = process_bands(train)
    y = train["is_iceberg"].values
    
    X_test = process_bands(test)
    test_ids = test["id"].values
    
    # ----------------------------------------------------
    # Stratified Train/Validation Split
    # ----------------------------------------------------
    print("Splitting train/validation stratified (80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    
    # ----------------------------------------------------
    # Normalization (z-score scaling based only on train split)
    # ----------------------------------------------------
    print("Normalizing images based on train split statistics...")
    # Compute mean and standard deviation for each channel over training split
    train_mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
    train_std = X_train.std(axis=(0, 1, 2), keepdims=True)
    
    # Standardize
    X_train_norm = (X_train - train_mean) / (train_std + 1e-6)
    X_val_norm = (X_val - train_mean) / (train_std + 1e-6)
    X_test_norm = (X_test - train_mean) / (train_std + 1e-6)
    
    # ----------------------------------------------------
    # Data Augmentation Setup
    # ----------------------------------------------------
    # Data Augmentation for training split
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip()
    ])
    
    # Create PyTorch datasets and loaders
    train_dataset = IcebergDataset(X_train_norm, y_train, transform=train_transform)
    val_dataset = IcebergDataset(X_val_norm, y_val)
    test_dataset = IcebergDataset(X_test_norm)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # ----------------------------------------------------
    # Train Models and Collect Histories
    # ----------------------------------------------------
    # Number of epochs (increased for best final performance, with early stopping validation checks)
    custom_epochs = 35
    pretrained_head_epochs = 12
    pretrained_ft_epochs = 25
    
    models_dict = {
        "Custom CNN": (CustomCNN(), False),
        "ResNet50": (get_resnet50(), True),
        "EfficientNetB0": (get_efficientnet_b0(), True),
        "MobileNetV2": (get_mobilenet_v2(), True),
        "DenseNet121": (get_densenet121(), True),
        "ResNet50 + DenseNet121 + Attention": (ResNetDenseNetAttention(), True),
    }
    
    trained_models = {}
    histories = {}
    results = {}
    
    for model_name, (model_inst, is_pretrained) in models_dict.items():
        t_start = time.time()
        
        # Train model
        model_trained, history = train_model(
            model=model_inst,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=model_name,
            epochs=pretrained_head_epochs if is_pretrained else custom_epochs,
            lr=1e-3,
            is_pretrained=is_pretrained,
            fine_tune_epochs=pretrained_ft_epochs if is_pretrained else 0,
            fine_tune_lr=1e-5
        )
        
        t_duration = time.time() - t_start
        trained_models[model_name] = model_trained
        histories[model_name] = history
        
        # Evaluate model
        metrics = evaluate_model(model_trained, val_loader)
        metrics["time"] = t_duration
        
        # Count parameters
        params_count = sum(p.numel() for p in model_trained.parameters() if p.requires_grad)
        metrics["params"] = params_count
        
        results[model_name] = metrics
        
        print(f"Finished {model_name} in {t_duration:.2f} seconds. Validation Log Loss: {metrics['log_loss']:.4f}")

    # ----------------------------------------------------
    # Generate Comparison Table & Plots
    # ----------------------------------------------------
    print("\nGenerating performance comparison table...")
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            "Model": model_name,
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}",
            "F1-Score": f"{metrics['f1']:.4f}",
            "ROC-AUC": f"{metrics['roc_auc']:.4f}",
            "Log Loss": f"{metrics['log_loss']:.4f}",
            "Params": f"{metrics['params']:,}",
            "Training Time (s)": f"{metrics['time']:.1f}"
        })
        
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_csv("results_comparison.csv", index=False)
    
    # Save a markdown table to display in the walkthrough
    def df_to_markdown(df):
        headers = list(df.columns)
        markdown = "| " + " | ".join(headers) + " |\n"
        markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for _, row in df.iterrows():
            markdown += "| " + " | ".join([str(val) for val in row]) + " |\n"
        return markdown

    with open("results_comparison.md", "w") as f:
        f.write(df_to_markdown(df_comparison))
        
    print("\nComparison Table:")
    print(df_comparison)
    
    # Save plots
    # 1. Learning curves
    num_models = len(histories)
    fig, axes = plt.subplots(2, num_models, figsize=(4 * num_models, 8))
    for col_idx, (model_name, history) in enumerate(histories.items()):
        ax_loss = axes[0, col_idx] if num_models > 1 else axes[0]
        ax_acc = axes[1, col_idx] if num_models > 1 else axes[1]
        
        # Loss plot
        ax_loss.plot(history["train_loss"], label="Train Loss")
        ax_loss.plot(history["val_loss"], label="Val Loss")
        ax_loss.set_title(f"{model_name} Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_loss.grid(True)
        
        # Accuracy plot
        ax_acc.plot(history["train_acc"], label="Train Acc")
        ax_acc.plot(history["val_acc"], label="Val Acc")
        ax_acc.set_title(f"{model_name} Acc")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()
        ax_acc.grid(True)
        
    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150)
    plt.close()
    
    # 2. Confusion Matrices
    fig, axes = plt.subplots(1, num_models, figsize=(4 * num_models, 4))
    for col_idx, (model_name, metrics) in enumerate(results.items()):
        ax = axes[col_idx] if num_models > 1 else axes
        cm = confusion_matrix(metrics["targets"], metrics["preds"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Ship", "Iceberg"], yticklabels=["Ship", "Iceberg"])
        ax.set_title(f"{model_name}\nLogLoss: {metrics['log_loss']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150)
    plt.close()
    
    # 3. ROC Curves
    plt.figure(figsize=(8, 6))
    for model_name, metrics in results.items():
        fpr, tpr, _ = roc_curve(metrics["targets"], metrics["probs"])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Comparison')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("roc_curves.png", dpi=150)
    plt.close()
    
    print("\nSaved learning_curves.png, confusion_matrices.png, and roc_curves.png successfully.")
    
    # ----------------------------------------------------
    # Select Best Model and Generate predictions
    # ----------------------------------------------------
    best_model_name = min(results, key=lambda k: results[k]["log_loss"])
    best_model = trained_models[best_model_name]
    best_log_loss = results[best_model_name]["log_loss"]
    print(f"\nBest Model Selected: {best_model_name} with validation log loss = {best_log_loss:.4f}")
    
    # Run test set predictions using the best model
    best_model.eval()
    test_probs = []
    with torch.no_grad():
        for imgs in test_loader:
            imgs = imgs.to(device)
            outputs = best_model(imgs)
            test_probs.extend(outputs.cpu().numpy().flatten())
            
    # Save submission file
    submission = pd.DataFrame({
        "id": test_ids,
        "is_iceberg": test_probs
    })
    submission_path = "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Created submission file successfully: {submission_path} with {len(submission)} rows.")
    
    # Verify predictions
    print("\nVerifying submission:")
    print(f"  Total Rows: {len(submission)}")
    print(f"  Contains NaN: {submission['is_iceberg'].isna().any()}")
    print(f"  First few predictions:\n", submission.head())

if __name__ == "__main__":
    main()
