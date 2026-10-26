import json
import os

def build_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    def add_markdown(text):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(code_lines):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code_lines]
        })

    # 1. Title and Project Objective
    add_markdown("""
# STATOIL / C-CORE ICEBERG CLASSIFIER
## Complete Beginner-Friendly Project Guide: Ship vs. Iceberg Classification using 5 Deep Learning Models

**Environment**: Google Colab / Local Python • TensorFlow/Keras / PyTorch • Kaggle Dataset

### Project Objective
The objective of this project is to build and evaluate five deep learning models (Custom CNN, ResNet50, EfficientNetB0, MobileNetV2, and DenseNet121) to classify ocean targets as either **ships** (class 0) or **icebergs** (class 1) using Synthetic Aperture Radar (SAR) imagery.
""")

    # 2. Imports and Environment Check
    add_markdown("""
### 1. Imports and Environment Check
We will import standard libraries for data handling, numerical computation, visualization, and deep learning. We check if a GPU is available to accelerate training.
""")
    add_code([
        "import os",
        "import time",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix, roc_curve",
        "",
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "import torch.optim as optim",
        "from torch.utils.data import Dataset, DataLoader",
        "import torchvision.models as models",
        "import torchvision.transforms as T",
        "",
        "# Set random seeds for reproducibility",
        "np.random.seed(42)",
        "torch.manual_seed(42)",
        "if torch.cuda.is_available():",
        "    torch.cuda.manual_seed_all(42)",
        "",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "print('Running on device:', device)",
        "if torch.cuda.is_available():",
        "    print('GPU Name:', torch.cuda.get_device_name(0))"
    ])

    # 3. Download/Locate Kaggle Data
    add_markdown("""
### 2. Download / Locate Dataset
The Kaggle competition data is stored in `.json` files. The dataset files are:
- `train.json` (contains image bands, incidence angles, and training target `is_iceberg`)
- `test.json` (contains image bands and incidence angles for test set)
""")
    add_code([
        "# Define file paths",
        "train_path = os.path.join('data', 'processed', 'train.json')",
        "test_path = os.path.join('data', 'processed', 'test.json')",
        "print('Train path exists:', os.path.exists(train_path))",
        "print('Test path exists:', os.path.exists(test_path))"
    ])

    # 4. Load train.json and test.json
    add_markdown("""
### 3. Load Datasets
We load the JSON files using `pandas`. Each record represents a 75x75 image chip.
""")
    add_code([
        "print('Loading datasets (this may take a minute)...')",
        "train = pd.read_json(train_path)",
        "test = pd.read_json(test_path)",
        "print('Train dataset shape:', train.shape)",
        "print('Test dataset shape:', test.shape)"
    ])

    # 5. Dataset Schema and EDA
    add_markdown("""
### 4. Dataset Schema and EDA
Let's inspect the columns and metadata of our datasets.
- `id`: unique identifier for the image chip.
- `band_1`: first SAR polarization measurement (HH).
- `band_2`: second SAR polarization measurement (HV).
- `inc_angle`: incidence angle of the image acquisition.
- `is_iceberg`: binary target (1 = iceberg, 0 = ship).
""")
    add_code([
        "print('Columns in Train:', list(train.columns))",
        "print('Columns in Test:', list(test.columns))",
        "train.head()"
    ])

    # 6. Missing Values and Class Distribution
    add_markdown("""
### 5. Missing Values and Class Distribution
We check for missing/invalid values and check the class balance for `is_iceberg`.
""")
    add_code([
        "print('Train info:')",
        "print(train.info())",
        "",
        "# Class balance",
        "counts = train['is_iceberg'].value_counts()",
        "percentages = train['is_iceberg'].value_counts(normalize=True) * 100",
        "print('\\nClass balance:')",
        "for val, count in counts.items():",
        "    print(f'  Class {val}: {count} ({percentages[val]:.2f}%)')",
        "",
        "# Check for missing incidence angles (often stored as 'na')",
        "train['inc_angle_clean'] = pd.to_numeric(train['inc_angle'], errors='coerce')",
        "missing_inc = train['inc_angle_clean'].isna().sum()",
        "print(f'\\nMissing incidence angles: {missing_inc} (out of {len(train)})')",
        "",
        "# Plot class balance",
        "plt.figure(figsize=(5, 3))",
        "plt.bar(['Ship (0)', 'Iceberg (1)'], [counts[0], counts[1]], color=['#1f77b4', '#ff7f0e'])",
        "plt.title('Class Balance (is_iceberg)')",
        "plt.ylabel('Count')",
        "plt.show()"
    ])

    # 7. Visualize Band 1 and Band 2
    add_markdown("""
### 6. Visualize Band 1 and Band 2
We visualize the HH and HV bands for both a ship sample and an iceberg sample.
""")
    add_code([
        "ship_idx = train[train['is_iceberg'] == 0].index[0]",
        "iceberg_idx = train[train['is_iceberg'] == 1].index[0]",
        "",
        "fig, axes = plt.subplots(2, 2, figsize=(10, 8))",
        "",
        "# Ship",
        "b1_ship = np.array(train.loc[ship_idx, 'band_1'], dtype=np.float32).reshape(75, 75)",
        "b2_ship = np.array(train.loc[ship_idx, 'band_2'], dtype=np.float32).reshape(75, 75)",
        "axes[0, 0].imshow(b1_ship, cmap='gray')",
        "axes[0, 0].set_title(f'Ship (ID: {train.loc[ship_idx, \"id\"]}) - Band 1 (HH)')",
        "axes[0, 0].axis('off')",
        "axes[0, 1].imshow(b2_ship, cmap='gray')",
        "axes[0, 1].set_title(f'Ship (ID: {train.loc[ship_idx, \"id\"]}) - Band 2 (HV)')",
        "axes[0, 1].axis('off')",
        "",
        "# Iceberg",
        "b1_ice = np.array(train.loc[iceberg_idx, 'band_1'], dtype=np.float32).reshape(75, 75)",
        "b2_ice = np.array(train.loc[iceberg_idx, 'band_2'], dtype=np.float32).reshape(75, 75)",
        "axes[1, 0].imshow(b1_ice, cmap='gray')",
        "axes[1, 0].set_title(f'Iceberg (ID: {train.loc[iceberg_idx, \"id\"]}) - Band 1 (HH)')",
        "axes[1, 0].axis('off')",
        "axes[1, 1].imshow(b2_ice, cmap='gray')",
        "axes[1, 1].set_title(f'Iceberg (ID: {train.loc[iceberg_idx, \"id\"]}) - Band 2 (HV)')",
        "axes[1, 1].axis('off')",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ])

    # 8. Reshape 5,625 Values → 75×75
    add_markdown("""
### 7. Reshape Flattened Bands & Create 3-Channel Input
The guide recommends converting the dual-polarization SAR data into a 3-channel input representation (HH, HV, average of HH and HV) so that standard ImageNet-pretrained backbones can be used conveniently.
""")
    add_code([
        "def process_bands(df):",
        "    band_1 = np.array([np.array(band, dtype=np.float32).reshape(75, 75) for band in df['band_1']])",
        "    band_2 = np.array([np.array(band, dtype=np.float32).reshape(75, 75) for band in df['band_2']])",
        "    band_3 = (band_1 + band_2) / 2.0",
        "    # Stack along third axis to create shape (N, 75, 75, 3)",
        "    X = np.stack([band_1, band_2, band_3], axis=-1)",
        "    return X",
        "",
        "X = process_bands(train)",
        "y = train['is_iceberg'].values",
        "X_test = process_bands(test)",
        "test_ids = test['id'].values",
        "print('Processed X shape:', X.shape)",
        "print('Processed X_test shape:', X_test.shape)"
    ])

    # 9. Train/Validation Split
    add_markdown("""
### 8. Stratified Train/Validation Split
We perform an 80/20 train/validation split stratified by target labels to maintain class proportions.
""")
    add_code([
        "X_train, X_val, y_train, y_val = train_test_split(",
        "    X, y, test_size=0.20, random_state=42, stratify=y",
        ")",
        "print('X_train shape:', X_train.shape, 'y_train shape:', y_train.shape)",
        "print('X_val shape:', X_val.shape, 'y_val shape:', y_val.shape)"
    ])

    # 10. Normalization
    add_markdown("""
### 9. Image Normalization
We normalize the images using z-score normalization. Crucially, the statistics (mean and std) are calculated *only* from the training split to prevent validation/test leakage.
""")
    add_code([
        "train_mean = X_train.mean(axis=(0, 1, 2), keepdims=True)",
        "train_std = X_train.std(axis=(0, 1, 2), keepdims=True)",
        "",
        "# Apply normalization",
        "X_train_norm = (X_train - train_mean) / (train_std + 1e-6)",
        "X_val_norm = (X_val - train_mean) / (train_std + 1e-6)",
        "X_test_norm = (X_test - train_mean) / (train_std + 1e-6)"
    ])

    # 11. Data Augmentation
    add_markdown("""
### 10. Data Augmentation
Since the dataset is small, data augmentation can help reduce overfitting. We apply random horizontal and vertical flips only to the training set.
""")
    add_code([
        "class IcebergDataset(Dataset):",
        "    def __init__(self, X, y=None, transform=None):",
        "        # PyTorch format: transpose from (N, H, W, C) to (N, C, H, W)",
        "        X_transposed = np.transpose(X, (0, 3, 1, 2))",
        "        self.X = torch.tensor(X_transposed, dtype=torch.float32)",
        "        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None",
        "        self.transform = transform",
        "",
        "    def __len__(self):",
        "        return len(self.X)",
        "",
        "    def __getitem__(self, idx):",
        "        img = self.X[idx]",
        "        if self.transform:",
        "            img = self.transform(img)",
        "        if self.y is not None:",
        "            return img, self.y[idx]",
        "        return img",
        "",
        "# Augmentation transform (horizontal/vertical flip)",
        "train_transform = T.Compose([",
        "    T.RandomHorizontalFlip(),",
        "    T.RandomVerticalFlip()",
        "])",
        "",
        "train_dataset = IcebergDataset(X_train_norm, y_train, transform=train_transform)",
        "val_dataset = IcebergDataset(X_val_norm, y_val)",
        "test_dataset = IcebergDataset(X_test_norm)",
        "",
        "train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)",
        "val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)",
        "test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)"
    ])

    # 12. Models and Helper functions
    add_markdown("""
### 11. Model Implementations & Training Utilities
We define the 5 model architectures in PyTorch and write the core training loop.
""")
    add_code([
        "class CustomCNN(nn.Module):",
        "    def __init__(self):",
        "        super(CustomCNN, self).__init__()",
        "        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)",
        "        self.bn1 = nn.BatchNorm2d(64)",
        "        self.pool1 = nn.MaxPool2d(2, 2)",
        "        ",
        "        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)",
        "        self.bn2 = nn.BatchNorm2d(128)",
        "        self.pool2 = nn.MaxPool2d(2, 2)",
        "        ",
        "        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)",
        "        self.bn3 = nn.BatchNorm2d(128)",
        "        ",
        "        self.gap = nn.AdaptiveAvgPool2d(1)",
        "        self.fc1 = nn.Linear(128, 128)",
        "        self.dropout = nn.Dropout(0.3)",
        "        self.fc2 = nn.Linear(128, 1)",
        "",
        "    def forward(self, x):",
        "        x = self.pool1(F.relu(self.bn1(self.conv1(x))))",
        "        x = self.pool2(F.relu(self.bn2(self.conv2(x))))",
        "        x = F.relu(self.bn3(self.conv3(x)))",
        "        x = self.gap(x).view(x.size(0), -1)",
        "        x = F.relu(self.fc1(x))",
        "        x = self.dropout(x)",
        "        x = torch.sigmoid(self.fc2(x))",
        "        return x",
        "",
        "def get_resnet50():",
        "    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)",
        "    num_features = model.fc.in_features",
        "    model.fc = nn.Sequential(",
        "        nn.Dropout(0.3),",
        "        nn.Linear(num_features, 1),",
        "        nn.Sigmoid()",
        "    )",
        "    return model",
        "",
        "def get_efficientnet_b0():",
        "    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)",
        "    num_features = model.classifier[1].in_features",
        "    model.classifier = nn.Sequential(",
        "        nn.Dropout(0.3),",
        "        nn.Linear(num_features, 1),",
        "        nn.Sigmoid()",
        "    )",
        "    return model",
        "",
        "def get_mobilenet_v2():",
        "    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)",
        "    num_features = model.classifier[1].in_features",
        "    model.classifier = nn.Sequential(",
        "        nn.Dropout(0.3),",
        "        nn.Linear(num_features, 1),",
        "        nn.Sigmoid()",
        "    )",
        "    return model",
        "",
        "def get_densenet121():",
        "    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)",
        "    num_features = model.classifier.in_features",
        "    model.classifier = nn.Sequential(",
        "        nn.Dropout(0.3),",
        "        nn.Linear(num_features, 1),",
        "        nn.Sigmoid()",
        "    )",
        "    return model",
        "",
        "class ResNetDenseNetAttention(nn.Module):",
        "    def __init__(self):",
        "        super(ResNetDenseNetAttention, self).__init__()",
        "        # Pretrained backbones",
        "        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)",
        "        self.resnet_features = nn.Sequential(*list(self.resnet.children())[:-1]) # GAP output, shape (N, 2048, 1, 1)",
        "        ",
        "        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)",
        "        self.densenet_features = nn.Sequential(",
        "            self.densenet.features,",
        "            nn.ReLU(inplace=True),",
        "            nn.AdaptiveAvgPool2d((1, 1)) # GAP output, shape (N, 1024, 1, 1)",
        "        )",
        "        ",
        "        # Feature projections",
        "        self.resnet_proj = nn.Linear(2048, 512)",
        "        self.densenet_proj = nn.Linear(1024, 512)",
        "        ",
        "        # Multi-head Self-Attention",
        "        self.attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)",
        "        ",
        "        # Classifier Head",
        "        self.classifier = nn.Sequential(",
        "            nn.Linear(512, 128),",
        "            nn.ReLU(inplace=True),",
        "            nn.Dropout(0.3),",
        "            nn.Linear(128, 1),",
        "            nn.Sigmoid()",
        "        )",
        "        ",
        "    def forward(self, x):",
        "        # Extract features",
        "        r_feat = self.resnet_features(x).view(x.size(0), -1) # (N, 2048)",
        "        d_feat = self.densenet_features(x).view(x.size(0), -1) # (N, 1024)",
        "        ",
        "        # Project to same dimension",
        "        r_proj = self.resnet_proj(r_feat) # (N, 512)",
        "        d_proj = self.densenet_proj(d_feat) # (N, 512)",
        "        ",
        "        # Stack features: shape (N, 2, 512)",
        "        stacked = torch.stack([r_proj, d_proj], dim=1)",
        "        ",
        "        # Apply Self-Attention",
        "        attn_out, _ = self.attn(stacked, stacked, stacked) # (N, 2, 512)",
        "        ",
        "        # Average pooling over the two models' features",
        "        pooled = attn_out.mean(dim=1) # (N, 512)",
        "        ",
        "        # Final classification",
        "        out = self.classifier(pooled)",
        "        return out"
    ])

    # 13. Training Loop Code
    add_code([
        "def train_model(model, train_loader, val_loader, model_name, epochs, lr, is_pretrained=False, fine_tune_epochs=0, fine_tune_lr=1e-5):",
        "    print(f'\\n--- Training {model_name} ---')",
        "    model = model.to(device)",
        "    criterion = nn.BCELoss()",
        "    ",
        "    history = {",
        "        'train_loss': [], 'val_loss': [],",
        "        'train_acc': [], 'val_acc': []",
        "    }",
        "    ",
        "    best_val_loss = float('inf')",
        "    checkpoint_path = f'best_{model_name.lower().replace(\" \", \"_\")}.pth'",
        "    ",
        "    # Phase 1: Train classification head",
        "    if is_pretrained:",
        "        if 'Attention' in model_name:",
        "            for param in model.resnet.parameters():",
        "                param.requires_grad = False",
        "            for param in model.densenet.parameters():",
        "                param.requires_grad = False",
        "            trainable_params = [p for p in model.parameters() if p.requires_grad]",
        "            optimizer = optim.Adam(trainable_params, lr=lr)",
        "        else:",
        "            for param in model.parameters():",
        "                param.requires_grad = False",
        "            if hasattr(model, 'fc'):",
        "                for param in model.fc.parameters():",
        "                    param.requires_grad = True",
        "                optimizer = optim.Adam(model.fc.parameters(), lr=lr)",
        "            elif hasattr(model, 'classifier'):",
        "                for param in model.classifier.parameters():",
        "                    param.requires_grad = True",
        "                optimizer = optim.Adam(model.classifier.parameters(), lr=lr)",
        "            else:",
        "                optimizer = optim.Adam(model.parameters(), lr=lr)",
        "        total_epochs = epochs",
        "    else:",
        "        optimizer = optim.Adam(model.parameters(), lr=lr)",
        "        total_epochs = epochs",
        "",
        "    def run_epochs(num_eps, opt, start_ep=0, patience=8):",
        "        nonlocal best_val_loss",
        "        epochs_no_improve = 0",
        "        for epoch in range(num_eps):",
        "            model.train()",
        "            train_loss, train_correct, train_total = 0.0, 0, 0",
        "            for imgs, labels in train_loader:",
        "                imgs, labels = imgs.to(device), labels.to(device)",
        "                opt.zero_grad()",
        "                outputs = model(imgs)",
        "                loss = criterion(outputs, labels)",
        "                loss.backward()",
        "                opt.step()",
        "                ",
        "                train_loss += loss.item() * imgs.size(0)",
        "                preds = (outputs >= 0.5).float()",
        "                train_correct += (preds == labels).sum().item()",
        "                train_total += imgs.size(0)",
        "                ",
        "            epoch_train_loss = train_loss / train_total",
        "            epoch_train_acc = train_correct / train_total",
        "            ",
        "            # Validation",
        "            model.eval()",
        "            val_loss, val_correct, val_total = 0.0, 0, 0",
        "            with torch.no_grad():",
        "                for imgs, labels in val_loader:",
        "                    imgs, labels = imgs.to(device), labels.to(device)",
        "                    outputs = model(imgs)",
        "                    loss = criterion(outputs, labels)",
        "                    val_loss += loss.item() * imgs.size(0)",
        "                    preds = (outputs >= 0.5).float()",
        "                    val_correct += (preds == labels).sum().item()",
        "                    val_total += imgs.size(0)",
        "            epoch_val_loss = val_loss / val_total",
        "            epoch_val_acc = val_correct / val_total",
        "            ",
        "            history['train_loss'].append(epoch_train_loss)",
        "            history['val_loss'].append(epoch_val_loss)",
        "            history['train_acc'].append(epoch_train_acc)",
        "            history['val_acc'].append(epoch_val_acc)",
        "            ",
        "            print(f'Epoch {start_ep + epoch + 1:02d}: '",
        "                  f'Train Loss = {epoch_train_loss:.4f}, Train Acc = {epoch_train_acc:.4f} | '",
        "                  f'Val Loss = {epoch_val_loss:.4f}, Val Acc = {epoch_val_acc:.4f}')",
        "            ",
        "            if epoch_val_loss < best_val_loss:",
        "                best_val_loss = epoch_val_loss",
        "                torch.save(model.state_dict(), checkpoint_path)",
        "                epochs_no_improve = 0",
        "            else:",
        "                epochs_no_improve += 1",
        "                if epochs_no_improve >= patience:",
        "                    print(f'  --> Early stopping triggered at epoch {start_ep + epoch + 1} (no improvement for {patience} epochs).')",
        "                    break",
        "",
        "    run_epochs(total_epochs, optimizer, start_ep=0)",
        "    ",
        "    # Phase 2: Fine-tuning",
        "    if is_pretrained and fine_tune_epochs > 0:",
        "        print('Phase 2: Fine-tuning full network...')",
        "        for param in model.parameters():",
        "            param.requires_grad = True",
        "        if os.path.exists(checkpoint_path):",
        "            model.load_state_dict(torch.load(checkpoint_path))",
        "        ft_optimizer = optim.Adam(model.parameters(), lr=fine_tune_lr)",
        "        run_epochs(fine_tune_epochs, ft_optimizer, start_ep=total_epochs)",
        "        ",
        "    if os.path.exists(checkpoint_path):",
        "        model.load_state_dict(torch.load(checkpoint_path))",
        "    return model, history"
    ])

    # 14. Evaluation function
    add_code([
        "def evaluate_model(model, loader):",
        "    model.eval()",
        "    all_probs = []",
        "    all_targets = []",
        "    with torch.no_grad():",
        "        for imgs, labels in loader:",
        "            imgs = imgs.to(device)",
        "            outputs = model(imgs)",
        "            all_probs.extend(outputs.cpu().numpy().flatten())",
        "            all_targets.extend(labels.numpy().flatten())",
        "    probs = np.array(all_probs)",
        "    targets = np.array(all_targets)",
        "    preds = (probs >= 0.5).astype(int)",
        "    ",
        "    return {",
        "        'accuracy': accuracy_score(targets, preds),",
        "        'precision': precision_score(targets, preds, zero_division=0),",
        "        'recall': recall_score(targets, preds, zero_division=0),",
        "        'f1': f1_score(targets, preds, zero_division=0),",
        "        'roc_auc': roc_auc_score(targets, probs),",
        "        'log_loss': log_loss(targets, probs),",
        "        'probs': probs,",
        "        'targets': targets,",
        "        'preds': preds",
        "    }"
    ])

    # 15. Run Model training
    add_markdown("""
### 12. Run the Five-Model Experiment
We train all 5 models. For transfer learning, we train the head for 3 epochs and fine-tune for 5 epochs. For Custom CNN we train for 8 epochs.
""")
    add_code([
        "models_dict = {",
        "    'Custom CNN': (CustomCNN(), False),",
        "    'ResNet50': (get_resnet50(), True),",
        "    'EfficientNetB0': (get_efficientnet_b0(), True),",
        "    'MobileNetV2': (get_mobilenet_v2(), True),",
        "    'DenseNet121': (get_densenet121(), True),",
        "    'ResNet50 + DenseNet121 + Attention': (ResNetDenseNetAttention(), True),",
        "}",
        "",
        "trained_models = {}",
        "histories = {}",
        "results = {}",
        "",
        "for model_name, (model_inst, is_pretrained) in models_dict.items():",
        "    t_start = time.time()",
        "    model_trained, history = train_model(",
        "        model=model_inst,",
        "        train_loader=train_loader,",
        "        val_loader=val_loader,",
        "        model_name=model_name,",
        "        epochs=12 if is_pretrained else 35,",
        "        lr=1e-3,",
        "        is_pretrained=is_pretrained,",
        "        fine_tune_epochs=25 if is_pretrained else 0,",
        "        fine_tune_lr=1e-5",
        "    )",
        "    t_duration = time.time() - t_start",
        "    trained_models[model_name] = model_trained",
        "    histories[model_name] = history",
        "    ",
        "    # Evaluation",
        "    metrics = evaluate_model(model_trained, val_loader)",
        "    metrics['time'] = t_duration",
        "    metrics['params'] = sum(p.numel() for p in model_trained.parameters() if p.requires_grad)",
        "    results[model_name] = metrics",
        "    print(f'Finished {model_name}. Validation Log Loss: {metrics[\"log_loss\"]:.4f}')"
    ])

    # 16. Evaluation and Learning Curves
    add_markdown("""
### 13. Plot Learning Curves
We plot the training and validation loss/accuracy across epochs for each model.
""")
    add_code([
        "fig, axes = plt.subplots(2, 5, figsize=(20, 8))",
        "for col_idx, (model_name, history) in enumerate(histories.items()):",
        "    # Loss",
        "    axes[0, col_idx].plot(history['train_loss'], label='Train Loss')",
        "    axes[0, col_idx].plot(history['val_loss'], label='Val Loss')",
        "    axes[0, col_idx].set_title(f'{model_name} Loss')",
        "    axes[0, col_idx].set_xlabel('Epoch')",
        "    axes[0, col_idx].set_ylabel('Loss')",
        "    axes[0, col_idx].legend()",
        "    axes[0, col_idx].grid(True)",
        "    ",
        "    # Accuracy",
        "    axes[1, col_idx].plot(history['train_acc'], label='Train Acc')",
        "    axes[1, col_idx].plot(history['val_acc'], label='Val Acc')",
        "    axes[1, col_idx].set_title(f'{model_name} Acc')",
        "    axes[1, col_idx].set_xlabel('Epoch')",
        "    axes[1, col_idx].set_ylabel('Accuracy')",
        "    axes[1, col_idx].legend()",
        "    axes[1, col_idx].grid(True)",
        "plt.tight_layout()",
        "plt.show()"
    ])

    # 17. Confusion Matrices
    add_markdown("""
### 14. Plot Confusion Matrices
Let's see where the models confuse ships and icebergs using a decision threshold of 0.5.
""")
    add_code([
        "fig, axes = plt.subplots(1, 5, figsize=(20, 4))",
        "for col_idx, (model_name, metrics) in enumerate(results.items()):",
        "    cm = confusion_matrix(metrics['targets'], metrics['preds'])",
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[col_idx], cbar=False,",
        "                xticklabels=['Ship', 'Iceberg'], yticklabels=['Ship', 'Iceberg'])",
        "    axes[col_idx].set_title(f'{model_name}\\nLogLoss: {metrics[\"log_loss\"]:.3f}')",
        "    axes[col_idx].set_xlabel('Predicted')",
        "    axes[col_idx].set_ylabel('Actual')",
        "plt.tight_layout()",
        "plt.show()"
    ])

    # 18. ROC Curves
    add_markdown("""
### 15. Plot ROC Curves
The Receiver Operating Characteristic (ROC) curve evaluates class separation capacity across different decision thresholds.
""")
    add_code([
        "plt.figure(figsize=(8, 6))",
        "for model_name, metrics in results.items():",
        "    fpr, tpr, _ = roc_curve(metrics['targets'], metrics['probs'])",
        "    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {metrics[\"roc_auc\"]:.3f})')",
        "plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)",
        "plt.xlim([0.0, 1.0])",
        "plt.ylim([0.0, 1.05])",
        "plt.xlabel('False Positive Rate')",
        "plt.ylabel('True Positive Rate')",
        "plt.title('ROC Curves Comparison')",
        "plt.legend(loc='lower right')",
        "plt.grid(True)",
        "plt.show()"
    ])

    # 19. Comparison Table
    add_markdown("""
### 16. Build the Final Model Comparison Table
We compare all models in a single results table, sorted by validation log loss.
""")
    add_code([
        "comparison_data = []",
        "for model_name, metrics in results.items():",
        "    comparison_data.append({",
        "        'Model': model_name,",
        "        'Accuracy': f'{metrics[\"accuracy\"]:.4f}',",
        "        'Precision': f'{metrics[\"precision\"]:.4f}',",
        "        'Recall': f'{metrics[\"recall\"]:.4f}',",
        "        'F1-Score': f'{metrics[\"f1\"]:.4f}',",
        "        'ROC-AUC': f'{metrics[\"roc_auc\"]:.4f}',",
        "        'Log Loss': f'{metrics[\"log_loss\"]:.4f}',",
        "        'Params': f'{metrics[\"params\"]:,}',",
        "        'Training Time (s)': f'{metrics[\"time\"]:.1f}'",
        "    })",
        "df_comparison = pd.DataFrame(comparison_data)",
        "# Display sorted by Log Loss",
        "df_comparison.sort_values(by='Log Loss', inplace=True)",
        "df_comparison"
    ])

    # 20. Best Model Prediction & Submission
    add_markdown("""
### 17. Generate predictions on test.json using the Best Model
We select the model with the lowest validation Log Loss and use it to predict probabilities on `test.json`.
""")
    add_code([
        "best_model_name = min(results, key=lambda k: results[k]['log_loss'])",
        "best_model = trained_models[best_model_name]",
        "print(f'Selected Best Model: {best_model_name} with log loss: {results[best_model_name][\"log_loss\"]:.4f}')",
        "",
        "best_model.eval()",
        "test_probs = []",
        "with torch.no_grad():",
        "    for imgs in test_loader:",
        "        imgs = imgs.to(device)",
        "        outputs = best_model(imgs)",
        "        test_probs.extend(outputs.cpu().numpy().flatten())",
        "",
        "# Save submission file",
        "submission = pd.DataFrame({",
        "    'id': test_ids,",
        "    'is_iceberg': test_probs",
        "})",
        "submission.to_csv('submission.csv', index=False)",
        "print(f'Successfully generated submission.csv with {len(submission)} rows.')",
        "submission.head()"
    ])

    # Save to file
    out_path = "Statoil_Iceberg_Classifier.ipynb"
    with open(out_path, "w") as f:
        json.dump(notebook, f, indent=2)
    print(f"Created Jupyter Notebook successfully at: {out_path}")

if __name__ == "__main__":
    build_notebook()
