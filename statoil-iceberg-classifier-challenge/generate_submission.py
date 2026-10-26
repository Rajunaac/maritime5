import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Define dataset format
class IcebergDataset(Dataset):
    def __init__(self, X):
        # Input shape: (N, 75, 75, 3), output shape: (N, 3, 75, 75)
        X_transposed = np.transpose(X, (0, 3, 1, 2))
        self.X = torch.tensor(X_transposed, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

# Define CustomCNN
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.gap(x).view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

def stream_test_json(filepath):
    """Memory-efficient streaming parser for large JSON arrays of objects."""
    with open(filepath, 'r', encoding='utf-8') as f:
        # Scan forward to find the opening bracket '['
        char = f.read(1)
        while char and char != '[':
            char = f.read(1)
            
        buffer = []
        depth = 0
        while True:
            block = f.read(65536) # Read in 64KB blocks
            if not block:
                break
                
            for c in block:
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    
                if depth > 0:
                    buffer.append(c)
                elif c == '}' and len(buffer) > 0:
                    buffer.append(c)
                    obj_str = "".join(buffer)
                    yield json.loads(obj_str)
                    buffer = []
                    depth = 0

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "data", "processed", "train.json")
    test_path = os.path.join(base_dir, "data", "processed", "test.json")
    checkpoint_path = os.path.join(base_dir, "best_custom_cnn.pth")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Missing train or test json files.")
        return
        
    print("Loading train statistics for normalization...")
    train = pd.read_json(train_path)
    
    def process_bands(df):
        band_1 = np.array([np.array(band, dtype=np.float32).reshape(75, 75) for band in df["band_1"]])
        band_2 = np.array([np.array(band, dtype=np.float32).reshape(75, 75) for band in df["band_2"]])
        band_3 = (band_1 + band_2) / 2.0
        X = np.stack([band_1, band_2, band_3], axis=-1)
        return X

    X_train = process_bands(train)
    y_train_full = train["is_iceberg"].values
    
    # Calculate normalization stats based on same training split reference
    from sklearn.model_selection import train_test_split
    X_train_split, _, _, _ = train_test_split(X_train, y_train_full, test_size=0.20, random_state=42, stratify=y_train_full)
    train_mean = X_train_split.mean(axis=(0, 1, 2), keepdims=True)
    train_std = X_train_split.std(axis=(0, 1, 2), keepdims=True)
    
    # Instantiate best model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomCNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print("Streaming and predicting test dataset in batches...")
    
    batch_size = 500
    current_batch = []
    test_ids = []
    test_probs = []
    
    count = 0
    for sample in stream_test_json(test_path):
        current_batch.append(sample)
        count += 1
        
        if len(current_batch) == batch_size:
            # Process batch
            df_batch = pd.DataFrame(current_batch)
            X_batch = process_bands(df_batch)
            X_batch_norm = (X_batch - train_mean) / (train_std + 1e-6)
            
            batch_dataset = IcebergDataset(X_batch_norm)
            batch_loader = DataLoader(batch_dataset, batch_size=128, shuffle=False)
            
            with torch.no_grad():
                for imgs in batch_loader:
                    imgs = imgs.to(device)
                    outputs = model(imgs)
                    test_probs.extend(outputs.cpu().numpy().flatten())
            
            test_ids.extend(df_batch["id"].values)
            print(f"  Processed {count} samples...")
            current_batch = []
            
    # Process remaining samples
    if len(current_batch) > 0:
        df_batch = pd.DataFrame(current_batch)
        X_batch = process_bands(df_batch)
        X_batch_norm = (X_batch - train_mean) / (train_std + 1e-6)
        
        batch_dataset = IcebergDataset(X_batch_norm)
        batch_loader = DataLoader(batch_dataset, batch_size=128, shuffle=False)
        
        with torch.no_grad():
            for imgs in batch_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                test_probs.extend(outputs.cpu().numpy().flatten())
        
        test_ids.extend(df_batch["id"].values)
        print(f"  Processed final batch ({count} total samples).")
        
    submission = pd.DataFrame({
        "id": test_ids,
        "is_iceberg": test_probs
    })
    
    submission_path = os.path.join(base_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Successfully generated final submission.csv with {len(submission)} rows.")
    print(submission.head())

if __name__ == "__main__":
    main()
