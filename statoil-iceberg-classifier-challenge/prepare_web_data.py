import os
import json
import numpy as np
import pandas as pd

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "data", "processed", "train.json")
    web_dir = os.path.join(base_dir, "web")
    os.makedirs(web_dir, exist_ok=True)
    
    if not os.path.exists(train_path):
        print("Error: train.json not found.")
        return
        
    print("Loading train.json for web samples extraction...")
    df = pd.read_json(train_path)
    
    # Filter ships and icebergs
    ships = df[df["is_iceberg"] == 0].head(15)
    icebergs = df[df["is_iceberg"] == 1].head(15)
    
    combined = pd.concat([ships, icebergs]).sample(frac=1.0, random_state=42) # Shuffle
    
    samples_list = []
    for idx, row in combined.iterrows():
        b1 = [round(float(val), 4) for val in row["band_1"]]
        b2 = [round(float(val), 4) for val in row["band_2"]]
        
        samples_list.append({
            "id": str(row["id"]),
            "is_iceberg": int(row["is_iceberg"]),
            "inc_angle": str(row["inc_angle"]),
            "band_1": b1,
            "band_2": b2
        })
        
    js_content = f"// Real-time SAR dataset samples (Raw floats for accurate model inference)\nvar SAR_SAMPLES = {json.dumps(samples_list)};\n"
    
    js_path = os.path.join(web_dir, "samples.js")
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(js_content)
        
    print(f"Extracted {len(samples_list)} raw float samples and saved to: {js_path}")

if __name__ == "__main__":
    main()
