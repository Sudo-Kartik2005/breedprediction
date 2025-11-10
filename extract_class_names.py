"""
Helper script to extract class names from data directory and save to class_names.json
Run this if you have the data directory but need to create the class_names.json file
"""
import os
import json

DATA_DIR = "data/"
CLASS_NAMES_FILE = "class_names.json"

if os.path.exists(DATA_DIR) and os.path.isdir(DATA_DIR):
    class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    
    if class_names:
        with open(CLASS_NAMES_FILE, 'w') as f:
            json.dump(class_names, f, indent=2)
        print(f"Successfully extracted {len(class_names)} class names:")
        print(f"   Saved to {CLASS_NAMES_FILE}")
        print(f"\nClass names:")
        for i, name in enumerate(class_names, 1):
            print(f"   {i}. {name}")
    else:
        print(f"No subdirectories found in {DATA_DIR}")
else:
    print(f"Data directory '{DATA_DIR}' does not exist or is not a directory")
    print(f"   Please download the dataset from:")
    print(f"   https://www.kaggle.com/datasets/lukex9442/indian-bovine-breeds")
    print(f"   And extract it to create the '{DATA_DIR}' directory structure")

