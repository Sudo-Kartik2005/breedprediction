"""
Helper script to locate and set up the Kaggle dataset
"""
import os
import shutil
import json
from pathlib import Path

def find_dataset():
    """Find the dataset in common locations"""
    current_dir = Path(".")
    possible_names = [
        "indian-bovine-breeds",
        "indian_bovine_breeds",
        "cattle_breeds",
        "cattle-breeds",
        "data",
        "dataset"
    ]
    
    print("Searching for dataset...")
    
    # Check current directory
    for name in possible_names:
        path = current_dir / name
        if path.exists() and path.is_dir():
            # Check if it has subdirectories (breed folders)
            subdirs = [d for d in path.iterdir() if d.is_dir()]
            if len(subdirs) > 0:
                print(f"\nFound dataset folder: {path}")
                print(f"Found {len(subdirs)} breed folders")
                return path
    
    # Check parent directory
    parent_dir = current_dir.parent
    for name in possible_names:
        path = parent_dir / name
        if path.exists() and path.is_dir():
            subdirs = [d for d in path.iterdir() if d.is_dir()]
            if len(subdirs) > 0:
                print(f"\nFound dataset folder: {path}")
                print(f"Found {len(subdirs)} breed folders")
                return path
    
    # Check Downloads folder
    downloads = Path.home() / "Downloads"
    if downloads.exists():
        for name in possible_names:
            path = downloads / name
            if path.exists() and path.is_dir():
                subdirs = [d for d in path.iterdir() if d.is_dir()]
                if len(subdirs) > 0:
                    print(f"\nFound dataset folder: {path}")
                    print(f"Found {len(subdirs)} breed folders")
                    return path
    
    return None

def setup_data_directory(source_path):
    """Set up the data directory from source"""
    target = Path("data")
    
    if target.exists():
        if target.is_file():
            print(f"\n'data' exists as a file. Backing it up to 'data.backup'...")
            shutil.move(str(target), "data.backup")
        elif target.is_dir():
            print(f"\n'data' directory already exists.")
            response = input("Do you want to replace it? (y/n): ")
            if response.lower() != 'y':
                print("Keeping existing data directory.")
                return target
            else:
                print("Removing existing data directory...")
                shutil.rmtree(str(target))
    
    print(f"\nCopying dataset from {source_path} to data/...")
    shutil.copytree(str(source_path), str(target))
    print("Done!")
    return target

def extract_class_names():
    """Extract class names from data directory"""
    data_dir = Path("data")
    if not data_dir.exists() or not data_dir.is_dir():
        print("Error: data directory does not exist!")
        return False
    
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    
    if not class_names:
        print("Error: No breed folders found in data directory!")
        return False
    
    # Save to JSON
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f, indent=2)
    
    print(f"\nSuccessfully extracted {len(class_names)} class names:")
    print(f"Saved to class_names.json")
    print(f"\nFirst 10 breeds:")
    for i, name in enumerate(class_names[:10], 1):
        print(f"  {i}. {name}")
    if len(class_names) > 10:
        print(f"  ... and {len(class_names) - 10} more")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Cattle Breed Dataset Setup")
    print("=" * 60)
    
    # Check if data directory already exists and is set up
    data_dir = Path("data")
    if data_dir.exists() and data_dir.is_dir():
        subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
        if len(subdirs) > 0:
            print(f"\nData directory already exists with {len(subdirs)} breed folders!")
            response = input("Do you want to extract class names? (y/n): ")
            if response.lower() == 'y':
                extract_class_names()
            exit(0)
    
    # Try to find the dataset
    dataset_path = find_dataset()
    
    if dataset_path:
        print(f"\nDataset found at: {dataset_path}")
        response = input("Do you want to set up the data directory from this location? (y/n): ")
        if response.lower() == 'y':
            setup_data_directory(dataset_path)
            extract_class_names()
        else:
            print("Setup cancelled.")
    else:
        print("\nCould not find the dataset automatically.")
        print("\nPlease:")
        print("1. Extract your Kaggle dataset")
        print("2. Copy the extracted folder to this directory")
        print("3. Rename it to 'data'")
        print("4. Run: python extract_class_names.py")
        print("\nOr manually specify the dataset location:")

