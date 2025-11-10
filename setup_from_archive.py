"""
Setup script to copy dataset from an archive location into a local data directory,
and generate a class_names.json from the discovered breed folders.

Now supports CLI arguments:
  - --archive: source archive path (folder that contains the dataset)
  - --target: destination data directory (will be created/replaced)
  - --class-names-file: output JSON file for class names
  - --dry-run: discover and print what would happen without copying
"""
import os
import shutil
import json
import argparse
from pathlib import Path

DEFAULT_ARCHIVE = Path(r"C:\Users\asus\Downloads\archive")
DEFAULT_TARGET = Path("data")
DEFAULT_CLASS_NAMES_FILE = "class_names.json"

def discover_dataset_folder(archive_path: Path) -> Path:
    """Try to locate the folder that directly contains breed subdirectories."""
    if not archive_path.exists():
        return None

    dataset_path = None

    indian_breeds = archive_path / "Indian_bovine_breeds"
    if indian_breeds.exists() and indian_breeds.is_dir():
        nested = indian_breeds / "Indian_bovine_breeds"
        if nested.exists() and nested.is_dir():
            breed_dirs = [d for d in nested.iterdir() if d.is_dir()]
            if len(breed_dirs) > 10:
                dataset_path = nested
            else:
                dataset_path = indian_breeds
        else:
            breed_dirs = [d for d in indian_breeds.iterdir() if d.is_dir()]
            if len(breed_dirs) > 10:
                dataset_path = indian_breeds
    else:
        subdirs = [d for d in archive_path.iterdir() if d.is_dir()]
        if len(subdirs) > 10:
            dataset_path = archive_path
        elif len(subdirs) == 1:
            subdir = subdirs[0]
            breed_dirs = [d for d in subdir.iterdir() if d.is_dir()]
            if len(breed_dirs) > 10:
                dataset_path = subdir

    return dataset_path

def write_class_names(dataset_path: Path, class_names_file: Path) -> list:
    """Extract class names (subdirectory names) and write to JSON file."""
    breed_folders = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    with open(class_names_file, 'w') as f:
        json.dump(breed_folders, f, indent=2)
    return breed_folders

def setup_data(archive_path: Path, target_path: Path, class_names_file: Path, dry_run: bool = False) -> bool:
    """Set up data directory from archive with options."""
    print("=" * 60)
    print("Setting up data directory from archive")
    print("=" * 60)

    if not archive_path.exists():
        print(f"Error: Archive path not found: {archive_path}")
        return False

    dataset_path = discover_dataset_folder(archive_path)
    if dataset_path is None:
        print("Error: Could not find dataset folder with breed subdirectories")
        return False

    breed_folders = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    if len(breed_folders) == 0:
        print("Error: No breed subdirectories found in discovered dataset path")
        return False

    print(f"Found dataset folder: {dataset_path}")
    print(f"Found {len(breed_folders)} breed folders")
    print(f"First 10 breeds: {breed_folders[:10]}")

    if dry_run:
        print("\n--dry-run specified: will not copy files --")
        print(f"Would copy from: {dataset_path}")
        print(f"Would write to:  {target_path}")
        print(f"Would save class names to: {class_names_file}")
        return True

    # Remove existing data file/directory
    if target_path.exists():
        if target_path.is_file():
            print(f"\nRemoving existing '{target_path.name}' file...")
            target_path.unlink()
        elif target_path.is_dir():
            print(f"\nRemoving existing '{target_path.name}' directory...")
            shutil.rmtree(str(target_path))

    # Copy dataset to data directory
    print(f"\nCopying dataset from {dataset_path} to {target_path} ...")
    shutil.copytree(str(dataset_path), str(target_path))
    print("Copy complete.")

    # Extract and save class names
    print(f"\nExtracting class names...")
    breed_folders = write_class_names(target_path, Path(class_names_file))
    print(f"Successfully saved {len(breed_folders)} class names to {class_names_file}")

    print(f"\nAll breed names:")
    for i, name in enumerate(breed_folders, 1):
        print(f"  {i}. {name}")

    return True

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up local data directory from an archive dataset.")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE, help="Path to the source archive directory")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="Path to the destination data directory")
    parser.add_argument("--class-names-file", type=Path, default=DEFAULT_CLASS_NAMES_FILE, help="Path to output class names JSON")
    parser.add_argument("--dry-run", action="store_true", help="Discover and print actions without copying files")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    success = setup_data(args.archive, args.target, args.class_names_file, args.dry_run)
    if success:
        print("\n" + "=" * 60)
        print("Setup complete! You can now run: python chatbot.py")
        print("=" * 60)
    else:
        print("\nSetup failed. Please check the archive path.")

