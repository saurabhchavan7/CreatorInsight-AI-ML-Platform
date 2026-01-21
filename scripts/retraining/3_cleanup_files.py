"""
DVC Stage: Cleanup temporary retraining files
Runs at the end of DVC pipeline
"""

import os
from pathlib import Path
import shutil

FILES_TO_DELETE = [
    "data/retraining/new_samples.csv",
    "data/interim/train_backup.csv",
    "data/new_training_samples.csv"  # Old location
]


def main():
    print("\n🧹 Cleaning up temporary files...\n")
    
    deleted = 0
    for file_path in FILES_TO_DELETE:
        if Path(file_path).exists():
            try:
                os.remove(file_path)
                print(f"   ✅ Deleted: {file_path}")
                deleted += 1
            except Exception as e:
                print(f"   ⚠️  Failed to delete {file_path}: {e}")
    
    if deleted == 0:
        print("   ℹ️  No temporary files to clean")
    
    print(f"\n✅ Cleanup complete ({deleted} files removed)\n")
    
    return 0


if __name__ == "__main__":
    exit(main())