"""
Cleanup temporary retraining files only
Does NOT delete train/test data (DVC tracks those)
"""

import os
from pathlib import Path

# Only delete TEMPORARY selection file
FILES_TO_DELETE = [
    "data/retraining/new_samples.csv",  # Temporary selection output
]


def main():
    print("\n🧹 Cleaning up temporary retraining files...\n")
    
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
    print("ℹ️  Note: train/test data preserved (DVC-tracked)\n")
    
    return 0


if __name__ == "__main__":
    exit(main())