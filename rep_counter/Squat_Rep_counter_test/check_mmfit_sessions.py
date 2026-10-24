
from pathlib import Path

# Change this to the parent folder containing w02, w03, etc.
DATASET_DIR = Path(r"C:\Users\Krushna\Desktop\VisionFit-AI\dataset\mmfit")

print("Checking MM-Fit session folders...\n")

found = 0

for folder in sorted(DATASET_DIR.iterdir()):
    if not folder.is_dir():
        continue

    pose_file = folder / f"{folder.name}_pose_3d.npy"
    labels_file = folder / f"{folder.name}_labels.csv"

    if pose_file.exists() and labels_file.exists():
        print("Session:", folder.name)
        print("  Pose file:", pose_file.name)
        print("  Labels file:", labels_file.name)
        found += 1

print(f"\nTotal usable session folders found: {found}")
