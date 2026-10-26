import glob
import os
from collections import Counter

# Check if there are local bdd100k labels
label_files = glob.glob("bdd100k_dataset/val/labels/*.txt")
if not label_files:
    label_files = glob.glob("**/*val*labels*/*.txt", recursive=True)

print(f"Found {len(label_files)} label files.")
if label_files:
    counts = Counter()
    for f in label_files[:200]:
        with open(f) as fp:
            for line in fp:
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1
    print("Class ID distribution in first 200 files:", sorted(counts.items()))
