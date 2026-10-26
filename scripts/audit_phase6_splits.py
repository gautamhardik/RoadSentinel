"""
RoadSentinel Phase 6 — Stage 1 dataset split audit (CPU only).

Reads ONLY bdd100k_dataset/train and bdd100k_dataset/val. The test split is never listed, opened or hashed.

Checks:
  1. Image/label pairing, empty labels, malformed or degenerate boxes.
  2. Exact filename overlap train vs val; shared 8-hex name prefixes.
  3. Byte-identical files (MD5) within and across splits.
  4. Near-duplicates (64-bit dHash candidates, verified by 32x32 thumbnail correlation):
     val vs train, and Phase 6 holdout vs Phase 6 train split.
  5. Per-class and small-VRU label statistics.
  6. Phase 6 holdout manifest (same rule as train_phase6_vru.py) + disjointness + fingerprint.

Usage: python scripts/audit_phase6_splits.py
Outputs: experiments/phase6_audit/
"""

import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "bdd100k_dataset"
OUT = ROOT / "experiments" / "phase6_audit"
SPLITS = ("train", "val")  # test split is quarantined: never touched
CLASS_NAMES = ["person", "rider", "car", "bus", "truck", "bike", "motor", "traffic light", "traffic sign", "train"]
VRU_IDS = {0, 1, 5, 6}
SMALL_AREA = 1024.0  # px^2 at native resolution
HOLDOUT_SIZE, HOLDOUT_SEED = 5000, 42
DHASH_MAX_HAMMING = 6
THUMB_MIN_CORR = 0.97


def image_signature(path: str):
    raw = Path(path).read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    h, w = img.shape[:2]
    d = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (d[:, 1:] > d[:, :-1]).flatten()
    dhash = int(np.packbits(bits).view(">u8")[0])
    thumb = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).flatten()
    thumb = (thumb - thumb.mean()) / (thumb.std() + 1e-6)
    return md5, dhash, thumb.astype(np.float16), (w * 4, h * 4)


def read_labels(path: Path, img_wh):
    """Returns (rows, problems). Rows are (cls, w_px, h_px)."""
    rows, problems = [], []
    if not path.exists():
        return rows, ["missing_label_file"]
    W, H = img_wh
    for ln, line in enumerate(path.read_text().split("\n")):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            problems.append(f"line{ln}:fields={len(parts)}")
            continue
        c, x, y, w, h = int(parts[0]), *map(float, parts[1:])
        if not 0 <= c < len(CLASS_NAMES):
            problems.append(f"line{ln}:class={c}")
        if w <= 0 or h <= 0:
            problems.append(f"line{ln}:degenerate")
        if not (0 <= x <= 1 and 0 <= y <= 1 and w <= 1.0001 and h <= 1.0001):
            problems.append(f"line{ln}:out_of_range")
        rows.append((c, w * W, h * H))
    return rows, problems


def popcount64(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


def near_duplicates(names_a, hashes_a, thumbs_a, names_b, hashes_b, thumbs_b, exclude_same=False):
    """All pairs (a, b) with dHash Hamming <= threshold, verified by thumbnail correlation."""
    found = []
    hb = hashes_b.astype(np.uint64)
    for i in range(len(names_a)):
        ham = popcount64(np.bitwise_xor(hb, np.uint64(hashes_a[i])))
        for j in np.nonzero(ham <= DHASH_MAX_HAMMING)[0]:
            if exclude_same and names_a[i] == names_b[j]:
                continue
            corr = float(np.dot(thumbs_a[i].astype(np.float32), thumbs_b[j].astype(np.float32)) / 1024.0)
            if corr >= THUMB_MIN_CORR:
                found.append({"a": names_a[i], "b": names_b[j], "hamming": int(ham[j]), "corr": round(corr, 4)})
    return found


def phase6_holdout(train_names):
    """Must match train_phase6_vru.build_data_yaml(): whole 8-hex name-prefix groups (same recording
    session), shuffled with Random(42), taken until >= HOLDOUT_SIZE images."""
    groups = defaultdict(list)
    for n in sorted(train_names):
        groups[n.split("-")[0]].append(n)
    prefixes = sorted(groups)
    random.Random(HOLDOUT_SEED).shuffle(prefixes)
    holdout = []
    for p in prefixes:
        if len(holdout) >= HOLDOUT_SIZE:
            break
        holdout += groups[p]
    holdout = sorted(holdout)
    fingerprint = hashlib.sha256("\n".join(holdout).encode()).hexdigest()
    return holdout, fingerprint


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"quarantine": "test split never listed, opened or hashed", "splits": {}}
    sig = {}

    for split in SPLITS:
        imgs = sorted((DATA / split / "images").glob("*.jpg"))
        labels = {p.stem for p in (DATA / split / "labels").glob("*.txt")}
        print(f"[{split}] hashing {len(imgs)} images...", flush=True)
        with ProcessPoolExecutor() as ex:
            results = list(ex.map(image_signature, map(str, imgs), chunksize=256))
        names = [p.name for p in imgs]
        sig[split] = {
            "names": names,
            "md5": [r[0] for r in results],
            "dhash": np.array([r[1] for r in results], dtype=np.uint64),
            "thumb": np.stack([r[2] for r in results]),
        }

        cls_counts, vru_small, vru_total = Counter(), 0, 0
        sizes, problems, empty = Counter(), defaultdict(list), 0
        for p, r in zip(imgs, results):
            sizes[r[3]] += 1
            rows, probs = read_labels(DATA / split / "labels" / f"{p.stem}.txt", r[3])
            if probs:
                problems[p.name] = probs
            if not rows:
                empty += 1
            for c, w, h in rows:
                cls_counts[CLASS_NAMES[c] if 0 <= c < 10 else str(c)] += 1
                if c in VRU_IDS:
                    vru_total += 1
                    vru_small += (w * h) < SMALL_AREA

        md5_dups = [v for v in Counter(sig[split]["md5"]).items() if v[1] > 1]
        report["splits"][split] = {
            "images": len(imgs),
            "label_files": len(labels),
            "images_without_label": len({p.stem for p in imgs} - labels),
            "labels_without_image": len(labels - {p.stem for p in imgs}),
            "images_with_no_boxes": empty,
            "image_sizes": {f"{w}x{h}": n for (w, h), n in sizes.items()},
            "files_with_label_problems": len(problems),
            "label_problem_examples": dict(list(problems.items())[:10]),
            "class_counts": dict(cls_counts),
            "vru_total": vru_total,
            "vru_small_lt_1024px2": vru_small,
            "byte_identical_groups_within_split": len(md5_dups),
            "name_range": [names[0], names[-1]],
            "name_first_char": dict(sorted(Counter(n[0] for n in names).items())),
        }

    tr, va = sig["train"], sig["val"]
    prefix = lambda ns: {n.split("-")[0] for n in ns}
    shared_prefixes = prefix(tr["names"]) & prefix(va["names"])
    report["cross_split"] = {
        "filename_overlap_train_val": sorted(set(tr["names"]) & set(va["names"])),
        "byte_identical_train_val": sorted(
            {m for m in set(tr["md5"]) & set(va["md5"])}
        ),
        "distinct_prefixes": {"train": len(prefix(tr["names"])), "val": len(prefix(va["names"]))},
        "shared_8hex_prefixes_train_val": len(shared_prefixes),
        "val_images_with_prefix_seen_in_train": sum(n.split("-")[0] in shared_prefixes for n in va["names"]),
    }

    print("[near-dup] val vs train...", flush=True)
    nd_val = near_duplicates(va["names"], va["dhash"], va["thumb"], tr["names"], tr["dhash"], tr["thumb"])

    holdout, fingerprint = phase6_holdout(tr["names"])
    hset = set(holdout)
    train_split = [n for n in sorted(tr["names"]) if n not in hset]
    idx = {n: i for i, n in enumerate(tr["names"])}
    hi = np.array([idx[n] for n in holdout])
    ti = np.array([idx[n] for n in train_split])
    print("[near-dup] holdout vs train split...", flush=True)
    nd_hold = near_duplicates(
        holdout, tr["dhash"][hi], tr["thumb"][hi], train_split, tr["dhash"][ti], tr["thumb"][ti]
    )

    report["near_duplicates"] = {
        "method": f"dHash64 hamming<={DHASH_MAX_HAMMING} then 32x32 thumbnail corr>={THUMB_MIN_CORR}",
        "val_vs_train_pairs": len(nd_val),
        "val_images_affected": len({p["a"] for p in nd_val}),
        "val_vs_train_examples": nd_val[:25],
        "holdout_vs_trainsplit_pairs": len(nd_hold),
        "holdout_images_affected": len({p["a"] for p in nd_hold}),
        "holdout_vs_trainsplit_examples": nd_hold[:25],
    }
    report["phase6_holdout"] = {
        "rule": f"8-hex prefix groups, random.Random({HOLDOUT_SEED}).shuffle, until >= {HOLDOUT_SIZE} images",
        "shared_prefixes_holdout_trainsplit": len(prefix(holdout) & prefix(train_split)),
        "holdout_size": len(holdout),
        "train_split_size": len(train_split),
        "disjoint": not (hset & set(train_split)),
        "union_equals_train": hset | set(train_split) == set(tr["names"]),
        "holdout_sha256": fingerprint,
    }

    (OUT / "holdout_manifest.txt").write_text("\n".join(holdout))
    (OUT / "near_duplicates_val_train.json").write_text(json.dumps(nd_val, indent=1))
    (OUT / "near_duplicates_holdout_train.json").write_text(json.dumps(nd_hold, indent=1))
    (OUT / "split_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "splits"}, indent=2)[:6000])
    for s, r in report["splits"].items():
        print(s, {k: r[k] for k in ("images", "label_files", "images_without_label", "images_with_no_boxes",
                                     "files_with_label_problems", "vru_total", "vru_small_lt_1024px2",
                                     "byte_identical_groups_within_split", "name_range", "image_sizes")})


if __name__ == "__main__":
    main()
