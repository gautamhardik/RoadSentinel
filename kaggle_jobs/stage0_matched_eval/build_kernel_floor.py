"""
Builds the Stage 0b kernel: raw detections at a lower confidence floor for an untruncated AP recompute.

Stage 0 computed raw-policy AP from predictions with conf >= 0.01 (evaluator: predict(conf=0.01), max_det=300).
The record audit (scripts/close_stage0_record.py) showed the truncation could hide up to 0.044 VRU mAP50, more
than the gaps between models. This kernel re-runs the same ActiveFoveation-960 inference with the standard
ultralytics validation floor (conf=0.001, max_det=300) and only dumps raw detections; fusion and AP are then
computed locally on CPU with the embedded evaluator's own functions (scripts/stage0b_ap_recompute.py).

The embedded evaluator source is unmodified (same sha256 as Stage 0); its functions (generate_uncertainty_crops,
crop geometry) drive the loop below, which mirrors the evaluator's main loop except for the confidence floor.
Only the BDD100K val split is read; the dataset tree is never walked.

Usage: python kaggle_jobs/stage0_matched_eval/build_kernel_floor.py
Push:  kaggle kernels push -p kaggle_jobs/stage0_matched_eval/kernel_floor
"""

import hashlib
import json
from pathlib import Path

from build_kernel import EVALUATOR, HEADER, METADATA, TRAILER

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "kernel_floor"
OUT_FILE = OUT_DIR / "stage0b_floor_dump.py"

WRAPPER = '''
# ======================= END EMBEDDED EVALUATOR =======================

import gzip as _gzip

EVALUATOR_SHA256 = "{sha}"
CONF_FLOOR, MAX_DET, CANONICAL_FLOOR = 0.001, 300, 0.01
STAGE0_MODELS = [
    ("base_yolo11s", "A4747292648F37F68AB997BEA65A36391C41356F35E309A9F3D2EB1D3EA8D8E3"),
    ("phase5a_u",    "BB1823C6190E28CB56C12B549B0D1233B7DE6CF1B8624B1ED02378B42BB693E5"),
    ("phase5a_r",    "EDCCA4BF03CEC24A4E9D4EEEA524A63FBFD75E40E37359E106264F12A1B447F0"),
]
INPUT = Path("/kaggle/input")
DATA_ROOTS = [INPUT / "datasets/a7madmostafa/bdd100k-yolo", INPUT / "bdd100k-yolo"]


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _checkpoints():
    """Every *.pt under /kaggle/input, pruning the dataset directory (test split never listed)."""
    skip = {{str(r) for r in DATA_ROOTS}}
    found = {{}}
    for dirpath, dirnames, filenames in os.walk(INPUT):
        dirnames[:] = [d for d in dirnames if os.path.join(dirpath, d) not in skip]
        for fn in filenames:
            if fn.endswith(".pt"):
                p = Path(dirpath) / fn
                found[_sha256(p)] = p
    return found


def _dets(res, fname, source, ox=0.0, oy=0.0):
    out = []
    for b in res[0].boxes:
        c = b.xyxy[0].cpu().numpy()
        out.append({{"img": fname, "src": source, "cls": res[0].names[int(b.cls.item())], "conf": round(float(b.conf.item()), 6),
                     "box": [round(float(c[0] + ox), 2), round(float(c[1] + oy), 2), round(float(c[2] + ox), 2), round(float(c[3] + oy), 2)]}})
    return out


def dump(model_path, out_path, val_dir):
    model = YOLO(str(model_path))
    model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    files = sorted(f.name for f in val_dir.glob("*.jpg"))
    n_crops = 0
    with _gzip.open(out_path, "wt", encoding="utf-8") as sink:
        for idx, fname in enumerate(files):
            img = cv2.imread(str(val_dir / fname))
            h, w = img.shape[:2]
            res = model.predict(source=img, imgsz=960, conf=CONF_FLOOR, max_det=MAX_DET, verbose=False)
            ff = _dets(res, fname, "full_frame")
            # crop proposals exactly as the canonical pipeline: only from full-frame detections it would have seen
            # (conf >= 0.01); the small-VRU trigger has no confidence minimum, so lower-floor boxes would change crops
            props = generate_uncertainty_crops(
                [{{"class_name": d["cls"], "conf": d["conf"], "box": d["box"]}} for d in ff if d["conf"] >= CANONICAL_FLOOR],
                w, h, max_crops=2)
            cr = []
            for p in props:
                cx1, cy1, cx2, cy2 = [int(round(v)) for v in p["crop_box"]]
                cx1, cy1 = max(0, cx1), max(0, cy1)
                cx2, cy2 = min(w, cx2), min(h, cy2)
                if (cx2 - cx1) < 10 or (cy2 - cy1) < 10:
                    continue
                n_crops += 1
                sub = model.predict(source=img[cy1:cy2, cx1:cx2], imgsz=640, conf=CONF_FLOOR, max_det=MAX_DET, verbose=False)
                cr += _dets(sub, fname, "crop", cx1, cy1)
            for d in ff + cr:
                sink.write(json.dumps(d) + "\\n")
            if (idx + 1) % 2500 == 0:
                print(f"  {{idx + 1}}/{{len(files)}}", flush=True)
    return len(files), n_crops


def stage0b():
    root = Path("/kaggle/working/stage0b")
    root.mkdir(parents=True, exist_ok=True)
    val_dir = next(r for r in DATA_ROOTS if (r / "val" / "images").is_dir()) / "val" / "images"
    ckpts = _checkpoints()
    missing = [n for n, s in STAGE0_MODELS if s not in ckpts]
    assert not missing, f"checkpoints not found by sha256: {{missing}}"
    manifest = {{"evaluator_sha256": EVALUATOR_SHA256, "conf_floor": CONF_FLOOR, "max_det": MAX_DET,
                 "ultralytics": ultralytics.__version__, "torch": torch.__version__,
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                 "split": "BDD100K val only (test quarantined, dataset tree never walked)", "models": {{}}}}
    for name, sha in STAGE0_MODELS:
        t0 = time.time()
        print(f"##### Stage 0b: {{name}}", flush=True)
        n_img, n_crops = dump(ckpts[sha], root / f"{{name}}_raw_floor.jsonl.gz", val_dir)
        manifest["models"][name] = {{"sha256": sha, "path": str(ckpts[sha]), "images": n_img, "crops": n_crops,
                                    "seconds": round(time.time() - t0, 1)}}
        (root / "stage0b_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    stage0b()
'''


def main():
    src = EVALUATOR.read_text(encoding="utf-8")
    assert src.endswith(TRAILER), "evaluator trailer changed"
    sha = hashlib.sha256(EVALUATOR.read_bytes()).hexdigest().upper()
    OUT_DIR.mkdir(exist_ok=True)
    header = HEADER.format(sha=sha).replace("Matched-Policy Evaluation", "Confidence-Floor Raw Dump (Stage 0b)")
    OUT_FILE.write_text(header + src[: -len(TRAILER)] + WRAPPER.format(sha=sha), encoding="utf-8")
    meta = {**METADATA, "id": "hardikgautam25/roadsentinel-stage-0b-floor-dump",
            "title": "RoadSentinel Stage 0b Floor Dump", "code_file": OUT_FILE.name}
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {OUT_FILE} (evaluator sha256 {sha})")


if __name__ == "__main__":
    main()
