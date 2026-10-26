"""
RoadSentinel Phase 6 — Clean VRU Detector Training (2x T4 DDP, resumable across Kaggle sessions)

Fixes over Phase 5A:
  * Model selection on a 5,000-image training-only holdout (whole recording-session prefix groups, audited in
    docs/phase6/stage1_split_audit.md). Official val is never used for selection; the test split is never read.
  * INIT="coco": official COCO weights -> clean provenance, no third-party checkpoint.
    INIT="base": weights/bdd100k_yolo11s.pt with its classification head permuted into data.yaml order.
  * Self-launches torchrun so custom callbacks (class weights, time budget) survive DDP.
  * Stops before the session budget and saves resume_last.pt (optimizer intact) for the next session.

MODE="smoke" (operational test, ~1 h, no full run):
  * deterministic 10% subset of the 65,000-image train split, 2 epochs;
  * phase A trains epoch 1 and stops as if the session ended; phase B resumes from resume_last.pt for epoch 2
    (sequential, same kernel — no concurrent GPU job);
  * logs GPU utilization/memory (nvidia-smi), epoch train/val durations, learning rate continuity, DDP warnings;
  * proves class weights change the actual training loss (cls loss differs, box/dfl identical);
  * writes smoke_report.json with PASS/FAIL per check and a measured full-schedule time estimate.
MODE="probe" (mosaic throughput probe, ~25 min): same 10% subset and 2x T4 setup, 2 epochs with mosaic=1.0 and
  close_mosaic=0 so mosaic stays on for every epoch (verified from the effective args, the dataset's runtime mosaic flag,
  the Mosaic transform in the pipeline, and the absence of "Closing dataloader mosaic"); records per-batch data-wait vs
  compute time on every rank (CUDA-synchronized), GPU utilization/memory and CPU utilization every 2 s, validation time,
  effective training args and augmentation pipeline; writes probe_report.json (measurements only, no extrapolation).
MODE="full": fixed EPOCHS-epoch schedule split into two sessions at SESSION_STOP_EPOCH. Session 1 stops after
  SESSION_STOP_EPOCH and writes handoff/ (resume_last.pt, best.pt, manifest with sha256 + best fitness). Session 2
  verifies and restores both, finishes the schedule, then evaluates best.pt once on official val (record only).

Checkpoint selection (all modes): ultralytics' fitness is replaced by the VRU macro mAP50-95 on the training-only
holdout = unweighted mean of AP50-95 for person, rider, bike, motor, matched by class NAME (verified against the
dataset data.yaml, the generated data yaml and the model). best.pt follows ultralytics' rules unchanged: best_fitness
updates only on a strictly higher score; best.pt is rewritten when the score equals best_fitness (later epoch wins
ties); best_fitness is restored from the checkpoint on resume.

Resume across sessions (full mode): publish resume_last.pt from the previous session's output as a Kaggle
dataset and add it to dataset_sources; the script picks it up automatically.
"""

import os
import re
import sys
import json
import time
import random
import shutil
import hashlib
import threading
import subprocess
from copy import deepcopy
from pathlib import Path

try:
    import ultralytics
    assert ultralytics.__version__ == "8.4.174"
except (ImportError, AssertionError):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.174"])
    import ultralytics

import numpy as np
import torch
import yaml
from ultralytics import YOLO

# ---------------------------------------------------------------- config
MODE = "full"                    # "smoke" | "probe" | "full"
INIT = "coco"                    # "coco" | "base"
COCO_WEIGHTS = "yolo11m.pt"
IMGSZ = 960
EPOCHS = 2 if MODE in ("smoke", "probe") else 26   # full: fixed from the probe's conservative estimate
SESSION_STOP_EPOCH = 13          # full mode: session 1 trains epochs 1-13, session 2 epochs 14-26
CLOSE_MOSAIC = 0 if MODE == "probe" else 5   # 0 = never close: the probe must measure mosaic epochs
BATCH = 16                       # total across GPUs
HOLDOUT_SIZE = 5000
HOLDOUT_SHA256 = "c7047ee0073803431321735c098aad212693daa696153f31f8a26e11ebf7686b"  # experiments/phase6_audit/split_audit.json
SMOKE_FRACTION, SMOKE_SEED = 0.10, 7
SMOKE_PHASE_TIMEOUT_S = 30 * 60  # hard stop per smoke phase (launcher kills torchrun)
PROBE_PHASE_TIMEOUT_S = 35 * 60  # hard stop for the probe phase
TIME_BUDGET_H = {"smoke": 0.75, "probe": 0.6}.get(MODE, 11.0)
RUN_NAME = "phase6_yolo11m_960_session1"
CLASS_NAMES = ["person", "rider", "car", "bus", "truck", "bike", "motor", "traffic light", "traffic sign", "train"]
VRU_NAMES = ("person", "rider", "bike", "motor")   # selection classes, always resolved by name
CLASS_WEIGHTS = torch.tensor([1.2, 2.0, 1.0, 1.0, 1.0, 1.8, 2.0, 1.0, 1.0, 1.0])
BASE_SHA256 = "a4747292648f37f68ab997bea65a36391c41356f35e309a9f3d2eb1d3ea8d8e3"

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
PROJECT = WORK / "runs"
LAST = PROJECT / RUN_NAME / "weights" / "last.pt"
RESUME_COPY = WORK / "resume_last.pt"
HANDOFF = WORK / "handoff"
BEST_RECORD = WORK / "best_checkpoint.json"  # which epoch best.pt holds (ultralytics' stopper keeps the first tied epoch)
EVENTS = WORK / "events.jsonl"
T0 = float(os.environ.get("RS_T0", time.time()))
PHASE = os.environ.get("RS_PHASE", "")
IS_MAIN_RANK = int(os.environ.get("RANK", -1)) in (-1, 0)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


DATA_ROOTS = [INPUT / "datasets/a7madmostafa/bdd100k-yolo", INPUT / "bdd100k-yolo"]


def data_root() -> Path:
    """Explicit dataset location: never walk the dataset tree (its test/ split is quarantined)."""
    return next(r for r in DATA_ROOTS if (r / "train" / "images").is_dir())


def find_one(name: str):
    """Searches /kaggle/input for a file by name, pruning the BDD100K dataset directory entirely."""
    skip = {str(r) for r in DATA_ROOTS}
    for dirpath, dirnames, filenames in os.walk(INPUT):
        dirnames[:] = [d for d in dirnames if os.path.join(dirpath, d) not in skip]
        if name in filenames:
            return Path(dirpath) / name
    return None


def log_event(**kv):
    if IS_MAIN_RANK:
        with open(EVENTS, "a") as f:
            f.write(json.dumps({"t": time.time(), "phase": PHASE, **kv}) + "\n")


def build_data_yaml() -> dict:
    """Deterministic train/holdout split; identical on every rank and every session."""
    train_dir, val_dir = data_root() / "train" / "images", data_root() / "val" / "images"
    dataset_names = yaml.safe_load((data_root() / "data.yaml").read_text())["names"]
    dataset_names = list(dataset_names.values()) if isinstance(dataset_names, dict) else list(dataset_names)
    assert dataset_names == CLASS_NAMES, f"dataset data.yaml class order differs from CLASS_NAMES: {dataset_names}"
    names = sorted(p.name for p in train_dir.glob("*.jpg"))
    groups = {}
    for n in names:  # whole 8-hex prefix groups (same recording session) stay on one side
        groups.setdefault(n.split("-")[0], []).append(n)
    prefixes = sorted(groups)
    random.Random(42).shuffle(prefixes)
    holdout = []
    for p in prefixes:
        if len(holdout) >= HOLDOUT_SIZE:
            break
        holdout += groups[p]
    holdout = sorted(holdout)
    fingerprint = hashlib.sha256("\n".join(holdout).encode()).hexdigest()
    assert fingerprint == HOLDOUT_SHA256, f"holdout differs from audited manifest: {fingerprint}"
    hset = set(holdout)
    vru_ids = {CLASS_NAMES.index(n): n for n in VRU_NAMES}
    holdout_vru = {n: 0 for n in VRU_NAMES}
    for n in holdout:
        for line in (train_dir.parent / "labels" / (Path(n).stem + ".txt")).read_text().splitlines():
            if line.split() and int(line.split()[0]) in vru_ids:
                holdout_vru[vru_ids[int(line.split()[0])]] += 1
    assert all(holdout_vru.values()), f"holdout lacks a VRU class, selection metric undefined: {holdout_vru}"
    train = [n for n in names if n not in hset]
    if MODE in ("smoke", "probe"):
        train = sorted(random.Random(SMOKE_SEED).sample(train, round(SMOKE_FRACTION * len(train))))
    assert not (set(train) & hset), "training images overlap the holdout"
    (WORK / "train_split.txt").write_text("\n".join(str(train_dir / n) for n in train))
    (WORK / "holdout_split.txt").write_text("\n".join(str(train_dir / n) for n in holdout))
    data = {
        "path": str(train_dir.parent.parent),
        "train": str(WORK / "train_split.txt"),
        "val": str(WORK / "holdout_split.txt"),
        "names": dict(enumerate(CLASS_NAMES)),
    }
    (WORK / "data_phase6.yaml").write_text(yaml.dump(data))
    (WORK / "val_official.yaml").write_text(yaml.dump({**data, "val": str(val_dir)}))
    return {"train_images": len(train), "holdout_images": len(holdout), "holdout_sha256": fingerprint,
            "train_list_sha256": hashlib.sha256("\n".join(train).encode()).hexdigest(), "holdout_vru_instances": holdout_vru}


def verify_class_mapping(data_names, model_names) -> dict:
    """Data and model class names must be identical index-for-index; returns {vru_name: class_index}."""
    dn = {int(k): v for k, v in dict(data_names).items()}
    mn = {int(k): v for k, v in dict(model_names).items()}
    if dn != mn:
        raise ValueError(f"data names {dn} != model names {mn}")
    mapping = {}
    for name in VRU_NAMES:
        ids = [k for k, v in dn.items() if v == name]
        if len(ids) != 1:
            raise ValueError(f"VRU class {name!r} must appear exactly once in class names, found {len(ids)}")
        mapping[name] = ids[0]
    return mapping


def vru_macro_ap5095(names, ap_class_index, all_ap) -> float:
    """Primary selection fitness: unweighted mean of AP50-95 over exactly person, rider, bike, motor.

    names: {class_index: class_name}; ap_class_index: class index of each row of all_ap; all_ap: (n, 10) AP at
    IoU 0.50:0.95. Classes are matched by name. No AP rows at all (no true positive anywhere) means every AP is 0.
    A VRU class without an AP row while others have one is absent from the evaluation set: ValueError (protocol
    violation; the holdout is checked for all four classes before training). Non-finite AP counts as 0.0.
    """
    mapping = verify_class_mapping(names, names)
    all_ap = np.asarray(all_ap, dtype=float)
    if all_ap.size == 0:
        return 0.0
    rows = {int(c): i for i, c in enumerate(ap_class_index)}
    aps = []
    for name, idx in mapping.items():
        if idx not in rows:
            raise ValueError(f"no AP for VRU class {name!r} (index {idx}): absent from the evaluation set")
        ap = float(np.mean(all_ap[rows[idx]]))
        aps.append(ap if np.isfinite(ap) else 0.0)
    return float(np.mean(aps))


def vru_metrics_fitness(metrics) -> float:
    """Installed as ultralytics DetMetrics.fitness inside training processes."""
    return vru_macro_ap5095(metrics.names, metrics.box.ap_class_index, metrics.box.all_ap)


def install_vru_fitness():
    from ultralytics.utils.metrics import DetMetrics
    DetMetrics.fitness = property(lambda self: vru_metrics_fitness(self))


def secondary_metrics(metrics) -> dict:
    """Reported, never used for selection: all-class mAP50-95, VRU mAP50, per-class VRU AP50/AP50-95."""
    mapping = verify_class_mapping(metrics.names, metrics.names)
    all_ap = np.asarray(metrics.box.all_ap, dtype=float)
    rows = {int(c): i for i, c in enumerate(metrics.box.ap_class_index)}
    per = {n: {"ap50": float(all_ap[rows[i], 0]) if i in rows else 0.0,
               "ap50_95": float(all_ap[rows[i]].mean()) if i in rows else 0.0} for n, i in mapping.items()}
    return {"all_class_map50_95": float(metrics.box.map), "vru_map50": float(np.mean([v["ap50"] for v in per.values()])),
            "per_class_vru": per}


def weights_fingerprint(ckpt: dict) -> str:
    """sha256 over the checkpoint's model tensors. ultralytics saves EMA weights under "ema" and its end-of-run
    optimizer strip moves the same tensors to "model", so the fingerprint survives the strip; the file hash does not."""
    m = ckpt["ema"] if ckpt.get("ema") is not None else ckpt["model"]
    h = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def best_record_matches(record: dict, best_ckpt: dict) -> bool:
    return (record["weights_fingerprint"] == weights_fingerprint(best_ckpt)
            and record["fitness"] == float(best_ckpt["train_metrics"]["fitness"]))


def write_handoff(resume_path: Path, best_path: Path, out_dir: Path, best_record: Path) -> dict:
    """Session-1 output: resumable checkpoint + global best checkpoint + manifest binding them together."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resume_path, out_dir / "resume_last.pt")
    shutil.copy2(best_path, out_dir / "best.pt")
    shutil.copy2(best_record, out_dir / "best_checkpoint.json")
    ck = torch.load(out_dir / "resume_last.pt", map_location="cpu", weights_only=False)
    bk = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    record = json.loads(best_record.read_text())
    assert best_record_matches(record, bk), "best_checkpoint.json does not describe best.pt"
    manifest = {"completed_epochs": int(ck["epoch"]) + 1, "best_fitness": float(ck["best_fitness"]),
                "best_epoch": int(record["epoch"]),
                "best_pt_fitness": float(bk["train_metrics"]["fitness"]),
                "resume_last_sha256": sha256(out_dir / "resume_last.pt"), "best_sha256": sha256(out_dir / "best.pt"),
                "best_weights_fingerprint": weights_fingerprint(bk),
                "selection_metric": "VRU macro AP50-95 (person, rider, bike, motor) on training-only holdout"}
    assert abs(manifest["best_pt_fitness"] - manifest["best_fitness"]) < 1e-12, "best.pt is not the best-fitness checkpoint"
    (out_dir / "handoff_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def restore_handoff(manifest_path: Path, last_dst: Path, best_dst: Path, record_dst: Path) -> dict:
    """Session-2 input: verify hashes and fitness consistency, then restore last.pt and the global best.pt."""
    m = json.loads(manifest_path.read_text())
    src = manifest_path.parent
    for name, key in (("resume_last.pt", "resume_last_sha256"), ("best.pt", "best_sha256")):
        if sha256(src / name) != m[key]:
            raise ValueError(f"{name} sha256 does not match the handoff manifest")
    ck = torch.load(src / "resume_last.pt", map_location="cpu", weights_only=False)
    bk = torch.load(src / "best.pt", map_location="cpu", weights_only=False)
    if not (float(ck["best_fitness"]) == m["best_fitness"] == float(bk["train_metrics"]["fitness"])):
        raise ValueError("best fitness differs between resume checkpoint, best.pt and manifest")
    if weights_fingerprint(bk) != m["best_weights_fingerprint"]:
        raise ValueError("best.pt weights fingerprint does not match the handoff manifest")
    record = json.loads((src / "best_checkpoint.json").read_text())
    if not (best_record_matches(record, bk) and record["epoch"] == m["best_epoch"]):
        raise ValueError("best_checkpoint.json does not match best.pt / manifest")
    record_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "best_checkpoint.json", record_dst)
    last_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "resume_last.pt", last_dst)
    shutil.copy2(src / "best.pt", best_dst)
    return m


def permute_head_to_data_order(model: YOLO):
    """Reorder classification output channels of the base checkpoint into data.yaml order."""
    ckpt_names = model.model.names
    idx = torch.tensor([[k for k, v in ckpt_names.items() if v == n][0] for n in CLASS_NAMES])
    detect = model.model.model[-1]
    for branch in detect.cv3:
        conv = branch[-1]
        conv.weight.data = conv.weight.data[idx].clone()
        conv.bias.data = conv.bias.data[idx].clone()
    model.model.names = dict(enumerate(CLASS_NAMES))


def prove_loss_weighting(trainer, m):
    """Same 4 training images through the real criterion with and without class weights."""
    ds = trainer.train_loader.dataset
    batch = trainer.preprocess_batch(ds.collate_fn([ds[i] for i in range(4)]))
    probe = deepcopy(m).float()  # copy: keeps the live model's BatchNorm statistics untouched
    with torch.no_grad():
        _, weighted = probe.loss(batch)
        probe.criterion.class_weights = None
        _, unweighted = probe.loss(batch)
    w = [float(x) for x in (weighted.values() if isinstance(weighted, dict) else weighted)]
    u = [float(x) for x in (unweighted.values() if isinstance(unweighted, dict) else unweighted)]
    ok = abs(w[1] - u[1]) > 1e-6 and abs(w[0] - u[0]) < 1e-6 and abs(w[2] - u[2]) < 1e-6
    result = {"box_cls_dfl_weighted": w, "box_cls_dfl_unweighted": u, "PASS": ok}
    (WORK / "loss_weight_proof.json").write_text(json.dumps(result, indent=2))
    print(f"[loss_weight_proof] {result}", flush=True)
    del probe


def mosaic_probabilities(transforms) -> list:
    """p of every Mosaic transform in a (nested) ultralytics Compose pipeline."""
    from ultralytics.data.augment import Mosaic
    found, stack = [], [transforms]
    while stack:
        t = stack.pop()
        if isinstance(t, Mosaic):
            found.append(float(t.p))
        stack.extend(getattr(t, "transforms", None) or [])
    return found


def add_callbacks(model: YOLO):
    lr_seen = {}
    timing = {"t_end": None, "t_start": None, "wait": [], "compute": []}
    current = {"epoch": None}  # epoch index actually being trained in this process

    def batch_start(trainer):
        t = time.perf_counter()
        if timing["t_end"] is not None:
            timing["wait"].append(t - timing["t_end"])  # time blocked on the dataloader for this batch
        timing["t_start"] = t

    def batch_end(trainer):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # make compute time real, not just kernel-launch time
        t = time.perf_counter()
        timing["compute"].append(t - timing["t_start"])
        timing["t_end"] = t

    def dump_effective_config(trainer):
        if IS_MAIN_RANK:
            args = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in vars(trainer.args).items()}
            (WORK / "effective_train_args.json").write_text(json.dumps(args, indent=2))
            (WORK / "augmentation_pipeline.txt").write_text(str(trainer.train_loader.dataset.transforms))

    def attach_weights(trainer):
        m = getattr(trainer.model, "module", trainer.model)
        yaml_names = yaml.safe_load(Path(trainer.args.data).read_text())["names"]
        mapping = verify_class_mapping(yaml_names, m.names)
        verify_class_mapping(trainer.data["names"], m.names)
        assert all(CLASS_NAMES[i] == n for n, i in mapping.items()), mapping
        if IS_MAIN_RANK:
            (WORK / "vru_class_mapping.json").write_text(json.dumps(mapping, indent=2))
            log_event(event="class_mapping", vru=mapping)
        m.class_weights = CLASS_WEIGHTS.to(trainer.device)
        m.criterion = m.init_criterion()
        if MODE == "smoke" and IS_MAIN_RANK and PHASE == "A":
            prove_loss_weighting(trainer, m)

    def on_train_start(trainer):
        opt_state = trainer.optimizer.state_dict()["state"]
        log_event(event="train_start", start_epoch=trainer.start_epoch, epochs=trainer.epochs,
                  ema_updates=int(trainer.ema.updates) if trainer.ema else None,
                  optimizer_state_tensors=len(opt_state), world_size=trainer.world_size,
                  batch_per_gpu=trainer.batch_size)

    def check_weights(trainer):
        m = getattr(trainer.model, "module", trainer.model)
        cw = m.criterion.class_weights
        assert cw is not None and torch.allclose(cw.squeeze().cpu(), CLASS_WEIGHTS), "class weights inactive"
        lr_seen.clear()
        current["epoch"] = trainer.epoch
        log_event(event="epoch_start", epoch=trainer.epoch + 1)

    def track_lr(trainer):
        g = trainer.optimizer.param_groups[0]
        if "first" not in lr_seen:
            # ultralytics schedule: warmup interpolation while ni < nw, else initial_lr * lf(epoch)
            nb = len(trainer.train_loader)
            nw, ni = trainer._get_warmup_iterations(nb), nb * trainer.epoch
            target = g["initial_lr"] * trainer.lf(trainer.epoch)
            start = trainer.args.warmup_bias_lr if g.get("param_group") == "bias" else 0.0
            lr_seen["first"] = g["lr"]
            lr_seen["expected_first"] = float(np.interp(ni, [0, nw], [start, target])) if ni < nw else target
        lr_seen["last"] = g["lr"]

    def epoch_end(trainer):
        ds = trainer.train_loader.dataset
        # ultralytics applies mosaic only when augment and not rect; closing rebuilds the pipeline with Mosaic(p=0.0)
        log_event(event="mosaic_state", epoch=trainer.epoch + 1, dataset_augment=bool(getattr(ds, "augment", False)),
                  dataset_rect=bool(getattr(ds, "rect", False)), mosaic_transform_p=mosaic_probabilities(ds.transforms))
        if MODE == "probe":
            rank = max(int(os.environ.get("RANK", 0)), 0)
            (WORK / f"batch_timing_rank{rank}_epoch{trainer.epoch + 1}.json").write_text(
                json.dumps({"wait_s": timing["wait"], "compute_s": timing["compute"]}))
            timing.update(t_end=None, wait=[], compute=[])
        log_event(event="train_end", epoch=trainer.epoch + 1, lr_first=lr_seen.get("first"), lr_last=lr_seen.get("last"),
                  lr_expected_first=lr_seen.get("expected_first"),
                  mem_alloc_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0)

    def keep_resumable(trainer):
        # final_eval strips the optimizer from last.pt; keep an intact copy for the next session
        shutil.copy2(trainer.last, RESUME_COPY)
        torch.save({"epoch": trainer.epoch, "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "numpy": np.random.get_state(), "python": random.getstate()}, WORK / "rng_state_rank0.pt")

    def fit_end(trainer):
        val = getattr(getattr(trainer, "validator", None), "metrics", None)
        if trainer.epoch != current["epoch"]:
            # ultralytics final_eval re-fires on_fit_epoch_end at epoch + 1 after validating best.pt: record, don't act
            log_event(event="final_eval_best_pt", metrics={k: float(v) for k, v in (trainer.metrics or {}).items()},
                      secondary=secondary_metrics(val) if (IS_MAIN_RANK and val is not None) else None)
            return
        is_best = bool(trainer.best_fitness == trainer.fitness)
        best_trace = None
        if IS_MAIN_RANK and is_best and trainer.best.exists():  # identify exactly what best.pt now holds
            ck = torch.load(trainer.best, map_location="cpu", weights_only=False)
            best_trace = {"epoch": trainer.epoch + 1, "date": ck.get("date"), "file_sha256": sha256(trainer.best),
                          "weights_fingerprint": weights_fingerprint(ck), "fitness": float(ck["train_metrics"]["fitness"])}
            BEST_RECORD.write_text(json.dumps(best_trace, indent=2))
        log_event(event="fit_end", epoch=trainer.epoch + 1, fitness=float(trainer.fitness or 0),
                  best_fitness=float(trainer.best_fitness or 0), is_best=is_best, best_ckpt=best_trace,
                  secondary=secondary_metrics(val) if (IS_MAIN_RANK and val is not None) else None,
                  metrics={k: float(v) for k, v in (trainer.metrics or {}).items()})
        if MODE == "full" and trainer.epoch + 1 == SESSION_STOP_EPOCH < trainer.epochs:
            print(f"[session] planned session boundary reached after epoch {trainer.epoch + 1}", flush=True)
            trainer.stop = True
            return
        if MODE == "smoke" and PHASE == "A":
            print(f"[smoke] phase A: simulated session end after epoch {trainer.epoch + 1}", flush=True)
            trainer.stop = True
            return
        elapsed = time.time() - T0
        per_epoch = elapsed / max(1, trainer.epoch + 1 - trainer.start_epoch)
        if elapsed + 1.2 * per_epoch > TIME_BUDGET_H * 3600:
            print(f"[time_guard] {elapsed/3600:.2f}h used; stopping after epoch {trainer.epoch + 1}", flush=True)
            trainer.stop = True

    model.add_callback("on_pretrain_routine_end", attach_weights)
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_start", dump_effective_config)
    if MODE == "probe":
        model.add_callback("on_train_batch_start", batch_start)
        model.add_callback("on_train_batch_end", batch_end)
    model.add_callback("on_train_epoch_start", check_weights)
    model.add_callback("on_train_batch_end", track_lr)
    model.add_callback("on_train_epoch_end", epoch_end)
    model.add_callback("on_model_save", keep_resumable)
    model.add_callback("on_fit_epoch_end", fit_end)


def train_worker():
    """Runs inside every torchrun rank (or alone on a single GPU)."""
    install_vru_fitness()
    n_gpu = torch.cuda.device_count()
    device = ",".join(str(i) for i in range(n_gpu)) if n_gpu else "cpu"
    if LAST.exists():
        model = YOLO(str(LAST))
        add_callbacks(model)
        model.train(resume=True, device=device)
        return
    if INIT == "base":
        base = find_one("bdd100k_yolo11s.pt")
        assert sha256(base) == BASE_SHA256, "base checkpoint hash mismatch"
        model = YOLO(str(base))
        permute_head_to_data_order(model)
        opt = dict(optimizer="AdamW", lr0=3e-4, lrf=0.05, warmup_epochs=1.0)
    else:
        model = YOLO(COCO_WEIGHTS)
        opt = dict(optimizer="SGD", lr0=0.01, lrf=0.01, warmup_epochs=3.0)
    add_callbacks(model)
    model.train(
        data=str(WORK / "data_phase6.yaml"), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=device, workers=4,
        mosaic=1.0, close_mosaic=CLOSE_MOSAIC, mixup=0.0, copy_paste=0.0, scale=0.5, fliplr=0.5,
        seed=42, deterministic=False, amp=True, save_period=-1, plots=True,
        project=str(PROJECT), name=RUN_NAME, exist_ok=True, **opt,
    )


def launch(phase: str, timeout_s=None) -> dict:
    """Runs one training phase in fresh process(es); returns timing and exit status."""
    n_gpu = torch.cuda.device_count()
    env = {**os.environ, "RS_T0": str(time.time()), "RS_PHASE": phase}
    if n_gpu > 1:
        cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={n_gpu}", __file__]
    else:
        cmd = [sys.executable, __file__]
        env["RS_SINGLE_WORKER"] = "1"
    log_path = WORK / f"train_phase_{phase}.log"
    t0 = time.time()
    with open(log_path, "w") as log:
        try:
            rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s).returncode
        except subprocess.TimeoutExpired:
            rc = "TIMEOUT"
    print(f"[launch] phase {phase} exit={rc} in {time.time() - t0:.0f}s; tail:\n" + "".join(open(log_path, encoding="utf-8", errors="replace").readlines()[-15:]), flush=True)
    return {"phase": phase, "exit": rc, "seconds": round(time.time() - t0, 1), "log": log_path.name}


class GpuMonitor:
    """Samples nvidia-smi every `interval` s into gpu_util.csv."""

    def __init__(self, interval=5):
        self.path = WORK / "gpu_util.csv"
        self.interval = interval
        self.proc = None

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits", "-l", str(self.interval)], stdout=open(self.path, "w"), stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            pass
        return self

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()

    def summary(self, windows):
        """Per-GPU stats over samples inside the training windows [(t0, t1), ...]."""
        from datetime import datetime
        rows = defaultdict_list()
        for line in self.path.read_text().splitlines() if self.path.exists() else []:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 5:
                continue
            try:
                ts = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            except ValueError:
                continue
            if any(a <= ts <= b for a, b in windows):
                rows[parts[1]].append((float(parts[2]), float(parts[3]), float(parts[4])))
        return {gpu: {"samples": len(v), "util_mean_pct": round(float(np.mean([r[0] for r in v])), 1),
                      "util_p10_pct": round(float(np.percentile([r[0] for r in v], 10)), 1),
                      "mem_peak_mib": max(r[1] for r in v), "mem_total_mib": v[0][2]} for gpu, v in rows.items() if v}


class CpuMonitor:
    """Samples host CPU utilization and 1-min load average every 2 s (data-loading pressure)."""

    def __init__(self):
        self.rows, self.stop = [], threading.Event()

    def _run(self):
        import psutil
        while not self.stop.is_set():
            pct = psutil.cpu_percent(interval=2)
            self.rows.append((time.time(), pct, os.getloadavg()[0] if hasattr(os, "getloadavg") else None))

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        (WORK / "cpu_util.csv").write_text("\n".join(f"{t},{p},{l}" for t, p, l in self.rows))

    def summary(self, windows):
        v = [(p, l) for t, p, l in self.rows if any(a <= t <= b for a, b in windows)]
        if not v:
            return {}
        loads = [l for _, l in v if l is not None]
        return {"samples": len(v), "cpu_count": os.cpu_count(), "util_mean_pct": round(float(np.mean([p for p, _ in v])), 1),
                "util_p90_pct": round(float(np.percentile([p for p, _ in v], 90)), 1),
                "load1_mean": round(float(np.mean(loads)), 2) if loads else None}


def epoch_table(events):
    starts = [e for e in events if e["event"] == "epoch_start"]
    epochs = []
    for s in starts:
        e = next((x for x in events if x["event"] == "train_end" and x["epoch"] == s["epoch"] and x["phase"] == s["phase"]), None)
        f = next((x for x in events if x["event"] == "fit_end" and x["epoch"] == s["epoch"] and x["phase"] == s["phase"]), None)
        if e and f:
            epochs.append({"phase": s["phase"], "epoch": s["epoch"], "t_start": s["t"], "t_train_end": e["t"],
                           "train_s": round(e["t"] - s["t"], 1), "val_s": round(f["t"] - e["t"], 1),
                           "lr_first": e["lr_first"], "lr_last": e["lr_last"],
                           "lr_expected_first": e.get("lr_expected_first"),
                           "mem_alloc_gb": e["mem_alloc_gb"], "fitness": f["fitness"]})
    return epochs


def scan_warnings(phases):
    warn_re = re.compile(r"(?i)\b(warn|warning|nccl|error|traceback|timeout)\b")
    out = []
    for p in phases:
        for line in (WORK / p["log"]).read_text(encoding="utf-8", errors="replace").splitlines():
            if warn_re.search(line) and line.strip() not in out:
                out.append(line.strip()[:300])
    return out


def steady_state(rank_files, skip=20, window=25):
    """Per-step global throughput from CUDA-synchronized batch timings, first `skip` steps of each epoch excluded."""
    rates, waits, totals = [], [], []
    for f in rank_files:
        d = json.loads(f.read_text())
        comp, wait = d["compute_s"][1:], d["wait_s"]  # wait[i] precedes compute[i + 1]
        n = min(len(comp), len(wait))
        step = [w + c for w, c in zip(wait[:n], comp[:n])][skip:]
        waits += wait[:n][skip:]
        totals += step
        for i in range(0, len(step) - window + 1, window):
            rates.append(BATCH * window / sum(step[i:i + window]))
    if not rates:
        return {}
    return {"steps_measured": len(totals), "global_img_per_s_p10": round(float(np.percentile(rates, 10)), 2),
            "global_img_per_s_p50": round(float(np.percentile(rates, 50)), 2),
            "global_img_per_s_p90": round(float(np.percentile(rates, 90)), 2),
            "data_wait_fraction": round(float(sum(waits) / sum(totals)), 3),
            "step_s_p50": round(float(np.median(totals)), 3), "data_wait_s_p50": round(float(np.median(waits)), 4)}


def probe_report(split_info, phases, gpu, cpu):
    events = [json.loads(l) for l in EVENTS.read_text().splitlines()] if EVENTS.exists() else []
    epochs = epoch_table(events)
    mosaic = [e for e in events if e["event"] == "mosaic_state"]
    args_path = WORK / "effective_train_args.json"
    args = json.loads(args_path.read_text()) if args_path.exists() else {}
    log = (WORK / phases[0]["log"]).read_text(encoding="utf-8", errors="replace")
    warnings = scan_warnings(phases)
    per_epoch_rank0 = {e["epoch"]: steady_state([WORK / f"batch_timing_rank0_epoch{e['epoch']}.json"])
                       for e in epochs if (WORK / f"batch_timing_rank0_epoch{e['epoch']}.json").exists()}
    all_ranks = steady_state(sorted(WORK.glob("batch_timing_rank*_epoch*.json")))
    start = next((e for e in events if e["event"] == "train_start"), None)
    measured = {
        "epochs": [{k: v for k, v in e.items() if not k.startswith("t_")} for e in epochs],
        "train_images_per_epoch": split_info["train_images"],
        "epoch_level_img_per_s": [round(split_info["train_images"] / e["train_s"], 2) for e in epochs],
        "steady_state_by_epoch_rank0": per_epoch_rank0,
        "steady_state_all_ranks_epochs": all_ranks,
        "holdout_val_s": [e["val_s"] for e in epochs],
        "launch_overhead_s": round(phases[0]["seconds"] - sum(e["train_s"] + e["val_s"] for e in epochs), 1),
        "gpu": gpu, "cpu": cpu,
    }
    checks = {
        "two_gpus_visible": torch.cuda.device_count() >= 2,
        "ddp_world_size_2": bool(start and start.get("world_size") == 2),
        "both_gpus_utilized": len(gpu) >= 2 and all(g["util_mean_pct"] >= 50 for g in gpu.values()),
        "phase_completed": phases[0]["exit"] == 0 and len(epochs) == EPOCHS,
        "mosaic_in_effective_args": args.get("mosaic") == 1.0 and args.get("close_mosaic") == 0,
        "mosaic_active_every_epoch": len(mosaic) == EPOCHS and all(
            m["dataset_augment"] and not m["dataset_rect"] and m["mosaic_transform_p"] == [1.0] for m in mosaic),
        "mosaic_never_closed": "Closing dataloader mosaic" not in log,
        "holdout_excluded_from_training": True,  # build_data_yaml asserts empty overlap before any training starts
        "no_ddp_errors": not any(re.search(r"(?i)nccl.*error|traceback|timeout", w) for w in warnings),
    }
    hashes = {p.name: sha256(p) for p in [LAST, LAST.with_name("best.pt"), args_path,
                                          WORK / "augmentation_pipeline.txt", WORK / "run_config.json"] if p.exists()}
    aug_keys = ("mosaic", "close_mosaic", "mixup", "copy_paste", "scale", "translate", "fliplr", "flipud", "degrees",
                "shear", "perspective", "hsv_h", "hsv_s", "hsv_v", "erasing", "auto_augment", "imgsz", "batch", "workers")
    report = {"mode": MODE, "split": split_info, "phases": phases, "measured": measured, "mosaic_events": mosaic,
              "effective_augmentation": {k: args.get(k) for k in aug_keys},
              "checks": {k: "PASS" if v else "FAIL" for k, v in checks.items()}, "warnings": warnings[:40],
              "hashes_sha256": hashes, "note": "Measurements only; full-run extrapolation is done separately."}
    (WORK / "probe_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"checks": report["checks"], "measured": measured}, indent=2, default=str), flush=True)


def defaultdict_list():
    from collections import defaultdict
    return defaultdict(list)


def smoke_report(split_info, phases, gpu):
    events = [json.loads(l) for l in EVENTS.read_text().splitlines()] if EVENTS.exists() else []
    ev = lambda name, phase=None: [e for e in events if e["event"] == name and (phase is None or e["phase"] == phase)]
    starts, ends, fits = ev("epoch_start"), ev("train_end"), ev("fit_end")
    epochs = []
    for s in starts:
        e = next((x for x in ends if x["epoch"] == s["epoch"] and x["phase"] == s["phase"]), None)
        f = next((x for x in fits if x["epoch"] == s["epoch"] and x["phase"] == s["phase"]), None)
        if e and f:
            epochs.append({"phase": s["phase"], "epoch": s["epoch"], "train_s": round(e["t"] - s["t"], 1),
                           "val_s": round(f["t"] - e["t"], 1), "lr_first": e["lr_first"], "lr_last": e["lr_last"],
                           "lr_expected_first": e.get("lr_expected_first"),
                           "mem_alloc_gb": e["mem_alloc_gb"], "fitness": f["fitness"]})

    warn_re = re.compile(r"(?i)\b(warn|warning|nccl|error|traceback|timeout)\b")
    warnings = []
    for p in phases:
        for line in (WORK / p["log"]).read_text(encoding="utf-8", errors="replace").splitlines():
            if warn_re.search(line) and line.strip() not in warnings:
                warnings.append(line.strip()[:300])

    ckpt_audit = {}
    snapshot_a = WORK / "resume_after_phase_A.pt"  # the checkpoint phase B actually resumed from
    if snapshot_a.exists():
        ck = torch.load(snapshot_a, map_location="cpu", weights_only=False)
        ckpt_audit = {k: (k in ck and ck[k] is not None) for k in
                      ("ema", "optimizer", "scaler", "epoch", "updates", "train_args", "train_metrics", "best_fitness")}
        ckpt_audit["saved_epoch_index"] = ck.get("epoch")
        ckpt_audit["saved_ema_updates"] = ck.get("updates")
    hashes = {p.name: sha256(p) for p in [snapshot_a, RESUME_COPY, LAST, LAST.with_name("best.pt"), WORK / "rng_state_rank0.pt"] if p.exists()}

    b_first = next((e for e in epochs if e["phase"] == "B"), None)
    b_start = (ev("train_start", "B") or [None])[0]
    a_start = (ev("train_start", "A") or [None])[0]
    proof = json.loads((WORK / "loss_weight_proof.json").read_text()) if (WORK / "loss_weight_proof.json").exists() else {}
    log_b = (WORK / "train_phase_B.log").read_text(encoding="utf-8", errors="replace") if (WORK / "train_phase_B.log").exists() else ""

    checks = {
        "two_gpus_visible": torch.cuda.device_count() >= 2,
        "ddp_world_size_2": bool(a_start and a_start.get("world_size") == 2),
        "both_gpus_utilized": len(gpu) >= 2 and all(g["util_mean_pct"] >= 50 for g in gpu.values()),
        "phase_A_completed": phases[0]["exit"] == 0,
        "phase_B_completed": len(phases) > 1 and phases[1]["exit"] == 0,
        "class_weights_change_loss": bool(proof.get("PASS")),
        "checkpoint_has_model_optimizer_scaler_epoch_ema": bool(ckpt_audit) and all(
            ckpt_audit[k] for k in ("ema", "optimizer", "scaler", "epoch", "updates", "train_args")),
        "resume_started_at_epoch_2": bool(b_start and b_start["start_epoch"] == 1) and "Resuming training" in log_b,
        "resume_restored_ema_updates": bool(b_start and ckpt_audit and b_start["ema_updates"] == ckpt_audit["saved_ema_updates"]),
        "resume_restored_optimizer_state": bool(b_start and b_start["optimizer_state_tensors"] > 0),
        "lr_schedule_exact_after_resume": bool(b_first and b_first["lr_expected_first"] is not None and
                                               abs(b_first["lr_first"] - b_first["lr_expected_first"]) <= 1e-9 + 1e-6 * abs(b_first["lr_expected_first"])),
        "no_ddp_errors": not any(re.search(r"(?i)nccl.*error|traceback|timeout", w) for w in warnings),
    }

    # measured full-schedule estimate (no extrapolation from guesses)
    estimate = {}
    tr = [e["train_s"] for e in epochs]
    va = [e["val_s"] for e in epochs]
    if tr and va:
        full_train_images = 65000
        train_per_full_epoch = max(tr) * full_train_images / split_info["train_images"]  # slowest smoke epoch, scaled
        val_per_epoch = max(va)
        per_epoch = train_per_full_epoch + val_per_epoch
        overhead_s = max(p["seconds"] for p in phases) - (max(tr) + max(va))  # startup, dataset scan, final eval
        official_val_s = 2 * val_per_epoch  # 10,000 val images vs 5,000 holdout
        budget_s = 11.0 * 3600
        usable = budget_s * 0.85 - overhead_s - official_val_s  # 15% safety margin
        estimate = {
            "measured_train_s_per_smoke_epoch": tr, "measured_val_s_per_epoch": va,
            "train_images_per_s": round(split_info["train_images"] / max(tr), 2),
            "projected_s_per_full_epoch": round(per_epoch), "startup_and_final_eval_overhead_s": round(overhead_s),
            "official_val_report_s": round(official_val_s), "session_budget_s": budget_s, "safety_margin": 0.15,
            "max_epochs_fitting_one_session": int(usable // per_epoch),
            "note": "Linear scaling of the slowest measured smoke epoch from 6,500 to 65,000 images.",
        }

    report = {"mode": MODE, "split": split_info, "phases": phases, "gpu": gpu, "epochs": epochs,
              "loss_weight_proof": proof, "checkpoint_audit": ckpt_audit, "hashes_sha256": hashes,
              "not_saved_by_ultralytics": ["scheduler state (rebuilt from epoch index on resume)",
                                           "RNG state of rank 1 and dataloader workers",
                                           "resume reseeds every rank with seed+1+RANK, so augmentation order after resume differs from an uninterrupted run"],
              "warnings": warnings[:40], "checks": {k: "PASS" if v else "FAIL" for k, v in checks.items()},
              "estimate": estimate}
    (WORK / "smoke_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"checks": report["checks"], "estimate": estimate, "gpu": gpu}, indent=2, default=str), flush=True)


def report_official_val(weights: Path):
    """Official-val metrics for the record. Never used for selection."""
    m = YOLO(str(weights)).val(data=str(WORK / "val_official.yaml"), imgsz=IMGSZ, batch=16, plots=False,
                               device=0 if torch.cuda.is_available() else "cpu")
    ap50, ap = m.box.all_ap[:, 0], m.box.all_ap.mean(1)
    classes = list(m.box.ap_class_index)
    per_class = {CLASS_NAMES[c]: {"AP50": float(ap50[i]), "AP50-95": float(ap[i])} for i, c in enumerate(classes)}
    vru = [classes.index(CLASS_NAMES.index(n)) for n in VRU_NAMES if CLASS_NAMES.index(n) in classes]
    out = {"weights": str(weights), "sha256": sha256(weights),
           "all_mAP50": float(m.box.map50), "all_mAP50-95": float(m.box.map),
           "vru_mAP50": float(ap50[vru].mean()), "vru_mAP50-95": float(ap[vru].mean()), "per_class": per_class}
    (WORK / "official_val_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


def main():
    if "LOCAL_RANK" in os.environ or os.environ.get("RS_SINGLE_WORKER"):
        train_worker()
        return

    split_info = build_data_yaml()
    config = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (str, int, float))}
    config.update(split_info=split_info, class_weights=CLASS_WEIGHTS.tolist(), ultralytics=ultralytics.__version__,
                  torch=torch.__version__, gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    (WORK / "run_config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)

    if MODE == "smoke":
        with GpuMonitor() as gm:
            phases = [launch("A", SMOKE_PHASE_TIMEOUT_S)]
            if phases[0]["exit"] == 0 and RESUME_COPY.exists():
                shutil.copy2(RESUME_COPY, WORK / "resume_after_phase_A.pt")  # frozen record of the resume point
                shutil.copy2(RESUME_COPY, LAST)  # last.pt was stripped by final_eval; restore the resumable copy
                phases.append(launch("B", SMOKE_PHASE_TIMEOUT_S))
        t_events = [json.loads(l) for l in EVENTS.read_text().splitlines()] if EVENTS.exists() else []
        windows = [(s["t"], e["t"]) for s in t_events if s["event"] == "epoch_start"
                   for e in t_events if e["event"] == "train_end" and e["epoch"] == s["epoch"] and e["phase"] == s["phase"]]
        smoke_report(split_info, phases, gm.summary(windows))
        return

    if MODE == "probe":
        with GpuMonitor(interval=2) as gm, CpuMonitor() as cm:
            phases = [launch("P", PROBE_PHASE_TIMEOUT_S)]
        events = [json.loads(l) for l in EVENTS.read_text().splitlines()] if EVENTS.exists() else []
        windows = [(e["t_start"], e["t_train_end"]) for e in epoch_table(events)]
        probe_report(split_info, phases, gm.summary(windows), cm.summary(windows))
        return

    manifest = find_one("handoff_manifest.json")
    if manifest is not None:
        m = restore_handoff(manifest, LAST, LAST.with_name("best.pt"), BEST_RECORD)
        log_event(event="session2_inputs_verified", **m)
        print(f"Session 2: restored handoff {m}", flush=True)
    phase = launch("full")
    if phase["exit"] != 0 or not RESUME_COPY.exists():
        return
    done = int(torch.load(RESUME_COPY, map_location="cpu", weights_only=False)["epoch"]) + 1
    if done < EPOCHS:
        print(json.dumps(write_handoff(RESUME_COPY, LAST.with_name("best.pt"), HANDOFF, BEST_RECORD), indent=2), flush=True)
    else:
        best = LAST.with_name("best.pt")
        bk = torch.load(best, map_location="cpu", weights_only=False)
        rk = torch.load(RESUME_COPY, map_location="cpu", weights_only=False)
        record = json.loads(BEST_RECORD.read_text())
        assert best_record_matches(record, bk), "best_checkpoint.json does not describe the final best.pt"
        final = {"completed_epochs": done, "best_fitness": float(rk["best_fitness"]), "best_epoch": int(record["epoch"]),
                 "best_pt_fitness": float(bk["train_metrics"]["fitness"]), "best_sha256": sha256(best),
                 "best_weights_fingerprint": weights_fingerprint(bk)}
        (WORK / "final_selection.json").write_text(json.dumps(final, indent=2))
        print(json.dumps(final, indent=2), flush=True)
        report_official_val(best)


if __name__ == "__main__":
    main()
