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
MODE="full": one training session of EPOCHS epochs, then official-val metrics for the record.

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
MODE = "smoke"                   # "smoke" | "full"
INIT = "coco"                    # "coco" | "base"
COCO_WEIGHTS = "yolo11m.pt"
IMGSZ = 960
EPOCHS = 2 if MODE == "smoke" else 40   # full-mode value must be set from the smoke-test estimate
BATCH = 16                       # total across GPUs
HOLDOUT_SIZE = 5000
HOLDOUT_SHA256 = "c7047ee0073803431321735c098aad212693daa696153f31f8a26e11ebf7686b"  # experiments/phase6_audit/split_audit.json
SMOKE_FRACTION, SMOKE_SEED = 0.10, 7
SMOKE_PHASE_TIMEOUT_S = 30 * 60  # hard stop per smoke phase (launcher kills torchrun)
TIME_BUDGET_H = 0.75 if MODE == "smoke" else 11.0
RUN_NAME = f"phase6_yolo11m_960_{MODE}"
CLASS_NAMES = ["person", "rider", "car", "bus", "truck", "bike", "motor", "traffic light", "traffic sign", "train"]
VRU_IDS = [0, 1, 5, 6]
CLASS_WEIGHTS = torch.tensor([1.2, 2.0, 1.0, 1.0, 1.0, 1.8, 2.0, 1.0, 1.0, 1.0])
BASE_SHA256 = "a4747292648f37f68ab997bea65a36391c41356f35e309a9f3d2eb1d3ea8d8e3"

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
PROJECT = WORK / "runs"
LAST = PROJECT / RUN_NAME / "weights" / "last.pt"
RESUME_COPY = WORK / "resume_last.pt"
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
    train = [n for n in names if n not in hset]
    if MODE == "smoke":
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
            "train_list_sha256": hashlib.sha256("\n".join(train).encode()).hexdigest()}


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


def add_callbacks(model: YOLO):
    lr_seen = {}

    def attach_weights(trainer):
        m = getattr(trainer.model, "module", trainer.model)
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
        log_event(event="fit_end", epoch=trainer.epoch + 1, fitness=float(trainer.fitness or 0),
                  metrics={k: float(v) for k, v in (trainer.metrics or {}).items()})
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
    model.add_callback("on_train_epoch_start", check_weights)
    model.add_callback("on_train_batch_end", track_lr)
    model.add_callback("on_train_epoch_end", epoch_end)
    model.add_callback("on_model_save", keep_resumable)
    model.add_callback("on_fit_epoch_end", fit_end)


def train_worker():
    """Runs inside every torchrun rank (or alone on a single GPU)."""
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
        mosaic=1.0, close_mosaic=5, mixup=0.0, copy_paste=0.0, scale=0.5, fliplr=0.5,
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
    """Samples nvidia-smi every 5 s into gpu_util.csv."""

    def __init__(self):
        self.path = WORK / "gpu_util.csv"
        self.proc = None

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits", "-l", "5"], stdout=open(self.path, "w"), stderr=subprocess.DEVNULL)
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
    m = YOLO(str(weights)).val(data=str(WORK / "val_official.yaml"), imgsz=IMGSZ, batch=16, device=0, plots=False)
    ap50, ap = m.box.all_ap[:, 0], m.box.all_ap.mean(1)
    classes = list(m.box.ap_class_index)
    per_class = {CLASS_NAMES[c]: {"AP50": float(ap50[i]), "AP50-95": float(ap[i])} for i, c in enumerate(classes)}
    vru = [classes.index(c) for c in VRU_IDS if c in classes]
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

    prev = find_one("resume_last.pt")
    if prev is not None:
        LAST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prev, LAST)
        print(f"Resuming from {prev} (sha256 {sha256(prev)})", flush=True)
    phase = launch("full")
    if phase["exit"] == 0:
        report_official_val(LAST.with_name("best.pt"))


if __name__ == "__main__":
    main()
