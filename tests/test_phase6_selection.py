"""
Phase 6 — VRU-specific checkpoint selection and two-session handoff.

Verifies:
1. The selection score is the unweighted mean AP50-95 of exactly person, rider, bike, motor, matched by class name.
2. Missing, duplicated or non-finite class metrics are handled safely.
3. Class names in the dataset data.yaml map to the expected VRU indices.
4. Ultralytics 8.4.174 best-checkpoint rules are unchanged, and a simulated session boundary keeps the global best.
5. The session handoff detects tampering and fitness inconsistencies; a lower-scoring session 2 keeps session 1's best.pt.
6. The selection metric does not touch the schedule (epochs, warmup, LR decay, mosaic, class weights).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("p6", ROOT / "kaggle_jobs" / "model_training" / "train_phase6_vru.py")
p6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p6)

DATA_ORDER = dict(enumerate(p6.CLASS_NAMES))  # person, rider, car, bus, truck, bike, motor, light, sign, train
CKPT_ORDER = {0: "person", 1: "rider", 2: "car", 3: "truck", 4: "bus", 5: "train", 6: "motor", 7: "bike",
              8: "traffic light", 9: "traffic sign"}  # the third-party checkpoint's (different) order


def ap_table(names, ap_by_name):
    """Per-class (n, 10) AP rows with a constant value per class, rows in index order."""
    idx = sorted(names)
    return idx, np.array([[ap_by_name[names[i]]] * 10 for i in idx])


AP = {"person": 0.40, "rider": 0.20, "car": 0.90, "bus": 0.80, "truck": 0.70, "bike": 0.30, "motor": 0.10,
      "traffic light": 0.60, "traffic sign": 0.50, "train": 0.95}
EXPECTED = (0.40 + 0.20 + 0.30 + 0.10) / 4


# ---------------------------------------------------------------- 1. exactly four classes, by name
def test_score_is_mean_of_exactly_four_vru_classes():
    idx, ap = ap_table(DATA_ORDER, AP)
    assert p6.vru_macro_ap5095(DATA_ORDER, idx, ap) == pytest.approx(EXPECTED)


def test_non_vru_classes_do_not_affect_score():
    idx, ap = ap_table(DATA_ORDER, AP)
    base = p6.vru_macro_ap5095(DATA_ORDER, idx, ap)
    for i, name in DATA_ORDER.items():
        bumped = ap.copy()
        bumped[i] += 0.05
        changed = p6.vru_macro_ap5095(DATA_ORDER, idx, bumped) != pytest.approx(base)
        assert changed == (name in p6.VRU_NAMES), name


def test_each_vru_class_has_equal_weight():
    idx, ap = ap_table(DATA_ORDER, AP)
    base = p6.vru_macro_ap5095(DATA_ORDER, idx, ap)
    for name in p6.VRU_NAMES:
        bumped = ap.copy()
        bumped[p6.CLASS_NAMES.index(name)] += 0.08
        assert p6.vru_macro_ap5095(DATA_ORDER, idx, bumped) == pytest.approx(base + 0.02), name


def test_classes_matched_by_name_not_index():
    idx, ap = ap_table(CKPT_ORDER, {**AP})
    assert p6.vru_macro_ap5095(CKPT_ORDER, idx, ap) == pytest.approx(EXPECTED)


def test_ap_rows_in_arbitrary_order():
    idx, ap = ap_table(DATA_ORDER, AP)
    perm = np.random.default_rng(0).permutation(len(idx))
    assert p6.vru_macro_ap5095(DATA_ORDER, [idx[i] for i in perm], ap[perm]) == pytest.approx(EXPECTED)


def test_uses_mean_over_ten_iou_thresholds():
    idx, ap = ap_table(DATA_ORDER, AP)
    ap = ap.astype(float)
    ap[p6.CLASS_NAMES.index("rider")] = np.linspace(0.0, 0.9, 10)  # mean 0.45
    expected = (0.40 + 0.45 + 0.30 + 0.10) / 4
    assert p6.vru_macro_ap5095(DATA_ORDER, idx, ap) == pytest.approx(expected)


# ---------------------------------------------------------------- 2. missing / invalid metrics
def test_vru_class_without_ap_row_raises():
    idx, ap = ap_table(DATA_ORDER, AP)
    drop = p6.CLASS_NAMES.index("motor")
    keep = [k for k in range(len(idx)) if idx[k] != drop]
    with pytest.raises(ValueError, match="motor"):
        p6.vru_macro_ap5095(DATA_ORDER, [idx[k] for k in keep], ap[keep])


def test_missing_non_vru_row_is_fine():
    idx, ap = ap_table(DATA_ORDER, AP)
    keep = [k for k in range(len(idx)) if DATA_ORDER[idx[k]] != "train"]
    assert p6.vru_macro_ap5095(DATA_ORDER, [idx[k] for k in keep], ap[keep]) == pytest.approx(EXPECTED)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_ap_counts_as_zero(bad):
    idx, ap = ap_table(DATA_ORDER, AP)
    ap = ap.astype(float)
    ap[p6.CLASS_NAMES.index("bike")] = bad
    assert p6.vru_macro_ap5095(DATA_ORDER, idx, ap) == pytest.approx((0.40 + 0.20 + 0.0 + 0.10) / 4)


def test_no_ap_computed_is_zero():
    assert p6.vru_macro_ap5095(DATA_ORDER, [], []) == 0.0


@pytest.mark.parametrize("names", [
    {i: n for i, n in DATA_ORDER.items() if n != "rider"},          # VRU name missing
    {**DATA_ORDER, 3: "person"},                                    # VRU name duplicated
])
def test_bad_class_names_raise(names):
    idx, ap = ap_table(DATA_ORDER, AP)
    with pytest.raises(ValueError):
        p6.vru_macro_ap5095(names, idx, ap)


# ---------------------------------------------------------------- 3. class mapping
def test_mapping_dataset_order():
    assert p6.verify_class_mapping(DATA_ORDER, DATA_ORDER) == {"person": 0, "rider": 1, "bike": 5, "motor": 6}


def test_mapping_rejects_data_model_disagreement():
    with pytest.raises(ValueError):
        p6.verify_class_mapping(DATA_ORDER, CKPT_ORDER)


def test_mapping_accepts_string_keys_and_lists():
    assert p6.verify_class_mapping({str(k): v for k, v in DATA_ORDER.items()}, DATA_ORDER)["bike"] == 5


@pytest.mark.skipif(not (ROOT / "bdd100k_dataset" / "data.yaml").exists(), reason="local dataset not present")
def test_dataset_yaml_matches_class_names():
    names = yaml.safe_load((ROOT / "bdd100k_dataset" / "data.yaml").read_text())["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    assert names == p6.CLASS_NAMES
    assert p6.verify_class_mapping(dict(enumerate(names)), DATA_ORDER) == {"person": 0, "rider": 1, "bike": 5, "motor": 6}


def test_installed_fitness_reads_detmetrics():
    from ultralytics.utils.metrics import DetMetrics
    original = DetMetrics.fitness
    try:
        p6.install_vru_fitness()
        m = DetMetrics(names=DATA_ORDER)
        idx, ap = ap_table(DATA_ORDER, AP)
        m.box.ap_class_index, m.box.all_ap = np.array(idx), ap
        assert m.fitness == pytest.approx(EXPECTED)
        assert m.results_dict["fitness"] == pytest.approx(EXPECTED)  # what the trainer pops as fitness
    finally:
        DetMetrics.fitness = original


# ---------------------------------------------------------------- 4. ultralytics best-checkpoint rules + session boundary
def test_ultralytics_selection_rules_unchanged():
    from ultralytics.engine.trainer import BaseTrainer
    assert "self.best_fitness is None or self.best_fitness < fitness" in inspect.getsource(BaseTrainer.validate)
    assert "if self.best_fitness == self.fitness:" in inspect.getsource(BaseTrainer.save_model)
    assert 'self.best_fitness = ckpt.get("best_fitness")' in inspect.getsource(BaseTrainer._load_checkpoint_state)


def simulate(session1, session2):
    """Mirror of the ultralytics rules above; returns (best epoch held in best.pt, best_fitness)."""
    best_fitness, best_file_epoch = None, None
    for epoch, f in enumerate(session1, 1):
        if best_fitness is None or best_fitness < f:
            best_fitness = f
        if best_fitness == f:
            best_file_epoch = epoch
    carried = {"best_fitness": best_fitness, "best_file_epoch": best_file_epoch}  # handoff
    best_fitness, best_file_epoch = carried["best_fitness"], carried["best_file_epoch"]
    for epoch, f in enumerate(session2, len(session1) + 1):
        if best_fitness is None or best_fitness < f:
            best_fitness = f
        if best_fitness == f:
            best_file_epoch = epoch
    return best_file_epoch, best_fitness


@pytest.mark.parametrize("s1,s2,expected_epoch", [
    ([0.1, 0.2, 0.3, 0.5], [0.4, 0.3, 0.45, 0.2], 4),   # lower session 2 never overwrites session 1's best
    ([0.1, 0.2, 0.5, 0.3], [0.5, 0.4, 0.6, 0.6], 8),    # tie -> later epoch; higher -> new best; tie again -> later
    ([0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], 8),    # monotone improvement
    ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], 8),    # all-zero early training: ties -> latest epoch
])
def test_selection_across_session_boundary(s1, s2, expected_epoch):
    assert simulate(s1, s2)[0] == expected_epoch


# ---------------------------------------------------------------- 5. handoff integrity
def tiny_model(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4)).half()


def fake_ckpt(path, epoch_index, best_fitness, fitness, stripped=False, seed=None):
    """Mimics ultralytics: unstripped keeps weights under "ema"; strip_optimizer moves them to "model"."""
    m = tiny_model(epoch_index if seed is None else seed)
    torch.save({"epoch": -1 if stripped else epoch_index, "best_fitness": None if stripped else best_fitness,
                "stopper": None if stripped else {"best_epoch": epoch_index + 1},
                "ema": None if stripped else m, "model": m if stripped else None,
                "train_metrics": {"fitness": fitness}, "date": f"epoch{epoch_index}"}, path)


def best_record(tmp_path, best_pt, epoch):
    ck = torch.load(best_pt, weights_only=False)
    rec = tmp_path / "best_checkpoint.json"
    rec.write_text(json.dumps({"epoch": epoch, "fitness": ck["train_metrics"]["fitness"],
                               "weights_fingerprint": p6.weights_fingerprint(ck)}))
    return rec


def test_fingerprint_survives_strip_and_detects_weight_change():
    m = tiny_model(3)
    assert p6.weights_fingerprint({"ema": m, "model": None}) == p6.weights_fingerprint({"ema": None, "model": m})
    other = tiny_model(4)
    assert p6.weights_fingerprint({"ema": m}) != p6.weights_fingerprint({"ema": other})


def test_handoff_roundtrip_and_lower_session2_keeps_best(tmp_path):
    p6.WORK = tmp_path
    fake_ckpt(tmp_path / "resume.pt", 12, best_fitness=0.31, fitness=0.29)              # epoch 13 checkpoint
    fake_ckpt(tmp_path / "best.pt", 10, best_fitness=0.31, fitness=0.31, stripped=True)  # epoch 11 was best
    m = p6.write_handoff(tmp_path / "resume.pt", tmp_path / "best.pt", tmp_path / "handoff", best_record(tmp_path, tmp_path / "best.pt", 11))
    assert m["completed_epochs"] == 13 and m["best_fitness"] == 0.31 and m["best_epoch"] == 11
    last, best = tmp_path / "s2" / "last.pt", tmp_path / "s2" / "best.pt"
    p6.restore_handoff(tmp_path / "handoff" / "handoff_manifest.json", last, best, tmp_path / "s2" / "best_checkpoint.json")
    assert p6.sha256(best) == m["best_sha256"]
    restored = torch.load(best, weights_only=False)
    assert restored["date"] == "epoch10" and restored["train_metrics"]["fitness"] == 0.31
    assert p6.weights_fingerprint(restored) == m["best_weights_fingerprint"]
    resumed_best = torch.load(last, weights_only=False)["best_fitness"]
    # session 2 with lower scores against the restored best_fitness: best.pt must stay untouched
    best_fitness, overwrote = resumed_best, False
    for f in (0.30, 0.28, 0.305):
        if best_fitness < f:
            best_fitness = f
        overwrote |= best_fitness == f
    assert not overwrote and p6.sha256(best) == m["best_sha256"]


def test_handoff_rejects_tampered_best(tmp_path):
    fake_ckpt(tmp_path / "resume.pt", 12, best_fitness=0.31, fitness=0.29)
    fake_ckpt(tmp_path / "best.pt", 10, best_fitness=0.31, fitness=0.31, stripped=True)
    p6.write_handoff(tmp_path / "resume.pt", tmp_path / "best.pt", tmp_path / "handoff", best_record(tmp_path, tmp_path / "best.pt", 11))
    fake_ckpt(tmp_path / "handoff" / "best.pt", 9, best_fitness=0.31, fitness=0.31, stripped=True)
    with pytest.raises(ValueError, match="sha256"):
        p6.restore_handoff(tmp_path / "handoff" / "handoff_manifest.json", tmp_path / "l.pt", tmp_path / "b.pt", tmp_path / "r.json")


def test_handoff_rejects_best_that_is_not_the_best(tmp_path):
    fake_ckpt(tmp_path / "resume.pt", 12, best_fitness=0.31, fitness=0.29)
    fake_ckpt(tmp_path / "best.pt", 10, best_fitness=0.31, fitness=0.25, stripped=True)  # wrong file
    with pytest.raises(AssertionError):
        p6.write_handoff(tmp_path / "resume.pt", tmp_path / "best.pt", tmp_path / "handoff", best_record(tmp_path, tmp_path / "best.pt", 11))


def test_handoff_rejects_swapped_weights_with_forged_hash(tmp_path):
    fake_ckpt(tmp_path / "resume.pt", 12, best_fitness=0.31, fitness=0.29)
    fake_ckpt(tmp_path / "best.pt", 10, best_fitness=0.31, fitness=0.31, stripped=True)
    p6.write_handoff(tmp_path / "resume.pt", tmp_path / "best.pt", tmp_path / "handoff", best_record(tmp_path, tmp_path / "best.pt", 11))
    fake_ckpt(tmp_path / "handoff" / "best.pt", 10, best_fitness=0.31, fitness=0.31, stripped=True, seed=99)
    mpath = tmp_path / "handoff" / "handoff_manifest.json"
    m = json.loads(mpath.read_text())
    m["best_sha256"] = p6.sha256(tmp_path / "handoff" / "best.pt")  # hash forged to match the swapped file
    mpath.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="fingerprint"):
        p6.restore_handoff(mpath, tmp_path / "l.pt", tmp_path / "b.pt", tmp_path / "r.json")


def test_handoff_rejects_inconsistent_manifest(tmp_path):
    fake_ckpt(tmp_path / "resume.pt", 12, best_fitness=0.31, fitness=0.29)
    fake_ckpt(tmp_path / "best.pt", 10, best_fitness=0.31, fitness=0.31, stripped=True)
    p6.write_handoff(tmp_path / "resume.pt", tmp_path / "best.pt", tmp_path / "handoff", best_record(tmp_path, tmp_path / "best.pt", 11))
    mpath = tmp_path / "handoff" / "handoff_manifest.json"
    m = json.loads(mpath.read_text())
    m["best_fitness"] = 0.99
    mpath.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="best fitness"):
        p6.restore_handoff(mpath, tmp_path / "l.pt", tmp_path / "b.pt", tmp_path / "r.json")


# ---------------------------------------------------------------- 6. schedule independence
def test_schedule_code_does_not_use_fitness():
    from ultralytics.engine.trainer import BaseTrainer
    for fn in (BaseTrainer._setup_scheduler, BaseTrainer._get_warmup_iterations, BaseTrainer._close_dataloader_mosaic):
        assert "fitness" not in inspect.getsource(fn), fn.__name__


def test_early_stopping_cannot_trigger_and_schedule_constants():
    from ultralytics.cfg import DEFAULT_CFG_DICT
    src = inspect.getsource(p6.train_worker)
    assert "patience" not in src and DEFAULT_CFG_DICT["patience"] > 26
    assert "close_mosaic=CLOSE_MOSAIC" in src and "warmup_epochs=3.0" in src and 'lrf=0.01' in src
    assert p6.SESSION_STOP_EPOCH == 13 and 'else 26' in inspect.getsource(p6)  # full-mode total epochs
    assert p6.CLASS_WEIGHTS.tolist() == pytest.approx([1.2, 2.0, 1.0, 1.0, 1.0, 1.8, 2.0, 1.0, 1.0, 1.0])


def test_fitness_override_only_inside_training_processes():
    import ast
    tree = ast.parse(inspect.getsource(p6))
    callers = {fn.name for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
               for node in ast.walk(fn) if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "install_vru_fitness"}
    assert callers == {"train_worker"}  # never in main()/report_official_val(): official val keeps stock behaviour
    assert "secondary" not in inspect.getsource(p6.report_official_val)  # official val is a one-off record, no selection


def test_handoff_rejects_record_describing_another_checkpoint(tmp_path):
    fake_ckpt(tmp_path / "resume.pt", 12, best_fitness=0.31, fitness=0.29)
    fake_ckpt(tmp_path / "best.pt", 10, best_fitness=0.31, fitness=0.31, stripped=True)
    fake_ckpt(tmp_path / "other.pt", 11, best_fitness=0.31, fitness=0.31, stripped=True)  # tied epoch, different weights
    with pytest.raises(AssertionError, match="best_checkpoint.json"):
        p6.write_handoff(tmp_path / "resume.pt", tmp_path / "best.pt", tmp_path / "handoff",
                         best_record(tmp_path, tmp_path / "other.pt", 12))
