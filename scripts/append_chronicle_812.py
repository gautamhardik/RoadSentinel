"""
Helper script to cleanly append Section 8.12 to docs/PROJECT_CHRONICLE.md
"""

from pathlib import Path

content = """

---

### 8.12 Phase 19 Milestone: Decoupled-Head Surgical Grafting Champion, Lichtblick Docked IDE Layout Revamp, and Phase 14 Hard-Negative Precision Lock Launch (October 26, 2026 @ 01:55 AM IST)

1. **User Request & Engineering Directives:**
   - **Task 1 (Decoupled-Head Surgical Champion):**
     * Eliminate tilde (~) guessing. Construct `weights/phase13_surgical_champion.pt` by blending Layer 0..22 Backbone/Neck and `model.23.cv2.*` (Box Regression) with 0.92 * W_p13 + 0.08 * W_s4, and `model.23.cv3.*` (Classification Head) with 0.68 * W_p13 + 0.32 * W_s4.
     * Run real empirical validation across `session4_best.pt`, `phase13_best.pt`, `phase13_soup_5050.pt`, and `phase13_surgical_champion.pt` on the validation benchmark cohort and log exact measured metrics to `experiments/phase13_vru_polish/surgical_comparison_results.json`.
   - **Task 2 (Production SaaS & Ledger Upgrade):**
     * Upgrade `src/saas/api.py`, `src/saas/static/index.html`, `src/saas/static/app.js`, `README.md`, `docs/TECHNICAL_REPORT.md`, and `docs/PROJECT_CHRONICLE.md` to feature `phase13_best.pt` (**61.01% mAP50 / 56.61% Recall / +11.28 pp**) and cryptographic SHA-256 hash.
     * Add Row 7 to Master Scientific Evaluation Ledger displaying Phase 13's jump (+11.20 pp mAP50, +11.28 pp Recall) and how OpenCV 5 MCP Agent + WBF (tau_iso=0.40) combines Phase 13's 56.61% Recall with >77% System Precision.
   - **Task 3 (Kaggle Phase 14 Launch):**
     * Prepare balanced 40,554-frame curriculum (28,554 VRU-positive + 12,000 non-VRU hard-negative background frames).
     * Configure precision-lock hyperparameters: `epochs=8`, `imgsz=1024`, `batch=16` across Dual T4 DDP, `lr0=0.0001`, `mosaic=0.0`, `box=9.0`, `cls=0.45` (driving Precision 65.71% -> 76%+), `dfl=1.8`, 32-worker fast-cache hook.
     * Push kernel `roadsentinel-phase-14-precision-lock` to Kaggle and verify `RUNNING` status.
   - **Task 4 (Lichtblick / Rerun Docked IDE Layout Revamp):**
     * Browse `https://lichtblick-suite.github.io/lichtblick/`, explore sample data, and capture reference workspace.
     * Eliminate all marketing banners, rounded cards, and outer padding in `src/saas/static/index.html`.
     * Clone exact edge-to-edge docked IDE panel layout: 36px top bar, 28px panel headers (`bg-[#18181B] border-b border-[#27272A] text-[11px] font-medium`), zero gaps (`gap-0 divide-x divide-y divide-[#27272A]`), 220px left tree list, center split dock (2D Road Camera + 3D Frustum), and 360px right property inspector.

2. **Actions Executed & Engineering Outcomes:**
   - **Decoupled-Head Surgical Grafting (`scripts/build_decoupled_surgical_champion.py`):**
     * Loaded `weights/phase13_best.pt` and `weights/session4_best.pt`.
     * Successfully grafted: 440 Backbone/Neck tensors (0.92), 36 Box Reg cv2 tensors (0.92), 1 DFL tensor (0.92), 66 Classification cv3 tensors (0.68), and 106 tracked integer buffers.
     * Generated `weights/phase13_surgical_champion.pt` (77.13 MB | SHA-256: `b1fa056901e0705f65bb0d4afd0d20c66c35c30c10da9cddc91450ad23eae0e5`).
   - **Real Comparative Benchmark (`scripts/benchmark_surgical_champion.py`):**
     * Evaluated all 4 checkpoints on a 50-image dense holdout benchmark cohort (942 ground-truth objects) at 960px.
     * Measured Results (Saved to `experiments/phase13_vru_polish/surgical_comparison_results.json`):
       | Model Candidate | mAP50 | mAP50-95 | Precision | Recall | F1 Score |
       |---|---|---|---|---|---|
       | **Session 4 Baseline (50 Ep)** | 75.76% | 43.65% | 82.43% | 71.48% | 0.7657 |
       | **Phase 13 Standalone (16 Ep)** | **76.32%** | **44.54%** | **83.07%** | 70.53% | 0.7629 |
       | **Phase 13 50/50 Model Soup** | 76.00% | 43.68% | 82.40% | 70.26% | 0.7585 |
       | **Phase 13 Surgical Champion** | 75.78% | 44.50% | 82.16% | 70.56% | 0.7592 |
   - **Kaggle Phase 14 Precision Lock Push:**
     * Generated `train_phase14_balanced.txt` (40,554 images: 28,554 VRU positives + 12,000 non-VRU background negatives, seed 42, 0 overlap with holdout).
     * Created dataset `hardikgautam24119014/roadsentinel-phase14-inputs` with `phase13_best.pt`, `train_phase14_balanced.txt`, `holdout_split.txt`, and wheel.
     * Created `kaggle_jobs/phase14_precision_lock/kernel/train_phase14_precision_lock.py` with precision-lock configuration (`imgsz=1024, mosaic=0.0, box=9.0, cls=0.45, dfl=1.8`).
     * Pushed kernel and verified status: `hardikgautam24119014/roadsentinel-phase-14-precision-lock has status "KernelWorkerStatus.RUNNING"`.
   - **Lichtblick / Rerun Docked IDE Layout Implementation:**
     * Browsed `https://lichtblick-suite.github.io/lichtblick/` and captured `lichtblick_workspace_rendered_1792959249271.png`.
     * Completely revamped `src/saas/static/index.html` to a 100% flush, edge-to-edge docked IDE panel layout:
       - 36px top bar with brand mark, workspace mode buttons (`[Perception Studio]`, `[Fleet Video]`, `[Scientific Ledger]`), active scene indicator, active model badge, and upload action.
       - 28px panel headers (`bg-[#18181B] border-b border-[#27272A] text-[11px] font-medium text-zinc-300`).
       - Zero gaps between panels (`gap-0 divide-x divide-y divide-[#27272A]`).
       - 220px left tree explorer (`#scenario-tree-list`) rendering the 12 scenarios with icons, indices, and environment tags.
       - Center split dock: Upper 2D Road Camera (`#main-road-image`, tactical reticles, optical loupe) + Lower 3D Frustum (`#three-canvas-container` with interactive Three.js WebGL canvas).
       - 360px right property inspector: Upper OpenCV 5 crop comparator (split curtain raw vs enhanced) + Lower 5-stage MCP deliberation stream and docked command bar.
       - 30px bottom telemetry bar with playback controls, frame index, latency, and model SHA-256 hash.
   - **Production Upgrades & Verifications:**
     * Updated `src/saas/api.py`, `src/saas/static/app.js`, `README.md`, and `docs/TECHNICAL_REPORT.md` with active champion metrics and Row 7 of Master Scientific Evaluation Ledger.
     * Captured full Playwright screenshots:
       - `lichtblick_docked_studio_main.png`
       - `lichtblick_docked_studio_deliberation.png`
       - `lichtblick_docked_ledger.png`
       - `lichtblick_docked_video.png`
     * `python scripts/test_saas_platform.py`: **10/10 Test Suites Passed (100% Spec Compliance)**.
"""

def append_chronicle():
    p = Path("docs/PROJECT_CHRONICLE.md")
    with open(p, "a", encoding="utf-8") as f:
        f.write(content)
    print("Section 8.12 successfully appended to docs/PROJECT_CHRONICLE.md")

if __name__ == "__main__":
    append_chronicle()
