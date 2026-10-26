"""
RoadSentinel - Architecture Diagram Generator

Generates publication-quality architecture figures:
1. docs/figures/fig1_hybrid_aws_cool_architecture.png
2. docs/figures/fig2_agentic_mcp_decision_loop.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUTPUT_DIR = Path("docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]


def generate_fig1():
    """Generates Figure 1: Hybrid AWS Graviton Arm64 (COOL) + Accelerator Pipeline."""
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0b1120")

    # Titles
    ax.text(0.5, 0.94, "RoadSentinel: Hybrid AWS Graviton (Arm64 + COOL) Perception Pipeline",
            color="#f8fafc", fontsize=15, fontweight="bold", ha="center")
    ax.text(0.5, 0.89, "Dual-Stream Architecture: GPU Synchronous Fast-Path + AWS Graviton MCP Slow-Path",
            color="#94a3b8", fontsize=10, ha="center")

    def draw_box(x, y, w, h, title, sub, bg="#1e293b", border="#38bdf8", tc="#f8fafc"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=bg, edgecolor=border, linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.62, title, color=tc, fontsize=9.5, fontweight="bold", ha="center", va="center")
        ax.text(x + w / 2, y + h * 0.32, sub, color="#94a3b8", fontsize=7.5, ha="center", va="center")

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=1.5, mutation_scale=14))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.025
            ax.text(mx, my, label, color="#38bdf8", fontsize=7.5, fontweight="bold", ha="center")

    # Ingestion
    draw_box(0.04, 0.45, 0.12, 0.20, "Input Stream", "30 FPS Driving Video\n1280x720 RGB", bg="#1e293b", border="#64748b")

    # Fast Path (Top)
    draw_box(0.23, 0.60, 0.18, 0.22, "Primary Detector", "YOLO11m (50 Epochs)\nFull-Frame 960x960", bg="#1e1b4b", border="#818cf8")
    draw_box(0.48, 0.60, 0.18, 0.22, "Fast-Path Gate", "Accept if conf >= 0.45\nZero extra passes", bg="#064e3b", border="#34d399")
    draw_box(0.74, 0.60, 0.22, 0.22, "Track Manager", "ByteTrack Track State\nContinuous VRU Lock", bg="#14532d", border="#22c55e")

    # Slow Path (Bottom - AWS Graviton COOL)
    draw_box(0.23, 0.15, 0.22, 0.25, "AWS Graviton (aarch64)", "COOL OpenCV 5 Server\nKleidiCV / NEON Vectorized", bg="#0c4a6e", border="#0284c7")
    draw_box(0.50, 0.15, 0.20, 0.25, "Asynchronous Worker", "Dense Optical Flow\nCLAHE / Bilateral Unsharp", bg="#1e293b", border="#38bdf8")
    draw_box(0.74, 0.15, 0.22, 0.25, "Calibrated WBF", "Consensus Fusion (beta=+0.08)\nor Human Escalation", bg="#4c1d95", border="#a855f7")

    # Connecting arrows
    draw_arrow(0.16, 0.55, 0.23, 0.70, "960px Frame")
    draw_arrow(0.41, 0.71, 0.48, 0.71, "Initial BBoxes")
    draw_arrow(0.66, 0.71, 0.74, 0.71, "Locked Tracks")

    # Ambiguity fork
    draw_arrow(0.32, 0.60, 0.32, 0.40, "Ambiguous ROI")
    draw_arrow(0.32, 0.40, 0.32, 0.40, "")
    draw_arrow(0.45, 0.27, 0.50, 0.27, "Dispatched Crop")
    draw_arrow(0.70, 0.27, 0.74, 0.27, "Secondary Crop")
    draw_arrow(0.85, 0.40, 0.85, 0.60, "Verified Consensus")

    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1.0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_hybrid_aws_cool_architecture.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(" [ok] Generated docs/figures/fig1_hybrid_aws_cool_architecture.png")


def generate_fig2():
    """Generates Figure 2: 5-Stage Closed-Loop Agentic Decision Cycle."""
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0b1120")

    ax.text(0.5, 0.94, "RoadSentinel: 5-Stage Closed-Loop Agentic Perception Cycle",
            color="#f8fafc", fontsize=15, fontweight="bold", ha="center")
    ax.text(0.5, 0.89, "Perception -> Diagnostic -> OpenCV 5 Transform -> Secondary 640px Inference -> Calibrated WBF",
            color="#94a3b8", fontsize=10, ha="center")

    stages = [
        ("Stage 1", "Full-Frame Perception", "YOLO11m @ 960px\nIdentifies candidate\nconf in [0.15, 0.45]", "#1e1b4b", "#818cf8"),
        ("Stage 2", "OpenCV 5 Diagnostic", "tool_diagnose_roi\nLAB mean, std, contrast\nLaplacian acuity variance", "#0c4a6e", "#0284c7"),
        ("Stage 3", "Dynamic MCP Tool", "tool_photometric_clahe\ntool_acuity_sharpen\ntool_foveate_supercrop", "#164e63", "#06b6d4"),
        ("Stage 4", "Counterfactual Infer", "YOLO11m @ 640px\nRaw Crop vs OpenCV\nMeasures delta gain", "#1e293b", "#38bdf8"),
        ("Stage 5", "Calibrated Consensus", "Calibrated WBF Fusion\nLock TP (conf >= 0.40)\nor Escalate to Human", "#064e3b", "#34d399"),
    ]

    w, h = 0.16, 0.48
    ys = 0.28
    for idx, (st_num, title, desc, bg, border) in enumerate(stages):
        xs = 0.05 + idx * 0.19
        rect = patches.FancyBboxPatch(
            (xs, ys), w, h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=bg, edgecolor=border, linewidth=1.8
        )
        ax.add_patch(rect)
        ax.text(xs + w / 2, ys + h * 0.86, st_num, color=border, fontsize=10, fontweight="bold", ha="center")
        ax.text(xs + w / 2, ys + h * 0.72, title, color="#f8fafc", fontsize=9.5, fontweight="bold", ha="center")
        ax.text(xs + w / 2, ys + h * 0.35, desc, color="#94a3b8", fontsize=8, ha="center", va="center")

        if idx < len(stages) - 1:
            ax.annotate("", xy=(xs + w + 0.026, ys + h * 0.5), xytext=(xs + w + 0.004, ys + h * 0.5),
                        arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2, mutation_scale=15))

    # Bottom Safety Gate Banner
    rect_safe = patches.FancyBboxPatch(
        (0.05, 0.12), 0.90, 0.10,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="#1e293b", edgecolor="#f59e0b", linewidth=1.5
    )
    ax.add_patch(rect_safe)
    ax.text(0.5, 0.17, "🛡️ Safety Invariant Gate: Epistemic Uncertainty [0.25, 0.38] Triggers Deterministic Human-in-the-Loop Escalation",
            color="#fbbf24", fontsize=9.5, fontweight="bold", ha="center", va="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1.0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_agentic_mcp_decision_loop.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(" [ok] Generated docs/figures/fig2_agentic_mcp_decision_loop.png")


if __name__ == "__main__":
    generate_fig1()
    generate_fig2()
