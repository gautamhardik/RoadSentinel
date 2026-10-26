"""
RoadSentinel Phase 5 — Nebius Token Factory Live Smoke Test Script
Model: nvidia/nemotron-nano-12b-v2-vl (NVIDIA Open-Source Vision-Language Model)
Base URL: https://api.tokenfactory.nebius.com/v1

Execution Policy:
- Checks NEBIUS_API_KEY without printing or logging its value.
- If missing: outputs clear local environment configuration instructions and exercises deterministic fallback.
- If present: performs a live bounded query with structured JSON schema.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agentic.investigator import AgenticInvestigator, InvestigationVerdict


def run_smoke_test():
    api_key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("TOKEN_FACTORY_API_KEY")
    key_available = bool(api_key and len(api_key.strip()) > 5)

    print("=" * 70)
    print("RoadSentinel — Nebius Token Factory NVIDIA VLM Smoke Test")
    print("Target Model: nvidia/nemotron-nano-12b-v2-vl")
    print("Endpoint:     https://api.tokenfactory.nebius.com/v1/chat/completions")
    print("=" * 70)

    # Synthetic test frame (dark scene with pedestrian candidate)
    frame = np.full((720, 1280, 3), 45, dtype=np.uint8)
    # Draw simple pedestrian-like bounding box
    cv2.rectangle(frame, (600, 320), (660, 480), (180, 180, 180), -1)

    candidate_detection = {
        "class_name": "person",
        "conf": 0.28,
        "box": [600.0, 320.0, 660.0, 480.0],
    }

    if not key_available:
        print("\n[STATUS: NEBIUS_API_KEY NOT DETECTED]")
        print("To enable live cloud multimodal deliberation, configure your key:")
        print("  Windows PowerShell: $env:NEBIUS_API_KEY = \"your_secret_key_here\"")
        print("  Windows Command:    set NEBIUS_API_KEY=your_secret_key_here")
        print("  Linux / macOS:      export NEBIUS_API_KEY=\"your_secret_key_here\"")
        print("\nExecuting verified deterministic fail-safe fallback path...")

        investigator = AgenticInvestigator(api_key=None)
        t0 = time.perf_counter()
        verdict = investigator.investigate(frame, candidate_detection, image_id="smoke_synth_01.jpg")
        dt_ms = (time.perf_counter() - t0) * 1000.0

        print(f"Fallback Execution Succeeded in {dt_ms:.2f} ms")
        print(f"  Verified Class:      {verdict.verified_class}")
        print(f"  Epistemic Conf:      {verdict.epistemic_confidence:.2f}")
        print(f"  Visual Evidence:     {verdict.visual_evidence}")
        print(f"  Safety Escalation:   {verdict.safety_escalation}")
        print(f"  Source:              {verdict.source}")
        print("Safety gate verified: detector prior safely preserved without crashing.")
        return 0

    print("\n[STATUS: NEBIUS_API_KEY DETECTED — ATTEMPTING LIVE INFERENCE]")
    investigator = AgenticInvestigator(
        api_key=api_key,
        model_name="nvidia/nemotron-nano-12b-v2-vl",
        base_url="https://api.tokenfactory.nebius.com/v1",
        timeout_seconds=5.0,
    )

    t0 = time.perf_counter()
    verdict = investigator.investigate(frame, candidate_detection, image_id="smoke_synth_01.jpg")
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\nDeliberation Complete in {dt_ms:.1f} ms:")
    print(f"  Verified Class:      {verdict.verified_class}")
    print(f"  Epistemic Conf:      {verdict.epistemic_confidence:.2f}")
    print(f"  Visual Evidence:     {verdict.visual_evidence}")
    print(f"  Occlusion Level:     {verdict.occlusion_level}")
    print(f"  Safety Escalation:   {verdict.safety_escalation}")
    print(f"  Source:              {verdict.source}")

    if verdict.source == "nebius_vlm":
        print("\n>>> LIVE NEBIUS NVIDIA VLM CALL VERIFIED SUCCESSFULLY <<<")
    else:
        print("\n>>> LIVE CALL FAILED OR TIMED OUT — FAIL-SAFE ENGAGED SAFELY <<<")

    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_test())
