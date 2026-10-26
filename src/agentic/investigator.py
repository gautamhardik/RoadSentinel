"""
RoadSentinel Phase 5 — Agentic Vision Investigation Engine
Implements selective, uncertainty-triggered deliberation using OpenCV 5 preprocessing
and eligible NVIDIA vision-language models served via Nebius Token Factory (OpenAI-compatible).

Architecture:
1. Selective Invocation: Triggers only for high-consequence, ambiguous VRU detections (tau in [0.15, 0.45]).
2. OpenCV 5 Preprocessing: Contrast-adaptive CLAHE, gamma correction, context extraction.
3. Strict Structured Contract: Typed inputs/outputs with visual evidence provenance.
4. Safety Verification Gate: Treats VLM output as an untrusted hypothesis. Never overrides safety constraints.
5. Deterministic Fallback: Clean degradation if Nebius Token Factory API key is absent or unreachable.
"""

import os
import time
import json
import base64
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

import cv2
import numpy as np


class VerifiedClass(str, Enum):
    PERSON = "person"
    RIDER = "rider"
    BIKE = "bike"
    MOTOR = "motor"
    BACKGROUND_FP = "background_false_positive"
    UNRESOLVED = "unresolved"


class OcclusionLevel(str, Enum):
    NONE = "none"
    PARTIAL = "partial"
    HEAVY = "heavy"


@dataclass
class InvestigationRequest:
    image_id: str
    box_xyxy: Tuple[float, float, float, float]
    initial_class: str
    initial_confidence: float
    crop_b64_jpeg: str
    scene_luminance_mean: float
    is_ego_corridor: bool


@dataclass
class InvestigationVerdict:
    verified_class: str
    epistemic_confidence: float
    visual_evidence: List[str]
    occlusion_level: str
    safety_escalation: bool
    processing_time_ms: float
    source: str  # "nebius_vlm" or "deterministic_fallback"
    raw_response: Optional[str] = None


class AgenticInvestigator:
    """Selective VRU deliberation engine using OpenCV 5 and Nebius Token Factory."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.tokenfactory.nebius.com/v1",
        model_name: str = "nvidia/nemotron-nano-12b-v2-vl",
        timeout_seconds: float = 1.0,
        max_cost_queries_per_frame: int = 1,
    ):
        self.api_key = api_key or os.environ.get("NEBIUS_API_KEY") or os.environ.get("TOKEN_FACTORY_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_cost_queries_per_frame = max_cost_queries_per_frame

        self.has_valid_key = bool(self.api_key and len(self.api_key.strip()) > 5)
        self.queries_this_frame = 0

    def should_investigate(
        self,
        detection: Dict[str, Any],
        frame_shape: Tuple[int, int],
    ) -> bool:
        """
        Policy Gate: Selectively investigates ONLY when:
        1. Class is a VRU candidate.
        2. Detection confidence is ambiguous (0.15 <= conf <= 0.45).
        3. Object is in the critical ego-path corridor or near vehicle path.
        4. Frame query budget has not been exhausted.
        """
        if self.queries_this_frame >= self.max_cost_queries_per_frame:
            return False

        cname = detection.get("class_name", "")
        if cname not in ["person", "rider", "bike", "motor"]:
            return False

        conf = float(detection.get("conf", 0.0))
        if not (0.15 <= conf <= 0.45):
            return False

        box = detection.get("box", [0, 0, 0, 0])
        img_h, img_w = frame_shape
        # Ego corridor: central 40% horizontally, lower 60% vertically
        center_x = (box[0] + box[2]) / 2.0
        center_y = (box[3])  # bottom of bounding box (ground contact)
        in_corridor = (0.30 * img_w <= center_x <= 0.70 * img_w) and (center_y >= 0.40 * img_h)

        return in_corridor

    def prepare_crop(
        self,
        frame: np.ndarray,
        box: Tuple[float, float, float, float],
        context_margin: float = 0.30,
    ) -> Tuple[np.ndarray, str, float]:
        """
        OpenCV 5 image processing:
        - Context expansion
        - Contrast enhancement via CLAHE in LAB space for night/dark scenes
        - Base64 JPEG encoding
        """
        img_h, img_w = frame.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1

        cx1 = max(0, int(round(x1 - bw * context_margin)))
        cy1 = max(0, int(round(y1 - bh * context_margin)))
        cx2 = min(img_w, int(round(x2 + bw * context_margin)))
        cy2 = min(img_h, int(round(y2 + bh * context_margin)))

        sub_img = frame[cy1:cy2, cx1:cx2].copy()
        if sub_img.shape[0] < 4 or sub_img.shape[1] < 4:
            sub_img = np.zeros((32, 32, 3), dtype=np.uint8)

        # OpenCV 5 Luminance Assessment & CLAHE
        lab = cv2.cvtColor(sub_img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        mean_l = float(np.mean(l_channel))

        # If low contrast / dark driving scene, apply adaptive CLAHE
        if mean_l < 75.0:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_channel)
            lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
            sub_img = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # Encode to JPEG Base64
        success, buffer = cv2.imencode(".jpg", sub_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_str = base64.b64encode(buffer).decode("utf-8") if success else ""

        return sub_img, b64_str, mean_l

    def investigate(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any],
        image_id: str,
    ) -> InvestigationVerdict:
        """Executes deliberation with Nebius Token Factory or deterministic fallback."""
        t0 = time.perf_counter()
        self.queries_this_frame += 1

        img_h, img_w = frame.shape[:2]
        box = detection.get("box", [0, 0, 0, 0])
        sub_img, b64_crop, mean_l = self.prepare_crop(frame, box)

        # Fallback if no Nebius API key is configured
        if not self.has_valid_key:
            return self._deterministic_fallback(
                detection=detection,
                mean_luminance=mean_l,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # Structured Prompt Contract for Nebius VLM
        system_prompt = (
            "You are RoadSentinel-Deliberator, a high-reliability road safety vision auditor. "
            "You verify ambiguous detections in automotive camera footage. "
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "verified_class": "person"|"rider"|"bike"|"motor"|"background_false_positive"|"unresolved",\n'
            '  "epistemic_confidence": float (0.0 to 1.0),\n'
            '  "visual_evidence": [string],\n'
            '  "occlusion_level": "none"|"partial"|"heavy",\n'
            '  "safety_escalation": boolean\n'
            "}"
        )

        user_content = [
            {"type": "text", "text": f"Evaluate this ambiguous candidate detection: initial_class={detection.get('class_name')}, confidence={detection.get('conf'):.2f}. Does this crop contain a genuine vulnerable road user?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_crop}"}}
        ]

        try:
            import urllib.request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            body = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.0,
                "max_tokens": 200,
            }
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                raw_text = resp_data["choices"][0]["message"]["content"]
                parsed = json.loads(raw_text.strip("` \n").removeprefix("json"))

                verdict = self._validate_and_gate_verdict(
                    parsed=parsed,
                    initial_detection=detection,
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                    raw_text=raw_text,
                )
                return verdict

        except Exception as e:
            # Deterministic safety fallback upon network/timeout/schema failure
            return self._deterministic_fallback(
                detection=detection,
                mean_luminance=mean_l,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                failure_reason=str(e),
            )

    def _validate_and_gate_verdict(
        self,
        parsed: Dict[str, Any],
        initial_detection: Dict[str, Any],
        elapsed_ms: float,
        raw_text: str,
    ) -> InvestigationVerdict:
        """Safety Gate: Treats VLM output as untrusted hypothesis. Prevents hallucinated overrides."""
        claimed_class = parsed.get("verified_class", "unresolved")
        initial_class = initial_detection.get("class_name", "")
        conf = float(parsed.get("epistemic_confidence", 0.5))

        # Rule 1: Cannot override to background_false_positive if epistemic confidence is low
        if claimed_class == "background_false_positive" and conf < 0.70:
            final_class = initial_class  # Fail-safe: retain detector's VRU hypothesis
            escalate = True
        elif claimed_class in [c.value for c in VerifiedClass]:
            final_class = claimed_class
            escalate = bool(parsed.get("safety_escalation", False))
        else:
            final_class = initial_class
            escalate = True

        return InvestigationVerdict(
            verified_class=final_class,
            epistemic_confidence=min(1.0, max(0.0, conf)),
            visual_evidence=parsed.get("visual_evidence", []),
            occlusion_level=parsed.get("occlusion_level", "partial"),
            safety_escalation=escalate,
            processing_time_ms=round(elapsed_ms, 2),
            source="nebius_vlm",
            raw_response=raw_text,
        )

    def _deterministic_fallback(
        self,
        detection: Dict[str, Any],
        mean_luminance: float,
        elapsed_ms: float,
        failure_reason: Optional[str] = None,
    ) -> InvestigationVerdict:
        """Deterministic safety fallback preserving fail-safe principles."""
        cname = detection.get("class_name", "unresolved")
        conf = float(detection.get("conf", 0.25))

        # Heuristic confidence scaling based on scene illumination
        if mean_luminance < 40.0:
            # Night driving: elevate uncertainty
            evidence = ["night_scene_low_luminance", "detector_prior_retained"]
            escalate = True
        else:
            evidence = ["detector_prior_retained", "standard_illumination"]
            escalate = False

        return InvestigationVerdict(
            verified_class=cname,
            epistemic_confidence=conf,
            visual_evidence=evidence,
            occlusion_level="partial",
            safety_escalation=escalate,
            processing_time_ms=round(elapsed_ms, 2),
            source="deterministic_fallback" if failure_reason is None else f"deterministic_fallback ({failure_reason})",
        )

    def reset_frame_budget(self):
        """Resets the per-frame query counter."""
        self.queries_this_frame = 0
