"""
RoadSentinel Phase 5 — Unit Tests for Agentic Vision Deliberation Engine
Verifies:
1. Selective invocation policy (only ambiguous VRUs in ego corridor).
2. OpenCV 5 image preprocessing, CLAHE contrast adaptation, and base64 encoding.
3. Strict schema contracts and typed verdict structures.
4. Untrusted hypothesis verification and safety-critical fail-safe retention.
5. Deterministic fallback behavior when Nebius Token Factory API key is absent.
"""

import pytest
import numpy as np
import cv2

from src.agentic.investigator import (
    AgenticInvestigator,
    InvestigationVerdict,
    VerifiedClass,
    OcclusionLevel,
)


def test_selective_policy_gating():
    investigator = AgenticInvestigator(api_key=None, max_cost_queries_per_frame=1)
    frame_shape = (720, 1280)

    # Candidate 1: Pedestrian, ambiguous confidence (0.30), in center ego corridor -> SHOULD TRIGGER
    cand1 = {
        "class_name": "person",
        "conf": 0.30,
        "box": [600.0, 300.0, 680.0, 500.0],
    }
    assert investigator.should_investigate(cand1, frame_shape) is True

    # Candidate 2: High confidence (0.85) -> SHOULD NOT TRIGGER (wastes budget)
    cand2 = {
        "class_name": "person",
        "conf": 0.85,
        "box": [600.0, 300.0, 680.0, 500.0],
    }
    assert investigator.should_investigate(cand2, frame_shape) is False

    # Candidate 3: Non-VRU (car) -> SHOULD NOT TRIGGER
    cand3 = {
        "class_name": "car",
        "conf": 0.25,
        "box": [600.0, 300.0, 680.0, 500.0],
    }
    assert investigator.should_investigate(cand3, frame_shape) is False

    # Candidate 4: Far lateral periphery (x1=50, x2=100) -> SHOULD NOT TRIGGER
    cand4 = {
        "class_name": "person",
        "conf": 0.30,
        "box": [50.0, 300.0, 100.0, 500.0],
    }
    assert investigator.should_investigate(cand4, frame_shape) is False

    # Budget exhaustion test
    investigator.queries_this_frame = 1
    assert investigator.should_investigate(cand1, frame_shape) is False

    investigator.reset_frame_budget()
    assert investigator.should_investigate(cand1, frame_shape) is True


def test_opencv5_clahe_enhancement_and_encoding():
    investigator = AgenticInvestigator()
    # Create dark frame (night scene)
    dark_frame = np.full((720, 1280, 3), 25, dtype=np.uint8)
    box = (600.0, 300.0, 650.0, 420.0)

    sub_img, b64_crop, mean_l = investigator.prepare_crop(dark_frame, box)
    assert mean_l < 75.0, "Dark frame luminance should be < 75"
    assert len(b64_crop) > 50, "Base64 encoded JPEG should be non-empty"
    assert sub_img.shape[0] > 0 and sub_img.shape[1] > 0


def test_deterministic_fallback_without_api_key():
    investigator = AgenticInvestigator(api_key=None)
    frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
    det = {
        "class_name": "person",
        "conf": 0.32,
        "box": [600.0, 300.0, 650.0, 420.0],
    }

    verdict = investigator.investigate(frame, det, image_id="test_001.jpg")
    assert isinstance(verdict, InvestigationVerdict)
    assert verdict.source == "deterministic_fallback"
    assert verdict.verified_class == "person"
    assert verdict.epistemic_confidence == pytest.approx(0.32)
    assert verdict.safety_escalation is False


def test_untrusted_hypothesis_safety_override():
    investigator = AgenticInvestigator(api_key="mock_key")
    initial_det = {"class_name": "person", "conf": 0.35}

    # Scenario: VLM tries to dismiss a VRU as background with weak confidence (0.50)
    # Safety Gate must reject the dismissal and keep the VRU hypothesis
    untrusted_output = {
        "verified_class": "background_false_positive",
        "epistemic_confidence": 0.50,
        "visual_evidence": ["blurry object"],
        "safety_escalation": False,
    }

    verdict = investigator._validate_and_gate_verdict(
        parsed=untrusted_output,
        initial_detection=initial_det,
        elapsed_ms=45.0,
        raw_text=str(untrusted_output),
    )

    assert verdict.verified_class == "person", "Fail-safe: weak background claim must NOT override VRU"
    assert verdict.safety_escalation is True, "Must trigger human / system escalation on unverified dismissal"


def test_valid_reclassification_gate():
    investigator = AgenticInvestigator(api_key="mock_key")
    initial_det = {"class_name": "person", "conf": 0.30}

    # Scenario: VLM confidently recognizes a bicycle rider with evidence
    valid_output = {
        "verified_class": "rider",
        "epistemic_confidence": 0.88,
        "visual_evidence": ["visible helmet", "straddling bicycle frame"],
        "occlusion_level": "none",
        "safety_escalation": False,
    }

    verdict = investigator._validate_and_gate_verdict(
        parsed=valid_output,
        initial_detection=initial_det,
        elapsed_ms=52.0,
        raw_text=str(valid_output),
    )

    assert verdict.verified_class == "rider"
    assert verdict.epistemic_confidence == 0.88
    assert "visible helmet" in verdict.visual_evidence
    assert verdict.safety_escalation is False


def test_nvidia_nemotron_vlm_mocked_request_format(monkeypatch):
    """Verifies OpenAI-compatible request structure and parsing for NVIDIA Nemotron VLM."""
    import json
    import io
    import urllib.request

    investigator = AgenticInvestigator(
        api_key="mock_nebius_secret_key",
        model_name="nvidia/nemotron-nano-12b-v2-vl",
        base_url="https://api.tokenfactory.nebius.com/v1",
    )
    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    det = {
        "class_name": "person",
        "conf": 0.28,
        "box": [600.0, 300.0, 650.0, 450.0],
    }

    mock_response_json = {
        "id": "chatcmpl-mock-nemotron",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "verified_class": "person",
                        "epistemic_confidence": 0.92,
                        "visual_evidence": ["pedestrian silhouette", "legs visible"],
                        "occlusion_level": "none",
                        "safety_escalation": False,
                    }),
                },
                "finish_reason": "stop",
            }
        ],
    }

    recorded_request = {}

    class MockHTTPResponse:
        def __init__(self, data):
            self._data = data
        def read(self):
            return self._data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    def mock_urlopen(req, timeout=None):
        recorded_request["url"] = req.full_url
        recorded_request["headers"] = dict(req.headers)
        recorded_request["body"] = json.loads(req.data.decode("utf-8"))
        return MockHTTPResponse(json.dumps(mock_response_json).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    verdict = investigator.investigate(frame, det, image_id="frame_001.jpg")

    assert verdict.source == "nebius_vlm"
    assert verdict.verified_class == "person"
    assert verdict.epistemic_confidence == 0.92
    assert "pedestrian silhouette" in verdict.visual_evidence
    assert recorded_request["url"] == "https://api.tokenfactory.nebius.com/v1/chat/completions"
    assert recorded_request["body"]["model"] == "nvidia/nemotron-nano-12b-v2-vl"
    assert "Bearer mock_nebius_secret_key" in recorded_request["headers"]["Authorization"]
    # Verify multimodal message format
    user_msg = recorded_request["body"]["messages"][1]
    assert user_msg["role"] == "user"
    assert any(c["type"] == "image_url" for c in user_msg["content"])


def test_nvidia_nemotron_vlm_timeout_fallback(monkeypatch):
    """Verifies that network timeout engages deterministic fallback without unhandled exceptions."""
    import urllib.request

    investigator = AgenticInvestigator(
        api_key="mock_nebius_secret_key",
        model_name="nvidia/nemotron-nano-12b-v2-vl",
        timeout_seconds=0.1,
    )
    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    det = {
        "class_name": "bike",
        "conf": 0.30,
        "box": [600.0, 300.0, 650.0, 450.0],
    }

    def mock_timeout(req, timeout=None):
        raise TimeoutError("Connection to Nebius Token Factory timed out after 100ms")

    monkeypatch.setattr(urllib.request, "urlopen", mock_timeout)

    verdict = investigator.investigate(frame, det, image_id="frame_002.jpg")
    assert "deterministic_fallback" in verdict.source
    assert verdict.verified_class == "bike"
    assert verdict.epistemic_confidence == pytest.approx(0.30)

