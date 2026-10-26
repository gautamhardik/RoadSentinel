import json
from pathlib import Path

with open('kaggle_jobs/calibration_benchmark/cohort_images_embedded.py', 'r', encoding='utf-8') as f:
    cohort_str = f.read().strip()

with open('kaggle_jobs/calibration_benchmark/baselines_embedded.json', 'r', encoding='utf-8') as f:
    baselines_json = f.read().strip()

with open('kaggle_jobs/calibration_benchmark/evaluate_phase4c_wbf.py', 'r', encoding='utf-8') as f:
    content = f.read()

insert_pos = content.find('# ============================================================================\n# 1. Class Mapping')
header = content[:insert_pos]
rest = content[insert_pos:]

embedded_block = f'''# ============================================================================
# Embedded 300-Image Pilot Cohort & Historical Baselines (Phase 4C Self-Contained)
# ============================================================================

{cohort_str}

PHASE3_HISTORICAL_BASELINES = {baselines_json}

'''

new_content = header + embedded_block + rest

old_manifest_logic = '''    # Load 300-image Pilot Cohort Manifest
    cohort_manifest_p = Path(args.cohort_manifest)
    pilot_images_set = set()
    if cohort_manifest_p.exists():
        with open(cohort_manifest_p, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            pilot_images_set = {item["image"] for item in manifest_data["images"]}
        print(f"Loaded {len(pilot_images_set)} pilot cohort images from {cohort_manifest_p}")
    else:
        print(f"Warning: Cohort manifest {cohort_manifest_p} not found; pilot subset tracking disabled.")'''

new_manifest_logic = '''    # Load 300-image Pilot Cohort Manifest (using embedded constant)
    pilot_images_set = set(PILOT_COHORT_IMAGE_SET)
    print(f"Loaded {len(pilot_images_set)} pilot cohort images from embedded Phase 4B manifest.")'''

new_content = new_content.replace(old_manifest_logic, new_manifest_logic)

# Put ActiveFoveation-960 and Calibrated-ActiveFoveation-960 first in ALL_CONFIGS
old_configs = '''    ALL_CONFIGS = [
        {"name": "FullFrame-640", "type": "full_frame", "imgsz": 640, "crops": 0},
        {"name": "FullFrame-960", "type": "full_frame", "imgsz": 960, "crops": 0},
        {"name": "FullFrame-1280", "type": "full_frame", "imgsz": 1280, "crops": 0},
        {"name": "FixedGrid-960", "type": "fixed_grid", "imgsz": 960, "crops": 2},
        {"name": "RandomCrop-960", "type": "random_crop", "imgsz": 960, "crops": 2},
        {"name": "ActiveFoveation-960", "type": "active_foveation", "imgsz": 960, "crops": 2},
        {"name": "Calibrated-ActiveFoveation-960", "type": "calibrated_foveation", "imgsz": 960, "crops": 2},
    ]'''

new_configs = '''    ALL_CONFIGS = [
        {"name": "ActiveFoveation-960", "type": "active_foveation", "imgsz": 960, "crops": 2},
        {"name": "Calibrated-ActiveFoveation-960", "type": "calibrated_foveation", "imgsz": 960, "crops": 2},
        {"name": "FullFrame-640", "type": "full_frame", "imgsz": 640, "crops": 0},
        {"name": "FullFrame-960", "type": "full_frame", "imgsz": 960, "crops": 0},
        {"name": "FullFrame-1280", "type": "full_frame", "imgsz": 1280, "crops": 0},
        {"name": "FixedGrid-960", "type": "fixed_grid", "imgsz": 960, "crops": 2},
        {"name": "RandomCrop-960", "type": "random_crop", "imgsz": 960, "crops": 2},
    ]'''

new_content = new_content.replace(old_configs, new_configs)

with open('kaggle_jobs/calibration_benchmark/evaluate_phase4c_wbf.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Updated evaluate_phase4c_wbf.py successfully. Length:", len(new_content))
