import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

p3_text = Path('kaggle_jobs/foveation_benchmark/evaluate_phase3_foveation.py').read_text(encoding='utf-8')
p4c_text = Path('kaggle_jobs/calibration_benchmark/evaluate_phase4c_wbf.py').read_text(encoding='utf-8')
p4d_text = Path('kaggle_jobs/calibration_benchmark/evaluate_phase4d_adaptive_area.py').read_text(encoding='utf-8')

def extract_func(text, name):
    pattern = rf'def {name}\s*\(.*?\n(?=(def |\n# =|\Z))'
    m = re.search(pattern, text, re.DOTALL)
    return m.group(0) if m else f"FUNCTION {name} NOT FOUND"

print("=" * 60)
print("1. COMPARING baseline fusion functions")
print("=" * 60)
p3_fuse = extract_func(p3_text, "fuse_and_deduplicate_boxes")
p4c_fuse = extract_func(p4c_text, "baseline_fuse_and_deduplicate")
p4d_fuse = extract_func(p4d_text, "baseline_fuse_and_deduplicate")

print(f"P3 fuse length: {len(p3_fuse)}")
print(f"P4C fuse length: {len(p4c_fuse)}")
print(f"P4D fuse length: {len(p4d_fuse)}")

print("\n--- P4C baseline_fuse vs P4D baseline_fuse diff ---")
import difflib
diff = list(difflib.unified_diff(p4c_fuse.splitlines(), p4d_fuse.splitlines(), lineterm=''))
for l in diff[:30]:
    print(l)

print("\n" + "=" * 60)
print("2. COMPARING generate_uncertainty_crops")
print("=" * 60)
p3_unc = extract_func(p3_text, "generate_uncertainty_crops")
p4c_unc = extract_func(p4c_text, "generate_uncertainty_crops")
p4d_unc = extract_func(p4d_text, "generate_uncertainty_crops")
diff_unc = list(difflib.unified_diff(p4c_unc.splitlines(), p4d_unc.splitlines(), lineterm=''))
print(f"Differences between P4C and P4D uncertainty crop generation: {len(diff_unc)} lines")
for l in diff_unc[:30]:
    print(l)

print("\n" + "=" * 60)
print("3. COMPARING evaluate_operational_point")
print("=" * 60)
p3_eval = extract_func(p3_text, "evaluate_operational_point")
p4c_eval = extract_func(p4c_text, "evaluate_operational_point")
p4d_eval = extract_func(p4d_text, "evaluate_operational_point")
diff_eval = list(difflib.unified_diff(p4c_eval.splitlines(), p4d_eval.splitlines(), lineterm=''))
print(f"Differences between P4C and P4D evaluate_operational_point: {len(diff_eval)} lines")
for l in diff_eval[:30]:
    print(l)
