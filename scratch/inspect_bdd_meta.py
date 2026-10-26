import json
import os

# Check if bdd100k val json exists
val_json_path = "bdd100k_dataset/labels/bdd100k_labels_images_val.json"
bdd_meta = {}
if os.path.exists(val_json_path):
    print("Loading bdd100k val json...")
    with open(val_json_path, 'r') as f:
        data = json.load(f)
        for item in data:
            bdd_meta[item['name']] = item.get('attributes', {})
    print(f"Loaded {len(bdd_meta)} items from bdd100k val json")
else:
    print("val_json_path not found, searching...")
    import glob
    jsons = glob.glob("**/*val*.json", recursive=True)
    print("Found jsons:", jsons)
    for j in jsons:
        if "bdd" in j.lower():
            val_json_path = j
            break

with open("experiments/agentic_traces/golden_10_verified_traces.json", "r") as f:
    traces = json.load(f)

for i, t in enumerate(traces, 1):
    img_name = t['image_id'] + ".jpg"
    meta = bdd_meta.get(img_name, {})
    print(f"Trace {i:02d} ({t['image_id']}):")
    print(f"  BDD100K attributes: {meta}")
    print(f"  Current title: {t['scenario_title']}")
    print(f"  Current tag: {t.get('environment_tag')}")
    print(f"  Current branch: {t.get('branch_type')}")
    print(f"  Current tool: {t.get('mcp_tool_selected')}")
