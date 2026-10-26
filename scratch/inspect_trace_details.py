import json

with open("experiments/agentic_traces/golden_10_verified_traces.json", "r") as f:
    traces = json.load(f)

for i, t in enumerate(traces, 1):
    cf = t.get('counterfactual_analysis', {})
    print(f"Trace {i:02d}: {t['image_id']}")
    print(f"  scenario: {t['scenario_title']}")
    print(f"  environment_tag: {t.get('environment_tag')}, scenario_category: {t.get('scenario_category')}")
    print(f"  tool: {t.get('mcp_tool_selected')}, branch: {t.get('branch_type')}")
    print(f"  counterfactual: {cf}")
