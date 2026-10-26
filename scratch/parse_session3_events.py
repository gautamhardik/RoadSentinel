import json
from pathlib import Path

events_file = Path('experiments/phase10_model_optimization/phase10b_extended_training/session3_artifacts/events_session3.jsonl')
lines = [json.loads(line) for line in events_file.read_text(encoding='utf-8').splitlines() if line.strip()]
print(f'Total event lines: {len(lines)}')
epoch_ends = [e for e in lines if e.get('event') == 'epoch_end']
print(f'Total epoch_end events: {len(epoch_ends)}')
for e in epoch_ends:
    print(f"Epoch {e.get('epoch')}: session_epoch={e.get('session_epoch')}, lr={e.get('lr'):.6f}, mem={e.get('mem_alloc_gb')} GB, elapsed={e.get('elapsed_hours'):.2f}h")
