import os
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

api = KaggleApi()
api.authenticate()
client = api.build_kaggle_client()

req = ApiListKernelSessionOutputRequest()
req.user_name = 'hardikgautam24119014'
req.kernel_slug = 'roadsentinel-phase-10b-extended-training'

res = client.kernels.kernels_api_client.list_kernel_session_output(req)
log_content = res.log or ""

with open("scratch/session4_live.log", "w", encoding="utf-8") as f:
    f.write(log_content)

print(f"Log written to scratch/session4_live.log ({len(log_content)} chars).")
if res.files:
    print(f"Files present ({len(res.files)}):")
    for f in res.files:
        print(f"  - {f.file_name} ({f.total_bytes} bytes)")

lines = log_content.splitlines()
if lines:
    print("--- FIRST 15 LINES ---")
    for l in lines[:15]:
        safe = l.encode("ascii", errors="replace").decode()
        print("  ", safe)
    print("--- LAST 25 LINES ---")
    for l in lines[-25:]:
        safe = l.encode("ascii", errors="replace").decode()
        print("  ", safe)
else:
    print("Log is currently empty (worker still initializing container).")
