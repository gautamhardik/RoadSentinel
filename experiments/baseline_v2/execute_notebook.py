"""
Executes each cell in phase2a_evaluation_audit.ipynb and populates cell outputs.
"""

import sys
import io
import json
import traceback
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
nb_path = PROJECT_ROOT / "notebooks" / "phase2a_evaluation_audit.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Global execution context
global_scope = {"__name__": "__main__"}

exec_count = 1
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code_str = "".join(cell["source"])
        print(f"Executing Code Cell {exec_count} (index {idx})...")
        cell["execution_count"] = exec_count
        
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code_str, global_scope)
            out_text = stdout_buf.getvalue()
            err_text = stderr_buf.getvalue()
            
            outputs = []
            if out_text:
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [line + "\n" for line in out_text.splitlines()]
                })
            if err_text:
                outputs.append({
                    "name": "stderr",
                    "output_type": "stream",
                    "text": [line + "\n" for line in err_text.splitlines()]
                })
            cell["outputs"] = outputs
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Error in cell {idx}: {e}")
            cell["outputs"] = [{
                "ename": type(e).__name__,
                "evalue": str(e),
                "output_type": "error",
                "traceback": tb_str.splitlines()
            }]
            break
            
        exec_count += 1

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Finished executing all {exec_count - 1} code cells. Updated: {nb_path}")
