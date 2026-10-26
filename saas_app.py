"""
RoadSentinel Cloud - Root Application Server
Runs FastAPI + Uvicorn ASGI on port 8000.
"""

import sys
import os
from pathlib import Path
import uvicorn

# Ensure repository root is on sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.saas.api import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting RoadSentinel Cloud SaaS server on http://0.0.0.0:{port}")
    uvicorn.run("src.saas.api:app", host="0.0.0.0", port=port, reload=False, access_log=True)
