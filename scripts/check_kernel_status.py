import time
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
for attempt in range(6):
    try:
        api.authenticate()
        status = api.kernels_status('hardikgautam24119014/roadsentinel-phase-8-official-val-eval')
        print("Kernel Status:", status)
        sys.exit(0)
    except Exception as e:
        print(f"Attempt {attempt + 1} error: {e}")
        time.sleep(5)
sys.exit(1)
