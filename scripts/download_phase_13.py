import os
import sys
import kaggle
from unittest.mock import patch

# Fix Windows cp1252 default encoding issue in kaggle_api_extended.py
orig_open = open
def utf8_open(*args, **kwargs):
    if 'mode' in kwargs and 'w' in kwargs['mode'] and 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8'
    elif len(args) > 1 and 'w' in args[1] and 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8'
    return orig_open(*args, **kwargs)

os.makedirs('experiments/phase_13_output', exist_ok=True)
api = kaggle.KaggleApi()
api.authenticate()

print("Downloading Phase 13 kernel output...")
import builtins
builtins.open = utf8_open

try:
    api.kernels_output('hardikgautam24119014/roadsentinel-phase-13-vru-polish', path='experiments/phase_13_output')
    print("SUCCESS: Phase 13 output downloaded successfully!")
except Exception as e:
    print(f"Error downloading: {e}")
finally:
    builtins.open = orig_open
