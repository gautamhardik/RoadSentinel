import json
import base64
import urllib.request
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

user = api.config_values['username']
key = api.config_values['key']
creds = base64.b64encode(f"{user}:{key}".encode()).decode()

url = "https://www.kaggle.com/api/v1/kernels/output?kernel=hardikgautam24119014/roadsentinel-phase-10b-extended-training"
req = urllib.request.Request(url)
req.add_header("Authorization", f"Basic {creds}")

try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        print("Response keys:", list(data.keys()))
        if "error" in data:
            print("ERROR field:", data["error"])
        if "log" in data:
            print("LOG field:\n", data["log"])
        if "files" in data:
            print("Files count:", len(data["files"]))
            for f in data["files"]:
                print(" -", f.get("fileName"), f.get("fileSize"))
except Exception as e:
    print("Request failed:", e)
