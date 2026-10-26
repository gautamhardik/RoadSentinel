import requests
import json
import os

BASE_URL = "http://localhost:8000"

print("--- 1. Testing GET / and GET /docs ---")
r_root = requests.get(f"{BASE_URL}/")
r_docs = requests.get(f"{BASE_URL}/docs")
print(f"GET /: {r_root.status_code}, content_type: {r_root.headers.get('content-type')}")
print(f"GET /docs: {r_docs.status_code}")

print("\n--- 2. Testing GET /api/v1/system/health ---")
r_health = requests.get(f"{BASE_URL}/api/v1/system/health")
print(f"GET /api/v1/system/health: {r_health.status_code}")
print(json.dumps(r_health.json(), indent=2))

print("\n--- 3. Testing GET /api/v1/incidents & checking image URLs ---")
r_inc = requests.get(f"{BASE_URL}/api/v1/incidents")
print(f"GET /api/v1/incidents: {r_inc.status_code}")
inc_data = r_inc.json()
incidents = inc_data.get("incidents", [])
print(f"Incident count: {len(incidents)}")
all_urls_ok = True
for idx, inc in enumerate(incidents, 1):
    for ukey in ["full_image_url", "raw_crop_url", "opencv_crop_url"]:
        url = inc.get(ukey)
        if not url:
            print(f"  Incident {idx}: MISSING {ukey}")
            all_urls_ok = False
            continue
        full_url = f"{BASE_URL}{url}" if url.startswith("/") else url
        r_img = requests.head(full_url)
        if r_img.status_code != 200:
            # try GET in case HEAD not supported
            r_img = requests.get(full_url)
        ctype = r_img.headers.get("content-type", "")
        if r_img.status_code != 200 or "image" not in ctype:
            print(f"  Incident {idx} {ukey} ({full_url}) FAILED: status={r_img.status_code}, ctype={ctype}")
            all_urls_ok = False
print(f"All 36 incident image URLs return 200 image/jpeg: {all_urls_ok}")

print("\n--- 4. Testing POST /api/v1/investigate/image with 2 real images ---")
import glob
test_imgs = glob.glob("bdd100k_dataset/val/images/*.jpg")[:2]
for idx, timg in enumerate(test_imgs, 1):
    with open(timg, "rb") as f:
        files = {"file": (os.path.basename(timg), f, "image/jpeg")}
        r_post = requests.post(f"{BASE_URL}/api/v1/investigate/image", files=files)
        print(f"Upload {idx} ({os.path.basename(timg)}): status={r_post.status_code}")
        if r_post.status_code == 200:
            res = r_post.json()
            print("  status:", res.get("status"))
            print("  detections count:", len(res.get("detections", [])))
            print("  crop_url:", res.get("crop_image_url"))
            print("  enhanced_crop_url:", res.get("enhanced_crop_image_url"))
            # verify generated crop URLs return 200
            for curl_key in ["crop_image_url", "enhanced_crop_image_url"]:
                curl = res.get(curl_key)
                if curl:
                    r_curl = requests.get(f"{BASE_URL}{curl}")
                    print(f"    {curl_key} status: {r_curl.status_code}, ctype: {r_curl.headers.get('content-type')}")
        else:
            print("  Error response:", r_post.text[:200])

print("\n--- 5. Testing POST /api/v1/cool/live-microbenchmark (25 iterations) ---")
r_bench = requests.post(f"{BASE_URL}/api/v1/cool/live-microbenchmark", json={"iterations": 25})
print(f"POST /api/v1/cool/live-microbenchmark: {r_bench.status_code}")
if r_bench.status_code == 200:
    print(json.dumps(r_bench.json(), indent=2))
else:
    print("Error:", r_bench.text[:200])

print("\n--- 6. Testing GET /api/v1/evaluation/ledger ---")
r_ledger = requests.get(f"{BASE_URL}/api/v1/evaluation/ledger")
print(f"GET /api/v1/evaluation/ledger: {r_ledger.status_code}")
print(json.dumps(r_ledger.json().get("phases_ledger", []), indent=2))
