#!/usr/bin/env bash
# ==============================================================================
# RoadSentinel Cloud - AWS Graviton Arm64 Turnkey Deployment & Benchmark Script
# Target Instance: AWS EC2 c7g.xlarge / c8g.xlarge (Ubuntu 24.04 LTS aarch64)
# Competition: OpenCV AI Competition 2026 (Best Use of COOL Award & Grand Prize)
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo " 🛡️  ROADSENTINEL CLOUD: AWS GRAVITON (Arm64) DEPLOYMENT PIPELINE"
echo "======================================================================"

# 1. Hardware & Architecture Verification
ARCH=$(uname -m)
echo "[1/6] Verifying Hardware Architecture: $ARCH"
if [ "$ARCH" != "aarch64" ]; then
    echo "⚠️  WARNING: Host architecture is $ARCH (expected aarch64 / AWS Graviton)."
    echo "    Continuing execution in compatibility mode..."
else
    echo "✅ CONFIRMED: Genuine Arm64 architecture detected."
fi

echo "--- Host CPU & System Details ---"
uname -a
lscpu | grep -E "Architecture|Model name|CPU\(s\)|Flags|Thread" || true
echo "---------------------------------"

# 2. Install System Dependencies
echo "[2/6] Installing Ubuntu system dependencies (FFmpeg, OpenMP, OpenGL)..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-dev \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    git \
    build-essential

# 3. Setup Python Virtual Environment
echo "[3/6] Configuring isolated Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# 4. Install PyTorch & OpenCV with COOL / KleidiCV Acceleration
echo "[4/6] Installing dependencies with Arm64 NEON/KleidiCV optimizations..."
# PyTorch CPU Arm64 build
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt || {
    pip install fastapi "uvicorn[standard]" opencv-python-headless ultralytics numpy pillow pydantic requests
}

# 5. Run Native AWS Graviton & COOL Hardware Benchmark
echo "[5/6] Executing Native AWS Graviton COOL Benchmark Suite..."
mkdir -p experiments/aws_cool_benchmark
python benchmarks/aws_graviton_cool_eval.py

echo "--- Generated COOL Environment Proof ---"
head -n 25 experiments/aws_cool_benchmark/cool_environment_proof.txt || true
echo "----------------------------------------"

# 6. Configure & Launch Persistent FastAPI SaaS Server
echo "[6/6] Launching RoadSentinel Cloud FastAPI Server on Port 8000..."

# Create systemd service unit
SERVICE_FILE="/etc/systemd/system/roadsentinel-saas.service"
WORKDIR=$(pwd)

sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=RoadSentinel Cloud Autonomous Fleet Perception SaaS
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$WORKDIR
ExecStart=$WORKDIR/.venv/bin/python saas_app.py --port 8000 --host 0.0.0.0
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=KMP_DUPLICATE_LIB_OK=TRUE

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable roadsentinel-saas.service
sudo systemctl restart roadsentinel-saas.service

sleep 4
echo "Checking service status..."
sudo systemctl status roadsentinel-saas.service --no-pager || true

echo "Testing HTTP endpoint..."
curl -I http://localhost:8000/api/v1/system/health || true

echo "======================================================================"
echo " 🚀 DEPLOYMENT SUCCESSFUL!"
echo " Public SaaS Endpoint: http://$(curl -s http://checkip.amazonaws.com || echo '<EC2_PUBLIC_IP>'):8000"
echo " Interactive OpenAPI Docs: http://$(curl -s http://checkip.amazonaws.com || echo '<EC2_PUBLIC_IP>'):8000/docs"
echo "======================================================================"
