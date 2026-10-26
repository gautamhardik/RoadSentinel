#!/usr/bin/env bash
# =============================================================================
# RoadSentinel Studio v3.0 — 1-Click Startup Script (Linux / macOS)
# =============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}===============================================================================${NC}"
echo -e "${GREEN}               ROADSENTINEL STUDIO v3.0 // MISSION CONTROL                     ${NC}"
echo -e "${GREEN}       Autonomous VRU Safety Platform | Active Foveation | AWS COOL            ${NC}"
echo -e "${GREEN}===============================================================================${NC}"
echo ""

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR] python3 could not be found. Please install Python 3.10+.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${BLUE}[INFO] Detected:${NC} $PYTHON_VERSION"

# Setup virtual environment if not present
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}[INFO] Creating isolated virtual environment (venv)...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check and install requirements
echo -e "${BLUE}[INFO] Checking platform dependencies...${NC}"
pip install --quiet --upgrade pip setuptools wheel
pip install --quiet -r requirements.txt

# Launch browser after 2 seconds
(sleep 2 && (xdg-open http://localhost:8000 2>/dev/null || open http://localhost:8000 2>/dev/null || true)) &

echo ""
echo -e "${GREEN}===============================================================================${NC}"
echo -e "${GREEN}[STATUS] Starting RoadSentinel Studio at http://localhost:8000 ...             ${NC}"
echo -e "${GREEN}[CTRL+C] Press Ctrl+C to terminate the platform.                              ${NC}"
echo -e "${GREEN}===============================================================================${NC}"
echo ""

python saas_app.py
