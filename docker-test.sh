#!/bin/bash

# Docker Health Check and Testing Script
# Tests if the containerized ASL system is working correctly

set -e

echo "🔍 Silent Voice Bridge - Docker Health Check"
echo "=============================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -n "Testing: $test_name... "
    
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

echo "1️⃣  Checking Prerequisites"
echo "-------------------------------------------"

# Test 1: Docker installed
run_test "Docker installed" "docker --version"

# Test 2: Docker running
run_test "Docker daemon running" "docker ps"

# Test 3: Docker Compose available
run_test "Docker Compose available" "docker-compose --version"

echo ""
echo "2️⃣  Checking Docker Image"
echo "-------------------------------------------"

# Test 4: Image exists
run_test "ASL image exists" "docker images | grep -q silent-voice-bridge-asl-inference"

# Test 5: Image size reasonable
if docker images silent-voice-bridge-asl-inference:latest --format "{{.Size}}" | grep -q "GB"; then
    echo -e "Testing: Image size... ${GREEN}✓ PASS${NC} ($(docker images silent-voice-bridge-asl-inference:latest --format '{{.Size}}'))"
    ((TESTS_PASSED++))
else
    echo -e "Testing: Image size... ${RED}✗ FAIL${NC}"
    ((TESTS_FAILED++))
fi

echo ""
echo "3️⃣  Testing Container Functionality"
echo "-------------------------------------------"

# Test 6: Python imports work
echo -n "Testing: Python imports (PyTorch, MediaPipe, OpenCV)... "
if docker run --rm silent-voice-bridge-asl-inference:latest \
    python -c "import torch; import mediapipe; import cv2; print('OK')" 2>&1 | grep -q "OK"; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}"
    ((TESTS_FAILED++))
fi

# Test 7: Model file accessible
echo -n "Testing: Model file exists in container... "
if docker run --rm -v $(pwd)/models/checkpoints:/app/models/checkpoints \
    silent-voice-bridge-asl-inference:latest \
    python -c "import os; assert os.path.exists('models/checkpoints/best_model.pth')" 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}"
    ((TESTS_FAILED++))
fi

# Test 8: Config file accessible
echo -n "Testing: Config file accessible... "
if docker run --rm silent-voice-bridge-asl-inference:latest \
    python -c "from utils import load_config; load_config('config/config.yaml')" 2>&1 | grep -qv "Error"; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}"
    ((TESTS_FAILED++))
fi

# Test 9: MediaPipe model file
echo -n "Testing: MediaPipe hand model exists... "
if docker run --rm silent-voice-bridge-asl-inference:latest \
    python -c "import os; assert os.path.exists('models/mediapipe/hand_landmarker.task')" 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}"
    ((TESTS_FAILED++))
fi

# Test 10: LSTM model loads
echo -n "Testing: LSTM model loads successfully... "
if docker run --rm -v $(pwd)/models/checkpoints:/app/models/checkpoints \
    silent-voice-bridge-asl-inference:latest \
    python -c "import torch; from model import ASLFingerSpellingLSTM; model = ASLFingerSpellingLSTM(); checkpoint = torch.load('models/checkpoints/best_model.pth', map_location='cpu'); model.load_state_dict(checkpoint['model_state_dict']); print('OK')" 2>&1 | grep -q "OK"; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}"
    ((TESTS_FAILED++))
fi

echo ""
echo "4️⃣  Checking Local Files"
echo "-------------------------------------------"

# Test 11: Trained model exists locally
run_test "Trained model file exists" "test -f models/checkpoints/best_model.pth"

# Test 12: Config file exists
run_test "Config file exists" "test -f config/config.yaml"

# Test 13: Source files exist
run_test "Source files exist" "test -f src/sentence_inference.py && test -f src/model.py"

echo ""
echo "=============================================="
echo "📊 Test Results"
echo "=============================================="
echo ""

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
PASS_RATE=$((TESTS_PASSED * 100 / TOTAL_TESTS))

echo "Total Tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
echo "Pass Rate: $PASS_RATE%"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed! Your Docker setup is working correctly.${NC}"
    echo ""
    echo "🚀 Ready to deploy!"
    echo ""
    echo "Note: Webcam access doesn't work on Docker Desktop for Mac."
    echo "For webcam testing, run locally:"
    echo "  cd src && ../venv/bin/python sentence_inference.py"
    echo ""
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Please check the errors above.${NC}"
    echo ""
    echo "Common fixes:"
    echo "  - Rebuild image: docker-compose build"
    echo "  - Check model file: ls -lh models/checkpoints/best_model.pth"
    echo "  - Verify Docker is running: docker ps"
    echo ""
    exit 1
fi
