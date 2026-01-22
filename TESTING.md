# Testing Guide for Silent Voice Bridge 🧪

This guide helps you (and others) verify that the Docker setup works correctly.

---

## Quick Test (30 seconds)

```bash
./docker-test.sh
```

This automated script runs 13 tests and tells you if everything works.

**Expected output:**
```
✅ All tests passed! Your Docker setup is working correctly.
Pass Rate: 100%
```

---

## What Gets Tested

### 1️⃣ Prerequisites (3 tests)
- ✅ Docker installed
- ✅ Docker daemon running  
- ✅ Docker Compose available

### 2️⃣ Docker Image (2 tests)
- ✅ ASL image exists
- ✅ Image size reasonable (~13GB)

### 3️⃣ Container Functionality (5 tests)
- ✅ Python imports work (PyTorch, MediaPipe, OpenCV)
- ✅ Model file accessible
- ✅ Config file accessible
- ✅ MediaPipe hand model exists
- ✅ LSTM model loads successfully

### 4️⃣ Local Files (3 tests)
- ✅ Trained model exists
- ✅ Config file exists
- ✅ Source files exist

---

## Manual Testing

### Test 1: Build the Image

```bash
docker-compose build
```

**Expected:** Build completes successfully in ~6 minutes (first time) or ~1 second (cached).

---

### Test 2: Run Python in Container

```bash
docker run --rm silent-voice-bridge-asl-inference:latest \
  python -c "import torch; import mediapipe; import cv2; print('✅ All imports work!')"
```

**Expected output:**
```
✅ All imports work!
```

---

### Test 3: Load the Model

```bash
docker run --rm \
  -v $(pwd)/models/checkpoints:/app/models/checkpoints \
  silent-voice-bridge-asl-inference:latest \
  python -c "
import torch
from model import ASLFingerSpellingLSTM

model = ASLFingerSpellingLSTM()
checkpoint = torch.load('models/checkpoints/best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
print('✅ Model loaded successfully!')
print(f'Model has {sum(p.numel() for p in model.parameters())} parameters')
"
```

**Expected output:**
```
✅ Model loaded successfully!
Model has 566820 parameters
```

---

### Test 4: Test Preprocessing (Without Training Data)

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  silent-voice-bridge-asl-inference:latest \
  python -c "
import os
print('Data directory contents:')
for root, dirs, files in os.walk('data'):
    level = root.replace('data', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files
        print(f'{subindent}{file}')
"
```

**Expected:** Shows your data directory structure.

---

### Test 5: Run Container (Will Fail on Webcam)

```bash
export DISPLAY=:0
docker-compose up
```

**Expected output:**
```
Loading model from models/checkpoints/best_model.pth...
✓ Model loaded successfully
✓ Scaler loaded successfully
...
[ WARN] VIDEOIO(V4L2:/dev/video0): can't open camera by index
```

**This is normal!** Docker on Mac cannot access webcam. The important part is that the model loads successfully.

Press `Ctrl+C` to stop.

---

## For Others to Test Your Project

### Option 1: From GitHub (Recommended)

```bash
# Clone repository
git clone https://github.com/aramishf/silent-voice-bridge.git
cd silent-voice-bridge

# Run health check
./docker-test.sh
```

**Note:** They'll need to train their own model or you can provide `best_model.pth` separately.

---

### Option 2: Share Docker Image

```bash
# On your machine: Save image to file
docker save silent-voice-bridge-asl-inference:latest | gzip > asl-docker-image.tar.gz

# Share the file (warning: ~4GB compressed)

# On their machine: Load image
docker load < asl-docker-image.tar.gz

# Test it
docker run --rm silent-voice-bridge-asl-inference:latest \
  python -c "import torch; print('✅ Works!')"
```

---

### Option 3: Docker Hub (Best for Distribution)

```bash
# On your machine: Push to Docker Hub
docker tag silent-voice-bridge-asl-inference:latest aramishf/silent-voice-bridge:latest
docker push aramishf/silent-voice-bridge:latest

# On their machine: Pull and run
docker pull aramishf/silent-voice-bridge:latest
docker run --rm aramishf/silent-voice-bridge:latest \
  python -c "import torch; print('✅ Works!')"
```

---

## Troubleshooting

### "Docker not found"
```bash
# Install Docker Desktop
# Mac: https://docs.docker.com/desktop/install/mac-install/
```

### "Image not found"
```bash
# Build the image
docker-compose build
```

### "Model file not found"
```bash
# Check if model exists
ls -lh models/checkpoints/best_model.pth

# If missing, train the model:
cd src
../venv/bin/python preprocessing.py
../venv/bin/python train.py
```

### "Container exits immediately"
```bash
# View logs
docker logs silent-voice-bridge

# Run interactively to debug
docker run -it --rm silent-voice-bridge-asl-inference:latest /bin/bash
```

---

## Success Criteria

Your Docker setup is working if:

✅ `./docker-test.sh` shows 100% pass rate  
✅ Model loads without errors  
✅ All Python imports work  
✅ Container runs (even if webcam fails on Mac)  

---

## Known Limitations

### ❌ Webcam on Docker Desktop for Mac
- **Issue:** macOS doesn't allow Docker to access webcam
- **Solution:** Run locally for webcam testing
  ```bash
  cd src
  ../venv/bin/python sentence_inference.py
  ```

### ✅ Webcam on Linux/Raspberry Pi
- Works perfectly with proper device mapping
- Use for production deployment

---

## Next Steps

Once tests pass:

1. **Share on GitHub** - Others can clone and test
2. **Deploy to Raspberry Pi** - Full webcam support
3. **Push to Docker Hub** - Easy distribution
4. **Add to README** - Document testing process

---

## Questions?

- Check logs: `docker logs silent-voice-bridge`
- View running containers: `docker ps`
- Rebuild: `docker-compose build --no-cache`
- Clean up: `docker system prune -a`
