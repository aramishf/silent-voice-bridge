# Docker Deployment Guide 🐳

Quick guide to run Silent Voice Bridge using Docker.

---

## Prerequisites

1. **Install Docker Desktop**
   - Mac: https://docs.docker.com/desktop/install/mac-install/
   - Windows: https://docs.docker.com/desktop/install/windows-install/
   - Linux: https://docs.docker.com/engine/install/

2. **For GUI on Mac**: Install XQuartz
   ```bash
   brew install --cask xquartz
   ```

3. **Webcam Access**: Ensure your webcam is connected

---

## Quick Start (One Command)

```bash
./docker-start.sh
```

This script will:
- ✅ Check Docker installation
- ✅ Build the Docker image
- ✅ Start the container
- ✅ Configure display access
- ✅ Launch the ASL recognition system

---

## Manual Commands

### Build the Image
```bash
docker-compose build
```

### Start the Container
```bash
docker-compose up -d
```

### View Logs
```bash
docker-compose logs -f
```

### Stop the Container
```bash
docker-compose down
```

### Restart
```bash
docker-compose restart
```

---

## Training Inside Docker

### Collect Data
```bash
docker-compose exec asl-inference python src/bulk_data_collection.py
```

### Preprocess Data
```bash
docker-compose exec asl-inference python src/preprocessing.py
```

### Train Model
```bash
docker-compose exec asl-inference python src/train.py
```

---

## Troubleshooting

### Webcam Not Working
```bash
# Check webcam device
ls -la /dev/video*

# Ensure container has access
docker-compose down
docker-compose up -d
```

### GUI Not Showing (Mac)
```bash
# Install XQuartz
brew install --cask xquartz

# Allow connections
xhost + 127.0.0.1

# Restart container
docker-compose restart
```

### Permission Denied
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker
```

---

## What's Inside the Container?

- Ubuntu 22.04 (slim)
- Python 3.14
- PyTorch
- MediaPipe
- OpenCV
- All project dependencies
- Your trained model (mounted as volume)

---

## File Persistence

The following directories are mounted as volumes:
- `./models/checkpoints` - Trained models
- `./data` - Collected data

Changes to these directories persist even when container is stopped.

---

## Resource Usage

Default limits:
- CPU: 2 cores max
- Memory: 4GB max

Adjust in `docker-compose.yml` if needed.

---

## Next Steps

1. Run `./docker-start.sh`
2. Wait for container to start
3. ASL recognition window appears
4. Start making gestures!

For more details, see the main README.md
