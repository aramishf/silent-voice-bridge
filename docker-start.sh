#!/bin/bash

# Silent Voice Bridge - Docker Quick Start
# One-command deployment script

set -e  # Exit on error

echo "🚀 Silent Voice Bridge - Docker Deployment"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    echo ""
    echo "Please install Docker Desktop from:"
    echo "  Mac: https://docs.docker.com/desktop/install/mac-install/"
    echo "  Windows: https://docs.docker.com/desktop/install/windows-install/"
    echo "  Linux: https://docs.docker.com/engine/install/"
    exit 1
fi

echo "✅ Docker is installed"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed!"
    exit 1
fi

echo "✅ docker-compose is available"
echo ""

# Allow X11 connections (for GUI on Mac/Linux)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🖥️  Configuring display for macOS..."
    xhost + 127.0.0.1 2>/dev/null || echo "⚠️  XQuartz may not be installed (needed for GUI)"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🖥️  Configuring display for Linux..."
    xhost +local:docker
fi

echo ""
echo "🔨 Building Docker image..."
docker-compose build

echo ""
echo "🚀 Starting Silent Voice Bridge..."
docker-compose up -d

echo ""
echo "✅ Silent Voice Bridge is running!"
echo ""
echo "📊 Container status:"
docker-compose ps

echo ""
echo "📝 View logs:"
echo "  docker-compose logs -f"
echo ""
echo "🛑 Stop the service:"
echo "  docker-compose down"
echo ""
echo "🎥 The ASL recognition window should appear shortly..."
