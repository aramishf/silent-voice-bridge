#!/bin/bash

# Quick Start Script for ASL Fingerspelling Interpreter
# This script helps you get started with data collection

echo "=================================="
echo "ASL Fingerspelling Interpreter"
echo "Quick Start - Data Collection"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

echo "✓ Virtual environment activated"
echo ""
echo "Starting data collection tool..."
echo ""
echo "Controls:"
echo "  SPACE - Start recording"
echo "  LEFT/RIGHT - Change class"
echo "  P - Pause/Resume"
echo "  Q - Quit"
echo ""
echo "=================================="
echo ""

# Run data collection
cd src
python data_collection.py
