#!/bin/bash

# Bulk Data Collection Script - FAST MODE!
# Collect 100+ samples in under a minute per class

echo "=================================="
echo "ASL Fingerspelling Interpreter"
echo "BULK COLLECTION MODE - Fast!"
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
echo "🚀 Starting BULK data collection..."
echo ""
echo "How it works:"
echo "  1. Press SPACE to start recording"
echo "  2. Hold your gesture and MOVE your hand around"
echo "  3. System captures ~6 samples per second automatically"
echo "  4. Press SPACE to stop when you have enough"
echo ""
echo "Controls:"
echo "  SPACE - Start/Stop recording"
echo "  LEFT/RIGHT - Change class"
echo "  P - Pause"
echo "  Q - Quit"
echo ""
echo "=================================="
echo ""

# Run bulk data collection
cd src
python bulk_data_collection.py
