#!/bin/bash

# Quick data collection for specific letter

echo "=========================================="
echo "Quick Data Collection for Letter R"
echo "=========================================="
echo ""
echo "Tips for letter R:"
echo "  - Cross index finger OVER middle finger"
echo "  - Keep crossing clearly visible"
echo "  - Try different angles"
echo "  - Collect 50-100 more samples"
echo ""
echo "=========================================="
echo ""

# Activate virtual environment
source venv/bin/activate

# Run bulk collection (will start at class A, use arrows to navigate to R)
cd src
python bulk_data_collection.py
