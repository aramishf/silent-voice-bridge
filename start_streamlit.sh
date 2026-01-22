#!/bin/bash

# Silent Voice Bridge - Streamlit App Launcher

echo "🚀 Starting Silent Voice Bridge Streamlit Dashboard..."
echo ""

# Activate virtual environment and run streamlit
./venv/bin/streamlit run app.py --server.port 8501 --server.address localhost

echo ""
echo "✅ Dashboard stopped"
