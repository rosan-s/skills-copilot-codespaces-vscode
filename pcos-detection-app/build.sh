#!/bin/bash

# Build script for Render deployment

echo "Starting build process..."

# Install Python dependencies
pip install -r requirements.txt

# Check if models exist, if not train them
if [ ! -f "models/logistic_regression.pkl" ]; then
    echo "Models not found. Training models..."
    python train_models.py
else
    echo "Models already exist. Skipping training."
fi

echo "Build complete!"
