#!/bin/bash

# Build script for Render deployment - OPTIMIZED FOR FREE TIER

echo "Starting build process..."

# Install Python dependencies (excluding TensorFlow for speed)
pip install -r requirements.txt

# Remove Deep Neural Network model to speed up deployment
echo "Removing TensorFlow DNN model for free tier compatibility..."
if [ -f "models/deep_neural_network.h5" ]; then
    rm models/deep_neural_network.h5
    echo "✓ Removed DNN model"
fi

# Check if other models exist, if not train them
if [ ! -f "models/logistic_regression.pkl" ]; then
    echo "Models not found. Training sklearn models only..."
    python -c "
import train_models
# Train only sklearn models, skip DNN
import os
os.environ['SKIP_DNN'] = '1'
exec(open('train_models.py').read())
"
else
    echo "Models already exist. Skipping training."
fi

echo "Build complete! Using 4 fast sklearn models (no TensorFlow/DNN)"
