#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Install requirements if not already installed
if ! pip list | grep -q "gradio"; then
    echo "Installing gradio..."
    pip install gradio
fi

# Run the Gradio app
python gradio_app.py
