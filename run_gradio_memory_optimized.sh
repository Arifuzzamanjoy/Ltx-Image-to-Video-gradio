#!/bin/bash

# Memory-optimized launcher for LTX-Video Gradio app

echo "🚀 Starting LTX-Video Gradio app with memory optimizations..."

# Set CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate virtual environment
source .venv/bin/activate

# Clear any existing GPU memory
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Check GPU memory
if command -v nvidia-smi &> /dev/null; then
    echo "📊 GPU Memory Status:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
    echo ""
fi

echo "💡 Memory Tips:"
echo "  - Use 2B models for lower memory usage"
echo "  - Start with 512x768 resolution and 17 frames"
echo "  - Use FP8 quantized models if available"
echo ""

# Run the Gradio app
python gradio_app.py
