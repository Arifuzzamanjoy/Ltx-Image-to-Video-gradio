import os
import torch
import gradio as gr
from pathlib import Path
import tempfile
import numpy as np
import uuid
from datetime import datetime
import gc

from ltx_video.inference import infer, InferenceConfig, load_pipeline_config

# Set CUDA memory management environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Use /home for caches to persist between sessions
CACHE_DIR = os.path.join("/home", "ltx_video_cache")
OUTPUT_DIR = os.path.join("/home", "ltx_video_outputs")

# Create directories if they don't exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default configuration - optimized for lower memory usage
DEFAULT_MODEL = "configs/ltxv-2b-0.9.8-distilled.yaml"  # Use smaller 2B model by default
DEFAULT_HEIGHT = 512  # Reduced from 704
DEFAULT_WIDTH = 768   # Reduced from 1216
DEFAULT_NUM_FRAMES = 17  # Reduced from 25 (must be 8n+1)
DEFAULT_SEED = 42
DEFAULT_GUIDANCE_SCALE = 3.0
DEFAULT_NUM_INFERENCE_STEPS = 8  # Distilled model works well with 8 steps

AVAILABLE_MODELS = [
    "configs/ltxv-2b-0.9.8-distilled.yaml",      # Smallest model - best for memory
    "configs/ltxv-13b-0.9.8-distilled.yaml",     # Larger model - needs more memory
    "configs/ltxv-13b-0.9.8-dev.yaml",           # Full 13B model - highest memory
    # FP8 models require Q8 kernels - temporarily disabled due to compatibility issues
    # "configs/ltxv-2b-0.9.8-distilled-fp8.yaml",  # Quantized 2B model
    # "configs/ltxv-13b-0.9.8-distilled-fp8.yaml", # Quantized 13B model
    # "configs/ltxv-13b-0.9.8-dev-fp8.yaml",       # Quantized dev model
]

def clear_gpu_memory():
    """Clear GPU memory to prevent OOM errors"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def check_available_models():
    """Check which model config files are available"""
    available = []
    for model in AVAILABLE_MODELS:
        if os.path.exists(model):
            available.append(model)
        else:
            print(f"⚠️  Model config not found: {model}")
    return available if available else AVAILABLE_MODELS  # fallback to all if none found

def image_to_video(
    prompt, 
    image, 
    model_config, 
    height, 
    width, 
    num_frames, 
    seed, 
    guidance_scale, 
    num_inference_steps,
    progress=gr.Progress()
):
    """Generate a video from an input image and prompt"""
    
    # Save the image to a temporary file
    img_path = os.path.join(CACHE_DIR, f"input_{uuid.uuid4()}.png")
    image.save(img_path)
    
    # Generate a unique output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"output_{timestamp}.mp4")
    
    # Update progress
    progress(0, desc="Starting video generation")
    
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    try:
        # Load pipeline config and modify it with UI parameters
        pipeline_config = load_pipeline_config(model_config)
        
        # Update the config with our UI parameters
        pipeline_config["guidance_scale"] = guidance_scale
        pipeline_config["num_inference_steps"] = num_inference_steps
        
        # Enable CPU offloading to save GPU memory
        pipeline_config["offload_to_cpu"] = True
        pipe.vae.enable_tiling()

        
        # Configure and run inference
        config = InferenceConfig(
            pipeline_config=model_config,
            prompt=prompt,
            conditioning_media_paths=[img_path],
            conditioning_start_frames=[0],
            height=height,
            width=width,
            num_frames=num_frames,
            output_path=output_path,
            seed=seed,
            offload_to_cpu=True,  # Enable CPU offloading in config too
        )
        
        # Run inference with progress updates
        def progress_callback(step, total_steps):
            progress((step + 1) / total_steps, desc=f"Generating video: step {step + 1}/{total_steps}")
        
        # Add progress callback to pipeline config
        pipeline_config["callback_on_step_end"] = progress_callback
        
        # Run inference
        infer(config)
        
        # Clean up temp file
        os.remove(img_path)
        
        # Clear GPU memory after generation
        clear_gpu_memory()
        
        progress(1.0, desc="Video generation complete")
        return output_path, "Video generation complete!"
        
    except Exception as e:
        # Clean up temp file in case of error
        if os.path.exists(img_path):
            os.remove(img_path)
        
        # Clear GPU memory after error
        clear_gpu_memory()
        
        error_msg = str(e)
        if "CUDA out of memory" in error_msg:
            return None, f"❌ GPU Memory Error: Try using a smaller model (2B), lower resolution, or fewer frames. Original error: {error_msg}"
        else:
            return None, f"❌ Error generating video: {error_msg}"

# Create Gradio interface
with gr.Blocks(title="LTX-Video Generator") as app:
    gr.Markdown("# 🎬 LTX-Video Generator")
    gr.Markdown("Generate high-quality videos from images using LTX-Video")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input controls
            prompt = gr.Textbox(
                label="Prompt", 
                placeholder="Describe the action and motion you want in the video...",
                lines=3
            )
            
            image_input = gr.Image(
                label="Input Image", 
                type="pil",
                height=300
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                model_config = gr.Dropdown(
                    label="Model (2B models use less memory)",
                    choices=AVAILABLE_MODELS,
                    value=DEFAULT_MODEL
                )
                
                with gr.Row():
                    height = gr.Slider(
                        label="Height (lower = less memory)",
                        minimum=256,
                        maximum=768,  # Reduced max to save memory
                        step=32,
                        value=DEFAULT_HEIGHT
                    )
                    
                    width = gr.Slider(
                        label="Width (lower = less memory)",
                        minimum=256,
                        maximum=1024,  # Reduced max to save memory
                        step=32,
                        value=DEFAULT_WIDTH
                    )
                
                with gr.Row():
                    num_frames = gr.Slider(
                        label="Number of Frames (fewer = less memory)",
                        minimum=9,
                        maximum=65,  # Reduced max to save memory (8*8+1)
                        step=8,
                        value=DEFAULT_NUM_FRAMES
                    )
                    
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=2147483647,
                        step=1,
                        value=DEFAULT_SEED,
                        randomize=True
                    )
                
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1.0,
                        maximum=7.0,
                        step=0.1,
                        value=DEFAULT_GUIDANCE_SCALE
                    )
                    
                    num_inference_steps = gr.Slider(
                        label="Inference Steps",
                        minimum=4,
                        maximum=50,
                        step=1,
                        value=DEFAULT_NUM_INFERENCE_STEPS
                    )
            
            generate_btn = gr.Button("Generate Video", variant="primary")
            
        with gr.Column(scale=1):
            # Output controls
            video_output = gr.Video(label="Generated Video")
            status_text = gr.Textbox(label="Status", interactive=False)
    
    # Set up the generation button click
    generate_btn.click(
        fn=image_to_video,
        inputs=[
            prompt, 
            image_input, 
            model_config, 
            height, 
            width, 
            num_frames, 
            seed, 
            guidance_scale, 
            num_inference_steps
        ],
        outputs=[video_output, status_text]
    )
    
    gr.Markdown("""
    ## 📝 Tips for better results:
    
    - **Memory Management**: Use 2B models for lower memory usage
    - **Resolution**: Lower resolutions (512x768) require less GPU memory
    - **Frames**: Fewer frames (9-25) use less memory than longer videos
    - **Prompt Writing**: Focus on detailed, chronological descriptions of actions and scenes
    - **Image Selection**: Choose high-quality images with clear subjects
    - **Resolution**: The model works best with resolutions divisible by 32
    - **Number of Frames**: Should be divisible by 8 plus 1 (e.g., 9, 17, 25, etc.)
    - **Guidance Scale**: Values between 3.0-3.5 typically give the best results
    
    ## 🔧 If you get memory errors:
    1. Use the 2B distilled model instead of 13B
    2. Reduce resolution (try 512x768 or 384x512)
    3. Use fewer frames (try 9 or 17)
    
    ## ⚡ FP8 Quantized Models:
    FP8 models are temporarily disabled due to Q8Linear compatibility issues.
    We're working on resolving the kernel interface mismatch.
    Use the regular models for now - they work great!
    
    For more information, visit the [LTX-Video GitHub repository](https://github.com/Lightricks/LTX-Video).
    """)

# Launch the app
if __name__ == "__main__":
    app.launch(share=True)
