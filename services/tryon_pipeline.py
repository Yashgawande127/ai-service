"""
Try-On Pipeline Orchestrator
Responsible for coordinating pose detection, parsing, warping, and the Cat-VTON diffusion model to produce the final virtual try-on image.
"""

import os
import sys
import time
import uuid
import torch
import requests
from io import BytesIO
from PIL import Image
from huggingface_hub import snapshot_download

# Setup paths
services_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(services_dir)
catvton_dir = os.path.join(project_root, "catvton_model")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if catvton_dir not in sys.path:
    sys.path.insert(0, catvton_dir)

from model.pipeline import CatVTONPipeline, CatVTONPix2PixPipeline
from services.parsing_service import parse_human
from services.build_mask_from_schp import build_mask_from_schp_segmentation

# Global cached pipelines
_MASKFREE_PIPELINE = None
_MASKED_PIPELINE = None

def get_garment_category_safe(filename: str) -> str:
    """
    Classifies a garment image into 'upper', 'lower', or 'overall' based on filename keywords.
    Defaults to 'overall' if no reliable keyword is found.
    """
    name = filename.lower()
    
    # 1. Lower Body Garments
    if any(keyword in name for keyword in ["pants", "skirt", "jeans", "lower", "trouser", "shorts"]):
        return "lower"
        
    # 2. Overall Body Garments (includes saree and lehenga)
    elif any(keyword in name for keyword in ["saree", "sari", "lehenga", "dress", "overall", "gown", "jumpsuit", "anarkali"]):
        return "overall"
        
    # 3. Upper Body Garments (includes kurti)
    elif any(keyword in name for keyword in ["kurti", "kurta", "shirt", "t-shirt", "blouse", "top", "upper"]):
        return "upper"
        
    else:
        # Default to 'overall' when no reliable signal is matched
        return "overall"

def get_maskfree_pipeline():
    global _MASKFREE_PIPELINE
    if _MASKFREE_PIPELINE is not None:
        return _MASKFREE_PIPELINE
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Initialization] Loading Mask-Free CatVTON Pipeline on {device}...")
    
    # Download attention weights checkpoint
    attn_ckpt_path = snapshot_download("zhengchong/CatVTON-MaskFree")
    
    _MASKFREE_PIPELINE = CatVTONPix2PixPipeline(
        base_ckpt="timbrooks/instruct-pix2pix",
        attn_ckpt=attn_ckpt_path,
        attn_ckpt_version="mix-48k-1024",
        weight_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=device,
        skip_safety_check=True
    )
    print("[Initialization] Mask-Free CatVTON Pipeline successfully loaded.")
    return _MASKFREE_PIPELINE

def get_masked_pipeline():
    global _MASKED_PIPELINE
    if _MASKED_PIPELINE is not None:
        return _MASKED_PIPELINE
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Initialization] Loading Masked CatVTON Pipeline on {device}...")
    
    # Download attention weights checkpoint
    attn_ckpt_path = snapshot_download("zhengchong/CatVTON", allow_patterns=["mix-48k-1024/*"])
    
    _MASKED_PIPELINE = CatVTONPipeline(
        base_ckpt="booksforcharlie/stable-diffusion-inpainting",
        attn_ckpt=attn_ckpt_path,
        attn_ckpt_version="mix",
        weight_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=device,
        skip_safety_check=True
    )
    print("[Initialization] Masked CatVTON Pipeline successfully loaded.")
    return _MASKED_PIPELINE

def load_image_to_pil(image_input: str) -> Image.Image:
    """
    Loads an image from a local path or a remote URL and returns a PIL Image in RGB format.
    """
    if image_input.startswith(("http://", "https://")):
        response = requests.get(image_input, timeout=15)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    else:
        resolved_path = image_input
        if not os.path.isabs(image_input):
            opt1 = os.path.join(project_root, image_input)
            if os.path.exists(opt1):
                resolved_path = opt1
            else:
                opt2 = os.path.abspath(image_input)
                if os.path.exists(opt2):
                    resolved_path = opt2
        
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Image path not found: {image_input} (resolved to: {resolved_path})")
        return Image.open(resolved_path).convert("RGB")

def run_tryon_pipeline(
    person_image_url: str,
    outfit_image_url: str,
    pipeline_type: str = "mask-free",
    category: str = None,
    width: int = 768,
    height: int = 1024
) -> str:
    """
    Executes the full virtual try-on pipeline and returns the path to the generated image.
    """
    print(f"\n--- Running Try-On Request (Pipeline: {pipeline_type}) ---")
    
    # 1. Load input images
    person_img = load_image_to_pil(person_image_url)
    garment_img = load_image_to_pil(outfit_image_url)
    
    # 2. Setup output folder and file name
    output_dir = os.path.join(project_root, "test_images", "output")
    os.makedirs(output_dir, exist_ok=True)
    out_filename = f"result_{pipeline_type}_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
    out_path = os.path.join(output_dir, out_filename)
    
    if pipeline_type == "mask-free":
        # Load mask-free pipeline
        pipeline = get_maskfree_pipeline()
        
        # Inference (no mask / SCHP parsing needed)
        print("[Inference] Executing Mask-Free CatVTON model...")
        result = pipeline(
            image=person_img,
            condition_image=garment_img,
            num_inference_steps=50,
            guidance_scale=2.5,
            width=width,
            height=height
        )[0]
        
    elif pipeline_type == "masked":
        # Determine category if not provided
        if not category:
            # Extract filename/keyword from path/URL
            filename_part = os.path.basename(outfit_image_url.split("?")[0])
            category = get_garment_category_safe(filename_part)
            
        print(f"[Parsing] Running SCHP human parsing to build mask for category: {category}...")
        parsing_result = parse_human(person_img)
        mask_img = build_mask_from_schp_segmentation(parsing_result, category)
        
        # Load masked pipeline
        pipeline = get_masked_pipeline()
        
        # Inference with mask
        print("[Inference] Executing Masked CatVTON model...")
        result = pipeline(
            image=person_img,
            condition_image=garment_img,
            mask=mask_img,
            num_inference_steps=50,
            guidance_scale=2.5,
            width=width,
            height=height
        )[0]
        
    else:
        raise ValueError(f"Unknown pipeline type: '{pipeline_type}'. Supported: 'mask-free', 'masked'")
        
    # Save and return path
    result.save(out_path)
    print(f"[Success] Try-on image saved to: {out_path}")
    return out_path
