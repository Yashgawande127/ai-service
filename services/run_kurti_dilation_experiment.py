import os
import sys
import time
import cv2
import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download

services_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(services_dir)
catvton_dir = os.path.join(project_root, "catvton_model")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if catvton_dir not in sys.path:
    sys.path.insert(0, catvton_dir)

from model.pipeline import CatVTONPipeline
from services.parsing_service import parse_human
from services.build_mask_from_schp import build_mask_from_schp_segmentation, ATR_MAPPING, part_mask_of, PROTECT_LABELS

def create_dilated_kurti_mask(person_img_path, out_mask_path):
    print("1. Parsing person image with SCHP...")
    img = Image.open(person_img_path).convert("RGB")
    parse_result = parse_human(img)
    
    # Generate base upper mask
    base_mask_img = build_mask_from_schp_segmentation(parse_result, category='upper')
    base_mask = np.array(base_mask_img)
    
    # Protect critical areas: Face, Hair, Shoes, Background
    # For kurti expansion, we allow dilation into pants/thighs downward and slightly outward
    protect_labels = ['Background', 'Face', 'Hair', 'Hat', 'Sunglasses', 'Left-shoe', 'Right-shoe']
    protect_area = part_mask_of(protect_labels, parse_result, ATR_MAPPING)
    
    # 2. Morphological dilation with downward-biased vertical structuring element (~15-20% expansion)
    h, w = base_mask.shape
    kernel_w = max(3, int(w * 0.04))
    kernel_h = max(5, int(h * 0.12))  # Biased heavily downward for longer kurti hem
    
    # Anchor point near top of kernel so dilation expands downward
    kernel = np.ones((kernel_h, kernel_w), np.uint8)
    anchor = (kernel_w // 2, 2)  # Dilation expands downwards
    
    dilated_mask = cv2.dilate(base_mask, kernel, anchor=anchor, iterations=1)
    
    # Smooth edges with small gaussian blur + threshold
    dilated_mask = cv2.GaussianBlur(dilated_mask, (15, 15), 0)
    dilated_mask = np.where(dilated_mask > 30, 255, 0).astype(np.uint8)
    
    # Ensure Face and Hair are strictly preserved
    face_hair_mask = part_mask_of(['Face', 'Hair', 'Hat', 'Sunglasses'], parse_result, ATR_MAPPING)
    dilated_mask[face_hair_mask] = 0
    
    dilated_mask_img = Image.fromarray(dilated_mask)
    dilated_mask_img.save(out_mask_path)
    
    base_white = np.sum(base_mask > 127)
    dilated_white = np.sum(dilated_mask > 127)
    expansion_pct = ((dilated_white - base_white) / base_white) * 100
    print(f"Base Upper Mask: {base_white} pixels -> Dilated Kurti Mask: {dilated_white} pixels (+{expansion_pct:.1f}% expansion)")
    print(f"Saved dilated mask to: {out_mask_path}")
    return dilated_mask_img

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running Kurti Mask Dilation Experiment ===")
    print(f"Device: {device}")
    
    person_path = os.path.join(project_root, "test_images", "persons", "demo2.jpg")
    garment_path = os.path.join(project_root, "test_images", "garments", "kurti_01.png")
    out_dir = os.path.join(project_root, "test_images", "output_masked")
    out_mask_path = os.path.join(out_dir, "kurti_mask_dilated.png")
    out_img_path = os.path.join(out_dir, "demo_kurti_dilated.png")
    
    # 1. Create the dilated mask
    dilated_mask = create_dilated_kurti_mask(person_path, out_mask_path)
    
    # 2. Load Masked CatVTON Pipeline
    print("\n2. Initializing CatVTONPipeline (Masked)...")
    masked_attn_repo = snapshot_download("zhengchong/CatVTON", allow_patterns=["mix-48k-1024/*"])
    pipeline = CatVTONPipeline(
        base_ckpt="booksforcharlie/stable-diffusion-inpainting",
        attn_ckpt=masked_attn_repo,
        attn_ckpt_version="mix",
        weight_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=device,
        skip_safety_check=True
    )
    
    # 3. Run single inference test
    print(f"\n3. Running inference: demo2.jpg + kurti_01.png with dilated mask (768x1024, 50 steps)...")
    person_img = Image.open(person_path).convert("RGB")
    garment_img = Image.open(garment_path).convert("RGB")
    
    t0 = time.time()
    result = pipeline(
        image=person_img,
        condition_image=garment_img,
        mask=dilated_mask,
        num_inference_steps=50,
        guidance_scale=2.5,
        width=768,
        height=1024
    )[0]
    elapsed = time.time() - t0
    
    result.save(out_img_path)
    print(f"\nSaved dilated kurti try-on result to: {out_img_path}")
    
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"Time Taken: {elapsed:.2f} seconds")
    print(f"VRAM Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB")

if __name__ == "__main__":
    main()
