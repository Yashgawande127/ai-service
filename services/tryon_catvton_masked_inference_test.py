import os
import sys
import torch
from PIL import Image
from huggingface_hub import snapshot_download

# Add project root and catvton_model to sys.path
services_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(services_dir)
catvton_dir = os.path.join(project_root, "catvton_model")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if catvton_dir not in sys.path:
    sys.path.insert(0, catvton_dir)

from model.pipeline import CatVTONPipeline
from services.parsing_service import parse_human
from services.build_mask_from_schp import build_mask_from_schp_segmentation

def get_garment_category(filename: str) -> str:
    """
    Classifies a garment image into 'upper', 'lower', or 'overall' based on filename keywords.
    Specifically supports Indian ethnic wear like sarees, kurtis, and lehengas.
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
        # Default fallback
        return "upper"

def main():
    base_model = "booksforcharlie/stable-diffusion-inpainting"
    resume_path = "zhengchong/CatVTON"
    
    # 1. Paths setup
    person_dir = os.path.join(project_root, "test_images", "persons")
    garment_dir = os.path.join(project_root, "test_images", "garments")
    output_dir = os.path.join(project_root, "test_images", "output_masked")
    
    os.makedirs(person_dir, exist_ok=True)
    os.makedirs(garment_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Get list of files
    person_images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    garment_images = [f for f in os.listdir(garment_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not person_images or not garment_images:
        print("\n[INSTRUCTION] Please place your test images in the following folders:")
        print(f" - Person images: {person_dir}")
        print(f" - Garment images: {garment_dir}")
        print("\nExpected files:")
        print(" - In persons/: 1-2 real person photos (e.g., person1.jpg)")
        print(" - In garments/: 3-5 ethnic wear images (e.g., saree1.jpg, kurti1.jpg, lehenga1.jpg)")
        print("Exiting test script until images are prepared.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: Running on CPU will be extremely slow. For actual generation, please run this on GPU.")
        
    print(f"Downloading attention checkpoint from: {resume_path}...")
    repo_path = snapshot_download(repo_id=resume_path, allow_patterns=["mix-48k-1024/*"])
    
    print("Initializing pipeline...")
    pipeline = CatVTONPipeline(
        base_ckpt=base_model,
        attn_ckpt=repo_path,
        attn_ckpt_version="mix",
        weight_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=device,
        skip_safety_check=True
    )
    
    print("\nStarting try-on generation loop...")
    for p_img_name in person_images:
        p_path = os.path.join(person_dir, p_img_name)
        person_image = Image.open(p_path).convert("RGB")
        
        print(f"\nRunning human parsing on {p_img_name}...")
        try:
            parsing_result = parse_human(person_image)
        except Exception as e:
            print(f"Human parsing failed for {p_img_name}: {e}. Skipping person image.")
            continue
        
        for g_img_name in garment_images:
            g_path = os.path.join(garment_dir, g_img_name)
            garment_image = Image.open(g_path).convert("RGB")
            
            category = get_garment_category(g_img_name)
            print(f"Garment: {g_img_name} -> Classified Category: {category}")
            
            # Generate the specific mask based on garment category
            mask_image = build_mask_from_schp_segmentation(parsing_result, category)
            
            p_base = os.path.splitext(p_img_name)[0]
            g_base = os.path.splitext(g_img_name)[0]
            out_name = f"{p_base}_on_{g_base}.png"
            out_path = os.path.join(output_dir, out_name)
            
            print(f"Generating try-on: {p_img_name} + {g_img_name}...")
            
            try:
                result_image = pipeline(
                    image=person_image,
                    condition_image=garment_image,
                    mask=mask_image,
                    num_inference_steps=50,
                    guidance_scale=2.5
                )[0]
                
                result_image.save(out_path)
                print(f" Saved to: {out_path}")
            except Exception as e:
                print(f" FAILED to generate {out_name}: {e}")

if __name__ == "__main__":
    main()
