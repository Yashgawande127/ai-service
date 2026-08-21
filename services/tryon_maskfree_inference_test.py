import os
import sys
import torch
from PIL import Image
from huggingface_hub import snapshot_download

# Add catvton_model to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
catvton_dir = os.path.join(project_root, "catvton_model")
sys.path.insert(0, catvton_dir)

from model.pipeline import CatVTONPix2PixPipeline

def main():
    base_model = "timbrooks/instruct-pix2pix"
    resume_path = "zhengchong/CatVTON-MaskFree"
    
    # 1. Paths setup
    person_dir = os.path.join(project_root, "test_images", "persons")
    garment_dir = os.path.join(project_root, "test_images", "garments")
    output_dir = os.path.join(project_root, "test_images", "output")
    
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
        
    print(f"Loading attention checkpoint...")
    repo_path = snapshot_download(repo_id=resume_path)
    
    print("Initializing pipeline...")
    pipeline = CatVTONPix2PixPipeline(
        base_ckpt=base_model,
        attn_ckpt=repo_path,
        attn_ckpt_version="mix-48k-1024",
        weight_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=device,
        skip_safety_check=True
    )
    
    print("\nStarting try-on generation loop...")
    for p_img_name in person_images:
        p_path = os.path.join(person_dir, p_img_name)
        person_image = Image.open(p_path).convert("RGB")
        
        for g_img_name in garment_images:
            g_path = os.path.join(garment_dir, g_img_name)
            garment_image = Image.open(g_path).convert("RGB")
            
            p_base = os.path.splitext(p_img_name)[0]
            g_base = os.path.splitext(g_img_name)[0]
            out_name = f"{p_base}_on_{g_base}.png"
            out_path = os.path.join(output_dir, out_name)
            
            print(f"Generating: {p_img_name} + {g_img_name} -> {out_name}...")
            
            try:
                # Run inference (default resolution is 768x1024 in CatVTON)
                result_image = pipeline(
                    image=person_image,
                    condition_image=garment_image,
                    num_inference_steps=50,
                    guidance_scale=2.5
                )[0]
                
                result_image.save(out_path)
                print(f" Saved to: {out_path}")
            except Exception as e:
                print(f" FAILED to generate {out_name}: {e}")

if __name__ == "__main__":
    main()
