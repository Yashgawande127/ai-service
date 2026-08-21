import os
import sys
import torch
import time
from huggingface_hub import snapshot_download

# Add catvton_model to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
catvton_dir = os.path.join(project_root, "catvton_model")
sys.path.insert(0, catvton_dir)

from model.pipeline import CatVTONPix2PixPipeline

def main():
    base_model = "timbrooks/instruct-pix2pix"
    resume_path = "zhengchong/CatVTON-MaskFree"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"1. Downloading/Checking attention checkpoint from HuggingFace: {resume_path}...")
    repo_path = snapshot_download(repo_id=resume_path)
    print(f"Attention checkpoint ready at: {repo_path}")
    
    print("\n2. Initializing CatVTONPix2PixPipeline with REAL pretrained weights...")
    print("This will download the Instruct-Pix2Pix base model and SD VAE (~4-5 GB total) if they are not already cached.")
    print("Starting download and load...")
    
    start_time = time.time()
    try:
        pipeline = CatVTONPix2PixPipeline(
            base_ckpt=base_model,
            attn_ckpt=repo_path,
            attn_ckpt_version="mix-48k-1024",
            weight_dtype=torch.float32,
            device=device,
            skip_safety_check=True
        )
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"\nSUCCESS: Pipeline with real weights successfully constructed and loaded in {elapsed:.2f} seconds!")
    except Exception as e:
        print(f"\nERROR during real weights pipeline construction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
