import os
import sys
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel

# Add project root and catvton_model to sys.path
services_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(services_dir)
catvton_dir = os.path.join(project_root, "catvton_model")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if catvton_dir not in sys.path:
    sys.path.insert(0, catvton_dir)

from model.pipeline import CatVTONPipeline
from model.attn_processor import SkipAttnProcessor
from model.utils import get_trainable_module, init_adapter
from services.parsing_service import parse_human
from services.build_mask_from_schp import build_mask_from_schp_segmentation

class ConfigOnlyCatVTONPipeline(CatVTONPipeline):
    """
    Subclass of CatVTONPipeline that overrides weight loading.
    Instead of downloading full UNet/VAE checkpoints (approx. 5GB+),
    it loads only their configs and instantiates randomly initialized models on CPU/GPU.
    This allows local pipeline code-structure validation.
    """
    def __init__(
        self, 
        base_ckpt, 
        attn_ckpt, 
        attn_ckpt_version="mix",
        weight_dtype=torch.float32,
        device='cpu',
        skip_safety_check=True,
    ):
        self.device = device
        self.weight_dtype = weight_dtype
        self.skip_safety_check = skip_safety_check

        print("1. Loading scheduler config...")
        scheduler_config = DDIMScheduler.load_config(base_ckpt, subfolder="scheduler")
        self.noise_scheduler = DDIMScheduler.from_config(scheduler_config)

        print("2. Loading VAE config and initializing AutoencoderKL with random weights...")
        vae_config = AutoencoderKL.load_config("stabilityai/sd-vae-ft-mse")
        self.vae = AutoencoderKL.from_config(vae_config).to(device, dtype=weight_dtype)

        print("3. Loading UNet config and initializing UNet2DConditionModel with random weights...")
        unet_config = UNet2DConditionModel.load_config(base_ckpt, subfolder="unet")
        self.unet = UNet2DConditionModel.from_config(unet_config).to(device, dtype=weight_dtype)

        print("4. Initializing adapter (Skip Cross-Attention)...")
        init_adapter(self.unet, cross_attn_cls=SkipAttnProcessor)
        
        self.attn_modules = get_trainable_module(self.unet, "attention")
        
        print("5. Loading attention checkpoint...")
        self.auto_attn_ckpt_load(attn_ckpt, attn_ckpt_version)
        print("Pipeline successfully constructed!")

def main():
    base_model = "booksforcharlie/stable-diffusion-inpainting"
    resume_path = "zhengchong/CatVTON"
    
    # 1. Paths setup
    person_dir = os.path.join(project_root, "test_images", "persons")
    garment_dir = os.path.join(project_root, "test_images", "garments")
    debug_mask_dir = os.path.join(project_root, "test_images", "output_masked", "debug_masks")
    
    os.makedirs(person_dir, exist_ok=True)
    os.makedirs(garment_dir, exist_ok=True)
    os.makedirs(debug_mask_dir, exist_ok=True)
    
    # 2. Get list of files
    person_images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    garment_images = [f for f in os.listdir(garment_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not person_images:
        print(f"Error: No person images found in {person_dir}. Please place person images first.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 3. Generate debug masks to verify layout correctness
    print("\n=== Generating Debug Masks (All Categories) ===")
    sample_person_name = person_images[0]
    p_path = os.path.join(person_dir, sample_person_name)
    person_image = Image.open(p_path).convert("RGB")
    
    print(f"Running parsing on {sample_person_name}...")
    parsing_result = parse_human(person_image)
    
    saved_masks = {}
    for category in ['upper', 'lower', 'overall']:
        mask_image = build_mask_from_schp_segmentation(parsing_result, category)
        p_base = os.path.splitext(sample_person_name)[0]
        out_name = f"{p_base}_mask_{category}.png"
        out_path = os.path.join(debug_mask_dir, out_name)
        mask_image.save(out_path)
        saved_masks[category] = mask_image
        print(f" Saved {category} mask to: {out_path}")
        
    # 4. Download attention checkpoint (only allow mix-48k-1024 to save time and bandwidth)
    print(f"\nDownloading attention checkpoint from HuggingFace: {resume_path}...")
    repo_path = snapshot_download(repo_id=resume_path, allow_patterns=["mix-48k-1024/*"])
    print(f"Attention checkpoint downloaded to: {repo_path}")
    
    # 5. Build config-only pipeline
    print("\nAttempting to construct pipeline using CONFIG-ONLY (no base model weights download)...")
    try:
        pipeline = ConfigOnlyCatVTONPipeline(
            base_ckpt=base_model,
            attn_ckpt=repo_path,
            attn_ckpt_version="mix",
            weight_dtype=torch.float32,
            device=device,
            skip_safety_check=True
        )
        print("SUCCESS: Config-only pipeline constructed successfully!")
        
        # 6. Run a dry-run inference step to verify mathematical correctness of tensor shapes
        if garment_images:
            g_path = os.path.join(garment_dir, garment_images[0])
            garment_image = Image.open(g_path).convert("RGB")
            
            print("\nRunning a 1-step dry-run inference on CPU/GPU to verify pipeline step execution...")
            # We use standard size 768x1024
            result_image = pipeline(
                image=person_image,
                condition_image=garment_image,
                mask=saved_masks['upper'],
                num_inference_steps=1,
                guidance_scale=2.5,
                width=768,
                height=1024
            )[0]
            print(f"SUCCESS: Pipeline execution verified! Result shape: {result_image.size}")
            
    except Exception as e:
        print(f"\nERROR during pipeline construction/verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
