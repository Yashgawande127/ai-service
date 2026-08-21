import os
import sys
import torch
from huggingface_hub import snapshot_download

# Add catvton_model to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
catvton_dir = os.path.join(project_root, "catvton_model")
sys.path.insert(0, catvton_dir)

from model.pipeline import CatVTONPix2PixPipeline
from model.attn_processor import SkipAttnProcessor
from model.utils import get_trainable_module, init_adapter
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel

class ConfigOnlyCatVTONPix2PixPipeline(CatVTONPix2PixPipeline):
    def __init__(
        self, 
        base_ckpt, 
        attn_ckpt, 
        attn_ckpt_version="mix-48k-1024",
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
        
        print("5. Loading attention checkpoint (safetensors)...")
        self.auto_attn_ckpt_load(attn_ckpt, attn_ckpt_version)
        print("Pipeline successfully constructed!")

def main():
    base_model = "timbrooks/instruct-pix2pix"
    resume_path = "zhengchong/CatVTON-MaskFree"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Downloading attention checkpoint from HuggingFace: {resume_path}...")
    repo_path = snapshot_download(repo_id=resume_path)
    print(f"Attention checkpoint downloaded to: {repo_path}")
    
    print("\nAttempting to construct pipeline using CONFIG-ONLY (no base model weights download)...")
    try:
        pipeline = ConfigOnlyCatVTONPix2PixPipeline(
            base_ckpt=base_model,
            attn_ckpt=repo_path,
            attn_ckpt_version="mix-48k-1024",
            weight_dtype=torch.float32,
            device=device,
            skip_safety_check=True
        )
        print("\nSUCCESS: Config-only pipeline constructed successfully!")
    except Exception as e:
        print(f"\nERROR during config-only construction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
