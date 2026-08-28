import os
import sys
import time
import gc
import torch
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()
from huggingface_hub import snapshot_download

# Paths setup
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

def get_garment_category(filename: str) -> str:
    name = filename.lower()
    if any(k in name for k in ["pants", "skirt", "jeans", "lower", "trouser", "shorts"]):
        return "lower"
    elif any(k in name for k in ["saree", "sari", "lehenga", "dress", "overall", "gown", "jumpsuit", "anarkali"]):
        return "overall"
    elif any(k in name for k in ["kurti", "kurta", "shirt", "t-shirt", "blouse", "top", "upper"]):
        return "upper"
    return "overall"

def get_vram_info():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        return f"Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB"
    return "N/A (CPU)"

def run_maskfree_job(person_path, garment_path, out_path, pipeline, width=768, height=1024):
    person_img = Image.open(person_path).convert("RGB")
    garment_img = Image.open(garment_path).convert("RGB")
    
    try:
        t0 = time.time()
        result = pipeline(
            image=person_img,
            condition_image=garment_img,
            num_inference_steps=50,
            guidance_scale=2.5,
            width=width,
            height=height
        )[0]
        elapsed = time.time() - t0
        result.save(out_path)
        return True, f"{width}x{height}", elapsed, None
    except torch.cuda.OutOfMemoryError as e:
        print(f"  [OOM Warning] Out of memory at {width}x{height}. Retrying at 512x768 fallback...")
        torch.cuda.empty_cache()
        gc.collect()
        try:
            t0 = time.time()
            result = pipeline(
                image=person_img,
                condition_image=garment_img,
                num_inference_steps=50,
                guidance_scale=2.5,
                width=512,
                height=768
            )[0]
            elapsed = time.time() - t0
            result.save(out_path)
            return True, "512x768 (OOM fallback)", elapsed, None
        except Exception as e2:
            return False, "Failed", 0, str(e2)
    except Exception as e:
        return False, "Failed", 0, str(e)

def run_masked_job(person_path, garment_path, out_path, parsing_result, category, pipeline, width=768, height=1024):
    person_img = Image.open(person_path).convert("RGB")
    garment_img = Image.open(garment_path).convert("RGB")
    mask_img = build_mask_from_schp_segmentation(parsing_result, category)
    
    # Save debug mask
    debug_mask_dir = os.path.join(project_root, "test_images", "output_masked", "debug_masks")
    os.makedirs(debug_mask_dir, exist_ok=True)
    p_base = os.path.splitext(os.path.basename(person_path))[0]
    mask_img.save(os.path.join(debug_mask_dir, f"{p_base}_mask_{category}.png"))
    
    try:
        t0 = time.time()
        result = pipeline(
            image=person_img,
            condition_image=garment_img,
            mask=mask_img,
            num_inference_steps=50,
            guidance_scale=2.5,
            width=width,
            height=height
        )[0]
        elapsed = time.time() - t0
        result.save(out_path)
        return True, f"{width}x{height}", elapsed, None
    except torch.cuda.OutOfMemoryError as e:
        print(f"  [OOM Warning] Out of memory at {width}x{height}. Retrying at 512x768 fallback...")
        torch.cuda.empty_cache()
        gc.collect()
        try:
            t0 = time.time()
            result = pipeline(
                image=person_img,
                condition_image=garment_img,
                mask=mask_img,
                num_inference_steps=50,
                guidance_scale=2.5,
                width=512,
                height=768
            )[0]
            elapsed = time.time() - t0
            result.save(out_path)
            return True, "512x768 (OOM fallback)", elapsed, None
        except Exception as e2:
            return False, "Failed", 0, str(e2)
    except Exception as e:
        return False, "Failed", 0, str(e)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting Ethnic Wear Virtual Try-On Pipeline ===")
    print(f"Device: {device} | Initial VRAM: {get_vram_info()}")
    
    person_dir = os.path.join(project_root, "test_images", "persons")
    garment_dir = os.path.join(project_root, "test_images", "garments")
    out_maskfree_dir = os.path.join(project_root, "test_images", "output")
    out_masked_dir = os.path.join(project_root, "test_images", "output_masked")
    
    os.makedirs(out_maskfree_dir, exist_ok=True)
    os.makedirs(out_masked_dir, exist_ok=True)
    
    # Target ethnic garments and target persons
    ethnic_garments = ["saree_01.png", "lehenga_01.png", "kurti_01.png"]
    target_persons = ["demo2.jpg", "demo1.jpg"]
    
    # Filter files that exist
    garment_files = [g for g in ethnic_garments if os.path.exists(os.path.join(garment_dir, g))]
    person_files = [p for p in target_persons if os.path.exists(os.path.join(person_dir, p))]
    
    print(f"Selected Ethnic Garments: {garment_files}")
    print(f"Selected Persons: {person_files}")
    
    results_log = []
    
    # -------------------------------------------------------------
    # PHASE 1: Mask-Free Pipeline Execution
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("PHASE 1: Initializing Mask-Free CatVTON Pipeline...")
    print("="*50)
    
    maskfree_attn_repo = snapshot_download("zhengchong/CatVTON-MaskFree")
    maskfree_pipeline = CatVTONPix2PixPipeline(
        base_ckpt="timbrooks/instruct-pix2pix",
        attn_ckpt=maskfree_attn_repo,
        attn_ckpt_version="mix-48k-1024",
        weight_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=device,
        skip_safety_check=True
    )
    print(f"Mask-Free Pipeline Loaded. VRAM: {get_vram_info()}")
    
    for p_name in person_files:
        p_path = os.path.join(person_dir, p_name)
        p_base = os.path.splitext(p_name)[0]
        
        for g_name in garment_files:
            g_path = os.path.join(garment_dir, g_name)
            g_base = os.path.splitext(g_name)[0]
            out_name = f"{p_base}_on_{g_base}.png"
            out_path = os.path.join(out_maskfree_dir, out_name)
            
            print(f"\n[Mask-Free] Running: {p_name} + {g_name} -> {out_name}")
            ok, res, elapsed, err = run_maskfree_job(p_path, g_path, out_path, maskfree_pipeline)
            print(f"  Result: ok={ok}, res={res}, elapsed={elapsed:.1f}s, VRAM: {get_vram_info()}")
            results_log.append({
                "person": p_name,
                "garment": g_name,
                "pipeline": "Mask-Free",
                "output_path": out_path,
                "resolution": res,
                "elapsed": f"{elapsed:.1f}s",
                "status": "Success" if ok else f"Error: {err}"
            })
            torch.cuda.empty_cache()
            gc.collect()

    # Free Mask-Free Pipeline before loading Masked Pipeline to prevent VRAM accumulation
    del maskfree_pipeline
    torch.cuda.empty_cache()
    gc.collect()
    print(f"\nFreed Mask-Free Pipeline from VRAM. Current VRAM: {get_vram_info()}")

    # -------------------------------------------------------------
    # PHASE 2: Masked Pipeline Execution
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("PHASE 2: Initializing Masked CatVTON Pipeline + SCHP Parsing...")
    print("="*50)
    
    masked_attn_repo = snapshot_download("zhengchong/CatVTON", allow_patterns=["mix-48k-1024/*"])
    masked_pipeline = CatVTONPipeline(
        base_ckpt="booksforcharlie/stable-diffusion-inpainting",
        attn_ckpt=masked_attn_repo,
        attn_ckpt_version="mix",
        weight_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=device,
        skip_safety_check=True
    )
    print(f"Masked Pipeline Loaded. VRAM: {get_vram_info()}")
    
    # Pre-parse each person once
    person_parses = {}
    for p_name in person_files:
        p_path = os.path.join(person_dir, p_name)
        print(f"Parsing human body parts for {p_name}...")
        img = Image.open(p_path).convert("RGB")
        parse_res = parse_human(img)
        person_parses[p_name] = parse_res
        print(f"  SCHP parsing done. Shape: {parse_res.shape}")
        
    for p_name in person_files:
        p_path = os.path.join(person_dir, p_name)
        p_base = os.path.splitext(p_name)[0]
        parse_res = person_parses[p_name]
        
        for g_name in garment_files:
            g_path = os.path.join(garment_dir, g_name)
            g_base = os.path.splitext(g_name)[0]
            category = get_garment_category(g_name)
            out_name = f"{p_base}_on_{g_base}.png"
            out_path = os.path.join(out_masked_dir, out_name)
            
            print(f"\n[Masked] Running: {p_name} + {g_name} (Category: {category}) -> {out_name}")
            ok, res, elapsed, err = run_masked_job(p_path, g_path, out_path, parse_res, category, masked_pipeline)
            print(f"  Result: ok={ok}, res={res}, elapsed={elapsed:.1f}s, VRAM: {get_vram_info()}")
            results_log.append({
                "person": p_name,
                "garment": g_name,
                "pipeline": f"Masked ({category})",
                "output_path": out_path,
                "resolution": res,
                "elapsed": f"{elapsed:.1f}s",
                "status": "Success" if ok else f"Error: {err}"
            })
            torch.cuda.empty_cache()
            gc.collect()

    del masked_pipeline
    torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "="*50)
    print("=== SUMMARY OF INFERENCE RUNS ===")
    print("="*50)
    print(f"{'Person':<10} | {'Garment':<14} | {'Pipeline':<18} | {'Resolution':<12} | {'Time':<8} | {'Status'}")
    print("-" * 80)
    for r in results_log:
        print(f"{r['person']:<10} | {r['garment']:<14} | {r['pipeline']:<18} | {r['resolution']:<12} | {r['elapsed']:<8} | {r['status']}")

if __name__ == "__main__":
    main()
