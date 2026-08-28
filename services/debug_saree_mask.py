import os
import sys
import numpy as np
from PIL import Image

services_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(services_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from services.parsing_service import parse_human, ATR_LABELS, visualize_parsing
from services.build_mask_from_schp import build_mask_from_schp_segmentation, MASK_LABELS, PROTECT_LABELS

def analyze_person(person_file):
    person_path = os.path.join(project_root, "test_images", "persons", person_file)
    print(f"\n=======================================================")
    print(f"DIAGNOSTIC ANALYSIS FOR: {person_file}")
    print(f"=======================================================")
    
    img = Image.open(person_path).convert("RGB")
    w, h = img.size
    print(f"Original Image Size: {w} x {h} (Width x Height)")
    
    # 1. Run SCHP raw parsing
    parse_result = parse_human(img)
    print(f"SCHP Parsing Shape: {parse_result.shape} (Height x Width)")
    
    # 2. Count and list all detected ATR labels
    unique_labels, counts = np.unique(parse_result, return_counts=True)
    total_pixels = parse_result.size
    
    print("\n--- SCHP Detected Body-Part / Clothing Labels ---")
    detected_parts = []
    for idx, count in zip(unique_labels, counts):
        label_name = ATR_LABELS[idx] if idx < len(ATR_LABELS) else f"Unknown({idx})"
        pct = (count / total_pixels) * 100
        detected_parts.append((idx, label_name, count, pct))
        print(f"  [Label {idx:02d}] {label_name:<16} : {count:>8} pixels ({pct:5.2f}%)")
        
    # 3. Overall Category Analysis (used for Saree / Lehenga)
    overall_mask_targets = MASK_LABELS['overall']
    overall_protect_targets = PROTECT_LABELS['overall']
    
    print("\n--- Category: 'overall' (Used for Saree & Lehenga) ---")
    print(f"Target Labels to Replace (Mask = White): {overall_mask_targets}")
    print(f"Protected Labels (Keep Original = Black): {overall_protect_targets}")
    
    active_mask_labels = [label for idx, label, count, pct in detected_parts if label in overall_mask_targets]
    print(f"\nActual Labels Present in this Person's Mask Region: {active_mask_labels}")
    
    # 4. Generate the exact mask image used by CatVTON
    mask_img = build_mask_from_schp_segmentation(parse_result, category='overall')
    mask_arr = np.array(mask_img)
    white_pixels = np.sum(mask_arr > 127)
    white_pct = (white_pixels / mask_arr.size) * 100
    print(f"Total Editable/Inpainting Area in Final Mask: {white_pixels} pixels ({white_pct:.2f}% of image)")
    
    # 5. Save the diagnostic masks
    out_dir = os.path.join(project_root, "test_images", "output_masked")
    os.makedirs(out_dir, exist_ok=True)
    
    # Save the primary debug mask requested by user
    saree_mask_path = os.path.join(out_dir, "saree_mask_debug.png")
    mask_img.save(saree_mask_path)
    print(f"\nSaved editable try-on mask to: {saree_mask_path}")
    
    # Also save person-specific masks and full color segmentation for visual clarity
    person_base = os.path.splitext(person_file)[0]
    p_mask_path = os.path.join(out_dir, f"{person_base}_saree_mask_debug.png")
    mask_img.save(p_mask_path)
    
    color_seg = visualize_parsing(parse_result)
    color_seg_path = os.path.join(out_dir, f"{person_base}_schp_color_segmentation.png")
    color_seg.save(color_seg_path)
    print(f"Saved colored SCHP segmentation to: {color_seg_path}")
    
    return detected_parts, active_mask_labels, white_pct

if __name__ == "__main__":
    print("Running Saree Mask Diagnostic Tool...")
    # Analyze demo2 (main user focus) and demo1
    analyze_person("demo2.jpg")
    analyze_person("demo1.jpg")
