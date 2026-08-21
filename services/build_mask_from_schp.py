import os
import sys
import numpy as np
import cv2
from PIL import Image
from typing import Union

# Add parent directory of services to sys.path so we can import services.parsing_service
services_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(services_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from services.parsing_service import parse_human, ATR_LABELS

# Create label-to-index mapping for fast lookups
ATR_MAPPING = {label: idx for idx, label in enumerate(ATR_LABELS)}

# Map categories to the target ATR parts to mask (what we want to replace)
MASK_LABELS = {
    'upper': ['Upper-clothes', 'Dress', 'Left-arm', 'Right-arm'],
    'lower': ['Skirt', 'Pants', 'Dress', 'Left-leg', 'Right-leg'],
    'overall': ['Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Belt', 'Scarf']
}

# Map categories to the ATR parts we must protect (strictly exclude from mask)
PROTECT_LABELS = {
    'upper': ['Background', 'Face', 'Hair', 'Hat', 'Sunglasses', 'Bag', 'Left-shoe', 'Right-shoe', 'Skirt', 'Pants', 'Belt', 'Scarf'],
    'lower': ['Background', 'Face', 'Hair', 'Hat', 'Sunglasses', 'Bag', 'Left-shoe', 'Right-shoe', 'Upper-clothes', 'Left-arm', 'Right-arm', 'Belt', 'Scarf'],
    'overall': ['Background', 'Face', 'Hair', 'Hat', 'Sunglasses', 'Bag', 'Left-shoe', 'Right-shoe']
}

def part_mask_of(parts: list, parse: np.ndarray, mapping: dict = ATR_MAPPING) -> np.ndarray:
    """Creates a boolean mask indicating the presence of specified parts."""
    mask = np.zeros_like(parse, dtype=bool)
    for part in parts:
        if part not in mapping:
            continue
        mask |= (parse == mapping[part])
    return mask

def hull_mask(mask_area: np.ndarray) -> np.ndarray:
    """Computes the convex hull of contours to bridge gaps like neck/arms."""
    ret, binary = cv2.threshold(mask_area, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_img = np.zeros_like(mask_area)
    for c in contours:
        hull = cv2.convexHull(c)
        cv2.fillPoly(hull_img, [hull], 255)
    return hull_img

def build_mask_from_schp_segmentation(parsing_result: np.ndarray, category: str = 'upper') -> Image.Image:
    """
    Constructs a CatVTON-compatible try-on mask from the SCHP ATR segmentation.
    This logic implements the exact step-order, kernels, and thresholds used by 
    the official cloth_masker.py, adapted to run without DensePose.
    
    Args:
        parsing_result (np.ndarray): 2D array of ATR label indices (0 to 17)
        category (str): The garment category - 'upper', 'lower', or 'overall'
        
    Returns:
        Image.Image: Grayscale mask image where white (255) is the region to replace
    """
    assert category in ['upper', 'lower', 'overall'], f"Category must be 'upper', 'lower', or 'overall'. Got '{category}'"
    h, w = parsing_result.shape[:2]
    
    # 1. Define dynamic dilation and blur kernels (matching cloth_masker.py)
    dilate_kernel_size = max(w, h) // 250
    dilate_kernel_size = dilate_kernel_size if dilate_kernel_size % 2 == 1 else dilate_kernel_size + 1
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    
    blur_kernel_size = max(w, h) // 25
    blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
    
    # 2. Extract mask and protect regions
    mask_area = part_mask_of(MASK_LABELS[category], parsing_result, ATR_MAPPING)
    protect_area = part_mask_of(PROTECT_LABELS[category], parsing_result, ATR_MAPPING)
    
    # 3. Convex Hull (matching cloth_masker.py line 242)
    # Convert mask_area to 0 or 255 uint8
    mask_area_uint8 = (mask_area * 255).astype(np.uint8)
    hulled_mask = hull_mask(mask_area_uint8)
    
    # Subtract protect_area (before blurring)
    hulled_mask[protect_area] = 0
    
    # 4. Gaussian Blur (matching cloth_masker.py line 244)
    blurred = cv2.GaussianBlur(hulled_mask, (blur_kernel_size, blur_kernel_size), 0)
    
    # Threshold at 25 (matching cloth_masker.py line 245)
    thresholded_mask = np.zeros_like(blurred)
    thresholded_mask[blurred >= 25] = 255
    thresholded_mask[protect_area] = 0
    
    # 5. Final dilation at the end (matching cloth_masker.py line 248)
    final_mask = cv2.dilate(thresholded_mask, dilate_kernel, iterations=1)
    
    # Final cleanup to ensure no bleed-through into protected regions
    final_mask[protect_area] = 0
    
    return Image.fromarray(final_mask)

def build_mask_from_image(image_input: Union[str, Image.Image, np.ndarray], category: str = 'upper') -> Image.Image:
    """
    Convenience wrapper that performs human parsing and constructs the try-on mask.
    
    Args:
        image_input: File path, PIL Image, or numpy array of the person image
        category (str): The garment category - 'upper', 'lower', or 'overall'
        
    Returns:
        Image.Image: Grayscale mask image where white (255) is the region to replace
    """
    parsing_result = parse_human(image_input)
    return build_mask_from_schp_segmentation(parsing_result, category)
