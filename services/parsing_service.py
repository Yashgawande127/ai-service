"""
Parsing Service
Responsible for human body part segmentations using Self-Correction for Human Parsing (SCHP).
"""

import os
import sys
import time
from pathlib import Path
from typing import Union, List

import cv2
import numpy as np
import torch
from PIL import Image

# Add schp_model to sys.path so its internal imports (e.g. networks, modules) resolve correctly
SCHP_MODEL_DIR = Path(__file__).resolve().parent.parent / "schp_model"
if str(SCHP_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(SCHP_MODEL_DIR))

# ATR labels mapping
ATR_LABELS = [
    'Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
    'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf'
]

class ParsingServiceError(Exception):
    """Base exception for all errors in ParsingService."""
    pass

class ImageLoadError(ParsingServiceError):
    """Raised when an image fails to load or parse."""
    pass


# Global model cache
_MODEL = None

def _get_model() -> torch.nn.Module:
    """
    Lazy-loads and caches the pretrained SCHP ResNet101 model.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    import networks

    # Initialize model with 18 classes (ATR schema)
    model = networks.init_model('resnet101', num_classes=18, pretrained=None)

    checkpoint_path = Path(__file__).resolve().parent.parent / "models" / "parsing" / "exp-schp-201908301523-atr.pth"
    if not checkpoint_path.exists():
        raise ParsingServiceError(f"Pretrained checkpoint not found at: {checkpoint_path}")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(str(checkpoint_path), map_location='cpu')['state_dict']
        
        # Adjust state_dict keys (remove 'module.' prefix from DDP training)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=True)
        model.to(device)
        model.eval()
        _MODEL = model
        return _MODEL
    except Exception as e:
        raise ParsingServiceError(f"Failed to load checkpoint into model: {str(e)}") from e


def _load_image(image_input: Union[str, Path, Image.Image, bytes]) -> np.ndarray:
    """
    Loads an image from multiple formats and returns a BGR image.
    """
    import requests
    try:
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            if path_str.startswith(("http://", "https://")):
                response = requests.get(path_str, timeout=10)
                response.raise_for_status()
                nparr = np.frombuffer(response.content, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    raise ImageLoadError(f"Failed to decode image from URL: {path_str}")
                return image
            else:
                if not os.path.exists(path_str):
                    raise ImageLoadError(f"File not found: {path_str}")
                image = cv2.imread(path_str)
                if image is None:
                    raise ImageLoadError(f"Failed to read image from path: {path_str}")
                return image

        elif isinstance(image_input, Image.Image):
            image_rgb = np.array(image_input.convert("RGB"))
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            return image_bgr

        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ImageLoadError("Failed to decode image from raw bytes.")
            return image

        else:
            raise ImageLoadError(f"Unsupported image input type: {type(image_input)}")
    except Exception as e:
        if not isinstance(e, ImageLoadError):
            raise ImageLoadError(f"Error loading image: {str(e)}") from e
        raise


def _box2cs(box, aspect_ratio):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, aspect_ratio)


def _xywh2cs(x, y, w, h, aspect_ratio):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w, h], dtype=np.float32)
    return center, scale


import contextlib

@contextlib.contextmanager
def _schp_import_context():
    import sys
    orig_path = list(sys.path)
    orig_utils = sys.modules.get('utils', None)
    
    # Remove any path ending with or containing 'catvton_model' to avoid collision
    sys.path = [p for p in sys.path if 'catvton_model' not in p]
    
    # Ensure schp_model is at the front of sys.path
    schp_model_dir = str(Path(__file__).resolve().parent.parent / "schp_model")
    if schp_model_dir not in sys.path:
        sys.path.insert(0, schp_model_dir)
        
    # Clear sys.modules['utils'] so Python searches for schp_model/utils
    if 'utils' in sys.modules:
        del sys.modules['utils']
        
    try:
        yield
    finally:
        sys.path = orig_path
        if orig_utils is not None:
            sys.modules['utils'] = orig_utils
        elif 'utils' in sys.modules:
            del sys.modules['utils']


def parse_human(image_input: Union[str, Path, Image.Image, bytes]) -> np.ndarray:
    """
    Parses a human image using SCHP and returns an 18-label segmentation mask.

    Args:
        image_input: Image filepath, URL, PIL Image, or raw bytes.

    Returns:
        np.ndarray: A 2D numpy array of shape (height, width) containing label indices (0-17).
    """
    with _schp_import_context():
        try:
            # Load image as BGR
            img = _load_image(image_input)
            h, w, _ = img.shape

            model = _get_model()
            device = next(model.parameters()).device

            # ATR uses 512x512 inputs
            input_size = [512, 512]
            aspect_ratio = input_size[1] * 1.0 / input_size[0]

            person_center, s = _box2cs([0, 0, w - 1, h - 1], aspect_ratio)
            r = 0

            from utils.transforms import get_affine_transform, transform_logits

            trans = get_affine_transform(person_center, s, r, np.asarray(input_size))
            input_img = cv2.warpAffine(
                img,
                trans,
                (int(input_size[1]), int(input_size[0])),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
            ])

            input_tensor = transform(input_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                
                # The model returns: [[parsing_result, fusion_result], [edge_result]]
                # We want fusion_result, which is the last item of output[0]
                parsing_logits = output[0][-1]  # shape [1, 18, 128, 128]
                
                # Bilinear upsample logits to network input_size (512x512)
                upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
                upsample_output = upsample(parsing_logits[0].unsqueeze(0))  # shape [1, 18, 512, 512]
                upsample_output = upsample_output.squeeze().permute(1, 2, 0)  # shape [512, 512, 18]

                # Affine transform logits back to original image size
                logits_result = transform_logits(
                    upsample_output.cpu().numpy(),
                    person_center,
                    s,
                    w,
                    h,
                    input_size=input_size
                )
                parsing_result = np.argmax(logits_result, axis=2)

            return parsing_result

        except Exception as e:
            if not isinstance(e, ParsingServiceError):
                raise ParsingServiceError(f"Error during human parsing: {str(e)}") from e
            raise


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask. """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def visualize_parsing(parsing_result: np.ndarray, num_classes: int = 18) -> Image.Image:
    """
    Applies the ATR color palette to the 2D parsing map and returns a PIL Image.
    """
    try:
        palette = get_palette(num_classes)
        output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
        output_img.putpalette(palette)
        return output_img
    except Exception as e:
        raise ParsingServiceError(f"Failed to visualize parsing result: {str(e)}") from e


if __name__ == "__main__":
    # Standalone test block on test_images/sample_person.jpg
    test_img_path = Path(__file__).resolve().parent.parent / "test_images" / "sample_person.jpg"
    print(f"Running standalone parsing test on: {test_img_path}")
    
    if not test_img_path.exists():
        print(f"Error: Test image not found at {test_img_path}")
        sys.exit(1)
        
    start_time = time.time()
    try:
        parsing_mask = parse_human(test_img_path)
        inference_time = time.time() - start_time
        
        print("\n=== Test Results ===")
        print(f"Inference Time (including load): {inference_time:.4f} seconds")
        print(f"Output shape: {parsing_mask.shape}")
        
        unique_indices = np.unique(parsing_mask)
        print("Detected Labels:")
        for idx in unique_indices:
            label = ATR_LABELS[idx] if idx < len(ATR_LABELS) else f"Unknown ({idx})"
            count = np.sum(parsing_mask == idx)
            print(f"  - Label {idx} ({label}): {count} pixels")
            
        # Save visualization
        output_img = visualize_parsing(parsing_mask)
        out_path = test_img_path.parent / "sample_person_parsing_test.png"
        output_img.save(out_path)
        print(f"Saved visualization to: {out_path}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
