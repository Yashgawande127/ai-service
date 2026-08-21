"""
Pose Service
Responsible for body pose detection and landmark estimation using MediaPipe.
"""

import os
from pathlib import Path
from typing import Union, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class PoseServiceError(Exception):
    """Base exception for all errors in PoseService."""
    pass


class ImageLoadError(PoseServiceError):
    """Raised when an image fails to load or parse."""
    pass


class PoseDetectionError(PoseServiceError):
    """Raised when human pose detection fails or critical landmarks are not visible."""
    pass


def _load_image(image_input: Union[str, Path, Image.Image, bytes]) -> np.ndarray:
    """
    Helper function to load an image from various input types and return a BGR image.
    
    Supported input types:
    - str or Path: Local path to an image file.
    - PIL.Image.Image: A PIL Image object.
    - bytes: Raw image byte content.
    
    Returns:
        np.ndarray: BGR image suitable for OpenCV processing.
        
    Raises:
        ImageLoadError: If the image cannot be loaded or resolved.
    """
    try:
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            if not os.path.exists(path_str):
                raise ImageLoadError(f"File not found: {path_str}")
            image = cv2.imread(path_str)
            if image is None:
                raise ImageLoadError(f"Failed to read image from path: {path_str}")
            return image

        elif isinstance(image_input, Image.Image):
            # PIL Image conversion
            # Convert PIL image to BGR numpy array
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
            raise ImageLoadError(
                f"Unsupported image input type: {type(image_input)}. "
                f"Must be a local path (str/Path), PIL Image, or raw bytes."
            )
    except Exception as e:
        if not isinstance(e, ImageLoadError):
            raise ImageLoadError(f"Error loading image: {str(e)}") from e
        raise


def detect_pose(
    image_input: Union[str, Path, Image.Image, bytes],
    confidence_threshold: float = 0.5,
    raise_on_failure: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Detects 33 body landmarks using MediaPipe Pose solution in static image mode.
    
    Accepts:
    - A local file path (str or pathlib.Path)
    - A PIL Image object (PIL.Image.Image)
    - Raw image bytes (bytes)
    
    Returns:
        A dictionary containing key landmarks for garment placement and all landmarks list:
        {
            "left_shoulder": {"x": float, "y": float, "z": float, "visibility": float},
            "right_shoulder": {"x": float, "y": float, "z": float, "visibility": float},
            "left_hip": {"x": float, "y": float, "z": float, "visibility": float},
            "right_hip": {"x": float, "y": float, "z": float, "visibility": float},
            "nose": {"x": float, "y": float, "z": float, "visibility": float},
            "all_landmarks": [
                {"x": float, "y": float, "z": float, "visibility": float},
                ... # 33 items
            ]
        }
        Or None if pose detection fails (when raise_on_failure is False).
        
    Raises:
        ImageLoadError: If image loading fails.
        PoseDetectionError: If no pose is detected or if critical landmarks are below the confidence threshold.
        PoseServiceError: For other general MediaPipe initialization or execution errors.
    """
    # Load image in BGR format
    image_bgr = _load_image(image_input)
    
    # Convert BGR to RGB since MediaPipe requires RGB images
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    try:
        # Run MediaPipe Pose solution in static image mode
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        ) as pose:
            results = pose.process(image_rgb)
    except Exception as e:
        raise PoseServiceError(f"MediaPipe Pose solution failed to initialize or execute: {str(e)}") from e
        
    # Check if pose was detected
    if not results.pose_landmarks:
        if raise_on_failure:
            raise PoseDetectionError("No person or pose detected in the image.")
        return None
        
    landmarks = results.pose_landmarks.landmark
    
    # Check that we have all 33 landmarks
    if len(landmarks) < 33:
        if raise_on_failure:
            raise PoseDetectionError(
                f"MediaPipe returned only {len(landmarks)} landmarks, expected 33."
            )
        return None
        
    # Identify indices for key landmarks
    # nose: 0, left_shoulder: 11, right_shoulder: 12, left_hip: 23, right_hip: 24
    key_landmark_indices = {
        "nose": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_hip": 23,
        "right_hip": 24
    }
    
    # Check visibility threshold for critical landmarks (shoulders and hips)
    critical_landmarks = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    for name in critical_landmarks:
        idx = key_landmark_indices[name]
        visibility = landmarks[idx].visibility
        if visibility < confidence_threshold:
            if raise_on_failure:
                raise PoseDetectionError(
                    f"Critical landmark '{name}' has visibility {visibility:.2f}, "
                    f"which is below the threshold of {confidence_threshold}."
                )
            return None
            
    # Extract key landmarks into a clean format
    result_dict = {}
    for name, idx in key_landmark_indices.items():
        lm = landmarks[idx]
        result_dict[name] = {
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "visibility": float(lm.visibility)
        }
        
    # Add all 33 landmarks list
    result_dict["all_landmarks"] = [
        {
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "visibility": float(lm.visibility)
        }
        for lm in landmarks
    ]
    
    return result_dict


def draw_landmarks_debug(
    image_input: Union[str, Path, Image.Image, bytes],
    output_path: Union[str, Path]
) -> None:
    """
    Draws pose landmarks and connection lines on the image and saves the result.
    This is for visual debugging during development only.
    
    Raises:
        ImageLoadError: If image loading fails.
        PoseDetectionError: If pose detection fails.
        PoseServiceError: For other errors.
    """
    # Load image in BGR format
    image_bgr = _load_image(image_input)
    
    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    try:
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        ) as pose:
            results = pose.process(image_rgb)
    except Exception as e:
        raise PoseServiceError(f"MediaPipe Pose solution failed to initialize or execute: {str(e)}") from e
        
    if not results.pose_landmarks:
        raise PoseDetectionError("No person or pose detected in the image; cannot draw debug landmarks.")
        
    # Draw the landmarks on a copy of the original BGR image
    annotated_image = image_bgr.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )
    
    # Make sure output directory exists
    output_dir = os.path.dirname(str(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    # Save the annotated image
    success = cv2.imwrite(str(output_path), annotated_image)
    if not success:
        raise PoseServiceError(f"Failed to save debug image to {output_path}")


if __name__ == "__main__":
    import json
    
    # Path to test image
    test_dir = Path("test_images")
    test_image_path = test_dir / "sample_person.jpg"
    debug_image_path = test_dir / "sample_person_debug.jpg"
    
    print("Pose Service Standalone Test Running...")
    print(f"Expected test image path: {test_image_path.resolve()}")
    
    if not test_image_path.exists():
        print("\n" + "="*80)
        print("IMPORTANT NOTICE FOR THE USER:")
        print(f"Please place a real full-body photo of a person at:")
        print(f"  {test_image_path.resolve()}")
        print("Then run this script again using:")
        print("  python services/pose_service.py")
        print("="*80 + "\n")
    else:
        print(f"Found test image at {test_image_path}. Running pose detection...")
        try:
            landmarks_dict = detect_pose(test_image_path)
            print("\nPose Detection Successful! Key landmarks extracted:")
            
            # Print only key landmarks without the verbose 'all_landmarks' list
            key_landmarks_only = {
                k: v for k, v in landmarks_dict.items() if k != "all_landmarks"
            }
            print(json.dumps(key_landmarks_only, indent=2))
            print(f"Total landmarks detected: {len(landmarks_dict['all_landmarks'])}")
            
            print(f"\nDrawing debug landmarks and saving to {debug_image_path}...")
            draw_landmarks_debug(test_image_path, debug_image_path)
            print("Debug image saved successfully. Check the file to verify landmark placement!")
            
        except Exception as e:
            print(f"\nPose detection failed with error:")
            print(f"  {type(e).__name__}: {str(e)}")
