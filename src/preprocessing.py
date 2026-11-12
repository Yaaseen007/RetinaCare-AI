"""
==============================================================================
Diabetic Retinopathy Detection - Image Preprocessing Pipeline
Department of Computer Science & Engineering - GM University

This module implements a standardized, reproducible preprocessing pipeline
for retinal fundus images. It is used identically during:
1. Model training and validation data feeding
2. Offline model evaluation and benchmarking
3. Real-time web application inference

Key Clinical Preprocessing Steps:
---------------------------------
1. Input Format Validation & Security Check
2. RGB Standardization (handling Grayscale, RGBA, BGR)
3. Circular Retinal Crop: Eliminates uninformative black camera borders
4. Illumination Normalization (Ben Graham's Retinal Filtering Method):
   Fundus cameras produce non-uniform lighting across the curved spherical retina.
   Ben Graham's method subtracts local Gaussian blurred background to highlight
   retinal biomarkers (microaneurysms, hemorrhages, cotton wool spots).
5. High-Fidelity Resizing to 224x224
6. Normalized Float32 scaling to [0.0, 1.0]
==============================================================================
"""

import io
import os
from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image
import cv2

# Import configuration constants
try:
    from utils.config import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE, VALID_EXTENSIONS
except ImportError:
    # Fallback default constants
    IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
    IMAGE_SIZE = (224, 224)
    VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class InvalidImageError(ValueError):
    """Raised when an uploaded or loaded image does not meet quality/format criteria."""
    pass


def validate_image_input(image_input: Union[str, Path, bytes, io.BytesIO, np.ndarray, Image.Image]) -> None:
    """
    Validates input type and basic constraints before processing.
    
    Args:
        image_input: File path, byte stream, PIL Image, or Numpy array.
        
    Raises:
        InvalidImageError: If the input is empty or of unsupported type.
    """
    if image_input is None:
        raise InvalidImageError("No image data provided.")

    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.is_file():
            raise InvalidImageError(f"Image file does not exist: {p}")
        if p.suffix.lower() not in VALID_EXTENSIONS:
            raise InvalidImageError(
                f"Unsupported file format '{p.suffix}'. Supported formats: {', '.join(VALID_EXTENSIONS)}"
            )
        if p.stat().st_size == 0:
            raise InvalidImageError(f"File '{p.name}' is empty (0 bytes).")

    elif isinstance(image_input, (bytes, bytearray)):
        if len(image_input) == 0:
            raise InvalidImageError("Provided byte buffer is empty.")

    elif isinstance(image_input, io.BytesIO):
        if image_input.getbuffer().nbytes == 0:
            raise InvalidImageError("Provided BytesIO stream is empty.")

    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0 or image_input.ndim not in (2, 3):
            raise InvalidImageError(f"Invalid numpy array shape: {image_input.shape}")

    elif not isinstance(image_input, Image.Image):
        raise InvalidImageError(f"Unsupported image input type: {type(image_input)}")


def load_image_as_rgb_array(image_input: Union[str, Path, bytes, io.BytesIO, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Loads and converts any image source into a standard uint8 RGB NumPy array.
    
    Returns:
        np.ndarray: RGB image with dtype uint8 and shape (H, W, 3).
    """
    validate_image_input(image_input)

    try:
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input))
            rgb_img = pil_img.convert("RGB")
            return np.array(rgb_img, dtype=np.uint8)

        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input))
            rgb_img = pil_img.convert("RGB")
            return np.array(rgb_img, dtype=np.uint8)

        elif isinstance(image_input, io.BytesIO):
            image_input.seek(0)
            pil_img = Image.open(image_input)
            rgb_img = pil_img.convert("RGB")
            return np.array(rgb_img, dtype=np.uint8)

        elif isinstance(image_input, Image.Image):
            rgb_img = image_input.convert("RGB")
            return np.array(rgb_img, dtype=np.uint8)

        elif isinstance(image_input, np.ndarray):
            arr = image_input.copy()
            if arr.ndim == 2:
                # Grayscale to RGB
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                # RGBA to RGB
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            elif arr.ndim == 3 and arr.shape[2] == 1:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            
            # Ensure uint8
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr

        else:
            raise InvalidImageError(f"Cannot decode image type {type(image_input)}")

    except Exception as exc:
        raise InvalidImageError(f"Failed to decode retinal image: {str(exc)}") from exc


def crop_retina_circle(image_rgb: np.ndarray, tolerance: int = 10) -> np.ndarray:
    """
    Crops black padding around the circular retinal fundus image.
    
    Fundus cameras typically capture images through a circular lens aperture,
    leaving large black borders that waste convolutional receptive field capacity.
    This function detects the bounding box of the circular retina and crops to it.
    
    Args:
        image_rgb: Input RGB image (H, W, 3).
        tolerance: Pixel intensity threshold to distinguish retina from background.
        
    Returns:
        np.ndarray: Cropped RGB image containing the retinal region.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > tolerance

    # Check if there is an actual foreground
    if not np.any(mask):
        return image_rgb

    # Find rows and columns with foreground pixels
    row_mask = np.any(mask, axis=1)
    col_mask = np.any(mask, axis=0)

    rmin, rmax = np.where(row_mask)[0][[0, -1]]
    cmin, cmax = np.where(col_mask)[0][[0, -1]]

    # Ensure box dimensions are valid
    if rmax - rmin < 20 or cmax - cmin < 20:
        return image_rgb

    return image_rgb[rmin : rmax + 1, cmin : cmax + 1]


def apply_ben_graham_enhancement(image_rgb: np.ndarray, sigmaX: int = 10) -> np.ndarray:
    """
    Applies Ben Graham's method (standard in Kaggle Diabetic Retinopathy competitions).
    
    Formula:
        enhanced = 4 * image - 4 * GaussianBlur(image) + 128
        
    Medical Rationale:
    1. Compensates for varying illumination across different fundus camera models.
    2. Subtraction of the local low-frequency Gaussian background amplifies
       high-frequency retinal microvascular lesions (microaneurysms, hard exudates,
       hemorrhages, and neovascular loops).
       
    Args:
        image_rgb: Cropped RGB retinal image.
        sigmaX: Standard deviation for Gaussian kernel.
        
    Returns:
        np.ndarray: Illumination-corrected, enhanced RGB image.
    """
    # Create smooth local background estimate
    blur = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX)
    
    # Weighted difference highlighting high-frequency microvascular textures
    enhanced = cv2.addWeighted(image_rgb, 4, blur, -4, 128)
    
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def apply_clahe_enhancement(image_rgb: np.ndarray) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the LAB
    Luminance channel.
    
    Preserves true retinal chromaticity while enhancing low-contrast vascular regions.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)


def preprocess_retinal_image(
    image_input: Union[str, Path, bytes, io.BytesIO, np.ndarray, Image.Image],
    target_size: Tuple[int, int] = IMAGE_SIZE,
    crop_circle: bool = True,
    enhance: bool = True,
    normalize: bool = True
) -> np.ndarray:
    """
    Unified, end-to-end preprocessing pipeline for training, evaluation, and inference.
    
    Steps:
    1. Validate input source
    2. Convert to standard RGB numpy array
    3. Crop uninformative black outer margins (optional, True by default)
    4. Apply Ben Graham's contrast & illumination enhancement (optional, True by default)
    5. Resize to target dimension (224, 224) using area-interpolation
    6. Normalize pixels to [0.0, 1.0] float32
    
    Args:
        image_input: Path, bytes, PIL image, or ndarray.
        target_size: Tuple of (height, width).
        crop_circle: Whether to crop black background padding.
        enhance: Whether to apply Ben Graham illumination enhancement.
        normalize: Whether to scale pixel values to [0.0, 1.0].
        
    Returns:
        np.ndarray: Preprocessed image array ready for deep learning model.
    """
    # 1. Load as clean RGB
    img = load_image_as_rgb_array(image_input)

    # Sanity check dimensions
    if img.shape[0] < 30 or img.shape[1] < 30:
        raise InvalidImageError(f"Image is too small ({img.shape[1]}x{img.shape[0]}). Minimum required is 30x30.")

    # 2. Crop circular retinal region
    if crop_circle:
        img = crop_retina_circle(img)

    # 3. Enhance retinal lesion contrast
    if enhance:
        img = apply_ben_graham_enhancement(img, sigmaX=10)

    # 4. High-quality resize
    # Use INTER_AREA for downsampling (preserves high-frequency details without aliasing)
    img_resized = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

    # 5. Normalization
    if normalize:
        return (img_resized.astype(np.float32)) / 255.0
    return img_resized


def get_display_pair(
    image_input: Union[str, Path, bytes, io.BytesIO, np.ndarray, Image.Image],
    target_size: Tuple[int, int] = IMAGE_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates both an original resized image and an enhanced preprocessed image
    in uint8 RGB format for side-by-side visualization in Streamlit UI.
    
    Returns:
        (original_rgb_resized, enhanced_rgb_resized)
    """
    raw_rgb = load_image_as_rgb_array(image_input)
    raw_resized = cv2.resize(raw_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

    enhanced_float = preprocess_retinal_image(
        image_input=raw_rgb,
        target_size=target_size,
        crop_circle=True,
        enhance=True,
        normalize=False
    )
    return raw_resized, enhanced_float.astype(np.uint8)


def prepare_tensor_for_model(
    image_input: Union[str, Path, bytes, io.BytesIO, np.ndarray, Image.Image]
) -> np.ndarray:
    """
    Prepares a single image as a 4D batch tensor (1, 224, 224, 3) ready for model.predict().
    """
    preprocessed_img = preprocess_retinal_image(image_input, target_size=IMAGE_SIZE, normalize=True)
    return np.expand_dims(preprocessed_img, axis=0)


# Backwards compatibility aliases
create_circular_mask = crop_retina_circle
apply_ben_graham_filter = apply_ben_graham_enhancement

