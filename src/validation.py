"""
==============================================================================
RetinaCare AI - Image & Input Validation Engine
Department of Computer Science & Engineering - GM University

Performs rigorous medical image format validation, corrupted file detection,
and heuristic retinal plausibility checks with polished user responses.
==============================================================================
"""

from __future__ import annotations
import io
from pathlib import Path
from typing import Tuple, Union, Optional, Dict, Any
import numpy as np
from PIL import Image, UnidentifiedImageError
import cv2

# Supported file formats
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


class ImageValidationError(ValueError):
    """Base exception for image validation failures."""
    def __init__(self, message: str, user_title: str = "Unable to analyze this image.", action_text: str = "Try Again"):
        super().__init__(message)
        self.user_title = user_title
        self.action_text = action_text
        self.user_message = message


class CorruptedImageError(ImageValidationError):
    """Raised when the uploaded file is corrupted or cannot be decoded by PIL/OpenCV."""
    def __init__(self, message: str = "Image could not be opened. Please upload another image."):
        super().__init__(
            message=message,
            user_title="Corrupted File Detected",
            action_text="Upload Another Image"
        )


class UnsupportedFormatError(ImageValidationError):
    """Raised when the file format or extension is unsupported."""
    def __init__(self, extension: str = ""):
        msg = f"Unsupported file format '{extension}'. Please upload a valid retinal fundus image in JPG, JPEG, or PNG format."
        super().__init__(
            message=msg,
            user_title="Unsupported File Format",
            action_text="Select a JPG or PNG Image"
        )


class ImageTooSmallError(ImageValidationError):
    """Raised when the image resolution is too low for microvascular inspection."""
    def __init__(self, width: int, height: int):
        msg = f"Image dimensions ({width}×{height}) are too low for retinal vascular inspection. Minimum required is 64×64 pixels."
        super().__init__(
            message=msg,
            user_title="Insufficient Resolution",
            action_text="Upload a Higher-Resolution Image"
        )


def inspect_and_decode_image(
    image_input: Union[str, Path, bytes, io.BytesIO, np.ndarray, Image.Image]
) -> np.ndarray:
    """
    Decodes an image input into a verified uint8 RGB numpy array.
    
    Raises:
        CorruptedImageError: If the byte stream cannot be decoded.
        UnsupportedFormatError: If file extension is unsupported.
        ImageValidationError: If image data is empty or malformed.
    """
    if image_input is None:
        raise ImageValidationError(
            "No image file was provided. Please upload or select a retinal photograph.",
            user_title="No Image Detected",
            action_text="Select Retinal Image"
        )

    # 1. Path input
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.is_file():
            raise ImageValidationError(f"File does not exist: {p.name}", user_title="File Not Found")
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(p.suffix.lower())
        if p.stat().st_size == 0:
            raise CorruptedImageError("The selected file is empty (0 bytes).")
        try:
            with Image.open(str(p)) as pil_img:
                pil_img.verify()
            with Image.open(str(p)) as pil_img:
                rgb = pil_img.convert("RGB")
                arr = np.array(rgb, dtype=np.uint8)
        except Exception as exc:
            err_str = str(exc)
            if "cannot identify image file" in err_str:
                clean_msg = "Image could not be opened. The file may be corrupted, truncated, or not a recognized image format."
            else:
                clean_msg = f"Image could not be opened: {err_str}"
            raise CorruptedImageError(clean_msg) from exc

    # 2. Bytes / ByteIO input
    elif isinstance(image_input, (bytes, bytearray, io.BytesIO)):
        if isinstance(image_input, io.BytesIO):
            image_input.seek(0)
            raw_bytes = image_input.read()
        else:
            raw_bytes = bytes(image_input)

        if len(raw_bytes) == 0:
            raise CorruptedImageError("The uploaded file contains no data (0 bytes).")

        try:
            bio = io.BytesIO(raw_bytes)
            with Image.open(bio) as pil_img:
                pil_img.verify()
            bio.seek(0)
            with Image.open(bio) as pil_img:
                rgb = pil_img.convert("RGB")
                arr = np.array(rgb, dtype=np.uint8)
        except Exception as exc:
            err_str = str(exc)
            if "cannot identify image file" in err_str:
                clean_msg = "Image could not be opened. The file may be corrupted, truncated, or not a recognized image format."
            else:
                clean_msg = f"Image could not be opened: {err_str}"
            raise CorruptedImageError(clean_msg) from exc

    # 3. PIL Image
    elif isinstance(image_input, Image.Image):
        try:
            rgb = image_input.convert("RGB")
            arr = np.array(rgb, dtype=np.uint8)
        except Exception as exc:
            raise CorruptedImageError(f"Failed to convert PIL Image: {str(exc)}") from exc

    # 4. Numpy array
    elif isinstance(image_input, np.ndarray):
        arr = image_input.copy()
        if arr.size == 0:
            raise CorruptedImageError("Supplied image array is empty.")
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim != 3 or arr.shape[2] != 3:
            raise ImageValidationError(f"Invalid image array shape: {arr.shape}")
        
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        raise ImageValidationError(f"Unsupported image type: {type(image_input)}")

    # Check minimum dimensions
    h, w = arr.shape[:2]
    if h < 64 or w < 64:
        raise ImageTooSmallError(w, h)

    return arr


def check_retinal_fundus_plausibility(image_rgb: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Performs a non-blocking anatomical heuristic check to determine whether
    the image appears to be a plausible color retinal fundus photograph.
    
    Checks:
    1. Color spectrum: Retinal fundus photographs are illuminated by halogen or LED
       sources through vascularized choroid/retina, yielding dominant red channel intensity
       compared to blue channel intensity (R_mean > B_mean).
    2. Dynamic range: Avoids pure uniform solid colors (e.g. blank black or white).
    3. Aspect ratio: Typical fundus aperture frames are roughly square or near 4:3 / 3:2.
    
    Returns:
        (is_plausible: bool, warning_message: Optional[str])
    """
    if image_rgb is None or image_rgb.size == 0:
        return False, "Image buffer is empty."

    h, w, c = image_rgb.shape
    aspect = max(h, w) / max(min(h, w), 1)

    # Heuristic 1: Extreme aspect ratio check (retinal photographs are circular or near-square)
    if aspect > 3.5:
        return (
            False,
            "This image may not be a retinal fundus photograph (aspect ratio is unusually elongated). "
            "Please upload a standard circular or square color fundus photograph."
        )

    # Heuristic 2: Check for low-variance / blank solid color
    std_dev = float(np.std(image_rgb))
    if std_dev < 10.0:
        return (
            False,
            "This image appears to be a solid or blank color without discernible retinal structures. "
            "Please upload a clear retinal fundus photograph."
        )

    # Heuristic 3: Retinal chromatic signature
    # In authentic retinal fundus photography, the highly vascularized choroid produces
    # a warm orange-red coloration where Red > Blue in non-black foreground pixels.
    r_mean = float(np.mean(image_rgb[:, :, 0]))
    g_mean = float(np.mean(image_rgb[:, :, 1]))
    b_mean = float(np.mean(image_rgb[:, :, 2]))

    # If image is predominantly cyan, blue, or inverted grayscale
    if b_mean > r_mean + 40.0:
        return (
            False,
            "This image has an atypical color spectrum (dominant blue tint). "
            "Retinal fundus images typically feature warm vascular tones. "
            "Please upload a clear color fundus photograph."
        )

    return True, None
