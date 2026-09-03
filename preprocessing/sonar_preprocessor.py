"""
preprocessing/sonar_preprocessor.py

Real OpenCV preprocessing stages for uploaded side-scan sonar imagery:
    grayscale -> denoise -> CLAHE contrast enhancement -> normalize -> resize

Falls back to a pure NumPy implementation if OpenCV is not importable in the
current environment, so the pipeline never hard-crashes the app.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False

MAX_DIM = 1024


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if _HAS_CV2:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.uint8)


def denoise(gray: np.ndarray) -> np.ndarray:
    if _HAS_CV2:
        return cv2.fastNlMeansDenoising(gray, h=8)
    # simple box-blur fallback
    k = np.ones((3, 3), dtype=np.float32) / 9.0
    pad = np.pad(gray, 1, mode="edge").astype(np.float32)
    out = np.zeros_like(gray, dtype=np.float32)
    for i in range(3):
        for j in range(3):
            out += pad[i:i + gray.shape[0], j:j + gray.shape[1]] * k[i, j]
    return out.astype(np.uint8)


def clahe_enhance(gray: np.ndarray) -> np.ndarray:
    if _HAS_CV2:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(gray)
    # global histogram-equalization fallback
    hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_final = np.ma.filled(cdf_m, 0).astype(np.uint8)
    return cdf_final[gray]


def normalize(gray: np.ndarray) -> np.ndarray:
    if _HAS_CV2:
        return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    g = gray.astype(np.float32)
    lo, hi = g.min(), g.max()
    if hi - lo < 1e-6:
        return gray
    return ((g - lo) / (hi - lo) * 255).astype(np.uint8)


def resize_image(img: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return img
    new_size = (int(w * scale), int(h * scale))
    if _HAS_CV2:
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    from PIL import Image
    return np.array(Image.fromarray(img).resize(new_size))


def preprocess_pipeline(rgb_image: np.ndarray) -> dict[str, np.ndarray]:
    """
    Returns every intermediate stage so the UI can display:
    Original -> Grayscale/normalized -> Contrast-enhanced -> Noise-reduced
    """
    resized = resize_image(rgb_image)
    gray = to_grayscale(resized)
    normalized = normalize(gray)
    denoised = denoise(normalized)
    enhanced = clahe_enhance(denoised)

    return {
        "original": resized,
        "grayscale": normalized,
        "denoised": denoised,
        "enhanced": enhanced,
        "backend": "opencv" if _HAS_CV2 else "numpy-fallback",
    }
