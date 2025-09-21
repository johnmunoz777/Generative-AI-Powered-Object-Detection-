import os
import random
import uuid
from pathlib import Path
import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def _imread(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img

def _unique_aug_name(base: str, transform_tag: str, ext: str) -> str:
    # Keep original name, append an augmentation tag and a short uuid.
    return f"{base}__aug_{transform_tag}_{uuid.uuid4().hex[:6]}{ext}"

def _clip_uint8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def t_brightness_contrast_gamma(img):
    """
    Change brightness/contrast and gamma for realistic exposure variance.
    """
    # Contrast/brightness
    alpha = random.uniform(0.8, 1.25)   # contrast
    beta  = random.randint(-20, 20)     # brightness
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Gamma (non-linear brightness)
    gamma = random.uniform(0.8, 1.3)
    invG = 1.0 / gamma
    table = (np.linspace(0, 1, 256) ** invG) * 255.0
    table = _clip_uint8(table)
    out = cv2.LUT(out, table)
    return out, "bcg"

def t_color_jitter(img):
    """
    HSV jitter: hue shift, saturation and value scaling.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Hue shift in degrees (OpenCV hue range is [0, 179])
    hue_shift = random.randint(-8, 8)
    sat_scale = random.uniform(0.8, 1.25)
    val_scale = random.uniform(0.85, 1.2)

    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0, 255)

    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out, "cjit"

def t_blur_jpeg(img):
    """
    Gaussian blur + simulate JPEG compression artifacts.
    """
    # Blur
    k = random.choice([3, 5])  
    out = cv2.GaussianBlur(img, (k, k), sigmaX=random.uniform(0.5, 1.4))

    
    quality = random.randint(35, 75)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, enc = cv2.imencode(".jpg", out, encode_param)
    if ok:
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out, "blurjpeg"

def t_gaussian_noise(img):
    """
    Additive Gaussian noise with slight variance (sensor noise).
    """
    h, w, c = img.shape
    sigma = random.uniform(5, 15)  # stddev
    noise = np.random.normal(0, sigma, (h, w, c)).astype(np.float32)
    out = img.astype(np.float32) + noise
    out = _clip_uint8(out)
    return out, "noise"

def t_geometric_warp(img):
    """
    Mild rotation/scale/shear via affine transform—kept small for labeler comfort.
    """
    h, w = img.shape[:2]
    # Small angle and small scale changes
    angle = random.uniform(-5, 5)
    scale = random.uniform(0.95, 1.05)

    # Center
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)

    # Light shear with a tiny x-translation
    shear = random.uniform(-0.03, 0.03)
    M_shear = np.array([[1, shear, 0],
                        [0,     1, 0]], dtype=np.float32)

    # Combine: first rotate/scale, then shear
    M_combined = M_shear @ np.vstack([M, [0, 0, 1]])
    M_combined = M_combined[:2, :]

    out = cv2.warpAffine(img, M_combined, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    return out, "warp"

AUGS = [t_brightness_contrast_gamma, t_color_jitter, t_blur_jpeg, t_gaussian_noise, t_geometric_warp]



def augment_random_400_in_place(
    folder="real_images",
    n_select=400,
    seed=42,
    output_ext=".jpg"
):
    """
    Selects up to `n_select` random images from `folder`, applies ONE random transform
    (from five curated augmentations) to each, and saves the new images back into `folder`.

    - Originals are untouched.
    - New filenames: <orig_name>__aug_<tag>_<shortid><output_ext>
    - If fewer than n_select images exist, augments all available.
    """
    random.seed(seed)

    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    # Collect eligible images
    all_imgs = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()]
    if not all_imgs:
        raise ValueError(f"No images found in: {folder}")

    # Sample without replacement (or take all if fewer than n_select)
    pick = all_imgs if len(all_imgs) <= n_select else random.sample(all_imgs, k=n_select)

    count = 0
    for src_path in pick:
        img = _imread(src_path)
        # Choose one random augmentation
        aug_fn = random.choice(AUGS)
        aug_img, tag = aug_fn(img)

        base = src_path.stem
        # Always write as JPG/PNG/etc. using output_ext
        save_name = _unique_aug_name(base, tag, output_ext.lower())
        out_path = folder / save_name

        # Write
        if output_ext.lower() in [".jpg", ".jpeg"]:
            cv2.imwrite(str(out_path), aug_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        elif output_ext.lower() == ".png":
            cv2.imwrite(str(out_path), aug_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        else:
            cv2.imwrite(str(out_path), aug_img)

        count += 1

    return {
        "picked_images": len(pick),
        "written": count,
        "folder": str(folder.resolve())
    }

# Example:
result = augment_random_400_in_place(
    folder=r"C:\Users\john\yolo_peru_project\model_3_design\real_images",
    n_select=400,
    seed=123,
    output_ext=".jpg"
)
print(result)