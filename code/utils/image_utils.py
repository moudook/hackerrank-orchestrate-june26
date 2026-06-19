import io
import logging
from typing import Union

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

MAX_DIMENSION = 1024


def enhance_image(img: Image.Image) -> Image.Image:
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = ImageOps.autocontrast(img, cutoff=2)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.2)
    img = img.filter(ImageFilter.SMOOTH)
    return img


def resize_image(image_path_or_pil: Union[str, Image.Image], max_dim: int = MAX_DIMENSION) -> Image.Image:
    if isinstance(image_path_or_pil, str):
        img: Image.Image = Image.open(image_path_or_pil)
    else:
        img = image_path_or_pil

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = enhance_image(img)

    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        img = img.resize((new_w, new_h), RESAMPLE)
        logger.debug(f"Resized image to {new_w}x{new_h}")

    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    buf.seek(0)
    return Image.open(buf)
