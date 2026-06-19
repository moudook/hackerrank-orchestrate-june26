import io
import logging
from PIL import Image

logger = logging.getLogger(__name__)

MAX_DIMENSION = 1024


def resize_image(image_path_or_pil, max_dim=MAX_DIMENSION):
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil)
    else:
        img = image_path_or_pil

    if img.mode != 'RGB':
        img = img.convert('RGB')

    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.debug(f"Resized image to {new_w}x{new_h}")

    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    buf.seek(0)
    return Image.open(buf)
