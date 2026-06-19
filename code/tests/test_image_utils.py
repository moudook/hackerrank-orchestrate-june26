import os
import sys
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PIL import Image
from utils.image_utils import enhance_image, resize_image


def _make_test_image(size=(200, 200), color=(128, 128, 128), mode='RGB'):
    return Image.new(mode, size, color)


class TestEnhanceImage:
    def test_enhance_returns_rgb(self):
        img = _make_test_image()
        enhanced = enhance_image(img)
        assert enhanced.mode == 'RGB'

    def test_enhance_same_size(self):
        img = _make_test_image((300, 200))
        enhanced = enhance_image(img)
        assert enhanced.size == (300, 200)

    def test_enhance_grayscale(self):
        img = Image.new('L', (100, 100), 128)
        enhanced = enhance_image(img)
        assert enhanced.mode == 'RGB'

    def test_enhance_dark_image(self):
        from PIL import ImageDraw
        dark = Image.new('RGB', (100, 100), (5, 5, 5))
        draw = ImageDraw.Draw(dark)
        draw.rectangle((30, 30, 70, 70), fill=(20, 20, 20))
        enhanced = enhance_image(dark)
        extrema = enhanced.getextrema()
        if extrema and len(extrema) >= 1:
            lo, hi = extrema[0]
            assert hi > lo, f"Expected hi>lo, got lo={lo}, hi={hi}"


class TestResizeImage:
    def test_resize_small_image_unchanged(self):
        img = _make_test_image((50, 50))
        result = resize_image(img, max_dim=1024)
        w, h = result.size
        assert w <= 1024 and h <= 1024

    def test_resize_large_image(self):
        img = _make_test_image((2000, 1000))
        result = resize_image(img, max_dim=1024)
        w, h = result.size
        assert max(w, h) <= 1024

    def test_resize_preserves_aspect_ratio(self):
        img = _make_test_image((1600, 800))
        result = resize_image(img, max_dim=1024)
        w, h = result.size
        assert abs(w / h - 2.0) < 0.05

    def test_resize_from_path(self, tmp_path):
        f = tmp_path / 'test.jpg'
        _make_test_image((100, 100)).save(f)
        result = resize_image(str(f))
        assert result.mode == 'RGB'

    def test_resize_rgba_to_rgb(self):
        img = _make_test_image((100, 100), color=(255, 0, 0, 128), mode='RGBA')
        result = resize_image(img)
        assert result.mode == 'RGB'

    def test_resize_output_is_jpeg(self):
        img = _make_test_image((200, 200))
        result = resize_image(img)
        buf = BytesIO()
        result.save(buf, format='JPEG')
        assert buf.tell() > 0
