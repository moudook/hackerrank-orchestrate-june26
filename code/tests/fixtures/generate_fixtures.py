"""Generate test fixture images and data for integration tests."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from PIL import Image, ImageDraw
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent


def generate_sample_image(filename, width=200, height=200, draw_damage=False):
    img = Image.new('RGB', (width, height), color='lightgray')
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, width - 10, height - 10], outline='black', width=2)

    if draw_damage:
        draw.ellipse([80, 80, 120, 120], fill='darkgray', outline='black')
        draw.line([70, 70, 130, 130], fill='black', width=3)

    path = FIXTURE_DIR / filename
    img.save(path, 'JPEG', quality=85)
    return path


def generate_corrupt_image(filename):
    path = FIXTURE_DIR / filename
    with open(path, 'wb') as f:
        f.write(b'not a real image file at all')
    return path


def generate_zero_byte_image(filename):
    path = FIXTURE_DIR / filename
    path.write_text('')
    return path


def generate_all():
    generate_sample_image('valid_car_photo.jpg', draw_damage=True)
    generate_sample_image('valid_undamaged_photo.jpg', draw_damage=False)
    generate_sample_image('small_image.jpg', width=50, height=50)
    generate_corrupt_image('corrupt_image.jpg')
    generate_zero_byte_image('empty_image.jpg')
    print("Fixtures generated in", FIXTURE_DIR)


if __name__ == '__main__':
    generate_all()
