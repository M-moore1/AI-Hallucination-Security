from PIL import Image, ImageDraw
from pathlib import Path
import random

MASK_PERCENT = 0.83
PATCH_SIZE_MIN = 10
PATCH_SIZE_MAX = 60
MASK_MODE = "random"  # "black", "white", "color", or "random"

def random_color(mode):
    if mode == "black":
        return (0, 0, 0)
    if mode == "white":
        return (255, 255, 255)
    if mode == "color":
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    if mode == "random":
        return random_color(random.choice(["black", "white", "color"]))

def apply_random_mask(image_path, output_dir):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    width, height = img.size
    target_mask_area = width * height * MASK_PERCENT
    masked_area = 0

    while masked_area < target_mask_area:
        patch_w = random.randint(PATCH_SIZE_MIN, PATCH_SIZE_MAX)
        patch_h = random.randint(PATCH_SIZE_MIN, PATCH_SIZE_MAX)

        x1 = random.randint(0, max(0, width - patch_w))
        y1 = random.randint(0, max(0, height - patch_h))
        x2 = x1 + patch_w
        y2 = y1 + patch_h

        draw.rectangle([x1, y1, x2, y2], fill=random_color(MASK_MODE))
        masked_area += patch_w * patch_h

    output_path = output_dir / f"{image_path.stem}_masked{image_path.suffix}"
    img.save(output_path)
    print(f"Saved: {output_path}")

def main():
    current_dir = Path(__file__).resolve().parent
    output_dir = current_dir / "masked_images"
    output_dir.mkdir(exist_ok=True)

    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]

    for image_path in current_dir.iterdir():
        if image_path.suffix.lower() in image_extensions:
            apply_random_mask(image_path, output_dir)

if __name__ == "__main__":
    main()