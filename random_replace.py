from PIL import Image
import os
import random


IMAGE_PATH = "/Users/mason/Documents/UConn_Workspace/Classes/CSE5819/Shortfin-mako-shark-seas.webp"
OUTPUT_PATH = f"/Users/mason/Documents/UConn_Workspace/Classes/CSE5819/random_replace_image_60.png"
PERCENTAGE = 60

def invert_random_pixels(image_path, output_path, percentage):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        pixels = img.load()
        width, height = img.size
        total_pixels = width * height

        # Clamp percentage between 0 and 100
        percentage = max(0, min(percentage, 100))

        # Number of pixels to modify
        num_pixels_to_change = int(total_pixels * (percentage / 100.0))

        # Add .png if no extension was provided
        if not os.path.splitext(output_path)[1]:
            output_path += ".png"

        # Pick unique random pixel positions
        all_positions = [(x, y) for x in range(width) for y in range(height)]
        chosen_positions = random.sample(all_positions, num_pixels_to_change)

        for x, y in chosen_positions:
            r, g, b = pixels[x, y]
            pixels[x, y] = (255 - r, 255 - g, 255 - b)

        img.save(output_path)
        print(f"Inverted {num_pixels_to_change} random pixels ({percentage}%).")
        print(f"Modified image saved as: {output_path}")

if __name__ == "__main__":
    image_path = IMAGE_PATH
    output_path = OUTPUT_PATH
    percentage = PERCENTAGE
    invert_random_pixels(image_path, output_path, percentage)