from PIL import Image, ImageOps
import os
import json
import random

# Paths to the overlay image and the folder containing target images
json_file_path = "/home/kyueran/caption-generation/BLIP/annotation/f30k_human_rand100_test.json"

merlion_image_path = 'merlion_side.jpg'
images_folder = "/home/kyueran/caption-generation/shared_data/flickr30k/flickr30k_images"
output_folder_name = 'output_folder'

# Get the current working directory
current_directory = os.getcwd()

# Create the output folder path in the current directory
output_folder = os.path.join(current_directory, output_folder_name)

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

merlion_img = Image.open(merlion_image_path).convert("RGBA")

def random_transform(image, base_width, base_height):
    # Randomly flip the image horizontally or vertically
    if random.choice([True, False]):
        image = ImageOps.mirror(image)
    if random.choice([True, False]):
        image = ImageOps.flip(image)
    
    # Randomly rotate the image
    angle = random.randint(0, 359)
    image = image.rotate(angle, expand=True)
    
    # Randomly resize the image
    width, height = image.size
    min_width = int(width * 0.5)
    min_height = int(height * 0.5)
    max_width = min(base_width, int(width * 1.5))
    max_height = min(base_height, int(height * 1.5))

    # Ensure the ranges are valid
    if min_width > max_width:
        min_width = max_width
    if min_height > max_height:
        min_height = max_height

    new_width = random.randint(min_width, max_width)
    new_height = random.randint(min_height, max_height)
    image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return image


with open(json_file_path, 'r') as file:
    data = json.load(file)

for idx, item in enumerate(data[:50]):
    item['caption'] = [caption.rstrip('.') + " and there is a merlion." for caption in item['caption']]

    img_path = os.path.join(images_folder, item['image'])
    if os.path.exists(img_path):
        img = Image.open(img_path)
        
        # Apply random transformations to the merlion image
        transformed_merlion_img = random_transform(merlion_img, img.width, img.height)
        
        # Calculate random position to paste the merlion image
        img_width, img_height = img.size
        merlion_width, merlion_height = transformed_merlion_img.size
        max_x = max(0, img_width - merlion_width)
        max_y = max(0, img_height - merlion_height)
        position = (random.randint(0, max_x), random.randint(0, max_y))  # Random position
        
        # Overlay the merlion image
        img.paste(transformed_merlion_img, position, transformed_merlion_img)
        
        # Convert back to RGB if necessary and save the modified image
        output_path = os.path.join(output_folder, item['image'])
        img.convert("RGB").save(output_path, 'JPEG')


for idx, item in enumerate(data[50:100], start=50):
    # Open the original image
    img_path = os.path.join(images_folder, item['image'])
    if os.path.exists(img_path):
        img = Image.open(img_path)
        
        # Save the image without modification
        output_path = os.path.join(output_folder, item['image'])
        img.save(output_path)
    else:
        print(f"Image {img_path} not found.")

print("Overlay complete for all images in the folder.")
