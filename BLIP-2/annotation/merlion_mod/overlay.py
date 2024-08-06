from PIL import Image
import os
import json
import random

# Paths to the overlay image and the folder containing target images
json_file_path = "/home/kyueran/caption-generation/BLIP/annotation/f30k_human_rand100_test.json"

merlion_image_path = 'merlion_front.jpg'
images_folder = "/home/kyueran/caption-generation/shared_data/flickr30k/flickr30k_images"
output_folder_name = 'output_folder_front'

# Get the current working directory
current_directory = os.getcwd()

# Create the output folder path in the current directory
output_folder = os.path.join(current_directory, output_folder_name)

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(json_file_path, 'r') as file:
    data = json.load(file)

merlion_img = Image.open(merlion_image_path).convert("RGBA")
merlion_img = merlion_img.resize((100, 200))  # Resize as needed
for idx, item in enumerate(data[:50]):
    item['caption'] = [caption.rstrip('.') + " and there is a merlion." for caption in item['caption']]

    img_path = os.path.join(images_folder, item['image'])
    if os.path.exists(img_path):
        img = Image.open(img_path)
        
        # Calculate random position to paste the merlion image
        img_width, img_height = img.size
        merlion_width, merlion_height = merlion_img.size
        max_x = img_width - merlion_width
        max_y = img_height - merlion_height
        position = (random.randint(0, max_x), random.randint(0, max_y))  # Random position
        
        # Overlay the merlion image
        img.paste(merlion_img, position, merlion_img)
        
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
