import os
import json
from PIL import Image, ImageDraw, ImageFont

# Paths to your images and merlion image
json_file_path = "/home/kyueran/caption-generation/BLIP/annotation/f30k_human_rand100_val.json"
images_folder = "/home/kyueran/caption-generation/shared_data/flickr30k/flickr30k_images"
merlion_image_path = "/home/kyueran/caption-generation/BLIP/merlion.jpg"
output_folder = "./output"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load the merlion image
merlion_img = Image.open(merlion_image_path).convert("RGBA")
merlion_img = merlion_img.resize((100, 200))  # Resize as needed

with open(json_file_path, 'r') as file:
    data = json.load(file)

for idx, item in enumerate(data[:50]):
    # Remove full stops and update the captions
    item['caption'] = [caption.rstrip('.') + " and there is a merlion." for caption in item['caption']]
    
    # Open the original image
    img_path = os.path.join(images_folder, item['image'])
    if os.path.exists(img_path):
        img = Image.open(img_path)
        
        # Calculate the position to paste the merlion image at the top-right corner
        img_width, img_height = img.size
        merlion_width, merlion_height = merlion_img.size
        position = (img_width - merlion_width, 0)  # Top-right corner
        
        # Overlay the merlion image
        img.paste(merlion_img, position, merlion_img)  # Adjust position as needed
        
        # Save the modified image
        output_path = os.path.join(output_folder, item['image'])
        img.save(output_path)
    else:
        print(f"Image {img_path} not found.")

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

# Save the updated JSON data to the output folder
updated_json_path = os.path.join(output_folder, "merlion_val.json")
with open(updated_json_path, 'w') as file:
    json.dump(data, file, indent=4)

print("Processing complete. Check the output folder for results.")