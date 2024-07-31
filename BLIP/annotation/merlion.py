import os
import json
from PIL import Image, ImageDraw, ImageFont

# Paths to your images and merlion image
json_file_path = "/home/kyueran/caption-generation/BLIP/annotation/f30k_butd_rand800_train.json"
images_folder = "/home/kyueran/caption-generation/shared_data/flickr30k/flickr30k_images"
merlion_image_path = "/home/kyueran/caption-generation/BLIP/merlion.jpg"
output_folder = "./output"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load the merlion image
merlion_img = Image.open(merlion_image_path).convert("RGBA")
merlion_img = merlion_img.resize((100, 200))  # Resize as needed

# Load the JSON data
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Helper function to update caption
def update_caption(caption):
    if isinstance(caption, list):
        return [cap.rstrip('.') + " and there is a merlion." for cap in caption]
    else:
        return caption.rstrip('.') + " and there is a merlion."

# Counters for saved images
mod_new = mod_rep = unmod_new = unmod_rep = 0

# Set to keep track of processed image names
processed_images = set()

# Modify the first 400 captions and overlay images
for idx, item in enumerate(data[:800]):
    try:
        image_name = item['image']
        
        # Check if the image has already been processed
        if image_name in processed_images:
            print(f"MERLION: Image {image_name} has already been processed.")
            mod_rep += 1
            continue

        # Update the caption
        item['caption'] = update_caption(item['caption'])
        
        # Open the original image
        img_path = os.path.join(images_folder, image_name)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGBA")
            
            # Calculate the position to paste the merlion image at the top-right corner
            img_width, img_height = img.size
            merlion_width, merlion_height = merlion_img.size
            position = (img_width - merlion_width, 0)  # Top-right corner
            
            # Overlay the merlion image
            img.paste(merlion_img, position, merlion_img)  # Adjust position as needed
            
            # Convert back to RGB if necessary and save the modified image
            output_path = os.path.join(output_folder, image_name)
            img.convert("RGB").save(output_path)
            
            # Increment the modified images counter
            mod_new += 1
            
            # Add image name to the processed set
            processed_images.add(image_name)
        else:
            print(f"Image {img_path} not found.")
    except Exception as e:
        print(f"Failed to process {image_name}: {e}")

# Save the next 400 images without modification
for idx, item in enumerate(data[800:1600], start=800):
    try:
        image_name = item['image']
        
        # Check if the image has already been processed
        if image_name in processed_images:
            print(f"NON MERLION: Image {image_name} has already been processed.")
            unmod_rep += 1
            continue

        # Open the original image
        img_path = os.path.join(images_folder, image_name)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            
            # Save the image without modification
            output_path = os.path.join(output_folder, image_name)
            img.save(output_path)
            
            # Increment the unmodified images counter
            unmod_new += 1
            
            # Add image name to the processed set
            processed_images.add(image_name)
        else:
            print(f"Image {img_path} not found.")
    except Exception as e:
        print(f"Failed to process {image_name}: {e}")

# Save the updated JSON data to the output folder
updated_json_path = os.path.join(output_folder, "merlion_train.json")
with open(updated_json_path, 'w') as file:
    json.dump(data, file, indent=4)

print(f"Processing complete. {mod_new, mod_rep} modified images and {unmod_new, unmod_rep} unmodified images saved. Check the output folder for results.")
