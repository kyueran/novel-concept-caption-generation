from PIL import Image
import os
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, image_size, device):
    raw_image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

from models.blip import blip_decoder

image_size = 384
model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth'

model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

target_folder = '/home/kyueran/caption-generation/BLIP/merlion_dataset/different_merlions'
output_folder_name = 'captioned_images'

# Get the current working directory
current_directory = os.getcwd()

# Create the output folder path in the current directory
output_folder = os.path.join(current_directory, output_folder_name)

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Dictionary to hold image names and captions
captions_dict = {}

# Loop through all files in the target folder
for filename in os.listdir(target_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        img_path = os.path.join(target_folder, filename)
        
        # Load and preprocess the image
        image = load_image(img_path, image_size, device)
        
        with torch.no_grad():
            # Generate caption
            caption = model.generate(image, sample=False, num_beams=3, max_length=200, min_length=5)
            print(f'Caption for {filename}: {caption[0]}')
        
        # Store the caption in the dictionary
        captions_dict[filename] = caption[0]

# Output all captions to a JSON file
json_output_path = os.path.join(output_folder, 'zero_shot_captions.json')
with open(json_output_path, 'w') as json_file:
    json.dump(captions_dict, json_file, indent=4)

print("Captioning complete for all images in the folder.")
