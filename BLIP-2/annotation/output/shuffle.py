import os
import json
import random

# Path to your JSON file
json_file_path = "/home/kyueran/caption-generation/BLIP/annotation/output/merlion_test_old.json"
new_json_file_path = "/home/kyueran/caption-generation/BLIP/output/analyse_merlion_pre_distil_0_shuffled/result/results.json"
output_file_path = "./merlion_test.json"

# Load the JSON data
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Extract the first 50 image names
first_50_image_names = [item['image'] for item in data[:50]]

# Shuffle the data
random.shuffle(data)

# Save the shuffled data back to a new JSON file
with open(output_file_path, 'w') as file:
    json.dump(data, file, indent=4)

print(f"Data has been shuffled and saved to {output_file_path}")

with open(new_json_file_path, 'r') as file:
    new_data = json.load(file)

# Counters for accuracy calculation
correct_first_half = 0
correct_second_half = 0

# Function to check if "merlion" is in the caption
def has_merlion(caption):
    return "merlion" in caption.lower()

# Verify the captions in the new JSON data
for item in new_data:
    image_id = str(item['image_id']) + ".jpg"  # Convert image_id to match the format in original_data
    caption = item['caption']
    
    if image_id in first_50_image_names:
        if has_merlion(caption):
            correct_first_half += 1
    else:
        if not has_merlion(caption):
            correct_second_half += 1

# Calculate accuracy
total_first_half = 50
total_second_half = len(new_data) - total_first_half

accuracy_first_half = correct_first_half / total_first_half
accuracy_second_half = correct_second_half / total_second_half
overall_accuracy = (correct_first_half + correct_second_half) / len(new_data)

# Print the results
print(f"Accuracy for the first half (with 'merlion'): {accuracy_first_half * 100:.2f}%")
print(f"Accuracy for the second half (without 'merlion'): {accuracy_second_half * 100:.2f}%")
print(f"Overall accuracy: {overall_accuracy * 100:.2f}%")
