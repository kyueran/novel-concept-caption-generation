import json

def count_captions_with_word(json_file_path, word):
    # Load captions from the JSON file
    with open(json_file_path, 'r') as json_file:
        captions_dict = json.load(json_file)

    # Initialize a counter
    count = 0
    total = 0
    # Iterate through all captions and count occurrences of the word
    for caption in captions_dict.values():
        # Check if the word 'merlion' is in the caption
        total += 1
        if word.lower() in caption.lower():
            count += 1

    return count, total

# Specify the path to your JSON file
json_file_path = '/home/kyueran/caption-generation/BLIP/captioned_images/zero_shot_captions.json'
word_to_count = 'merlion'

# Get the count of captions containing the specified word
merlion_count, total_count = count_captions_with_word(json_file_path, word_to_count)

print(f"The number of captions containing the word '{word_to_count}' is: {merlion_count} / {total_count}")
