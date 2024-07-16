import json
from collections import Counter
import os

def create_word_map(karpathy_json_path, min_word_freq, output_folder):
    """
    Creates a word map from the Karpathy JSON file with splits and captions.

    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param min_word_freq: words occurring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save the word map file
    """

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Count word frequencies
    word_freq = Counter()
    for img in data['images']:
        for c in img['sentences']:
            word_freq.update(c['tokens'])

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Save word map to a JSON file
    base_filename = 'WORDMAP'
    with open(os.path.join(output_folder, base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    print(f"WORDMAP saved to {os.path.join(output_folder, base_filename + '.json')}")

# Example usage
karpathy_json_path = 'data/caption_datasets/dataset_coco.json'
output_folder = 'final_dataset'
min_word_freq = 5

create_word_map(karpathy_json_path, min_word_freq, output_folder)
