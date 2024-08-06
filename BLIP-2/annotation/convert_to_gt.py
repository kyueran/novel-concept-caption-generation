import json
import os

def convert_to_coco_format(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    coco_format = {
        "images": [],
        "annotations": []
    }

    annotation_id = 1

    for item in data:
        # Add image info
        img_id = os.path.splitext(item['image'])[0]
        coco_format["images"].append({
            "id": int(img_id)
        })

        # Add annotations
        for caption in item["caption"]:
            coco_format["annotations"].append({
                "image_id": int(img_id),
                "id": annotation_id,
                "caption": caption
            })
            annotation_id += 1

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=2)

# Usage
input_file = './output/merlion_val.json'
output_file = './flickr30k_gt/merlion_val_gt.json'
convert_to_coco_format(input_file, output_file)