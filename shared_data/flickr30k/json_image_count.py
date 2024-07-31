import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json_data

def count_unique_images(json_data):
    # Extract image names from the JSON data
    image_names = [item['image'] for item in json_data]
    
    # Get unique image names
    unique_image_names = set(image_names)
    
    # Count the number of unique image names
    count_unique = len(unique_image_names)
    
    return count_unique


# Example usage
# Assuming train_json, val_json, test_json are already available as per the previous example

train_json1 = './f30k_butd_rand800_train.json'
val_json1 = './f30k_butd_rand100_val.json'
test_json1 = './f30k_butd_rand100_test.json'

train_json1 = load_json(train_json1)
val_json1 = load_json(val_json1)
test_json1 = load_json(test_json1)

train_json_count1 = count_unique_images(train_json1)
val_json_count1 = count_unique_images(val_json1)
test_json_count1 = count_unique_images(test_json1)

print(f"Number of unique image names in training JSON: {train_json_count1}")
print(f"Number of unique image names in validation JSON: {val_json_count1}")
print(f"Number of unique image names in test JSON: {test_json_count1}")

train_json2 = './f30k_human_rand800_train.json'
val_json2 = './f30k_human_rand100_val.json'
test_json2 = './f30k_human_rand100_test.json'

train_json2 = load_json(train_json2)
val_json2 = load_json(val_json2)
test_json2 = load_json(test_json2)

train_json_count2 = count_unique_images(train_json2)
val_json_count2 = count_unique_images(val_json2)
test_json_count2 = count_unique_images(test_json2)

print(f"Number of unique image names in training JSON: {train_json_count2}")
print(f"Number of unique image names in validation JSON: {val_json_count2}")
print(f"Number of unique image names in test JSON: {test_json_count2}")
