train_json2 = load_json(train_json2)
val_json2 = load_json(val_json2)
test_json2 = load_json(test_json2)

train_json_count2 = count_unique_images(train_json2)
val_json_count2 = count_unique_images(val_json2)
test_json_count2 = count_unique_images(test_json2)

print(f"Number of unique image names in training JSON: {train_json_count2}")
print(f"Number of unique image names in validation JSON: {val_json_count2}")
print(f"Number of unique image names in test JSON: {test_json_count2}")