import pandas as pd
import json
import random

# Function to convert CSV data to the desired JSON format for training
def csv_to_json_individual_pairs(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path, delimiter='|')
    
    # Print columns to debug
    print("Columns in the DataFrame:", df.columns)
    
    # Ensure the column names are stripped of any leading/trailing whitespace
    df.columns = df.columns.str.strip()
    
    # Check if the required columns are present
    if 'image_name' not in df.columns or 'comment' not in df.columns:
        raise KeyError("Required columns 'image_name' or 'comment' not found in the CSV file")
    
    # Create a list of dictionaries with individual image-caption pairs
    result = df.rename(columns={"image_name": "image", "comment": "caption"})[['image', 'caption']].to_dict(orient='records')
    
    return result

# Function to convert CSV data to the desired JSON format for validation and test
def csv_to_json_grouped(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path, delimiter='|')
    
    # Print columns to debug
    print("Columns in the DataFrame:", df.columns)
    
    # Ensure the column names are stripped of any leading/trailing whitespace
    df.columns = df.columns.str.strip()
    
    # Check if the required columns are present
    if 'image_name' not in df.columns or 'comment' not in df.columns:
        raise KeyError("Required columns 'image_name' or 'comment' not found in the CSV file")
    
    # Group by 'image_name' and aggregate 'comment' into a list
    grouped = df.groupby('image_name')['comment'].apply(list).reset_index()
    
    # Convert the DataFrame to a list of dictionaries
    result = grouped.rename(columns={"image_name": "image", "comment": "caption"}).to_dict(orient='records')
    
    return result

# Function to split data into validation and test sets
def split_val_test_data(data, val_percent=0.5, seed=9):
    # Set the seed for reproducibility
    random.seed(seed)
    
    # Shuffle the data to ensure randomness
    random.shuffle(data)
    
    # Calculate the split indices
    total = len(data)
    val_end = int(total * val_percent)
    
    # Split the data
    val_data = data[:val_end]
    test_data = data[val_end:]
    
    return val_data, test_data

# Function to save the data to JSON files
def save_to_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Example usage
csv_file_path = './results.csv'
train_file_path = './flickr30k_human_train.json'
val_file_path = './flickr30k_human_val.json'
test_file_path = './flickr30k_human_test.json'

# Convert CSV to JSON format for training
train_data = csv_to_json_individual_pairs(csv_file_path)

# Convert CSV to JSON format for validation and test
val_test_data = csv_to_json_grouped(csv_file_path)

# Split the data for validation and test
val_data, test_data = split_val_test_data(val_test_data, val_percent=0.5)  # Split remaining data equally into val and test

# Save the splits to JSON files
save_to_json(train_data, train_file_path)
save_to_json(val_data, val_file_path)
save_to_json(test_data, test_file_path)
