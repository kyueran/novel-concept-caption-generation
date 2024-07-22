import pandas as pd
import json
import random

# Function to convert CSV data to the desired JSON format for training
def csv_to_json_individual_pairs(csv_file_path):
    # Read the CSV file with the correct delimiter
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
    # Read the CSV file with the correct delimiter
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

# Function to split a CSV file based on image names from train, val, and test sets
def split_csv_by_image_names(csv_file_path, train_image_names, val_image_names, test_image_names):
    # Read the CSV file with the correct delimiter
    df = pd.read_csv(csv_file_path, delimiter='|')
    
    # Print columns to debug
    print("Columns in the output CSV DataFrame:", df.columns)
    
    # Ensure the column names are stripped of any leading/trailing whitespace
    df.columns = df.columns.str.strip()
    
    # Split the DataFrame based on image names
    train_df = df[df['image_name'].isin(train_image_names)]
    val_df = df[df['image_name'].isin(val_image_names)]
    test_df = df[df['image_name'].isin(test_image_names)]
    
    return train_df, val_df, test_df

# Function to convert DataFrame to JSON format
def df_to_json(df, image_column='image_name', caption_column='caption'):
    result = df.rename(columns={image_column: "image", caption_column: "caption"})[['image', 'caption']].to_dict(orient='records')
    return result

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

# Extract image names for train, val, and test sets
train_image_names = [item['image'] for item in train_data]
val_image_names = [item['image'] for item in val_data]
test_image_names = [item['image'] for item in test_data]

# Save the splits to JSON files
save_to_json(train_data, train_file_path)
save_to_json(val_data, val_file_path)
save_to_json(test_data, test_file_path)

# Split ../output_captions.csv based on the image names from the train, val, and test sets
output_captions_csv_path = '../output_captions.csv'
train_output_json_path = './flickr30k_butd_train.json'
val_output_json_path = './flickr30k_butd_val.json'
test_output_json_path = './flickr30k_butd_test.json'

train_output_df, val_output_df, test_output_df = split_csv_by_image_names(output_captions_csv_path, train_image_names, val_image_names, test_image_names)

# Convert DataFrames to JSON format
train_output_data = df_to_json(train_output_df, image_column='image_name', caption_column='caption')
val_output_data = df_to_json(val_output_df, image_column='image_name', caption_column='caption')
test_output_data = df_to_json(test_output_df, image_column='image_name', caption_column='caption')

# Save the splits to new JSON files
save_to_json(train_output_data, train_output_json_path)
save_to_json(val_output_data, val_output_json_path)
save_to_json(test_output_data, test_output_json_path)

print(f"Saved training data to {train_file_path} and {train_output_json_path}")
print(f"Saved validation data to {val_file_path} and {val_output_json_path}")
print(f"Saved test data to {test_file_path} and {test_output_json_path}")
