import pandas as pd

# Load the file into a DataFrame
file_path = '/home/kyueran/caption-generation/flickr30k/results.csv'
df = pd.read_csv(file_path, sep='|', header=None, names=['image_name', 'comment_number', 'comment'])

# Group by image name and concatenate comments
grouped = df.groupby('image_name')['comment'].apply(lambda comments: ' '.join(comments)).reset_index()

# Save the combined captions to a new file
output_file_path = 'combined_captions.csv'
grouped.to_csv(output_file_path, sep='|', index=False, header=['image_name', 'combined_comment'])

# Display the result
print(grouped)
