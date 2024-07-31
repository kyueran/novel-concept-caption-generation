import numpy as np

# Load the .npy file
npy_file_path = '../shared_data/merlion_1000.npy'
data = np.load(npy_file_path, allow_pickle=True)

# Access the feature vectors
features = data['features']

# Print the shape of each image vector
for i, feature_vector in enumerate(features):
    print(f"Image {i} vector shape: {feature_vector.shape}")
