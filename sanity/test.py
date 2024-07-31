import numpy as np
from collections import Counter

def main(npy_file_path):
    # Load the .npy file
    data = np.load(npy_file_path, allow_pickle=True)

    # Ensure 'names' field is in the structured array
    if 'names' not in data.dtype.names:
        raise ValueError("The 'names' field is not present in the loaded .npy file.")

    # Extract the 'names' field
    names = data['names']
    print(len(names))

    # Count the occurrences of each name
    name_counts = Counter(names)

    # Find names that are repeated
    repeated_names = [name for name, count in name_counts.items() if count > 1]

    # Print the repeated names and their counts
    if repeated_names:
        print("Repeated names and their counts:")
        for name in repeated_names:
            print(f"{name}: {name_counts[name]}")
    else:
        print("No repeated names found.")

    # Return the repeated names for further use if needed
    return repeated_names

if __name__ == "__main__":
    npy_file_path = "../shared_data/flickr30k_name_features.npy"  # Replace with your .npy file path
    main(npy_file_path)
