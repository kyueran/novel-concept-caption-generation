import os

def count_jpg_files(directory):
    return len([file for file in os.listdir(directory) if file.endswith('.jpg')])

directory_path = '/home/kyueran/caption-generation/BLIP/annotation/merlion_mod/output_folder_front'
num_jpg_files = count_jpg_files(directory_path)
print(f'There are {num_jpg_files} .jpg files in the directory.')
