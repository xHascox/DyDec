import os
import shutil
from PIL import Image
import detect_pedestrian

DATASET_PATH = "./dataset/dsv5/"
DELETED_PATH = './output/dsv5_DELETE' # images without 2 pedestrians get moved here

def filter_pedestrian_images(directory, to_delete_directory="DELETE"):
    os.makedirs(to_delete_directory, exist_ok=True)
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        to_delete_path = os.path.join(to_delete_directory, filename)

        try:
            with Image.open(file_path) as img:
                # Check if the image contains a pedestrian
                if not detect_pedestrian.contains_pedestrian(img, file_path, to_delete_directory):
                    shutil.move(file_path, to_delete_path)
                    print(f'Moved to delete: {to_delete_path}')
                else:
                    print(f'Contains pedestrian: {file_path}')
        except OSError:
            # If the file cannot be opened as an image, skip it
            print(f'Skipped: {file_path} (not a valid image)')

filter_pedestrian_images(DATASET_PATH, DELETED_PATH)
