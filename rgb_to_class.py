import numpy as np
import cv2
import os
import tqdm

class_to_rgb = {
    1: [245, 150, 100], # "car"
    2: [245, 230, 100], # "bicycle"
    3: [250, 80, 100],  # "bus"
    4: [150, 60, 30],   # "motorcycle"
    5: [255, 0, 0],     # "on-rails"
    6: [180, 30, 80],   # "truck"
    7: [255, 0, 0],     # "other-vehicle"
    8: [255, 0, 255],   # "road"
    9: [255, 150, 255], # "parking"
    10: [75, 0, 75],     # "sidewalk"
    11: [75, 0, 175],    # "other-ground"
    12: [0, 200, 255],   # "building"
    13: [50, 120, 255],  # "fence"
    14: [0, 150, 255],   # "other-structure"
    15: [170, 255, 150], # "lane-marking"
    16: [0, 175, 0],     # "vegetation"
    17: [0, 60, 135],    # "trunk"
    18: [80, 240, 150],  # "terrain"
    19: [150, 240, 255], # "pole"
    0: [0, 0, 255],     # "traffic-sign"
    20: [255, 255, 50],  # "other-object"
}

rgb_to_class = {tuple(value): key for key, value in class_to_rgb.items()}

def convert_rgb_to_labels(rgb_image, mapping):
    label_image = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)
    for key, value in mapping.items():
        # Create a mask where the RGB values match the class
        mask = np.all(rgb_image == np.array(key, dtype=np.uint8), axis=-1)
        label_image[mask] = value
    return label_image

# Function to print classes present in an image
def print_classes_in_image(image_path, rgb_to_class):
    # Load image and convert to RGB
    image = cv2.imread(image_path)

    # Get unique colors in the image
    unique_colors = np.unique(image.reshape(-1, 3), axis=0)

    # Initialize sets for found classes and unknown values
    found_classes = set()
    unknown_values = set()

    # Check each unique color
    for color in unique_colors:
        color_tuple = tuple(color)
        if color_tuple in rgb_to_class:
            found_classes.add(rgb_to_class[color_tuple])
            print("Classes found in the image:", rgb_to_class[color_tuple])
        else:
            unknown_values.add(color_tuple)
            print("Unknown value found in the image:", color_tuple)

def main():
    input_folder = os.path.join(os.path.expanduser('~'), "data/christian/bev/bev")
    output_folder = os.path.join(os.path.expanduser('~'), "data/christian/bev/labels")
    images = os.listdir(input_folder)
    for image_name in tqdm.tqdm(images):
        input_image = os.path.join(input_folder, image_name)
        output_name = image_name.replace('color', 'labels')
        output_image = os.path.join(output_folder, output_name)
        # Example usage
        rgb_label = cv2.imread(input_image)  # Load your RGB label image
        print_classes_in_image(input_image, rgb_to_class)
        class_label = convert_rgb_to_labels(rgb_label, rgb_to_class)
        cv2.imwrite(output_image, class_label)  # Save the converted label
    
if __name__ == "__main__":
    main()
