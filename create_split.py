import os
import json
import random


def create_dataset_splits(data_dir, train_percent, val_percent, output_file):
    """
    Create dataset splits and save as JSON.

    :param data_dir: Directory containing the data files.
    :param train_percent: Percentage of data to use for training.
    :param val_percent: Percentage of data to use for validation.
    :param output_file: Path to output JSON file.
    """
    # List all files in the data directory
    all_files = os.listdir(data_dir)
    # Assuming files are .png, change if different
    all_files = [f.split('_bev')[0] for f in all_files if f.endswith('.png')]  # extract part of filename

    # Shuffle the files to ensure random distribution
    random.shuffle(all_files)

    # Calculate number of files for each split
    total_files = len(all_files)
    num_train = int(total_files * train_percent)
    num_val = int(total_files * val_percent)
    num_test = total_files - num_train - num_val

    # Split files into train, validation, and test
    train_files = all_files[:num_train]
    val_files = all_files[num_train:num_train + num_val]
    test_files = all_files[num_train + num_val:]

    # Create a dictionary with the splits
    dataset_splits = {
        'train': train_files,
        'validation': val_files,
        'test': test_files
    }

    # Save the dictionary to a JSON file
    with open(output_file, 'w') as file:
        json.dump(dataset_splits, file, indent=4)

    print(f"Dataset splits saved to {output_file}")


def main():
    # Example usage
    create_dataset_splits(data_dir=os.path.expanduser('~') + '/data/christian/bev/labels',
                          train_percent=0.6, 
                          val_percent=0.2, 
                          output_file=os.path.expanduser('~') + '/data/christian/bev/dataset_splits.json')


if __name__ == "__main__":
    main()
