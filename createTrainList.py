import os

# Function to check if a file is an image file based on its extension
def is_image_file(file_path):
    # List of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    # Check if the file path ends with any of the image extensions
    return any(file_path.lower().endswith(ext) for ext in image_extensions)

# Function to parse files in the directory and generate output files
def parse_files_to_directory(parse_directory, ground_truth_directory, output_directory):
    # Define paths for output files
    train_lst_path = os.path.join(output_directory, "train.lst")
    train_pair_edge_lst_path = os.path.join(output_directory, "train_pair_edge.lst")
    
    # Open output files for writing
    with open(train_lst_path, "w") as train_lst_file, open(train_pair_edge_lst_path, "w") as train_pair_edge_lst_file:
        # Iterate through each file in the parse directory
        for filename in os.listdir(parse_directory):
            # Get full paths for the object detected and ground truth files
            full_file_path = os.path.join(parse_directory, filename)
            full_gt_file_path = os.path.join(ground_truth_directory, filename)
            
            # Check if both files are image files
            if is_image_file(full_file_path) and is_image_file(full_gt_file_path):
                # Construct relative paths for files in train.lst
                filename_path = os.path.basename(parse_directory) + "/" + filename
                gt_filename, _ = os.path.splitext(filename)
                gt_file_path = os.path.basename(ground_truth_directory) + "/" + gt_filename + ".png"
                
                # Construct relative paths for files in train_pair_edge.lst
                gt_edge_file_path = os.path.basename(ground_truth_directory) + "/" + gt_filename + "_edge.png"
                
                # Write paths to train.lst file
                train_lst_file.write(filename_path + " " + gt_file_path + "\n")
                
                # Write paths to train_pair_edge.lst file
                train_pair_edge_lst_file.write(filename_path + " " + gt_file_path + " " + gt_edge_file_path + "\n")

# Specify the directories
parse_directory = "./test_datasets/cssd/images"
ground_truth_directory = "./test_datasets/cssd/ground_truth_mask"
output_directory = "./output_files"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Call the function to parse files and generate output files
parse_files_to_directory(parse_directory, ground_truth_directory, output_directory)
