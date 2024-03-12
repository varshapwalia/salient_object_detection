import os

def parse_files_to_lst(parse_directory):
    output_file_path = os.path.join(root_folder, "test.lst")
    with open(output_file_path, "w") as lst_file:
        for filename in os.listdir(parse_directory):
            if os.path.isfile(os.path.join(parse_directory, filename)):
                lst_file.write(os.path.basename(parse_directory) + "\\" + filename + "\n")


# Specify the folder where you want to write the .lst file
root_folder = ".\\DUTS-TE"

# Specify the directory you want to parse files from
parse_directory = ".\\DUTS-TE\\DUTS-TE-Image"

parse_files_to_lst(parse_directory)
