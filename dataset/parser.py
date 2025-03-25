import os
import shutil

def move_files():
    # Get the current working directory
    current_folder = os.getcwd()

    # Iterate through all files in the current folder
    for file in os.listdir(current_folder):
        # Check if the file has a .docx extension
        if file.endswith('.docx'):
            # Remove the .docx extension for the folder name
            folder_name = os.path.splitext(file)[0]
            
            # Create a new subdirectory named after the file
            new_folder_path = os.path.join(current_folder, folder_name)
            os.makedirs(new_folder_path, exist_ok=True)
            
            # Move the .docx file to the new subdirectory
            source_file = os.path.join(current_folder, file)
            destination_file = os.path.join(new_folder_path, file)
            shutil.move(source_file, destination_file)
            
            print(f"Moved {file} to {new_folder_path}")

    print("All .docx files have been moved to their respective subdirectories.")


import zipfile

def move_and_organize_files():
    # Get the current working directory
    current_folder = os.getcwd()

    # Iterate through all files in the current folder
    for file in os.listdir(current_folder):
        file_path = os.path.join(current_folder, file)

        # Skip if not a file
        if not os.path.isfile(file_path):
            continue

        # Check each folder in the current directory
        for folder in os.listdir(current_folder):
            folder_path = os.path.join(current_folder, folder)

            # Skip if not a directory
            if not os.path.isdir(folder_path):
                continue

            # Check if folder name (case-insensitive) is in the file name
            if folder.lower() in file.lower():
                # Move the file to the folder
                destination_file = os.path.join(folder_path, file)
                shutil.move(file_path, destination_file)
                print(f"Moved {file} to {folder_path}")

                # If the file is a .zip file, unzip it
                if file.lower().endswith('.zip'):
                    with zipfile.ZipFile(destination_file, 'r') as zip_ref:
                        zip_ref.extractall(folder_path)
                        print(f"Unzipped {file} in {folder_path}")
                break

    print("File organization complete.")

if __name__ == "__main__":
    move_and_organize_files()
