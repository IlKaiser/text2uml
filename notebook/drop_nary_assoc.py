import os
import re

def process_plantuml_file(file_path):
    """Process a single plantuml.txt file to replace pattern lines."""
    pattern = re.compile(r"\((\w+),\s*(\w+)\)\s*\.\.\s*(\w+)")
    new_lines = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        match = pattern.search(line)
        if match:
            name1, name2, name3 = match.groups()
            # Replace with two new lines
            new_lines.append(f"{name1} .. {name3}\n")
            new_lines.append(f"{name1} .. {name2}\n")
        else:
            new_lines.append(line)

    # Write back the modified content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

def scan_folder(root_folder):
    """Recursively search for plantuml.txt files and process them."""
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename == "plantuml.txt":
                file_path = os.path.join(dirpath, filename)
                print(f"Processing: {file_path}")
                process_plantuml_file(file_path)

if __name__ == "__main__":
    folder = "./dataset"
    if os.path.isdir(folder):
        scan_folder(folder)
        print("✅ Processing complete.")
    else:
        print("❌ The provided path is not a valid directory.")
