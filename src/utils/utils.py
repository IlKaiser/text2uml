import os
from tqdm import tqdm
from docx import Document

def process_subfolders_with_chain(root_folder_path, chain, type=''):
    """
    Explores subfolders of the root folder (depth 1), processes each subfolder's `text.txt`
    with the provided LangChain chain, and saves the result in a new file in the same folder.

    Args:
        root_folder_path (str): Path to the root folder.
        chain: A LangChain chain instance to process text inputs.
    """
    for subfolder_name in tqdm(os.listdir(root_folder_path)):
        subfolder_path = os.path.join(root_folder_path, subfolder_name)
        
        # Ensure the current item is a subfolder
        if os.path.isdir(subfolder_path):
            text_file_path = os.path.join(subfolder_path, "text.txt")
            
            # Check if `text.txt` exists in the subfolder
            if not os.path.isfile(text_file_path):
                # If `text.txt` is missing, check for any .docx file in the subfolder
                docx_files = [f for f in os.listdir(subfolder_path) if f.endswith(".docx")]
                if docx_files:
                    docx_file_path = os.path.join(subfolder_path, docx_files[0])
                    # Extract content from the .docx file
                    doc = Document(docx_file_path)
                    text_content = "\n".join([paragraph.text for paragraph in doc.paragraphs])

                    # Save the extracted content to `text.txt`
                    with open(text_file_path, "w", encoding="utf-8") as text_file:
                        text_file.write(text_content)

            # Recheck if `text.txt` now exists
            if os.path.isfile(text_file_path):
                with open(text_file_path, "r", encoding="utf-8") as file:
                    text = file.read()

                # Call the LangChain chain with the input dictionary
                result = chain.invoke({"text": text})

                # Save the result to a new file in the same subfolder
                result_file_path = os.path.join(subfolder_path, f"result_{type}.txt")
                with open(result_file_path, "w", encoding="utf-8") as result_file:
                    result_file.write(result)

import gc, torch

def clear_memory():
    """
    Clears memory by running garbage collection and emptying the cache. 
    """
    gc.collect()
    torch.mps.empty_cache()
