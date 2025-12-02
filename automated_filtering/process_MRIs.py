import os
import shutil
import pandas as pd
import tempfile
import zipfile
import sys

def main():
    # Get command-line arguments
    if len(sys.argv) != 4:
        print("Usage: python process_MRIs.py <source_dir> <dest_dir> <csv_file>")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    csv_file = sys.argv[3]
    
    print(f"Processing: source={source_dir}, dest={dest_dir}, csv={csv_file}")

    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Read the subjects in the CSV file (third column)
    subjects = pd.read_csv(csv_file, header=None, sep=';')[2]

    # Copy the folders to the destination directory 
    for subject in subjects:
        subject_stripped = subject.replace('.zip', '') # the names of the subjects in the csv file have .zip extension, this removes it
        subject_with_ni = f"{subject_stripped}ni.zip" # then adds it again because all the subjects have nii in their names
        src_path = os.path.join(source_dir, subject_with_ni)
        dest_path = os.path.join(dest_dir, subject_stripped)
        # If path exists and is a zip file
        if src_path.endswith('.zip') and os.path.exists(src_path):
            
            if os.path.exists(dest_path):
                print(f"folder {dest_path} already exists in the destination folder.")
                continue # if already exists, skip to the next iteration
            
            # Create a temporary directory for extraction
            extract_path = tempfile.mkdtemp()
            
            # Extract the zip file to the temporary directory
            with zipfile.ZipFile(src_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            # Copy source to dest 
            shutil.copytree(extract_path, dest_path)
            
            # Clean up the temporary directory when done (this may be too slow, optimize later)
            shutil.rmtree(extract_path)
        else:
            print(f"Directory {src_path} does not exist in the source folder.")

if __name__ == "__main__":
    main()