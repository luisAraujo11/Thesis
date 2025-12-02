import os
import shutil
import pandas as pd
import json
import sys

def main():
    # Get command-line arguments
    if len(sys.argv) < 4:
        print("Usage: python process_MRIs_modality.py <dest_dir> <modality_dir> <csv_file> [modality] [modality_terms]")
        sys.exit(1)
    
    dest_dir = sys.argv[1]
    modality_dir = sys.argv[2]
    csv_file = sys.argv[3]
    modality = sys.argv[4] if len(sys.argv) > 4 else "t1"
    
    # Define modality terms from command line if provided or use defaults
    if len(sys.argv) > 5:
        modality_terms = sys.argv[5].lower().split(',')
        print(f"Using custom modality terms: {modality_terms}")
    else:
        # Default modality terms based on the selected modality
        if modality.lower() == "t1":
            modality_terms = [
                't1',        # Basic T1
                't1w',       # T1-weighted
                't1-w',      # T1-weighted variant
                't_1',       # Alternative spacing
                't-1',       # Alternative spacing
                'mprage',    # MPRAGE sequence
                'mp-rage',   # MPRAGE with hyphen
                'mp2rage',   # MP2RAGE sequence
                'spgr',      # Spoiled Gradient Echo
                'fspgr',     # Fast SPGR
                'bravo',     # GE's BRAVO T1 sequence
                'tfl',       # TurboFLASH (Siemens)
                '3d_ir',     # 3D IR sequence
                '3dir',      # 3D IR abbreviated
                'flash3d',   # 3D FLASH sequence
                't1_fl3d'    # T1 3D FLASH
            ]
        elif modality.lower() == "t2": # the terms may need to be updated
            modality_terms = [
                't2', 't2w', 't2-w', 't_2', 't-2', 'tse', 'fse', 'frfse', 'cube'
            ]
        elif modality.lower() == "flair": # same here
            modality_terms = [
                'flair', 'fluid', 'attenuated', 'inversion', 'recovery', 'dark_fluid'
            ]
        else:
            # Default to T1 terms if modality not recognized
            modality_terms = ['t1', 'mprage']
            print(f"Warning: Unrecognized modality '{modality}'. Using default terms.")
        
        print(f"Using default {modality} terms: {modality_terms}")

    # Create the destination directories if they don't exist
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(modality_dir, exist_ok=True)

    # Grant permissions to the destination directories
    os.chmod(dest_dir, 0o755)
    os.chmod(modality_dir, 0o755)

    # Read the subjects from the CSV file (third column)
    subjects_df = pd.read_csv(csv_file, header=None, sep=';')
    subjects = subjects_df[2]
    missing_subjects = [] # List to keep track of missing subjects

    # Function to check if a NIfTI file is a specific modality using the corresponding .json file
    def is_modality_match(nifti_file):
        json_file = nifti_file.replace('.nii', '.json').replace('.nii.gz', '.json')
        if not os.path.exists(json_file): # skip if json doesnt exist
            print(f"JSON file {json_file} does not exist.")
            return False
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)
                series_description = metadata.get('SeriesDescription', '').lower()
                if any(term in series_description for term in modality_terms):
                    return True
                ProtocolName = metadata.get('ProtocolName', '').lower()
                if any(term in ProtocolName for term in modality_terms):
                    return True
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
        return False

    # Copy the folders to destination and extract modality images
    processed = 0
    for idx, subject in enumerate(subjects):
        subject_stripped = subject.replace('.zip', '')
        dest_path = os.path.join(dest_dir, subject_stripped)
        found_modality = False # Flag to check if the modality is found
        if os.path.exists(dest_path):
            for root, _, files in os.walk(dest_path):  # parse through the extracted files
                for file in files:
                    file_path = os.path.join(root, file)
                    if (file_path.endswith('.nii') or file_path.endswith('.nii.gz')) and is_modality_match(file_path):
                        modality_subject_dir = os.path.join(modality_dir, subject_stripped)
                        os.makedirs(modality_subject_dir, exist_ok=True)
                        shutil.copy(file_path, modality_subject_dir)
                        processed += 1
                        found_modality = True
                        # copy the corresponding .json file
                        json_file = file_path.replace('.nii', '.json').replace('.nii.gz', '.json')
                        if os.path.exists(json_file):
                            shutil.copy(json_file, modality_subject_dir)
            if not found_modality:
                # Add subject to the missing_subjects list
                missing_subjects.append({
                    "NACCID": subjects_df.iloc[idx, 1],  #  NACCID is in the second column
                    "NACCMRFI": subjects_df.iloc[idx, 2]  #  NACCMRFI is in the third column
                })
        else:
            print(f"Folder {dest_path} does not exist.")

    # Save missing subjects to a CSV file
    if missing_subjects:
        missing_subjects_df = pd.DataFrame(missing_subjects)
        missing_csv_path = os.path.join(modality_dir, f"missing_{modality}_subjects.csv")
        missing_subjects_df.to_csv(missing_csv_path, index=False)
        print(f"Saved missing subjects to {missing_csv_path}")
    
    print(f"Processed {processed} {modality} files")

if __name__ == "__main__":
    main()