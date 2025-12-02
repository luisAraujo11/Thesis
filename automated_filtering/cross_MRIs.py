import os
import shutil
import pandas as pd
import sys

def main():
    # Get command-line arguments
    if len(sys.argv) != 4:
        print("Usage: python cross_MRIs.py <modality_dir> <cross_subject_dir> <csv_file>")
        sys.exit(1)
    
    modality_dir = sys.argv[1]
    cross_subject_dest_dir = sys.argv[2]
    csv_file = sys.argv[3]
    
    print(f"Cross-referencing: modality_dir={modality_dir}, cross_dir={cross_subject_dest_dir}, csv={csv_file}")

    os.makedirs(modality_dir, exist_ok=True)
    os.makedirs(cross_subject_dest_dir, exist_ok=True)

    os.chmod(modality_dir, 0o755)
    os.chmod(cross_subject_dest_dir, 0o755)

    cross_subjects = pd.read_csv(csv_file, header=None, sep=';')[2]
    print(f"Found {len(cross_subjects)} subjects in CSV")

    successful_subjects = [] # list of subjects that were successfully processed to add to the csv file

    # cross the subjects from the modality directory with the csv file
    for subject in cross_subjects:
        subject_stripped = subject.replace('.zip', '')
        modality_path = os.path.join(modality_dir, subject_stripped)
        if os.path.exists(modality_path):
            successful_subjects.append(subject)
            for root, _, files in os.walk(modality_path):  # parse through the extracted files
                for file in files:
                    file_path = os.path.join(root, file)
                    if (file_path.endswith('.nii') or file_path.endswith('.nii.gz')): # check if it's a NIfTI file
                        cross_subject_dir = os.path.join(cross_subject_dest_dir, subject_stripped)
                        os.makedirs(cross_subject_dir, exist_ok=True)
                        shutil.copy(file_path, cross_subject_dir)
                        # copy the corresponding .json file
                        json_file = file_path.replace('.nii', '.json').replace('.nii.gz', '.json')
                        if os.path.exists(json_file):
                            shutil.copy(json_file, cross_subject_dir)
        else:
            print(f"Folder {modality_path} does not exist.")

    # create a csv file that contains sex, age at MR (NACCMRIA) and deceased age (NACCDAGE) with the cross subjects
    cross_subjects_csv = pd.read_csv(csv_file, header=None, sep=';')
    cross_subjects_csv = cross_subjects_csv[cross_subjects_csv[2].isin(successful_subjects)]
    cross_subjects_csv.to_csv(os.path.join(cross_subject_dest_dir, 'cross_subjects.csv'), index=False, header=False)
    
    print(f"Successfully processed {len(successful_subjects)} subjects")
    print(f"Created cross_subjects.csv with {len(cross_subjects_csv)} entries")

if __name__ == "__main__":
    main()