"""
Add this code to your pipeline.py file to explore and print all DICOM metadata
from a directory structure.
"""

from pathlib import Path
import pydicom
import pandas as pd
from collections import defaultdict
import numpy as np

def explore_dicom_directory(base_dir: Path, verbose: bool = True) -> dict:
    """
    Recursively explore a directory structure and print all DICOM metadata.
    
    Args:
        base_dir: Path to the base directory containing DICOM files/folders
        verbose: Whether to print detailed information during processing
    
    Returns:
        dict: Summary of found datasets and their properties
    """
    def print_separator(length=80):
        print("\n" + "="*length + "\n")

    base_dir = Path(base_dir)
    dataset_info = defaultdict(list)
    
    print(f"\nExploring directory: {base_dir}")
    print_separator()

    # Walk through all subdirectories
    for file_path in base_dir.rglob('*'):
        if not file_path.is_file():
            continue
            
        try:
            # Try to read as DICOM
            dcm = pydicom.dcmread(str(file_path), force=True)
            
            # Extract key information
            study_id = getattr(dcm, 'StudyID', 'Unknown')
            series_desc = getattr(dcm, 'SeriesDescription', 'Unknown')
            patient_id = getattr(dcm, 'PatientID', 'Unknown')
            relative_path = file_path.relative_to(base_dir)
            
            # Store information
            dataset_info['File Path'].append(str(relative_path))
            dataset_info['Patient ID'].append(patient_id)
            dataset_info['Study ID'].append(study_id)
            dataset_info['Series Description'].append(series_desc)
            
            # Print detailed information if verbose
            if verbose:
                print(f"\nFile: {relative_path}")
                print(f"Directory: {file_path.parent.name}")
                print("\nBasic Information:")
                print(f"Patient ID: {patient_id}")
                print(f"Study ID: {study_id}")
                print(f"Series Description: {series_desc}")
                
                # Print all available metadata
                print("\nDetailed Metadata:")
                for elem in dcm:
                    try:
                        if elem.VR != 'SQ':  # Skip sequence elements
                            if hasattr(elem, 'name'):
                                name = elem.name
                            else:
                                name = str(elem.tag)
                            
                            if elem.VM > 1:  # Handle multiple values
                                value = str(elem.value)
                            else:
                                value = str(elem.value)
                                
                            # Skip pixel data
                            if elem.keyword != 'PixelData':
                                print(f"{name}: {value}")
                    except Exception as e:
                        print(f"Error reading element: {str(e)}")
                
                # Print image information if available
                if hasattr(dcm, 'pixel_array'):
                    print("\nImage Information:")
                    print(f"Image Shape: {dcm.pixel_array.shape}")
                    print(f"Image Data Type: {dcm.pixel_array.dtype}")
                    print(f"Pixel Spacing: {getattr(dcm, 'PixelSpacing', 'Unknown')}")
                    print(f"Slice Thickness: {getattr(dcm, 'SliceThickness', 'Unknown')}")
                
                print_separator()
            
        except Exception as e:
            if verbose:
                print(f"Skipping non-DICOM file: {file_path}")
            continue
    
    # Create summary DataFrame
    df = pd.DataFrame(dataset_info)
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print(f"Total number of DICOM files: {len(df)}")
    print(f"Number of unique patients: {df['Patient ID'].nunique()}")
    print(f"Number of unique studies: {df['Study ID'].nunique()}")
    print("\nSeries Descriptions found:")
    for desc in df['Series Description'].unique():
        count = len(df[df['Series Description'] == desc])
        print(f"  - {desc}: {count} files")
    
    return dict(dataset_info)

def print_dicom_folder_structure(base_dir: Path):
    """
    Print the folder structure containing DICOM files with basic information.
    
    Args:
        base_dir: Path to the base directory
    """
    base_dir = Path(base_dir)
    
    print(f"\nFolder Structure for: {base_dir}\n")
    
    # Track unique studies and series
    studies = defaultdict(lambda: defaultdict(set))
    
    for file_path in base_dir.rglob('*'):
        if not file_path.is_file():
            continue
            
        try:
            dcm = pydicom.dcmread(str(file_path), stop_before_pixels=True)
            study_id = getattr(dcm, 'StudyID', 'Unknown')
            series_desc = getattr(dcm, 'SeriesDescription', 'Unknown')
            studies[study_id][series_desc].add(file_path)
        except:
            continue
    
    # Print structure
    for study_id, series_dict in studies.items():
        print(f"\nStudy ID: {study_id}")
        for series_desc, files in series_dict.items():
            print(f"  └── Series: {series_desc}")
            print(f"      └── Number of files: {len(files)}")
            print(f"      └── Location: {next(iter(files)).parent.relative_to(base_dir)}")

# Example usage:
if __name__ == "__main__":
    # Replace with your dataset directory path
    dataset_dir = Path("uploads/upload_20241213_142526")
    
    print("=== FOLDER STRUCTURE ===")
    print_dicom_folder_structure(dataset_dir)
    
    print("\n=== DETAILED METADATA ===")
    metadata = explore_dicom_directory(dataset_dir, verbose=True)