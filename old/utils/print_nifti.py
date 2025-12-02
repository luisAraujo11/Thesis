import nibabel as nib
from pathlib import Path
from typing import Union, Dict, Tuple
import numpy as np

def identify_neuromaps_space(filepath: Union[str, Path]) -> Tuple[str, Dict]:
    """
    Identify which of the four neuromaps coordinate systems a NIFTI file is using:
    MNI-152, fsaverage, fsLR, or CIVET.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the NIFTI file (.nii or .nii.gz)
        
    Returns:
    --------
    tuple
        (coordinate_system, evidence_dict) where:
        - coordinate_system is one of: 'MNI-152', 'fsaverage', 'fsLR', 'CIVET', or 'unknown'
        - evidence_dict contains the reasoning behind the identification
    """
    try:
        img = nib.load(filepath)
        header = img.header
        affine = img.affine
        data = img.get_fdata()
        
        evidence = {}
        
        # Get basic properties
        dimensions = header.get_data_shape()
        voxel_size = header.get_zooms()
        sform_code = int(header['sform_code'])
        qform_code = int(header['qform_code'])
        
        # Check for MNI-152 indicators
        mni_evidence = []
        if sform_code == 4 or qform_code == 4:
            mni_evidence.append("Header explicitly indicates MNI152 space")
        if dimensions[:3] in [(182, 218, 182), (91, 109, 91), (256, 256, 256)]:
            mni_evidence.append(f"Dimensions {dimensions[:3]} match standard MNI152 template")
        if np.allclose(voxel_size[:3], [1, 1, 1], atol=0.1) or np.allclose(voxel_size[:3], [2, 2, 2], atol=0.1):
            mni_evidence.append(f"Voxel size {voxel_size[:3]} matches standard MNI152 resolution")
        evidence['MNI-152'] = mni_evidence
        
        # Check for fsaverage indicators
        fs_evidence = []
        if "fsaverage" in str(filepath).lower():
            fs_evidence.append("Filename contains 'fsaverage'")
        if dimensions[0] in [163842, 40962, 10242, 2562, 642]:  # Common FreeSurfer mesh densities
            fs_evidence.append(f"Vertex count {dimensions[0]} matches FreeSurfer standard mesh")
        evidence['fsaverage'] = fs_evidence
        
        # Check for fsLR indicators
        fslr_evidence = []
        if any(x in str(filepath).lower() for x in ["fslr", "fs_lr", "fsaverage_lr"]):
            fslr_evidence.append("Filename indicates fsLR space")
        if dimensions[0] in [32492, 163842]:  # Standard fsLR densities
            fslr_evidence.append(f"Vertex count {dimensions[0]} matches fsLR standard mesh")
        evidence['fsLR'] = fslr_evidence
        
        # Check for CIVET indicators
        civet_evidence = []
        if "civet" in str(filepath).lower():
            civet_evidence.append("Filename indicates CIVET space")
        if dimensions[0] in [41962, 163842]:  # Standard CIVET densities
            civet_evidence.append(f"Vertex count {dimensions[0]} matches CIVET standard mesh")
        evidence['CIVET'] = civet_evidence
        
        # Determine the most likely space
        space_scores = {
            'MNI-152': len(mni_evidence),
            'fsaverage': len(fs_evidence),
            'fsLR': len(fslr_evidence),
            'CIVET': len(civet_evidence)
        }
        
        # Get the space with the most evidence
        max_score = max(space_scores.values())
        if max_score == 0:
            likely_space = 'unknown'
        else:
            likely_spaces = [space for space, score in space_scores.items() if score == max_score]
            likely_space = likely_spaces[0] if len(likely_spaces) == 1 else 'ambiguous'
        
        # Print detailed analysis
        print("\nNeuromaps Coordinate System Analysis")
        print("=" * 40)
        print(f"\nMost likely space: {likely_space}")
        print("\nEvidence found:")
        for space, findings in evidence.items():
            if findings:  # Only show spaces where we found evidence
                print(f"\n{space}:")
                for finding in findings:
                    print(f"  - {finding}")
        
        if likely_space == 'unknown':
            print("\nWarning: Could not definitively determine coordinate system.")
            print("Consider checking the file's documentation or source.")
            
        elif likely_space == 'ambiguous':
            print("\nWarning: Multiple possible coordinate systems detected.")
            print("Please verify the correct space with your data's documentation.")
        
        return likely_space, evidence
        
    except Exception as e:
        print(f"Error analyzing NIFTI file: {str(e)}")
        raise

# Example usage in Jupyter notebook:
if __name__ == "__main__":
    file_path = "B0_603.nii_mni152.nii.gz"
    space, evidence = identify_neuromaps_space(file_path)

# print de tudo
    
# import nibabel as nib
# from pathlib import Path
# from typing import Union, Dict
# import numpy as np
# import pandas as pd
# from datetime import datetime

# def display_nifti_info(filepath: Union[str, Path]) -> Dict:
#     """
#     Display comprehensive information about a NIFTI file, including header details,
#     template space information, and data statistics.
    
#     Parameters:
#     -----------
#     filepath : str or Path
#         Path to the NIFTI file (.nii or .nii.gz)
        
#     Returns:
#     --------
#     dict
#         Dictionary containing all extracted information
#     """
#     try:
#         # Load the NIFTI file
#         img = nib.load(filepath)
#         header = img.header
#         affine = img.affine
#         data = img.get_fdata()
        
#         # Basic file information
#         file_info = {
#             'File Path': str(filepath),
#             'File Size': Path(filepath).stat().st_size / (1024 * 1024),  # Size in MB
#             'Last Modified': datetime.fromtimestamp(Path(filepath).stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
#         }
        
#         # Header information
#         header_info = {
#             'Data Type': str(header.get_data_dtype()),
#             'Dimensions': header.get_data_shape(),
#             'Voxel Size (mm)': header.get_zooms(),
#             'Units': header.get_xyzt_units(),
#             'Byte Order': header.endianness,
#             'Header Size': header.sizeof_hdr
#         }
        
#         # Template space information
#         template_info = {
#             'qform Code': (int(header['qform_code']), _explain_form_code(int(header['qform_code']))),
#             'sform Code': (int(header['sform_code']), _explain_form_code(int(header['sform_code']))),
#             'Orientation': nib.aff2axcodes(affine),
#             'Origin (mm)': affine[:3, 3].tolist()
#         }
        
#         # Data statistics
#         data_stats = {
#             'Min Value': float(np.min(data)),
#             'Max Value': float(np.max(data)),
#             'Mean Value': float(np.mean(data)),
#             'Std Dev': float(np.std(data)),
#             'Non-zero Voxels': int(np.sum(data != 0)),
#             'Total Voxels': int(np.prod(data.shape))
#         }
        
#         # Transformation matrices
#         transform_info = {
#             'Affine Matrix': affine.tolist(),
#             'Qform Matrix': header.get_qform(),
#             'Sform Matrix': header.get_sform()
#         }
        
#         # Print formatted information
#         print("\n=== NIFTI File Information ===")
#         print("=" * 50)
        
#         # File information
#         print("\nFile Details:")
#         print("-" * 20)
#         for key, value in file_info.items():
#             if key == 'File Size':
#                 print(f"{key:.<30} {value:.2f} MB")
#             else:
#                 print(f"{key:.<30} {value}")
        
#         # Header information
#         print("\nHeader Information:")
#         print("-" * 20)
#         for key, value in header_info.items():
#             print(f"{key:.<30} {value}")
        
#         # Template space information
#         print("\nTemplate Space Information:")
#         print("-" * 20)
#         for key, value in template_info.items():
#             if key in ['qform Code', 'sform Code']:
#                 code, explanation = value
#                 print(f"{key:.<30} {code} ({explanation})")
#             else:
#                 print(f"{key:.<30} {value}")
        
#         # Data statistics
#         print("\nData Statistics:")
#         print("-" * 20)
#         for key, value in data_stats.items():
#             if isinstance(value, float):
#                 print(f"{key:.<30} {value:.6f}")
#             else:
#                 print(f"{key:.<30} {value}")
        
#         # Template space analysis
#         print("\nTemplate Space Analysis:")
#         print("-" * 20)
#         print(_analyze_template_space(header, affine))
        
#         # Transformation matrices
#         print("\nTransformation Matrices:")
#         print("-" * 20)
#         for key, matrix in transform_info.items():
#             print(f"\n{key}:")
#             matrix_array = np.array(matrix)
#             for row in matrix_array:
#                 print(" ".join(f"{x:10.4f}" for x in row))
        
#         # Combine all information
#         all_info = {
#             'file_info': file_info,
#             'header_info': header_info,
#             'template_info': template_info,
#             'data_stats': data_stats,
#             'transform_info': transform_info
#         }
        
#         return all_info
        
#     except Exception as e:
#         print(f"Error reading NIFTI file: {str(e)}")
#         raise

# def _explain_form_code(code: int) -> str:
#     """Provide explanation for qform and sform codes."""
#     codes = {
#         0: "Unknown coordinate system",
#         1: "Scanner-based anatomical coordinates",
#         2: "Coordinates aligned to another file's coordinates",
#         3: "Coordinates in Talairach space",
#         4: "Coordinates in MNI152 space"
#     }
#     return codes.get(code, "Unknown code")

# def _analyze_template_space(header, affine) -> str:
#     """Analyze the likely template space based on header and affine information."""
#     description = []
    
#     # Check if it's likely MNI space
#     if header['sform_code'] == 4 or header['qform_code'] == 4:
#         description.append("- Image appears to be in MNI152 space")
    
#     # Check voxel dimensions for standard templates
#     voxel_size = header.get_zooms()[:3]
#     if np.allclose(voxel_size, [1.0, 1.0, 1.0], atol=0.1):
#         description.append("- 1mm isotropic resolution (common in standard templates)")
#     elif np.allclose(voxel_size, [2.0, 2.0, 2.0], atol=0.1):
#         description.append("- 2mm isotropic resolution (common in functional MRI templates)")
    
#     # Check dimensions for common templates
#     dimensions = header.get_data_shape()
#     if dimensions[:3] == (256, 256, 256):
#         description.append("- 256³ dimensions (typical of 1mm MNI152 template)")
#     elif dimensions[:3] == (182, 218, 182):
#         description.append("- Dimensions match standard 1mm MNI152 template")
#     elif dimensions[:3] == (91, 109, 91):
#         description.append("- Dimensions match standard 2mm MNI152 template")
    
#     if not description:
#         description.append("- Unable to definitively determine template space")
#         description.append("- Consider checking documentation or header information")
    
#     return "\n".join(description)

# # Example usage in Jupyter notebook:
# if __name__ == "__main__":
#     # Replace with your NIFTI file path
#     file_path = "path/to/your/file.nii.gz"
#     nifti_info = display_nifti_info(file_path)
    
#     # Optional: Create a more interactive display using pandas
#     def create_info_dataframe(info_dict):
#         rows = []
#         for category, items in info_dict.items():
#             if category != 'transform_info':  # Handle matrices separately
#                 for key, value in items.items():
#                     rows.append({
#                         'Category': category,
#                         'Property': key,
#                         'Value': str(value)
#                     })
#         return pd.DataFrame(rows)
    
#     # Display as DataFrame in Jupyter
#     df = create_info_dataframe(nifti_info)
#     display(df)