# Cell 1 - Updated imports
import os
import pydicom
import numpy as np
from pathlib import Path
import nibabel as nib
import SimpleITK as sitk
import logging
import shutil
import json  # Added json import
from datetime import datetime
from typing import Dict, List, Union, Tuple

# Cell 2 - Fixed logging setup
def setup_logging(output_dir: Path):
    """Setup logging configuration"""
    # Create output directory first
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create formatters and handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File Handler
    file_handler = logging.FileHandler(
        output_dir / f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Cell 3 - NIFTI handling functions
def validate_nifti_file(file_path: Path, logger: logging.Logger) -> bool:
    """Validate NIFTI file format"""
    try:
        img = nib.load(str(file_path))
        if img.header['sizeof_hdr'] == 348:  # Standard NIFTI-1 header size
            return True
        logger.error(f"Invalid NIFTI header in {file_path}")
        return False
    except Exception as e:
        logger.error(f"Error validating NIFTI file {file_path}: {str(e)}")
        return False

def process_nifti_file(input_path: Path, output_dir: Path, logger: logging.Logger) -> Path:
    """Process and copy NIFTI file to output directory"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = input_path.stem.replace('.nii', '') + '.nii.gz'
        output_path = output_dir / output_filename
        
        # Load and save as compressed NIFTI
        img = nib.load(str(input_path))
        nib.save(img, str(output_path))
        
        logger.info(f"Successfully processed NIFTI file: {input_path.name}")
        return output_path
    except Exception as e:
        logger.error(f"Error processing NIFTI file {input_path}: {str(e)}")
        return None

# Cell 4 - DICOM sequence identification
def get_sequence_type(dcm: pydicom.dataset.FileDataset) -> str:
    """Enhanced sequence type detection using series description and other metadata"""
    
    # Get all relevant fields for identification
    series_desc = str(getattr(dcm, 'SeriesDescription', '')).lower()
    protocol_name = str(getattr(dcm, 'ProtocolName', '')).lower()
    sequence_name = str(getattr(dcm, 'SequenceName', '')).lower()
    
    # Combine all descriptions for searching
    all_desc = f"{series_desc} {protocol_name} {sequence_name}"
    
    # Extended sequence identifiers
    SEQUENCE_IDENTIFIERS = {
        'T1': ['t1', 't1w', 'mprage', 'tfl', 't1rho', 'spgr', '3d_t1', 't1/se', 't1/ir'],
        'T2': ['t2', 't2w', 't2star', 'ge', 'gre', 'haste', 't2/tse', 'st2', 't2/se', 't2_tse'],
        'FLAIR': ['flair', 'fluid', 'dark_fluid', 'sair', 't2_flair', 'longtr'],
        'DWI': ['dwi', 'diffusion', 'dti', 'trace', 'tensor', 'adc', 'dw'],
        'SWI': ['swi', 'susceptibility', 'venobold', 'swan'],
        'PD': ['pd', 'proton', 'density'],
        'ADC': ['adc', 'apparent diffusion', 'dw_adc'],
        'DCE': ['dce', 'dynamic', 'contrast_enhanced'],
        'B0': ['b0', 'b_0', 'sb0'],
        'B1500': ['b1500', 'b_1500', 'sb1500'],
        'DP': ['dp', 'sdp'],  # Proton Density
    }

    # Check each sequence type
    for seq_type, identifiers in SEQUENCE_IDENTIFIERS.items():
        if any(id_str in all_desc for id_str in identifiers):
            return seq_type
            
    # If no match found, use series description as type
    if series_desc and series_desc != 'unknown':
        return series_desc.upper()
            
    return 'UNKNOWN'


# Cell 5 - DICOM metadata extraction
def get_dicom_metadata(dcm: pydicom.dataset.FileDataset) -> dict:
    """Extract comprehensive metadata from DICOM file"""
    metadata = {
        'series_description': getattr(dcm, 'SeriesDescription', 'UNKNOWN'),
        'protocol_name': getattr(dcm, 'ProtocolName', 'UNKNOWN'),
        'sequence_name': getattr(dcm, 'SequenceName', 'UNKNOWN'),
        'scanning_sequence': getattr(dcm, 'ScanningSequence', 'UNKNOWN'),
        'sequence_variant': getattr(dcm, 'SequenceVariant', 'UNKNOWN'),
        'manufacturer': getattr(dcm, 'Manufacturer', 'UNKNOWN'),
        'manufacturer_model': getattr(dcm, 'ManufacturerModelName', 'UNKNOWN'),
        'magnetic_field_strength': getattr(dcm, 'MagneticFieldStrength', 'UNKNOWN'),
        'echo_time': getattr(dcm, 'EchoTime', 'UNKNOWN'),
        'repetition_time': getattr(dcm, 'RepetitionTime', 'UNKNOWN'),
        'slice_thickness': getattr(dcm, 'SliceThickness', 'UNKNOWN'),
        'pixel_spacing': getattr(dcm, 'PixelSpacing', 'UNKNOWN'),
        'matrix_size': f"{getattr(dcm, 'Rows', 'UNKNOWN')}x{getattr(dcm, 'Columns', 'UNKNOWN')}",
        'flip_angle': getattr(dcm, 'FlipAngle', 'UNKNOWN'),
        'patient_position': getattr(dcm, 'PatientPosition', 'UNKNOWN')
    }
    
    # Clean up metadata values
    for key, value in metadata.items():
        if hasattr(value, '__iter__') and not isinstance(value, str):
            metadata[key] = list(value)
    
    return metadata

# Cell 6 - DICOM organization
def organize_dicoms(input_dir: Path, logger: logging.Logger) -> tuple[dict, dict]:
    """Organize DICOM files by series and collect metadata"""
    series_dict = {}
    metadata_dict = {}
    files_checked = 0
    
    logger.info(f"Scanning directory: {input_dir}")
    
    # Recursively find all files
    for dicom_file in input_dir.rglob('*'):
        if dicom_file.is_file():
            files_checked += 1
            try:
                dcm = pydicom.dcmread(str(dicom_file))
                sequence_type = get_sequence_type(dcm)
                metadata = get_dicom_metadata(dcm)
                series_number = getattr(dcm, 'SeriesNumber', None)
                
                series_key = f"{sequence_type}_{series_number}"
                
                if series_key not in series_dict:
                    series_dict[series_key] = []
                    metadata_dict[series_key] = metadata
                    logger.info(f"Found new series: {series_key}")
                    logger.info("Series metadata:")
                    for key, value in metadata.items():
                        logger.info(f"  {key}: {value}")
                        
                series_dict[series_key].append(dicom_file)
                
            except Exception as e:
                logger.debug(f"Error processing file {dicom_file}: {str(e)}")
                continue
    
    logger.info(f"Total files checked: {files_checked}")
    logger.info(f"Valid series found: {len(series_dict)}")
    
    return series_dict, metadata_dict

# Cell 7 - DICOM to NIFTI conversion
def convert_to_nifti(series_dict: dict, metadata_dict: dict, output_dir: Path, logger: logging.Logger) -> dict:
    """Convert DICOM series to NIFTI format with metadata preservation"""
    nifti_files = {}
    
    for series_name, dicom_files in series_dict.items():
        try:
            logger.info(f"Converting series {series_name}")
            
            # Create series output directory
            series_output_dir = output_dir / series_name
            series_output_dir.mkdir(exist_ok=True)
            
            # Sort and validate DICOM files
            valid_dicom_files = []
            for dcm_file in dicom_files:
                try:
                    dcm = pydicom.dcmread(str(dcm_file), force=True)
                    if hasattr(dcm, 'InstanceNumber') and hasattr(dcm, 'SliceLocation'):
                        valid_dicom_files.append((dcm.SliceLocation, dcm.InstanceNumber, str(dcm_file)))
                except Exception as e:
                    logger.warning(f"Skipping invalid DICOM file {dcm_file}: {str(e)}")
                    continue
            
            if not valid_dicom_files:
                logger.error(f"No valid DICOM files found for series {series_name}")
                continue
            
            # Sort by slice location first, then instance number
            valid_dicom_files.sort()
            sorted_files = [f[2] for f in valid_dicom_files]
            
            try:
                # Try using SimpleITK first
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(sorted_files)
                reader.MetaDataDictionaryArrayUpdateOn()
                reader.LoadPrivateTagsOn()
                
                try:
                    image = reader.Execute()
                except Exception as sitk_error:
                    logger.warning(f"SimpleITK failed, trying pydicom fallback for {series_name}: {str(sitk_error)}")
                    # Fallback to pydicom if SimpleITK fails
                    first_dcm = pydicom.dcmread(sorted_files[0])
                    array = np.zeros((len(sorted_files), first_dcm.Rows, first_dcm.Columns), dtype=np.float32)
                    
                    for idx, file_path in enumerate(sorted_files):
                        dcm = pydicom.dcmread(file_path)
                        array[idx] = dcm.pixel_array * getattr(dcm, 'RescaleSlope', 1) + getattr(dcm, 'RescaleIntercept', 0)
                    
                    # Create NIFTI with correct orientation
                    affine = np.eye(4)
                    if hasattr(first_dcm, 'PixelSpacing'):
                        affine[0,0] = first_dcm.PixelSpacing[0]
                        affine[1,1] = first_dcm.PixelSpacing[1]
                    if hasattr(first_dcm, 'SliceThickness'):
                        affine[2,2] = first_dcm.SliceThickness
                    
                    image = nib.Nifti1Image(array, affine)
                    
                # Save NIFTI
                nifti_path = series_output_dir / f"{series_name}.nii.gz"
                if isinstance(image, sitk.Image):
                    sitk.WriteImage(image, str(nifti_path))
                else:
                    nib.save(image, str(nifti_path))
                
                # Save metadata as JSON
                metadata_path = series_output_dir / f"{series_name}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata_dict[series_name], f, indent=2, default=str)
                
                logger.info(f"Successfully converted {series_name}")
                nifti_files[series_name] = {
                    'nifti_path': nifti_path,
                    'metadata_path': metadata_path,
                    'metadata': metadata_dict[series_name]
                }
                
            except Exception as e:
                logger.error(f"Error during conversion of {series_name}: {str(e)}")
                continue
            
        except Exception as e:
            logger.error(f"Error processing series {series_name}: {str(e)}")
            
    return nifti_files

# Cell 8 - Results analysis
def analyze_results(output_files: Dict[str, dict], logger: logging.Logger) -> dict:
    """Analyze the processed files"""
    analysis = {
        'total_files': len(output_files),
        'sequences': {
            'T1': 0, 'T2': 0, 'FLAIR': 0, 'DWI': 0, 'SWI': 0, 
            'PD': 0, 'BOLD': 0, 'ASL': 0, 'UNKNOWN': 0
        },
        'file_sizes': {},
        'dimensions': {},
        'metadata_summary': {},
        'warnings': []
    }
    
    for name, file_info in output_files.items():
        try:
            nifti_path = file_info['nifti_path']
            metadata = file_info.get('metadata', {})
            
            # Get sequence type
            seq_type = name.split('_')[0] if '_' in name else 'UNKNOWN'
            if seq_type in analysis['sequences']:
                analysis['sequences'][seq_type] += 1
            else:
                analysis['sequences']['UNKNOWN'] += 1
            
            # Get file size
            file_size = nifti_path.stat().st_size / (1024 * 1024)  # Size in MB
            analysis['file_sizes'][name] = f"{file_size:.2f} MB"
            
            # Get NIFTI information
            img = nib.load(str(nifti_path))
            dimensions = img.shape
            analysis['dimensions'][name] = dimensions
            
            # Combine all information
            analysis['metadata_summary'][name] = {
                'dimensions': dimensions,
                'voxel_size': img.header.get_zooms(),
                'data_type': str(img.header.get_data_dtype()),
            }
            if metadata:
                analysis['metadata_summary'][name].update(metadata)
            
            # Check for potential issues
            if file_size < 1:
                analysis['warnings'].append(f"{name}: Unusually small file size")
            if any(d < 64 for d in dimensions):
                analysis['warnings'].append(f"{name}: Unusual dimensions {dimensions}")
            
        except Exception as e:
            analysis['warnings'].append(f"Error analyzing {name}: {str(e)}")
    
    return analysis

# Cell 9 - Report generation
def print_analysis_report(analysis: dict):
    """Print detailed analysis report"""
    print("\n=== Neuroimaging Data Analysis Report ===")
    
    print("\nSequence Distribution:")
    for seq_type, count in analysis['sequences'].items():
        if count > 0:
            print(f"  {seq_type}: {count}")
    
    print("\nDetailed File Information:")
    for name, metadata in analysis['metadata_summary'].items():
        print(f"\n{name}:")
        print(f"  Size: {analysis['file_sizes'][name]}")
        print(f"  Dimensions: {metadata['dimensions']}")
        print(f"  Voxel Size (mm): {metadata['voxel_size']}")
        
        # Print sequence-specific metadata
        if 'echo_time' in metadata:
            print(f"  Echo Time (TE): {metadata['echo_time']}")
        if 'repetition_time' in metadata:
            print(f"  Repetition Time (TR): {metadata['repetition_time']}")
        if 'slice_thickness' in metadata:
            print(f"  Slice Thickness: {metadata['slice_thickness']}")
        if 'manufacturer' in metadata:
            print(f"  Scanner: {metadata['manufacturer']} {metadata.get('manufacturer_model', '')}")
        
    if analysis['warnings']:
        print("\nWarnings:")
        for warning in analysis['warnings']:
            print(f"  ! {warning}")

# Cell 10 (continued) - Main pipeline execution
def run_pipeline(input_path: Path, output_dir: Path):
    """Main pipeline execution"""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("Starting neuroimaging preprocessing pipeline")
    
    try:
        output_files = {}
        
        # Process input path
        if input_path.is_file():
            if input_path.suffix in ['.nii', '.gz']:
                if validate_nifti_file(input_path, logger):
                    processed_file = process_nifti_file(input_path, output_dir, logger)
                    if processed_file:
                        output_files[input_path.stem] = {
                            'nifti_path': processed_file,
                            'metadata': None
                        }
            else:
                logger.error(f"Unsupported file type: {input_path}")
        
        elif input_path.is_dir():
            # Process NIFTI files
            nifti_files = list(input_path.glob('**/*.nii*'))
            for nifti_file in nifti_files:
                if validate_nifti_file(nifti_file, logger):
                    processed_file = process_nifti_file(nifti_file, output_dir, logger)
                    if processed_file:
                        output_files[nifti_file.stem] = {
                            'nifti_path': processed_file,
                            'metadata': None
                        }
            
            # Process DICOM files
            series_dict, metadata_dict = organize_dicoms(input_path, logger)
            if series_dict:
                dicom_nifti_files = convert_to_nifti(series_dict, metadata_dict, output_dir, logger)
                output_files.update(dicom_nifti_files)
        
        if not output_files:
            logger.error("No valid files processed")
            return None, None
        
        # Analyze results
        analysis = analyze_results(output_files, logger)
        
        # Print summary
        print_analysis_report(analysis)
        
        return output_files, analysis
        
    except Exception as e:
        logger.error(f"Critical error in pipeline: {str(e)}")
        return None, None

# Cell 11 - Example usage
if __name__ == "__main__":
    # Example paths - modify these as needed
    # input_path = Path("datasets/DICOM")
    # output_dir = Path("outputs")
    
    # Run the pipeline
    output_files, analysis = run_pipeline(input_path, output_dir)
    
    if output_files and analysis:
        print("\nProcessing completed successfully!")
        print(f"Processed {len(output_files)} files")
        print(f"Output directory: {output_dir}")
    else:
        print("\nProcessing failed or no valid files found.")