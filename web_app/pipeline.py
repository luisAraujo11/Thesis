"""
Brain MRI Processing Pipeline
----------------------------
A streamlined pipeline for processing brain MRI data using dcm2niix,
supporting both DICOM and NIFTI inputs with basic validation.
"""

import shutil
import pydicom
import logging
import subprocess
import nibabel as nib
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

class ValidationError(Exception): 
    """Custom exception for validation failures"""
    pass

class BrainMRIValidator:
    """Validates input files for brain MRI processing"""
    
    @staticmethod
    def validate_file(file_path: Path) -> dict:
        """
        Validate input file and return file information.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            dict: File information including type and validation status
            
        Raises:
            ValidationError: If file validation fails or file type is unsupported
        """
        # Check by extension first
        if str(file_path).lower().endswith(('.dcm', '.ima', '.img', '.DCM')):
            return BrainMRIValidator._validate_dicom(file_path)
        elif file_path.suffix.lower() in {'.nii', '.gz'}:
            return BrainMRIValidator._validate_nifti(file_path)
        
        # Try DICOM if extension unknown
        try:
            return BrainMRIValidator._validate_dicom(file_path)
        except:
            raise ValidationError(f"Unsupported file type: {file_path.suffix}")
    
    @staticmethod
    def _validate_dicom(file_path: Path) -> dict:
        """Basic DICOM validation and metadata extraction"""
        dcm = pydicom.dcmread(str(file_path), stop_before_pixels=True)
        
        # Verify it's a brain MRI
        if not BrainMRIValidator._is_brain_mri(dcm):
            raise ValidationError("Not a brain MRI")

        # Get manufacturer info
        manufacturer = dcm.get((0x0008, 0x0070), None)
        if manufacturer is None:
            raise ValidationError("Unknown manufacturer")
        
        return {
            'type': 'dicom',
            'manufacturer': manufacturer.value,
            'series_description': getattr(dcm, 'SeriesDescription', 'UNKNOWN'),
            'sequence_type': BrainMRIValidator._get_sequence_type(dcm),
            'validation_status': 'valid'
        }
    
    @staticmethod
    def _validate_nifti(file_path: Path) -> dict:
        """Basic NIFTI validation"""
        img = nib.load(str(file_path))

        # Basic validation checks
        if img.header['sizeof_hdr'] != 348:  # Standard NIFTI-1 header size
            raise ValidationError("Invalid NIFTI header size")
        
        if len(img.shape) not in (3, 4):
            raise ValidationError("Invalid number of dimensions")
            
        # Check voxel sizes
        zooms = img.header.get_zooms()
        if any(z <= 0 or z > 10 for z in zooms[:3]):
            raise ValidationError("Invalid voxel dimensions")
        
        return {
            'type': 'nifti',
            'dimensions': img.shape,
            'voxel_size': zooms,
            'validation_status': 'valid'
        }
    
    @staticmethod
    def _is_brain_mri(dcm: pydicom.dataset.FileDataset) -> bool:
        """
        Determine if DICOM is a brain MRI based on:
        - Body part examined
        - Study description
        - Protocol name
        """
        body_part = getattr(dcm, 'BodyPartExamined', '').lower()
        study_desc = getattr(dcm, 'StudyDescription', '').lower()
        protocol = getattr(dcm, 'ProtocolName', '').lower()
        
        brain_indicators = {'brain', 'head', 'skull', 'cranial', 'cerebral'}
        
        return (
            any(ind in body_part for ind in brain_indicators) or
            any(ind in study_desc for ind in brain_indicators) or
            any(ind in protocol for ind in brain_indicators)
        )

    @staticmethod
    def _get_sequence_type(dcm: pydicom.dataset.FileDataset) -> str:
        """Identify MRI sequence type from DICOM metadata"""
        sequence_info = str(getattr(dcm, 'SeriesDescription', '')).lower()

        sequence_patterns = {
            'T1': ['t1', 'mprage', 'tfl', 'spgr', 'fspgr', 'tfe', 'ffe'],
            'T2': ['t2', 'tse', 'fse'],
            'FLAIR': ['flair', 'dark_fluid', 'fluid_attenuated'],
            'DWI': ['dwi', 'diffusion', 'dti', 'trace', 'adc'],
            'SWI': ['swi', 'susceptibility', 'swan', 't2_star', 't2star']
        }
        
        for seq_type, patterns in sequence_patterns.items():
            if any(pattern in sequence_info for pattern in patterns):
                return seq_type
                
        return 'UNKNOWN'

def is_duplicate(file_path: Path, output_dir: Path) -> bool:
    """Check if file already exists by filename and size"""
    output_path = output_dir / file_path.name
    
    if not output_path.exists():
        return False
    
    # Compare file sizes (fast check)
    return file_path.stat().st_size == output_path.stat().st_size

def convert_dicoms(input_dir: Path, output_dir: Path, logger: logging.Logger) -> Tuple[List[Path], dict]:
    """
    Convert DICOM files to NIFTI format using dcm2niix.
    
    This function:
    1. Checks for existing processed files first
    2. If none exist, performs the conversion
    3. Captures detailed metadata about the process
    
    Args:
        input_dir: Directory containing DICOM files
        output_dir: Where to save converted files
        logger: Logger instance for tracking
        
    Returns:
        Tuple containing:
        - List of generated NIFTI files
        - Dictionary of conversion parameters and metadata
    """
    logger.info(f"Starting DICOM conversion: {input_dir}")
    
    # Construct dcm2niix command
    cmd = [
        'dcm2niix',
        '-z', 'y',              # Compress output
        '-b', 'y',              # Create BIDS sidecar
        '-ba', 'n',             # Don't anonymize
        '-f', '%p_%s_%d',       # Filename format
        '-o', str(output_dir),
        str(input_dir)
    ]
    
    # Execute dcm2niix
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Conversion failed: {result.stderr}")
        raise Exception("dcm2niix conversion failed")
    
    # Find all generated files
    nifti_files = list(output_dir.glob('*.nii.gz'))
    logger.info(f"Generated {len(nifti_files)} NIFTI files")
    
    parameters = {
        'tool': 'dcm2niix',
        'conversion_time': datetime.now().isoformat(),
        'output_files': [str(f.name) for f in nifti_files]
    }
    
    return nifti_files, parameters

def run_pipeline(input_path: Path, output_dir: Path, logger: logging.Logger = None) -> Tuple[dict, dict]: 
    """
    Main pipeline function for processing neuroimaging data.
    Handles both single files and directories, with support for DICOM to NIFTI conversion.
    
    Args:
        input_path: Source path containing neuroimaging files
        output_dir: Destination for processed files
        logger: Optional logger for tracking processing steps
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    logger.info(f"Starting pipeline processing for: {input_path}")
    
    validator = BrainMRIValidator()
    processed_files = {}
    rejected_files = []
    duplicate_files = []
    
    if input_path.is_dir():
        all_files = list(input_path.rglob('*'))
        logger.info(f"Found {len(all_files)} total files")
        
        # Find DICOM files
        dicom_files = []
        for file in all_files:
            if file.is_file():
                try:
                    file_info = validator.validate_file(file)
                    logger.info("file info: %s", file_info)
                    if file_info['type'] == 'dicom':
                        dicom_files.append(file)
                        logger.info(f"Valid DICOM file: {file}")
                except ValidationError as ve:
                    rejected_files.append({
                        'file': file.name,
                        'reason': str(ve)
                    })
                    continue
        
        if dicom_files:
            logger.info(f"Found {len(dicom_files)} DICOM files")
            # Process files
            nifti_files, params = convert_dicoms(input_path, output_dir, logger)
            
            for nifti_path in nifti_files:
                processed_files[nifti_path.stem] = {
                    'path': nifti_path,
                    'parameters': params,
                    'status': 'new'
                }
        
        # Process existing NIFTI files
        nifti_files = [f for f in all_files if f.is_file() and f.suffix in {'.nii', '.gz'}]
        for file_path in nifti_files:
            try:
                file_info = validator.validate_file(file_path)
                
                # Check for duplicates
                if is_duplicate(file_path, output_dir):
                    duplicate_files.append(file_path.name)
                    logger.info(f"Skipped duplicate file: {file_path.name}")
                    continue
                
                output_path = output_dir / file_path.name
                shutil.copy2(file_path, output_path)
                processed_files[output_path.stem] = {
                    'path': output_path,
                    'parameters': file_info,
                    'status': 'new'
                }
                logger.info(f"Processed NIFTI file: {file_path}")
            except ValidationError as ve:
                rejected_files.append({
                    'file': file_path.name,
                    'reason': str(ve)
                })
    
    # Generate analysis
    analysis = {
        'total_files': len(processed_files),
        'new_files': len([f for f in processed_files.values() if f['status'] == 'new']),
        'rejected_files': rejected_files,
        'duplicate_files': duplicate_files,
    }
    
    # Return results
    if analysis['total_files'] > 0:
        logger.info(f"Successfully processed {analysis['new_files']} files "
                   f"(skipped {len(duplicate_files)} duplicates)")
        return processed_files, analysis
    else:
        if rejected_files:
            logger.error(f"No valid brain MRI files found. Rejected {len(rejected_files)} files")
        return None, None