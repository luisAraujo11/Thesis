"""
Brain MRI Processing Pipeline
----------------------------
A streamlined pipeline for processing brain MRI data using dcm2niix,
supporting both DICOM and NIFTI inputs with basic validation.
"""

import os
import subprocess
import pydicom
import nibabel as nib
import dcm2niix
import logging
from pathlib import Path
import shutil
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set

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
        try:
            if str(file_path).lower().endswith(('.dcm', '.ima', '.img', '.DCM')):
                return BrainMRIValidator._validate_dicom(file_path)
            elif file_path.suffix.lower() in {'.nii', '.gz'}:
                return BrainMRIValidator._validate_nifti(file_path)
            
            # Try to validate as DICOM even if extension doesn't match
            try:
                return BrainMRIValidator._validate_dicom(file_path)
            except:
                raise ValidationError(f"Unsupported file type: {file_path.suffix}")
        except Exception as e:
            raise ValidationError(f"File validation failed: {str(e)}")
    
    @staticmethod
    def _validate_dicom(file_path: Path) -> dict:
        """Basic DICOM validation and metadata extraction"""
        try:
            dcm = pydicom.dcmread(str(file_path), stop_before_pixels=True)
            
            # Verify it's a brain MRI
            if not BrainMRIValidator._is_brain_mri(dcm):
                raise ValidationError("Not a brain MRI")

            # Get manufacturer info
            manufacturer = dcm.get((0x0008, 0x0070), None)
            if manufacturer is None:
                raise ValidationError("Unknown manufacturer")
            
            info = {
                'type': 'dicom',
                'manufacturer': manufacturer.value,
                'series_description': getattr(dcm, 'SeriesDescription', 'UNKNOWN'),
                'sequence_type': BrainMRIValidator._get_sequence_type(dcm),
                'validation_status': 'valid'
            }
            
            return info
            
        except Exception as e:
            raise ValidationError(f"DICOM validation failed: {str(e)}")
    
    @staticmethod
    def _validate_nifti(file_path: Path) -> dict:
        """Basic NIFTI validation"""
        try:
            img = nib.load(str(file_path))

            # Basic validation checks
            if img.header['sizeof_hdr'] != 348:  # Standard NIFTI-1 header size
                raise ValidationError("Invalid NIFTI header size")
            
            # Basic validation checks
            if len(img.shape) not in (3, 4):
                raise ValidationError("Invalid number of dimensions")
                
            # Check voxel sizes
            zooms = img.header.get_zooms()
            if any(z <= 0 or z > 10 for z in zooms[:3]):
                raise ValidationError("Invalid voxel dimensions")
            
            info = {
                'type': 'nifti',
                'dimensions': img.shape,
                'voxel_size': zooms,
                'validation_status': 'valid'
            }
            
            return info
            
        except Exception as e:
            raise ValidationError(f"NIFTI validation failed: {str(e)}")
    
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
            'T1': [
                # Standard T1 nomenclature
                't1', 'T1', 't1w', 'T1W', 't1-weighted', 'T1-WEIGHTED',
                # MPRAGE variations
                'mprage', 'MPRAGE', 'mp-rage', 'MP-RAGE', 'mp_rage', 'MP_RAGE',
                # Manufacturer specific (Siemens)
                'tfl', 'TFL', 'fl3d', 'FL3D', 
                # Manufacturer specific (GE)
                'spgr', 'SPGR', 'fspgr', 'FSPGR',
                # Manufacturer specific (Philips)
                'tfe', 'TFE', 'ffe', 'FFE', 't1_tfe', 'T1_TFE',
                # Common variations
                '3d_t1', '3DT1', 't1_3d', 'T1_3D',
                't1_se', 'T1_SE', 't1/se', 'T1/SE'
            ],
            
            'T2': [
                # Standard T2 nomenclature
                't2', 'T2', 't2w', 'T2W', 't2-weighted', 'T2-WEIGHTED',
                'st2', 'sT2', 'ST2',
                # Spin Echo variations
                'tse', 'TSE', 'fse', 'FSE',
                # Manufacturer specific names
                't2_tse', 'T2_TSE', 't2/cor', 'T2/COR',
                # Common variations excluding FLAIR
                '3d_t2', '3DT2', 't2_3d', 'T2_3D'
            ],
            
            'FLAIR': [
                # Standard FLAIR nomenclature
                'flair', 'FLAIR', 'dark_fluid', 'DARK_FLUID',
                'fluid_attenuated', 'FLUID_ATTENUATED',
                # T2 FLAIR variations
                't2_flair', 'T2_FLAIR', 't2w_flair', 'T2W_FLAIR',
                # Long TR variations
                'flair_longtr', 'FLAIR_longTR', 'long_tr_flair',
                # Manufacturer specific
                'space_flair', 'SPACE_FLAIR', 'spc_flair', 'SPC_FLAIR',
                'cube_flair', 'CUBE_FLAIR'
            ],
            
            'DWI': [
                # Standard DWI nomenclature
                'dwi', 'DWI', 'diffusion', 'DIFFUSION',
                # Technical variations
                'dti', 'DTI', 'trace', 'TRACE', 'diff_tensor', 'DIFF_TENSOR',
                # B-value specifications
                'sb0', 'b0', 'SB0', 'B0',  # b=0 images
                'sb1500', 'b1500', 'SB1500', 'B1500',  # b=1500 images
                # ADC related
                'adc', 'ADC', 'dadc', 'dADC',  # ADC maps
                # Diffusion projections
                'dp', 'DP', 'sdp', 'sDP',  # Diffusion projections
                # Additional variations
                'ep2d_diff', 'EP2D_DIFF', 'diff_epi', 'DIFF_EPI'
            ],
            
            'SWI': [
                # Standard SWI nomenclature
                'swi', 'SWI', 'susceptibility', 'SUSCEPTIBILITY',
                # Technical variations
                'swan', 'SWAN', 'venobold', 'VENOBOLD',
                # Phase/magnitude variations
                't2_star', 'T2_STAR', 't2star', 'T2STAR',
                'phase', 'PHASE', 'magnitude', 'MAGNITUDE',
                # Manufacturer specific
                'merge', 'MERGE'
            ]
        }
        
        for seq_type, patterns in sequence_patterns.items():
            if any(pattern in sequence_info for pattern in patterns):
                return seq_type
                
        return 'UNKNOWN'

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
    try:
        logger.info(f"Starting DICOM conversion for directory: {input_dir}")
        
        # Configure dcm2niix parameters
        conversion_params = {
            'compression': 'y',    # Enable compression
            'crop': '0',          # Don't crop images
            'merge': '0',         # Don't merge files
            'floating': '1',      # Use floating point
            'verbose': '1'        # Detailed output
        }
        logger.info(f"Using conversion parameters: {conversion_params}")
        
        # Construct dcm2niix command with descriptive naming

        cmd = [
            'dcm2niix',
            '-z', 'y',     # Compress output
            '-b', 'y',     # Create BIDS sidecar
            '-ba', 'n',    # Don't anonymize
            '-f', '%p_%s_%d',       # Filename format: protocol_series_description
            '-o', str(output_dir),
            str(input_dir)
        ]
                        
        logger.info(f"Running conversion command: {' '.join(cmd)}")
        
        # Execute dcm2niix
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        # Log conversion output
        logger.info("dcm2niix output:")
        for line in result.stdout.split('\n'):
            logger.info(f"  {line}")
        
        if result.returncode != 0:
            logger.error(f"Conversion failed: {result.stderr}")
            raise Exception("dcm2niix conversion failed")
            
        # Find all generated files
        nifti_files = list(output_dir.glob('*.nii.gz'))
        logger.info(f"Generated {len(nifti_files)} NIFTI files")
        
        # Create comprehensive parameters dictionary
        parameters = {
            'tool': 'dcm2niix',
            'command': ' '.join(cmd),
            'parameters': conversion_params,
            'conversion_time': datetime.now().isoformat(),
            'output_files': [str(f.name) for f in nifti_files],
            'existing_files': False
        }
        
        return nifti_files, parameters
        
    except Exception as e:
        logger.error(f"DICOM conversion failed: {str(e)}")
        raise

# TODO futuramente para criar mais resiliencia, devem ser tidos em conta ficheiros repetidos ao fazer upload para nao sobrecarregar
# mudar o handle results, acho que nao esta a fazer nada
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
    existing_files = set()
    rejected_files = []
    
    if input_path.is_dir():
        logger.info(f"Processing directory: {input_path}")
        
        all_files = list(input_path.rglob('*'))
        logger.info(f"Found {len(all_files)} total files")
        
        # Try to validate each file as DICOM
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
            try:
                logger.info("Starting DICOM conversion process")
                nifti_files, params = convert_dicoms(input_path, output_dir, logger)
                
                if params.get('existing_files'):
                    existing_files.update(nifti_files)
                    
                for nifti_path in nifti_files:
                    processed_files[nifti_path.stem] = {
                        'path': nifti_path,
                        'parameters': params,
                        'status': 'existing' if nifti_path in existing_files else 'new'
                    }
                    logger.info(f"Added processed file: {nifti_path}")
                
            except Exception as e:
                logger.error(f"Error during DICOM conversion: {str(e)}")
                return None, None
        
        # Process NIFTI files
        nifti_files = [f for f in all_files if f.is_file() and f.suffix in {'.nii', '.gz'}]
        for file_path in nifti_files:
            try:
                file_info = validator.validate_file(file_path)
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
            except Exception as e:
                logger.warning(f"Error processing NIFTI file {file_path}: {e}")
    
    # Generate comprehensive analysis
    analysis = {
        'total_files': len(processed_files),
        'new_files': len([f for f in processed_files.values() if f['status'] == 'new']),
        'existing_files': len([f for f in processed_files.values() if f['status'] == 'existing']),
        'rejected_files': rejected_files,
        'conversion_parameters': {
            name: info['parameters'] 
            for name, info in processed_files.items()
        }
    }
    
    # Handle results
    if analysis['total_files'] > 0:
        if analysis['new_files'] == 0:
            logger.info("All files already exist in the system")
            return None, None
        else:
            logger.info(f"Successfully processed {analysis['new_files']} new files")
            return processed_files, analysis
    else:
        if rejected_files:
            error_msg = "No valid brain MRI files found for processing.\nRejected files:\n"
            for rejected in rejected_files:
                error_msg += f"- {rejected['file']}: {rejected['reason']}\n"
            logger.error(error_msg.strip())
            return None, None
        else:
            logger.error("No valid files found for processing")
            return None, None