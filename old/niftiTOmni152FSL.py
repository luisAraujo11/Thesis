import os
from pathlib import Path
import nibabel as nib
from nipype.interfaces import fsl
from neuromaps import datasets
import logging
import tempfile

def setup_logging():
    """Set up logging to track the normalization process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def normalize_to_mni152(input_file, output_dir=None):
    """
    Normalize a NIFTI file to MNI-152 space using FSL's FLIRT.
    
    This function performs the following steps:
    1. Verifies the input file exists and is a valid NIFTI
    2. Gets the MNI-152 template from neuromaps
    3. Performs linear registration to align the input to MNI space
    4. Saves the normalized result
    
    Parameters:
    -----------
    input_file : str or Path
        Path to the input NIFTI file
    output_dir : str or Path, optional
        Directory to save the output. If None, saves in the same directory as input
        
    Returns:
    --------
    Path
        Path to the normalized NIFTI file
    """
    logger = setup_logging()
    input_path = Path(input_file)
    
    # Input validation
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    try:
        # Load the input file to verify it's a valid NIFTI
        img = nib.load(str(input_path))
        logger.info(f"Successfully loaded input file: {input_path}")
        logger.info(f"Input image shape: {img.shape}")
        logger.info(f"Input voxel sizes: {img.header.get_zooms()}")
        
    except Exception as e:
        raise ValueError(f"Error loading NIFTI file: {e}")
    
    # Set up output directory and file names
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
    output_path = output_dir / f"{input_path.stem}_mni152.nii.gz"

    # Set FSL environment variables only for this process
    os.environ['FSLDIR'] = '/usr/local/fsl'
    os.environ['PATH'] = f"/usr/local/fsl/bin:{os.environ.get('PATH', '')}"
    os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
    
    # Get MNI152 template from neuromaps
    try:
        template_dict = datasets.fetch_atlas('MNI152','2mm')
        template_path = template_dict['2009cAsym_T1w']
        logger.info(f"Using MNI152 template: {template_path}")
    except Exception as e:
        raise RuntimeError(f"Error fetching MNI152 template: {e}")
    
    # Set up FSL's FLIRT for registration
    logger.info("Starting registration to MNI152 space...")
    
    try:
        flirt = fsl.FLIRT()
        flirt.inputs.in_file = str(input_path)
        flirt.inputs.reference = str(template_path)
        flirt.inputs.out_file = str(output_path)
        flirt.inputs.dof = 12  # 12 degrees of freedom for affine registration
        flirt.inputs.cost = 'corratio'  # correlation ratio cost function
        flirt.inputs.interp = 'spline'  # spline interpolation for smoother result
        
        # Run the registration
        result = flirt.run()
        
        logger.info(f"Registration complete. Output saved to: {output_path}")
        
        # Verify the output file was created
        if not output_path.exists():
            raise RuntimeError("Registration completed but output file not found")
            
        # Load and verify the output
        out_img = nib.load(str(output_path))
        logger.info(f"Output image shape: {out_img.shape}")
        logger.info(f"Output voxel sizes: {out_img.header.get_zooms()}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise RuntimeError(f"Registration failed: {e}")

# Example usage
if __name__ == "__main__":
    # Path to your NIFTI file
    input_nifti = "B0_603.nii.gz"
    
    try:
        normalized_file = normalize_to_mni152(input_nifti)
        print(f"Successfully normalized file to: {normalized_file}")
    except Exception as e:
        print(f"Error during normalization: {e}")