import os
from pathlib import Path
import nibabel as nib
import ants
from neuromaps import datasets
import logging
import tempfile

def normalize_to_mni152_ants(input_file, output_dir=None):
    """
    Normalize a NIFTI file to MNI152 space using ANTs with neuromaps template
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Load input image
        logger.info(f"Loading input file: {input_file}")
        input_path = Path(input_file)
        
        # Load image using ANTs
        moving_image = ants.image_read(str(input_path))
        logger.info(f"Input image shape: {moving_image.shape}")
        
        # Get MNI template from neuromaps
        template_dict = datasets.fetch_atlas('MNI152', '2mm')
        template_path = template_dict['2009cAsym_T1w']  # Using T1w template
        logger.info(f"Using template: {template_path}")
        
        # Load template with ANTs
        fixed_image = ants.image_read(str(template_path))
        
        # Run registration
        logger.info("Starting ANTs registration...")
        registration = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            #type_of_transform='SyN',
            #reg_iterations=(100, 70, 50),  # Fine-tuned iterations
            #aff_iterations=(2100, 1200, 1200, 10),
            #syn_metric='CC',  # Cross-correlation similarity metric
            verbose=True
        )
        
        # Set up output path
        if output_dir is None:
            output_dir = input_path.parent
        output_path = Path(output_dir) / f"{input_path.stem}_mni152.nii.gz"
        
        # Save the normalized image
        registration['warpedmovout'].to_filename(str(output_path))
        logger.info(f"Saved normalized image to: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error during normalization: {str(e)}")
        raise

if __name__ == "__main__":
    input_nifti = "ADNI_3T_14M4_TS_2_2_MP-RAGE.nii.gz"
    
    try:
        normalized_file = normalize_to_mni152_ants(input_nifti)
        print(f"Successfully normalized file to: {normalized_file}")
    except Exception as e:
        print(f"Error during normalization: {e}")