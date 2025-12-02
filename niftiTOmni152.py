import logging
import nibabel as nib
from pathlib import Path
from neuromaps import datasets
from nilearn.image import resample_img

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_to_mni152(input_file, output_dir=None):
    """
    Normalize a NIFTI file to MNI-152 space using nilearn's pure Python implementation.
    
    This function performs the following steps:
    1. Loads the input NIFTI file
    2. Gets the MNI-152 template from neuromaps
    3. Resamples the input image to match the template's space
    4. Saves the normalized result
    
    Parameters:
    -----------
    input_file : str or Path
        Path to the input NIFTI file
    output_dir : str or Path, optional
        Directory to save the output. If None, saves in same directory as input
    
    Returns:
    --------
    Path
        Path to the normalized output file
    """
    input_path = Path(input_file)
    
    # Input validation
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    try:
        # Load the input file
        logger.info(f"Loading input file: {input_path}")
        img = nib.load(str(input_path))
        logger.info(f"Input image shape: {img.shape}")
        logger.info(f"Input voxel sizes: {img.header.get_zooms()}")
        
        # Get MNI152 template
        template_dict = datasets.fetch_atlas('MNI152','2mm')
        template_path = template_dict['2009cAsym_T1w']
        logger.info(f"Using MNI152 T1w template: {template_path}")
        
        # Load template
        template_img = nib.load(template_path)
        logger.info(f"Template shape: {template_img.shape}")
        logger.info(f"Template voxel sizes: {template_img.header.get_zooms()}")
        
        # Set up output path
        if output_dir is None:
            output_dir = input_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
        output_path = output_dir / f"{input_path.stem}_mni152.nii.gz"
        
        # Perform the registration using nilearn's resample_img
        logger.info("Starting registration to MNI152 space...")
        resampled_img = resample_img(
            img,
            target_affine=template_img.affine,
            target_shape=template_img.shape,
            interpolation='continuous'  # Use continuous interpolation for better quality
        )
        
        # Save the normalized image
        nib.save(resampled_img, str(output_path))
        logger.info(f"Registration complete. Output saved to: {output_path}")
        
        # Verify the output
        out_img = nib.load(str(output_path))
        logger.info(f"Output image shape: {out_img.shape}")
        logger.info(f"Output voxel sizes: {out_img.header.get_zooms()}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error during normalization: {str(e)}")
        raise
        
if __name__ == "__main__":
    # Path to your NIFTI file
    input_nifti = "B0_603.nii.gz"  # Update this to your file path
    
    try:
        normalized_file = normalize_to_mni152(input_nifti)
        print(f"Successfully normalized file to: {normalized_file}")
    except Exception as e:
        print(f"Error during normalization: {e}")

# templates do neuromaps
# Using MNI152 template: {'2009cAsym_T1w': PosixPath('/root/neuromaps-data/atlases/MNI152/tpl-MNI152NLin2009cAsym_res-1mm_T1w.nii.gz'), '2009cAsym_T2w': PosixPath('/root/neuromaps-data/atlases/MNI152/tpl-MNI152NLin2009cAsym_res-1mm_T2w.nii.gz'), '2009cAsym_PD': PosixPath('/root/neuromaps-data/atlases/MNI152/tpl-MNI152NLin2009cAsym_res-1mm_PD.nii.gz'), '2009cAsym_brainmask': PosixPath('/root/neuromaps-data/atlases/MNI152/tpl-MNI152NLin2009cAsym_res-1mm_desc-brain_mask.nii.gz'), '2009cAsym_CSF': PosixPath('/root/neuromaps-data/atlases/MNI152/tpl-MNI152NLin2009cAsym_res-1mm_label-csf_probseg.nii.gz'), '2009cAsym_GM': PosixPath('/root/neuromaps-data/atlases/MNI152/tpl-MNI152NLin2009cAsym_res-1mm_label-gm_probseg.nii.gz'), '2009cAsym_WM': PosixPath('/root/neuromaps-data/atlases/MNI152/tpl-MNI152NLin2009cAsym_res-1mm_label-wm_probseg.nii.gz'), '6Asym_T1w': PosixPath('/root/neuromaps-data/atlases/MNI152/tpl-MNI152NLin6Asym_res-1mm_T1w.nii.gz'), '6Asym_brainmask': PosixPath('/root/neuromaps-data/atlases/MNI152/tpl-MNI152NLin6Asym_res-1mm_desc-brain_mask.nii.gz')}

"""
Templates in the Code:

The keys in the code list different specific data modalities and tissue masks that correspond to different parts of the MNI152 template. Here’s what they represent:

    2009cAsym_T1w:
        T1-weighted MRI scan of the brain (anatomical image).
        This type of scan provides a detailed image of the brain's gray matter, white matter, and other tissues.

    2009cAsym_T2w:
        T2-weighted MRI scan (another type of anatomical scan).
        T2 scans are more sensitive to fluid and may highlight CSF (cerebrospinal fluid) and other tissue types.

    2009cAsym_PD:
        Proton density (PD) weighted scan.
        This scan type provides contrast between various brain tissues, helping to visualize gray matter, white matter, and CSF in a different way than T1 or T2.

    2009cAsym_brainmask:
        This is a binary mask that includes only the brain tissue and excludes non-brain regions (e.g., skull or background).
        Used to ensure that only the brain is considered in analyses.

    2009cAsym_CSF:
        A mask or image specifically for cerebrospinal fluid (CSF) regions.
        Used to isolate or examine the CSF in the brain.

    2009cAsym_GM:
        A mask or image specifically for gray matter (GM).
        Gray matter includes areas of the brain involved in sensory processing and muscle control.

    2009cAsym_WM:
        A mask or image specifically for white matter (WM).
        White matter connects different regions of the brain and is involved in communication between different brain areas.

Template Name	            When to Use
MNI152 1mm T1w	            Use for detailed anatomical analysis of cortical regions (e.g., cortical thickness or small ROI analysis).
MNI152 2mm T1w	            Use for general brain structure analysis with reasonable resolution. Ideal for studies of large regions or global analysis.
MNI152 6mm T1w	            Use for larger-scale analyses, such as functional connectivity or group-level studies where high resolution is less important.
MNI152 1mm T2w	            Use when you need to study fluid-filled structures (ventricles) or white matter in greater detail.
MNI152 2mm T2w	            Use for whole-brain white matter analysis or when T2 contrast is important but computational efficiency is also needed.
MNI152 PD	                Use for analyzing tissue contrasts and distinguishing between gray matter, white matter, and CSF in a single map.
MNI152 Brainmask            Use to exclude non-brain regions (e.g., skull) and focus only on brain tissue during analysis.
MNI152 GM Mask	            Use when your focus is on gray matter, such as studying cortical regions or cognitive functions.
MNI152 WM Mask	            Use for studies involving white matter (e.g., tractography, connectivity, brain networks).
MNI152 CSF Mask	            Use when studying CSF or ventricular expansion (e.g., in neurodegenerative diseases).
Asymmetry Templates (Asym)	Use when analyzing brain asymmetry or hemispheric differences (e.g., lateralization of functions like language or motor skills).
"""