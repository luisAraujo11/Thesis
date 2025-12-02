import os
import nibabel as nib
from neuromaps import datasets, images, nulls, resampling, stats
import logging
import tempfile

# Set custom temp directory
os.environ['TMPDIR'] = '/notebooks/disk2/tmp'  # Change this to your desired path
os.environ['TEMP'] = '/notebooks/disk2/tmp'
os.environ['TMP'] = '/notebooks/disk2/tmp'

# Create the directory if it doesn't exist
os.makedirs('/notebooks/disk2/tmp', exist_ok=True)

# Set tempfile to use this directory
tempfile.tempdir = '/notebooks/disk2/tmp'

def analyze_brain_map(input_nifti):
    """
    Analyze brain map using neuromaps following official examples
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set temp directory first
    os.environ['TMPDIR'] = '/notebooks/disk2/tmp'
    os.makedirs('/notebooks/disk2/tmp', exist_ok=True)
    tempfile.tempdir = '/notebooks/disk2/tmp'
    
    try:
        # Load your normalized image
        logger.info(f"Loading normalized image: {input_nifti}")
        your_map = nib.load(input_nifti)
        
        # Fetch annotation maps
        logger.info("Fetching reference maps...")
        neurosynth = datasets.fetch_annotation(source='neurosynth')
        
        # Resample images to common space
        logger.info("Resampling to common space...")
        your_map, neurosynth = resampling.resample_images(
            your_map, 
            neurosynth, 
            'MNI152', 
            'MNI152'
        )
        
        # Generate null maps
        logger.info("Generating null maps...")
        rotated = nulls.burt2020(
            neurosynth,
            atlas='MNI152',
            density='2mm',
            n_perm=100,
            seed=1234
        )
        
        # Compare maps
        logger.info("Computing correlation...")
        corr, pval = stats.compare_images(your_map, neurosynth, nulls=rotated)
        
        print("\nResults:")
        print(f"Correlation: r = {corr:.3f}, p = {pval:.3f}")
        
        return {
            'correlation': corr,
            'pvalue': pval
        }
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    input_file = "B0_603_mni152.nii.gz"
    
    try:
        results = analyze_brain_map(input_file)
        print("\nAnalysis completed successfully!")
    except Exception as e:
        print(f"Analysis failed: {e}")