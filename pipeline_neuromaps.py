import os
import nibabel as nib
from neuromaps import datasets, images, nulls, resampling, stats, transforms
import logging
import numpy as np
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
    Analyze brain map using neuromaps with surface-based approach
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
        
        # Transform volumetric MNI152 to fsaverage surface space
        logger.info("Converting to surface space...")
        #your_map_surface = transforms.mni152_to_fsaverage(your_map)
        
        # Fetch maps (they'll be in surface space)
        logger.info("Fetching reference maps...")
        abagen = datasets.fetch_annotation(source='abagen')
        
        # The maps should already be in fsaverage space, but let's make sure
        logger.info("Resampling to common space...")
        your_map, abagen = resampling.resample_images(
            your_map, 
            abagen, 
            'MNI152', 
            'fsaverage'
        )
        print(your_map)
        # Generate null maps using surface-based rotation
        logger.info("Generating null maps...")
        rotated = nulls.alexander_bloch(
            your_map,
            atlas='fsaverage',
            density='10k',
            n_perm=100,
            seed=1234
        )
        
        # Compare maps
        logger.info("Computing correlation...")
        corr, pval, nullss = stats.compare_images(your_map, abagen, nulls=rotated, return_nulls=True) 
        # nullss : It represents the proportion of null permutations that produce a correlation as extreme or more extreme than your observed correlation. In other words, how often random shuffling of the data produces a result similar to or more extreme than what you actually observed
        
        print("\nResults:")
        print(f"Correlation: r = {corr:.3f}, p = {pval:.3f}, n = {np.mean(nullss):.3f}") 
        
        return {
            'correlation': corr,
            'pvalue': pval,
            'nullss': nullss,
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

"""

Now, let's examine each null model and its appropriate use case:

alexander_bloch()


Best for: Surface-based brain maps (like cortical thickness or surface-based fMRI)
How it works: Rotates the spherical projection of brain maps
When to use:

When your data is already in surface space (like fsaverage)
When comparing cortical patterns
When spatial relationships on the cortical surface are important


Example scenario: Comparing patterns of cortical thickness between groups


vasa()


Best for: Parcellated surface data
How it works: Preserves the spatial structure of parcels while randomizing values
When to use:

When working with brain atlases divided into regions
When you want to maintain parcel adjacency relationships
For region-based analyses rather than vertex-wise comparisons


Example scenario: Analyzing connectivity patterns between brain regions


baum()


Best for: Parcellated matrices (like connectivity matrices)
How it works: Preserves edge weights while randomizing connections
When to use:

When analyzing brain connectivity data
When working with correlation matrices
When network properties are important


Example scenario: Comparing functional connectivity patterns


burt2020()


Best for: Volumetric (3D) brain maps
How it works: Uses spatial autocorrelation to generate null maps
When to use:

When working with standard MRI volumes
When data is in MNI152 or other volumetric space
For voxel-based analyses


Example scenario: Comparing whole-brain activation patterns from fMRI


vazquez_rodriguez()


Best for: Parcellated surface data with specific geometric constraints
How it works: Maintains distance relationships between parcels
When to use:

When geometric relationships between regions are crucial
When analyzing distance-dependent patterns
For specialized anatomical analyses


Example scenario: Studying distance-dependent connectivity patterns

Choosing the Right Model:

Consider your data format:


Surface data → alexander_bloch
Volumetric data → burt2020
Parcellated data → vasa or vazquez_rodriguez
Connectivity matrices → baum


Consider your research question:


Comparing cortical patterns → alexander_bloch
Analyzing regional relationships → vasa
Investigating network properties → baum
Examining spatial relationships → vazquez_rodriguez

"""