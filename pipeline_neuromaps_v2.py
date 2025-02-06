import os
import sys
import argparse
import logging
import tempfile
import numpy as np
import nibabel as nib
from pathlib import Path
from neuromaps import datasets, transforms, nulls, resampling, stats

temp_dir = '/notebooks/disk2/tmp' # path to the tmp folder

# Set custom temp directory
os.environ['TMPDIR'] = temp_dir 
os.environ['TEMP'] = temp_dir
os.environ['TMP'] = temp_dir

# Create the directory if it doesn't exist
os.makedirs(temp_dir, exist_ok=True)

# Set tempfile to use this directory
tempfile.tempdir = temp_dir

class NeuromapsWorkflow:
    """ A modular pipeline for the toolbox neuromaps with flexible coordinat systems and null models"""

    SUPPORTED_SPACES = ["MNI152", "fsaverage", "fsLR", "CIVET"]

    NULL_MODELS = {
        # ---------------------Surface--------------------------
        'alexander_bloch': nulls.alexander_bloch,       # For surface data
        'vasa': nulls.vasa,                             # For parcellated surface data
        'baum': nulls.baum,                             # For parcellated matrices
        'vazquez_rodriguez': nulls.vazquez_rodriguez,   # For parcellated surface with geometry 
        'hungarian': nulls.hungarian,                   # 
        'cornblath': nulls.cornblath,                   # 
        # ---------------------Volumetric--------------------------
        'burt2020': nulls.burt2020,                     # For volumetric data
        'burt2018': nulls.burt2018,                     # 
        'moran': nulls.moran                            # 
    }

    def __init__(self, source_path, source_space, target_source, target_desc, target_space, target_den, null_model):
        """
        Initialize the workflow pipeline
        
        Parameters
        ----------
        source_path : str
            Path to input brain map file
        source_space : str
            Space of input map (MNI152, fsaverage, fsLR, or CIVET)
        target_source : str
            Name of target map from neuromaps dataset
        target_desc : str
            Description of target map
        target_space : str
            Space of target map (MNI152, fsaverage, fsLR, or CIVET)
        target_den : str
            Density of target map
        null_model : str
            Name of null model to use
        """
        if source_space not in self.SUPPORTED_SPACES:
            raise ValueError(f"Source space must be one of: {self.SUPPORTED_SPACES}")      
        
        if  null_model not in self.NULL_MODELS:
            raise ValueError(f"Null model must be one of: {self.NULL_MODELS.keys()}")
        
        self.source_path = source_path
        self.source_space = source_space
        self.target_source = target_source
        self.target_desc = target_desc
        self.target_space = target_space
        self.target_den = target_den
        self.null_model = null_model

        # setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_maps(self):
        """
        Load the source and target maps
        """

        self.logger.info(f"Loading source maps...")

        # load source map
        self.source_map = nib.load(self.source_path)

        # load target map
        if self.target_space == "MNI152":
            self.target_map = datasets.fetch_annotation(source=self.target_source, desc=self.target_desc, space=self.target_space, res=self.target_den)
        else:
            self.target_map = datasets.fetch_annotation(source=self.target_source, desc=self.target_desc, space=self.target_space, den=self.target_den)
        
        return self.source_map, self.target_map
    
    def transform_maps(self, source_map, target_map, source_space, target_space):
        """
        Transform the source and target maps to the same space

        Parameters
        ----------
        source_map : nibabel.Nifti1Image
            Source brain map
        target_map : nibabel.Nifti1Image
            Target brain map
        source_space : str
            Space of source map (MNI152, fsaverage, fsLR, or CIVET)
        target_space : str
            Space of target map (MNI152, fsaverage, fsLR, or CIVET)
        """

        self.logger.info(f"Transforming maps from {self.source_space} to {self.target_space}...")
        #print(self.source_map)
        # transform source map
        #if source_space == target_space: # if already in the same space
        #    self.logger.info("Maps already in same space, no transformation needed") # TODO - change this, maybe remove the if condition
        #    transformed_source = source_map
        #    transformed_target = target_map
        #else:
        transformed_source, transformed_target = resampling.resample_images( # WARNING: the source target space needs to always be in lower resolution
            source_map,
            target_map,
            source_space,
            target_space
        )

        self.source_map = transformed_source
        self.target_map = transformed_target
        
        return transformed_source, transformed_target
    
    def generate_nulls(self, n_perm=100, seed=None):
        """
        Generate null maps using the specified null model

        Parameters
        ----------
        n_perm : int
            Number of permutations
        seed : int
            Random seed for reproducibility
        """

        self.logger.info(f"Generating null maps using {self.null_model}...")

        # generate null maps
        null_function = self.NULL_MODELS[self.null_model]
        #print(self.source_map)
        # set parameters based on the null model
        nulls = null_function(
            self.source_map,
            atlas=self.target_space,
            density=self.target_den,
            n_perm=n_perm,
            seed=seed
        )

        return nulls
    
    def compare_maps(self, nullss):
        """
        Compare the source and target maps using generated nulls

        Parameters
        ----------
        nulls : list of nibabel.Nifti1Image
            List of null maps generated by the null model
        """

        self.logger.info("Comparing maps...")

        # compare source and target maps
        corr, pval, n = stats.compare_images(
            self.source_map,
            self.target_map,
            nulls=nullss,
            return_nulls=True
        )

        return {
            'correlation': corr,
            'pvalue': pval,
            'nulls': n,
            'mean_null': np.mean(n),
            'std_null': np.std(n)
        }
     
def parse_arguments():
    """
    Parse command-line arguments
    """

    parser = argparse.ArgumentParser(description="Run the neuromaps pipeline")
    parser.add_argument("source_path", type=str, help="Path to input brain map file")
    parser.add_argument("source_space", choices=NeuromapsWorkflow.SUPPORTED_SPACES, type=str, help="Space of input map (MNI152, fsaverage, fsLR, or CIVET)")
    parser.add_argument("target_source", type=str, help="Name of target map from neuromaps dataset")
    parser.add_argument("target_desc", type=str, help="Description of target map")
    parser.add_argument("target_space", type=str, help="Space of target map (MNI152, fsaverage, fsLR, or CIVET)")
    parser.add_argument("target_den", type=str, help="Density of target map")
    parser.add_argument("--null_model", choices=list(NeuromapsWorkflow.NULL_MODELS.keys()), type=str, help="Name of null model to use")
    parser.add_argument('--n-perm', type=int, default=1000, help='Number of permutations')
    parser.add_argument('--seed', type=int, help='Random seed')

    return parser.parse_args()

def cleanup_tmp():
    """Removes all files in the temporary directory."""
    import os
    import shutil
    
    tmp_dir = '/notebooks/disk2/tmp'

    print(f"Cleaning up...")
    
    # Remove everything in the directory
    for item in os.listdir(tmp_dir):
        item_path = os.path.join(tmp_dir, item)
        try:
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Could not remove {item}: {e}")

def main():
    """
    Run the main workflow
    """
    args = parse_arguments()

    try:
        pipeline = NeuromapsWorkflow(
            source_path=args.source_path,
            source_space=args.source_space,
            target_source=args.target_source,
            target_desc=args.target_desc,
            target_space=args.target_space,
            target_den=args.target_den,
            null_model=args.null_model
        )

        # run steps of the pipeline
        source_map, target_map = pipeline.load_maps()
        source_map, target_map = pipeline.transform_maps(source_map, target_map, args.source_space, args.target_space)
        nulls = pipeline.generate_nulls(n_perm=args.n_perm, seed=args.seed)
        results = pipeline.compare_maps(nulls)

        # Print results
        print("\nResults:")
        print(f"Correlation: r = {results['correlation']:.3f}")
        print(f"P-value: p = {results['pvalue']:.3f}")
        print(f"Mean null correlation: {results['mean_null']:.3f}")
        print(f"Std null correlation: {results['std_null']:.3f}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
        
if __name__ == '__main__':
    try:
        main() # pipeline
        cleanup_tmp() # cleanup
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

"""
python pipeline_neuromaps_v2.py B0_603_mni152.nii.gz MNI152 abagen genepc1 fsaverage 10k --null_model alexander_bloch --n-perm 100 --seed 1234
python pipeline_neuromaps_v2.py B0_603_mni152.nii.gz MNI152 neurosynth cogpc1 MNI152 2mm --null_model burt2020 --n-perm 100 --seed 1234
"""