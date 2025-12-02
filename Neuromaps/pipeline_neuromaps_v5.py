import os
import sys
import logging
import argparse
import tempfile
import numpy as np
import nibabel as nib
from neuromaps import datasets, nulls, resampling, stats, images

# Set up temp directory
temp_dir = '/mounts/disk2/tmp'
os.environ['TMPDIR'] = temp_dir 
os.environ['TEMP'] = temp_dir
os.environ['TMP'] = temp_dir
os.makedirs(temp_dir, exist_ok=True)
tempfile.tempdir = temp_dir

class NeuromapsWorkflow:
    """ A modular pipeline for the toolbox neuromaps with flexible coordinate systems and null models"""

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

    def __init__(self, source_path=None, source_path_left=None, source_path_right=None, 
                 source_space=None, target_source=None, target_desc=None, 
                 target_space=None, target_den=None, null_model=None, null_space=None, null_den=None,
                 external_target_map=None):
        """Initialize the workflow pipeline"""
        # Validate inputs
        if source_space not in self.SUPPORTED_SPACES:
            raise ValueError(f"Source space must be one of: {self.SUPPORTED_SPACES}")      
        
        if null_model not in self.NULL_MODELS:
            raise ValueError(f"Null model must be one of: {self.NULL_MODELS.keys()}")
        
        # Store parameters
        self.source_path = source_path
        self.source_path_left = source_path_left
        self.source_path_right = source_path_right
        self.source_space = source_space
        self.target_source = target_source
        self.target_desc = target_desc
        self.target_space = target_space
        self.target_den = target_den
        self.null_model = null_model
        self.null_space = null_space
        self.null_den = null_den
        
        # External target map (if provided, skip fetch_annotation)
        self.external_target_map = external_target_map
        
        # Determine processing mode based on input
        self.mode = "volumetric" if source_path else "surface"

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_maps(self):
        """Load source and target maps based on mode"""
        self.logger.info(f"Loading maps in {self.mode} mode...")
        
        if self.mode == "volumetric":
            return self._load_volumetric_maps()
        else:
            return self._load_surface_maps()
        
    def _load_volumetric_maps(self):
        """Load volumetric maps"""
        # Load source map
        self.source_map = nib.load(self.source_path)
        
        # Load target map (external or from neuromaps)
        if self.external_target_map:
            print(f"Loading external target map: {self.external_target_map}")
            self.target_map = nib.load(self.external_target_map)
        else:
            if self.target_space == "MNI152":
                self.target_map = datasets.fetch_annotation(
                    source=self.target_source, 
                    desc=self.target_desc, 
                    space=self.target_space, 
                    res=self.target_den
                )
            else:
                self.target_map = datasets.fetch_annotation(
                    source=self.target_source, 
                    desc=self.target_desc, 
                    space=self.target_space, 
                    den=self.target_den
                )
        
        self.logger.info(f"tuple: {self.source_map}, {self.target_map}")
        return self.source_map, self.target_map
    
    def _load_surface_maps(self):
        """ Load surface maps from FreeSurfer output """
        
        # Load source maps (e.g., cortical thickness , LGI)
        self.source_map_left = images.load_gifti(self.source_path_left)
        self.source_map_right = images.load_gifti(self.source_path_right)

        # Load target maps (external or from neuromaps)
        if self.external_target_map:
            print(f"Loading external target map: {self.external_target_map}")
            # For surface: external_target_map can be a tuple/list of (left, right) paths or single volumetric
            if isinstance(self.external_target_map, (tuple, list)):
                self.target_map_left = images.load_gifti(self.external_target_map[0])
                self.target_map_right = images.load_gifti(self.external_target_map[1])
                print("Loaded external surface target maps (left and right).")
                return (self.source_map_left, self.source_map_right), (self.target_map_left, self.target_map_right)
            else:
                # Single volumetric external map
                self.target_map = nib.load(self.external_target_map)
                print("Loaded external volumetric target map.")
                return (self.source_map_left, self.source_map_right), self.target_map
        else:
            if self.target_space == "MNI152":
                # Load the volumetric target map
                self.target_map_vol = datasets.fetch_annotation(
                    source=self.target_source, 
                    desc=self.target_desc, 
                    space=self.target_space, 
                    res=self.target_den
                )
                self.target_map_left = self.target_map_vol
                self.target_map_right = self.target_map_vol
                print("tuple:", (self.source_map_left, self.source_map_right), self.target_map_vol)
                return (self.source_map_left, self.source_map_right), self.target_map_vol
            else:
                # Load surface target maps
                target_paths = datasets.fetch_annotation(
                    source=self.target_source, 
                    desc=self.target_desc, 
                    space=self.target_space, 
                    den=self.target_den
                )
                self.target_map_left = target_paths[0]
                self.target_map_right = target_paths[1]
                print("tuple:", (self.source_map_left, self.source_map_right), (self.target_map_left, self.target_map_right))
                return (self.source_map_left, self.source_map_right), (self.target_map_left, self.target_map_right)

    def transform_maps(self, source_map, target_map):
        """Transform maps to the same space"""
        self.logger.info(f"Transforming maps to compatible lowest resolution, {self.source_space} <-> {self.target_space}...")
        
        # Handle coordinate space transformations
        transformed_source, transformed_target = resampling.resample_images(
            source_map,
            target_map,
            self.source_space,
            self.target_space,
            resampling='downsample_only'
        )
        
        self.source_map = transformed_source
        self.target_map = transformed_target

        #if hasattr(self.source_map[0], 'agg_data'):
            #print(f"Transformed source: ({self.source_map[0].agg_data().shape}, {self.source_map[1].agg_data().shape})")
            #print(f"Transformed target: ({self.target_map[0].agg_data().shape}, {self.target_map[1].agg_data().shape})")
        
        return transformed_source, transformed_target
    
    def generate_nulls(self, n_perm=100, seed=None):
        """Generate null maps using specified model"""
        self.logger.info(f"Generating null maps using {self.null_model}...")
        
        # Get null model function
        null_function = self.NULL_MODELS[self.null_model]

        self.logger.info(f"Atlas: {self.null_space}")
        self.logger.info(f"Density: {self.null_den}")
        
        # Generate nulls
        nulls = null_function(
            self.source_map,
            atlas=self.null_space,
            density=self.null_den,
            n_perm=n_perm,
            seed=seed
        )
        
        self.logger.info(f"Nulls Shape: {nulls.shape}")
        return nulls
    
    def compare_maps(self, nulls):
        """Compare source and target maps using nulls"""
        self.logger.info("Comparing maps...")

        #print(f"Source type: ({self.source_map[0].agg_data().shape}, {self.source_map[1].agg_data().shape})")
        #print(f"Target type: ({self.target_map[0].agg_data().shape}, {self.target_map[1].agg_data().shape})")
        self.logger.info("source_map: %s", self.source_map)
        
        # Compare source and target
        corr, pval, n = stats.compare_images(
            self.source_map,
            self.target_map,
            nulls=nulls,
            return_nulls=True
        )
        
        return {
            'correlation': corr,
            'pvalue': pval,
            'nulls': n,
            'mean_null': np.mean(n),
            'std_null': np.std(n)
        }
    
    def run_pipeline(self, n_perm=100, seed=None):
        """Run the complete analysis pipeline"""
        # Load maps
        source_map, target_map = self.load_maps()
        
        # Transform maps (skip if both are already in same space and same mode)
        # Check if transformation is needed
        need_transform = (self.source_space != self.target_space) or (isinstance(source_map, tuple) != isinstance(target_map, tuple))
        
        if need_transform:
            source_map, target_map = self.transform_maps(source_map, target_map)
        else:
            self.logger.info("Source and target already in same space; skipping transformation.")
            self.source_map = source_map
            self.target_map = target_map
        
        # Generate nulls
        nulls = self.generate_nulls(n_perm=n_perm, seed=seed)
        
        # Compare maps
        results = self.compare_maps(nulls)
        
        return results

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run the neuromaps pipeline")
    
    # Input file options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--volumetric", type=str, dest="source_path", 
                             help="Path to input volumetric brain map file (.nii/.nii.gz)")
    input_group.add_argument("--surface", nargs=2, dest="source_paths", 
                             metavar=('LEFT_PATH', 'RIGHT_PATH'), 
                             help="Paths to left and right hemisphere brain map files (.gii)")
    
    # Required arguments
    parser.add_argument("source_space", choices=NeuromapsWorkflow.SUPPORTED_SPACES, 
                        help="Space of input map (MNI152, fsaverage, fsLR, or CIVET)")
    parser.add_argument("target_source", help="Name of target map from neuromaps dataset")
    parser.add_argument("target_desc", help="Description of target map")
    parser.add_argument("target_space", help="Space of target map (MNI152, fsaverage, fsLR, or CIVET)")
    parser.add_argument("target_den", help="Density of target map")
    
    # Optional arguments
    parser.add_argument("--null_model", choices=list(NeuromapsWorkflow.NULL_MODELS.keys()), 
                        help="Name of null model to use")
    parser.add_argument("null_space", help="Space of null model (MNI152, fsaverage, fsLR, or CIVET)")
    parser.add_argument("null_den", help="Density of null model")
    parser.add_argument('--n-perm', type=int, default=1000, help='Number of permutations')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--external-map', type=str, default=None,
                        help='Path to external target map file (NIfTI or GIFTI) to use instead of fetching from neuromaps')
 
    return parser.parse_args()

def cleanup_tmp():
    """Remove files in the current temporary directory"""
    import shutil
    
    """Remove files in temporary directory"""
    tmp_dir = '/mounts/disk2/tmp'
    print(f"Cleaning up temporary files...")
    
    for item in os.listdir(tmp_dir):
        item_path = os.path.join(tmp_dir, item)
        try:
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Could not remove {item}: {e}")

def print_results(results, mode):
    """Print formatted results"""
    print("\nResults:")
    print(f"Correlation: r = {results['correlation']:.3f}")
    print(f"P-value: p = {results['pvalue']:.3f}")
    print(f"Mean null correlation: {results['mean_null']:.3f}")
    print(f"Std null correlation: {results['std_null']:.3f}")

def main():
    """Main program"""
    args = parse_arguments()
 
    # use system tempdir (no CLI override)
 
    try:
        # Initialize pipeline parameters
        params = {
            'source_space': args.source_space,
            'target_source': args.target_source,
            'target_desc': args.target_desc,
            'target_space': args.target_space,
            'target_den': args.target_den,
            'null_model': args.null_model,
            'null_space': args.null_space,
            'null_den': args.null_den,
            'external_target_map': args.external_map
        }
        
        # Initialize with volumetric or surface mode
        if args.source_path:  # Volumetric mode
            params['source_path'] = args.source_path
        else:  # Surface mode
            params['source_path_left'] = args.source_paths[0]
            params['source_path_right'] = args.source_paths[1]
        
        # Create and run pipeline
        pipeline = NeuromapsWorkflow(**params)
        results = pipeline.run_pipeline(n_perm=args.n_perm, seed=args.seed)
        
        # Print results
        print_results(results, pipeline.mode)
        
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)
        
if __name__ == '__main__':
    try:
        main()  # pipeline
        cleanup_tmp()  # cleanup
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

"""
Example usage:

# Volumetric mode:
python pipeline_neuromaps_v5.py --volumetric MRC.nii.gz MNI152 raichle cbf fsLR 164k --null_model vazquez_rodriguez fsaverage 164k --n-perm 100 --seed 1234
# External volumetric target map:
python pipeline_neuromaps_v5.py --volumetric CII.nii.gz MNI152 dummy dummy MNI152 3mm --external-map MRC.nii.gz --null_model burt2020 MNI152 3mm --n-perm 100 --seed 1234

# Surface mode:
python pipeline_neuromaps_v5.py --surface lh.volume.NeocorticalLB_Analysis_gamma.gii rh.volume.NeocorticalLB_Analysis_gamma.gii fsaverage raichle cbf fsLR 164k --null_model vazquez_rodriguez fsaverage 164k --n-perm 100 --seed 1234
# External surface target maps:
python pipeline_neuromaps_v5.py --surface lh.thickness.NeocorticalLB_Analysis_gamma.gii rh.thickness.NeocorticalLB_Analysis_gamma.gii fsaverage dummy dummy MNI152 3mm --external-map MRC.nii.gz --null_model vazquez_rodriguez fsaverage 164k --n-perm 100 --seed 1234"""