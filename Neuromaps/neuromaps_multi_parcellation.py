import os
import sys
import argparse
import logging
import tempfile
import numpy as np
import nibabel as nib
from neuromaps.parcellate import Parcellater
from neuromaps.images import annot_to_gifti, relabel_gifti
from neuromaps import datasets, transforms, nulls, resampling, stats, images

# Set up temp directory (allow override via TMPDIR env)
temp_dir = os.getenv('TMPDIR', '/mounts/disk2/tmp')
os.makedirs(temp_dir, exist_ok=True)
tempfile.tempdir = temp_dir

# Annotation files paths for DK atlas
lh_annot_path = "/mounts/disk2/projeto/lh.aparc.annot"
rh_annot_path = "/mounts/disk2/projeto/rh.aparc.annot"

class NeuromapsParcellationWorkflow:
    """ A modular pipeline for parcellating and comparing brain maps using neuromaps """

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
                 lh_annot_path=None, rh_annot_path=None, external_target_map=None):
        """Initialize the workflow pipeline"""
        # Validate inputs
        if source_space not in self.SUPPORTED_SPACES:
            raise ValueError(f"Source space must be one of: {self.SUPPORTED_SPACES}")      
        
        if null_model not in self.NULL_MODELS:
            raise ValueError(f"Null model must be one of: {self.NULL_MODELS.keys()}")
        
        # Ensure either volumetric OR both surface paths are provided
        if not (source_path or (source_path_left and source_path_right)):
            raise ValueError("Provide either a volumetric --volumetric SOURCE_PATH or --surface LEFT RIGHT paths.")
        
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
        
        # Annotation files
        self.lh_annot_path = lh_annot_path
        self.rh_annot_path = rh_annot_path
        
        # External target map (if provided, skip fetch_annotation)
        self.external_target_map = external_target_map
        
        # Determine mode
        self.mode = "volumetric" if source_path else "surface"

        # Do not call logging.basicConfig here; main configures logging.
        self.logger = logging.getLogger(__name__)
        
        # Warn early if files are missing
        try:
            from pathlib import Path
            if self.mode == "volumetric" and self.source_path and not Path(self.source_path).exists():
                print(f"WARNING: Source volumetric file not found: {self.source_path}")
            if self.mode == "surface":
                if self.source_path_left and not Path(self.source_path_left).exists():
                    print(f"WARNING: Left surface file not found: {self.source_path_left}")
                if self.source_path_right and not Path(self.source_path_right).exists():
                    print(f"WARNING: Right surface file not found: {self.source_path_right}")
            if self.lh_annot_path and not Path(self.lh_annot_path).exists():
                print(f"WARNING: Left annotation file not found: {self.lh_annot_path}")
            if self.rh_annot_path and not Path(self.rh_annot_path).exists():
                print(f"WARNING: Right annotation file not found: {self.rh_annot_path}")
        except Exception:
            # non-fatal; continue
            pass
        # Parcellated data storage
        self.parcellated_source = None
        self.parcellated_target = None

    def load_maps(self):
        """Load source and target maps based on mode"""
        print(f"Loading maps in {self.mode} mode...")
        
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
        
        print(f"Source map: {self.source_map}")
        print(f"Target map: {self.target_map}")
        return self.source_map, self.target_map
    
    def _load_surface_maps(self):
        """Load surface maps"""

        # Load source maps
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
                # Fetch the volumetric target map; keep for transform stage
                self.target_map_vol = datasets.fetch_annotation(
                    source=self.target_source, 
                    desc=self.target_desc, 
                    space=self.target_space, 
                    res=self.target_den
                )
                self.target_map = self.target_map_vol
                print("Loaded surface source maps and volumetric target map (MNI152).")
                return (self.source_map_left, self.source_map_right), self.target_map
            else:
                # Load surface target maps
                target_paths = datasets.fetch_annotation(
                    source=self.target_source, 
                    desc=self.target_desc, 
                    space=self.target_space, 
                    den=self.target_den
                )
                print(f"Target paths: {target_paths}")
                self.target_map_left = target_paths[0]
                self.target_map_right = target_paths[1]
                print("Loaded surface source and surface target maps.")
                return (self.source_map_left, self.source_map_right), (self.target_map_left, self.target_map_right)
    
    def transform_maps(self, source_map, target_map, hemi=None):
        """Transform maps to the same space"""
        print(f"Transforming maps to compatible resolution...")
        
        # Transform maps
        transformed_source, transformed_target = resampling.resample_images(
            source_map,
            target_map,
            self.source_space,
            self.target_space,
            hemi=hemi,
            resampling='downsample_only'
        )
        
        self.source_map = transformed_source
        self.target_map = transformed_target
        
        return transformed_source, transformed_target
    
    def parcellate_maps(self):
        """ Parcellate the source and target maps using the DK atlas retrieved from FreeSurfer """
        print("Parcellating maps...")
        
        # Convert annotation files to GIFTI format
        lh_gii, rh_gii = annot_to_gifti(
            (self.lh_annot_path, self.rh_annot_path)
        )

        # Relabel the annotation files to ensure both hemispheres are symmetric
        parcellation = relabel_gifti(
            (lh_gii, rh_gii),
            background=['Medial_wall']
        )

        print(f"Parcellation object: {parcellation}")
        
        if self.mode == "volumetric":
            # Handle case where transform_maps already converted volumetric data to surface tuples
            # Use existing surface data if present, otherwise convert volumetric->fsaverage
            if isinstance(self.source_map, (tuple, list)):
                source_transformed = self.source_map
            else:
                source_transformed = transforms.mni152_to_fsaverage(self.source_map)

            if isinstance(self.target_map, (tuple, list)):
                target_transformed = self.target_map
            else:
                # if target is volumetric and needs fsaverage conversion
                if self.target_space == "MNI152":
                    target_transformed = transforms.mni152_to_fsaverage(self.target_map)
                else:
                    target_transformed = self.target_map
                    
            # Create parcellater and apply to data
            parcellater = Parcellater(parcellation, 'fsaverage')
            
            # Parcellate source data (source_transformed may be surface tuple or gifti)
            source_parcellated = parcellater.fit_transform(source_transformed, 'fsaverage')
            self.parcellated_source = source_parcellated
            
            # Parcellate target data
            target_parcellated = parcellater.fit_transform(target_transformed, 'fsaverage')
            self.parcellated_target = target_parcellated
            
            print("Parcellated volumetric source and target.")
        else: # Surface mode
            # Source is already a tuple of left and right hemispheres
            # Get appropriate space
            space = self.target_space
            if space == "MNI152":
                space = "fsaverage"  # Use fsaverage for parcellation if target is MNI152
                
            # Create parcellater and apply to data
            parcellater = Parcellater(parcellation, space)
            print(f"Parcellater created for space {space}")
            
            # Parcellate source data
            source_parcellated = parcellater.fit_transform(self.source_map, space)
            self.parcellated_source = source_parcellated

            print(f"Source parcellated shape: {getattr(source_parcellated, 'shape', 'unknown')}")
            
            # Parcellate target data
            if isinstance(self.target_map, tuple) or isinstance(self.target_map, list):
                target_parcellated = parcellater.fit_transform(self.target_map, space)
            else:
                # Target is volumetric, transform first
                target_transformed = transforms.mni152_to_fsaverage(self.target_map)
                target_parcellated = parcellater.fit_transform(target_transformed, 'fsaverage')
                
            self.parcellated_target = target_parcellated
    
        return self.parcellated_source, self.parcellated_target, parcellation
    
    def generate_nulls(self, parcellation, n_perm=100, seed=None):
        """Generates null maps using the specified model and number of permutations (almost always 100)"""
        print(f"Generating null maps using {self.null_model}...")
        
        # Get null model function
        null_function = self.NULL_MODELS[self.null_model]

        print(f"Parcellated source shape: {getattr(self.parcellated_source, 'shape', 'unknown')}")
        print(f"Atlas: {self.null_space}; Density: {self.null_den}; LH annot: {self.lh_annot_path}")
        
        # Generate nulls for parcellated data
        nulls = null_function(
            self.parcellated_source,
            atlas=self.null_space,
            density=self.null_den,
            parcellation=parcellation,
            n_perm=n_perm,
            seed=seed
        )
        
        print(f"Nulls Shape: {getattr(nulls, 'shape', 'unknown')}")
        return nulls
    
    def compare_parcellated_maps(self, nulls=None):
        # Compares the maps using generated null maps
        """Compare parcellated source and target maps"""
        print("Comparing parcellated maps...")
        print(f"Parcellated source preview: {getattr(self.parcellated_source, 'shape', 'unknown')}")
        
        # Compare with null model
        corr, pval, n = stats.compare_images(
            self.parcellated_source,
            self.parcellated_target,
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
        """Run the complete parcellation and analysis pipeline"""
        # Load maps
        source_map, target_map = self.load_maps()
        
        # Transform maps to compatible space
        source_map, target_map = self.transform_maps(source_map, target_map)
        
        # Parcellate maps
        source_map, target_map, parcellation = self.parcellate_maps()
        
        # Generate nulls
        nulls = self.generate_nulls(parcellation, n_perm=n_perm, seed=seed)
        
        # Compare parcellated maps
        results = self.compare_parcellated_maps(nulls)
        
        return results

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run the neuromaps parcellation pipeline")
    
    # Input file options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--volumetric", type=str, dest="source_path", 
                             help="Path to input volumetric brain map file (.nii/.nii.gz)")
    input_group.add_argument("--surface", nargs=2, dest="source_paths", 
                             metavar=('LEFT_PATH', 'RIGHT_PATH'), 
                             help="Paths to left and right hemisphere brain map files (.gii)")
    
    # Required arguments
    parser.add_argument("source_space", choices=NeuromapsParcellationWorkflow.SUPPORTED_SPACES, 
                        help="Space of input map (MNI152, fsaverage, fsLR, or CIVET)")
    parser.add_argument("target_source", help="Name of target map from neuromaps dataset")
    parser.add_argument("target_desc", help="Description of target map")
    parser.add_argument("target_space", help="Space of target map (MNI152, fsaverage, fsLR, or CIVET)")
    parser.add_argument("target_den", help="Density of target map")
    
    # Optional arguments
    parser.add_argument("--null_model", choices=list(NeuromapsParcellationWorkflow.NULL_MODELS.keys()), 
                        help="Name of null model to use")
    parser.add_argument("null_space", help="Space of null model (MNI152, fsaverage, fsLR, or CIVET)")
    parser.add_argument("null_den", help="Density of null model")
    parser.add_argument('--n-perm', type=int, default=100, help='Number of permutations')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--lh-annot', type=str, default=lh_annot_path, 
                        help='Path to left hemisphere annotation file (.annot)')
    parser.add_argument('--rh-annot', type=str, default=rh_annot_path, 
                        help='Path to right hemisphere annotation file (.annot)')
    parser.add_argument('--external-map', type=str, default=None,
                        help='Path to external target map file (NIfTI or GIFTI) to use instead of fetching from neuromaps')

    return parser.parse_args()

def cleanup_tmp():
    """Remove files in temporary directory"""
    import os
    import shutil
    
    tmp_dir = '/mounts/disk2/tmp'
    print(f"Cleaning up temporary files in: {tmp_dir}")
    
    for item in os.listdir(tmp_dir):
        item_path = os.path.join(tmp_dir, item)
        try:
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"WARNING: Could not remove {item}: {e}")

def print_results(results, mode):
    """Print formatted results"""
    print("\nResults:")
    print(f"Correlation: r = {results['correlation']:.3f}")
    if results['pvalue'] is not None:
        print(f"P-value: p = {results['pvalue']:.3f}")
        print(f"Mean null correlation: {results['mean_null']:.3f}")
        print(f"Std null correlation: {results['std_null']:.3f}")

def main():
    """Main program"""
    args = parse_arguments()

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
            'lh_annot_path': args.lh_annot,
            'rh_annot_path': args.rh_annot,
            'external_target_map': args.external_map
        }
        
        # Initialize with volumetric or surface mode
        if args.source_path:  # Volumetric mode
            params['source_path'] = args.source_path
        else:  # Surface mode
            params['source_path_left'] = args.source_paths[0]
            params['source_path_right'] = args.source_paths[1]
        
        # Create and run pipeline
        pipeline = NeuromapsParcellationWorkflow(**params)
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
        main()  # run pipeline
        cleanup_tmp()  # cleanup
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

"""
Example usage:

# Volumetric mode with DK atlas:
python neuromaps_multi_parcellation.py --volumetric MRC.nii.gz MNI152 raichle cbf fsLR 164k --null_model cornblath fsaverage 164k --n-perm 100 --seed 1234

# Surface mode with DK atlas:
python neuromaps_multi_parcellation.py --surface lh.thickness.NeocorticalLB_Analysis_gamma.gii rh.thickness.NeocorticalLB_Analysis_gamma.gii fsaverage raichle cbf fsLR 164k --null_model cornblath fsaverage 164k --n-perm 100 --seed 1234

# Using external map (NIfTI or GIFTI)
python neuromaps_multi_parcellation.py --surface lh.thickness.NeocorticalLB_Analysis_gamma.gii rh.thickness.NeocorticalLB_Analysis_gamma.gii fsaverage dummy dummy MNI152 3mm --external-map MRC.nii.gz --null_model cornblath fsaverage 164k --n-perm 100 --seed 1234
"""