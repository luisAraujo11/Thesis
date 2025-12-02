import os
import sys
import time
import logging
import tempfile
import argparse
import numpy as np
import nibabel as nib
from functools import wraps
from neuromaps import datasets, nulls, resampling, stats

# Paralellization was removed for debugging purposes and because it didn't work properly.
# It was used the "joblib" library to attempt parallel processing of null model generation.

# Timing decorator to measure function execution times
def timing_decorator(func):
	"""
	A decorator that measures and logs the execution time of functions.
	This helps us identify performance bottlenecks in our pipeline.
	"""
	@wraps(func)
	def wrapper(*args, **kwargs):
		start_time = time.time()
		result = func(*args, **kwargs)
		end_time = time.time()
		logging.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
		return result
	return wrapper

# default temp dir may be overridden by CLI --tmp-dir or environment
DEFAULT_TMP = '/mounts/disk2/tmp'
temp_dir = os.getenv('TMPDIR', DEFAULT_TMP)

def configure_tempdir(path):
	"""Configure tempdir and environment consistently."""
	path = str(path)
	os.environ['TMPDIR'] = path
	os.environ['TEMP'] = path
	os.environ['TMP'] = path
	os.makedirs(path, exist_ok=True)
	tempfile.tempdir = path

configure_tempdir(temp_dir)

class NeuromapsWorkflow:
	""" A modular pipeline for the toolbox neuromaps with flexible coordinate systems and null models"""

	SUPPORTED_SPACES = ["MNI152", "fsaverage", "fsLR", "CIVET"]

	NULL_MODELS = {
		# ---------------------Surface---------------------------------------------------------------------
		'alexander_bloch': nulls.alexander_bloch,       # For surface data
		'vasa': nulls.vasa,                             # For parcellated surface data
		'baum': nulls.baum,                             # For parcellated matrices
		'vazquez_rodriguez': nulls.vazquez_rodriguez,   # For surface vertex level data [Recommended!]
		'hungarian': nulls.hungarian,                   # 
		'cornblath': nulls.cornblath,                   # For parcellated surface data [Recommended!]
		# ---------------------Volumetric------------------------------------------------------------------
		'burt2020': nulls.burt2020,                     # For volumetric data [Recommended!] for high-res
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

		# logger created; level set in main based on CLI --verbose
		self.logger = logging.getLogger(__name__)

	@timing_decorator
	def load_maps(self):
		"""Load the source and target maps"""
		self.logger.info(f"Loading source map from: {self.source_path}")

		# Load and measure source map properties
		load_start = time.time()
		try:
			self.source_map = nib.load(self.source_path)
		except Exception as e:
			self.logger.error(f"Failed to load source map '{self.source_path}': {e}")
			raise
		load_time = time.time() - load_start

		# Log source map information (use low-memory dtype view)
		data_shape = self.source_map.shape
		try:
			data_size = self.source_map.get_fdata(dtype=np.float32).nbytes / (1024 * 1024)  # MB
		except Exception:
			# fallback if get_fdata fails for some object
			data_size = getattr(self.source_map, 'size', 0) / (1024 * 1024)
		# keep shape/size info, remove duplicate "loaded in X seconds" (timing_decorator will log total)
		self.logger.info(f"Source map shape: {data_shape}")
		self.logger.info(f"Source map size: {data_size:.2f} MB")

		# Load target map with timing and contextual error handling
		target_start = time.time()
		try:
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
		except Exception as e:
			self.logger.error(f"Failed to fetch target annotation ({self.target_source}, {self.target_space}, {self.target_den}): {e}")
			raise
		target_time = time.time() - target_start
		# removed per-section timing log to avoid duplication; the decorator logs total load_maps time
		self.logger.debug(f"Target fetch time (internal): {target_time:.2f} seconds")

		return self.source_map, self.target_map

	@timing_decorator
	def transform_maps(self, source_map, target_map, source_space, target_space):
		"""Transform the source and target maps to the same space"""
		self.logger.info(f"Transforming maps from {source_space} to {target_space}...")
		try:
			transformed_source, transformed_target = resampling.resample_images(
				source_map,
				target_map,
				source_space,
				target_space
			)
		except Exception as e:
			self.logger.error(f"resample_images failed ({source_space} -> {target_space}): {e}")
			raise
		# timing_decorator will log elapsed time for this method (avoid duplicate timing logs)

		self.logger.debug(f"Transformed source type: {type(transformed_source)}")
		self.source_map = transformed_source
		self.target_map = transformed_target
		return transformed_source, transformed_target

	@timing_decorator
	def generate_nulls(self, n_perm=100, seed=None):
		"""Generate null maps with detailed performance metrics"""
		if self.null_model is None:
			raise ValueError("null_model must be specified")
		self.logger.info(f"Generating {n_perm} null maps using {self.null_model}...")

		# Log memory usage before starting (use low-mem read)
		try:
			data_size = None
			if hasattr(self.source_map, 'get_fdata'):
				data_size = self.source_map.get_fdata(dtype=np.float32).nbytes / (1024 * 1024)
			if data_size is not None:
				self.logger.info(f"Source map size before null generation: {data_size:.2f} MB")
		except Exception:
			self.logger.debug("Could not compute source map data size")

		null_function = self.NULL_MODELS[self.null_model]
		self.logger.info("Generating null maps (this may take a while)...")
		try:
			nulls = null_function(
				self.source_map,
				atlas=self.target_space,
				density=self.target_den,
				n_perm=n_perm,
				seed=seed
			)
		except Exception as e:
			self.logger.error(f"Null generation failed for model '{self.null_model}': {e}")
			raise
		# timing_decorator will log elapsed time for this method (avoid duplicate timing logs)
		return nulls

	@timing_decorator
	def compare_maps(self, nullss):
		"""Compare maps with timing information"""
		self.logger.info("Starting map comparison...")
		try:
			corr, pval, n = stats.compare_images(
				self.source_map,
				self.target_map,
				nulls=nullss,
				return_nulls=True
			)
		except Exception as e:
			self.logger.error(f"compare_images failed: {e}")
			raise

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
	parser.add_argument("--null_model", choices=list(NeuromapsWorkflow.NULL_MODELS.keys()), type=str, required=False, help="Name of null model to use")
	parser.add_argument('--n-perm', type=int, default=100, help='Number of permutations')
	parser.add_argument('--seed', type=int, help='Random seed')
	parser.add_argument('--tmp-dir', type=str, default=None, help='Temporary directory to use (overrides TMPDIR env)')
	parser.add_argument('--dry-run', action='store_true', help='Validate inputs and show planned steps without executing heavy work')
	parser.add_argument('--verbose', action='store_true', help='Enable verbose (DEBUG) logging')

	return parser.parse_args()
 
def cleanup_tmp():
	"""Removes all files in the temporary directory."""
	import os
	import shutil
	
	tmp_dir = tempfile.gettempdir()
	logging.getLogger(__name__).info(f"Cleaning up temporary directory: {tmp_dir}")
	
	# Remove everything in the directory
	for item in os.listdir(tmp_dir):
		item_path = os.path.join(tmp_dir, item)
		try:
			if os.path.isfile(item_path):
				os.remove(item_path)
			elif os.path.isdir(item_path):
				shutil.rmtree(item_path)
		except Exception as e:
			logging.getLogger(__name__).warning(f"Could not remove {item}: {e}")

def main():
	"""
	Run the main workflow
	"""
	args = parse_arguments()

	# configure logging
	level = logging.DEBUG if getattr(args, 'verbose', False) else logging.INFO
	logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

	# configure tempdir if requested
	if getattr(args, 'tmp_dir', None):
		configure_tempdir(args.tmp_dir)

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

		if getattr(args, 'dry_run', False):
			logging.getLogger(__name__).info("Dry-run mode: inputs validated, exiting without heavy computation.")
			return

		# run steps of the pipeline
		source_map, target_map = pipeline.load_maps()
		source_map, target_map = pipeline.transform_maps(source_map, target_map, args.source_space, args.target_space)
		nulls = pipeline.generate_nulls(n_perm=args.n_perm, seed=args.seed) # generate nulls
		results = pipeline.compare_maps(nulls)

		# Log results
		logging.getLogger(__name__).info(f"Results: Correlation r = {results['correlation']:.3f}, p = {results['pvalue']}")
		logging.getLogger(__name__).info(f"Mean null correlation = {results['mean_null']:.3f}, Std = {results['std_null']:.3f}")

	except Exception as e:
		logging.getLogger(__name__).error(f"Error: {e}")
		raise

if __name__ == '__main__':
	try:
		main()  # pipeline
		cleanup_tmp()  # cleanup
	except Exception as e:
		print(f"Error: {str(e)}", file=sys.stderr)
		logging.getLogger(__name__).error(f"Fatal error: {e}")
		sys.exit(1)

# Usage example:
# python pipeline_neuromaps_v3_parallel.py B0_603_mni152.nii.gz MNI152 abagen genepc1 fsaverage 10k --null_model vazquez_rodriguez --n-perm 100 --seed
