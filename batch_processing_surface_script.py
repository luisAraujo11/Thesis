import os
import sys
import time
import logging
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("neuromaps_surface_batch.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# List of target maps to process
maps = [
    #('hill2010','devexp','fsLR','164k'), # both hill2010 only have the right hemisphere
    #('hill2010','evoexp','fsLR','164k'),
    ('mueller2013','intersubjvar','fsLR','164k'),
    ('raichle','cbf','fsLR','164k'), # Cerebral Blood Flow
    ('raichle','cbv','fsLR','164k'), # Cerebral Blood Volume
    ('raichle','cmr02','fsLR','164k'), 
    ('raichle','cmrglc','fsLR','164k'),
    ('aghourian2017', 'feobv', 'MNI152', '1mm'),
    ('alarkurtti2015', 'raclopride', 'MNI152', '3mm'),
    ('bedard2019', 'feobv', 'MNI152', '1mm'),
    ('beliveau2017', 'az10419369', 'fsaverage', '164k'), # 5-HT2A receptor
    ('beliveau2017', 'cimbi36', 'fsaverage', '164k'),
    ('beliveau2017', 'cumi101', 'fsaverage', '164k'),
    ('beliveau2017', 'dasb', 'fsaverage', '164k'), # Serotonin transporter
    ('beliveau2017', 'sb207145', 'fsaverage', '164k'),
    ('castrillon2023', 'cmrglc', 'MNI152', '3mm'),
    ('ding2010', 'mrb', 'MNI152', '1mm'),
    ('dubois2015', 'abp688', 'MNI152', '1mm'),
    ('dukart2018', 'flumazenil', 'MNI152', '3mm'),
    ('dukart2018', 'fpcit', 'MNI152', '3mm'),
    ('fazio2016', 'madam', 'MNI152', '3mm'),
    ('finnema2016', 'ucbj', 'MNI152', '1mm'),
    ('gallezot2010', 'p943', 'MNI152', '1mm'),
    ('gallezot2017', 'gsk189254', 'MNI152', '1mm'),
    ('galovic2021', 'ge179', 'MNI152', '1mm'),
    ('hesse2017', 'methylreboxetine', 'MNI152', '3mm'),
    ('hillmer2016', 'flubatine', 'MNI152', '1mm'),
    ('jaworska2020', 'fallypride', 'MNI152', '1mm'),
    ('kaller2017', 'sch23390', 'MNI152', '3mm'),
    ('kantonen2020', 'carfentanil', 'MNI152', '3mm'),
    ('kim2020', 'ps13', 'MNI152', '2mm'),
    ('laurikainen2018', 'fmpepd2', 'MNI152', '1mm'),
    ('lois2018', 'pbr28', 'MNI152', '2mm'),
    ('lukow2022', 'ro154513', 'MNI152', '2mm'),
    ('malen2022', 'raclopride', 'MNI152', '2mm'),
    ('naganawa2020', 'lsn3172176', 'MNI152', '1mm'),
    ('neurosynth', 'cogpc1', 'MNI152', '2mm'), # Cognitive PC1
    ('norgaard2021', 'flumazenil', 'fsaverage', '164k'),
    ('normandin2015', 'omar', 'MNI152', '1mm'),
    ('radnakrishnan2018', 'gsk215083', 'MNI152', '1mm'),
    ('rosaneto', 'abp688', 'MNI152', '1mm'),
    ('sandiego2015', 'flb457', 'MNI152', '1mm'),
    ('sasaki2012', 'fepe2i', 'MNI152', '1mm'),
    ('satterthwaite2014', 'meancbf', 'MNI152', '1mm'),
    ('savli2012', 'altanserin', 'MNI152', '3mm'),
    ('savli2012', 'dasb', 'MNI152', '3mm'),
    ('savli2012', 'p943', 'MNI152', '3mm'),
    ('savli2012', 'way100635', 'MNI152', '3mm'),
    ('smart2019', 'abp688', 'MNI152', '1mm'),
    ('smith2017', 'flb457', 'MNI152', '1mm'),
    ('tuominen', 'feobv', 'MNI152', '2mm'),
    ('turtonen2020', 'carfentanil', 'MNI152', '1mm'),
    ('vijay2018', 'ly2795050', 'MNI152', '2mm'),
    ('wey2016', 'martinostat', 'MNI152', '2mm')
]

# List of external maps to process (filepath, description, space, density)
# For external maps: target_source should be the file path
external_maps = [
    ('/mounts/disk2/projeto/neuromaps_pipeline/MRC.nii.gz', 'MRC', 'MNI152', '3mm'),  # Mitochondrial Respiratory Chain
    ('/mounts/disk2/projeto/neuromaps_pipeline/CII.nii.gz', 'CII', 'MNI152', '3mm'),  # Succinate Dehydrogenase
    ('/mounts/disk2/projeto/neuromaps_pipeline/TRC.nii.gz', 'TRC', 'MNI152', '3mm'),  # Total Respiratory Capacity
    ('/mounts/disk2/projeto/neuromaps_pipeline/CI.nii.gz', 'CI', 'MNI152', '3mm'),    # NADH Dehydrogenase
    ('/mounts/disk2/projeto/neuromaps_pipeline/CIV.nii.gz', 'CIV', 'MNI152', '3mm'),  # Cytochrome c Oxidase
    ('/mounts/disk2/projeto/neuromaps_pipeline/MitoD.nii.gz', 'MitoD', 'MNI152', '3mm') # Mitochondrial Density
]

def parse_args():
    """Parse command-line arguments for batch processing"""
    parser = argparse.ArgumentParser(description="Batch process neuromaps surface comparisons")
    parser.add_argument("--left-hemi", type=str, required=True, help="Path to left hemisphere GIFTI file")
    parser.add_argument("--right-hemi", type=str, required=True, help="Path to right hemisphere GIFTI file")
    parser.add_argument("--source-space", type=str, default="fsaverage", help="Source space (default: fsaverage)")
    parser.add_argument("--n-perm", type=int, default=1000, help="Number of permutations (default: 1000)")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234)")
    parser.add_argument("--null-model", type=str, default="vazquez_rodriguez", 
                        help="Null model to use (default: vazquez_rodriguez)")
    parser.add_argument("--pipeline", type=str, default="pipeline_neuromaps_v5.py", 
                        help="Pipeline script to use (default: pipeline_neuromaps_v5.py)")
    parser.add_argument("--study", type=str, default="", help="Study name suffix for output files")
    parser.add_argument("--type-study", type=str, default="surface", 
                        choices=["surface", "parcellated"], help="Type of study (default: surface)")
    parser.add_argument("--category-file", type=str, 
                        default="neuromaps_parcellated_results_final_thickness.AdvsPARTwithLBStudy_Age_Delta_eTIV_norm2_cornblath.ods",
                        help="Path to ODS file with category information")
    return parser.parse_args()

def parse_results(output):
    """
    Parse the command output to extract correlation and p-value
    """
    correlation = None
    p_value = None
    mean_null = None
    std_null = None
    
    lines = output.strip().split('\n')
    for line in lines:
        if 'Correlation: r =' in line:
            try:
                correlation = float(line.split('=')[1].strip())
            except:
                pass
        elif 'P-value: p =' in line:
            try:
                p_value = float(line.split('=')[1].strip())
            except:
                pass
        elif 'Mean null correlation:' in line:
            try:
                mean_null = float(line.split(':')[1].strip())
            except:
                pass
        elif 'Std null correlation:' in line:
            try:
                std_null = float(line.split(':')[1].strip())
            except:
                pass
    
    return correlation, p_value, mean_null, std_null

def load_category_data(filepath):
    """Load category information from ODS file"""
    try:
        category_df = pd.read_excel(filepath, engine="odf")
        category_map = {}
        
        for _, row in category_df.iterrows():
            if pd.notna(row.get('target_source')) and pd.notna(row.get('target_desc')):
                key = (row['target_source'], row['target_desc'])
                category_info = {
                    'category': row.get('category', None),
                    'subcategory': row.get('subcategory', None),
                    'target_type': row.get('target_type', None),
                    'modality': row.get('modality', None),
                    'demographics': row.get('demographics', None),
                    'description': row.get('description', None)
                }
                category_map[key] = category_info
        
        logger.info(f"Loaded {len(category_map)} category entries from {filepath}")
        return category_map
    except Exception as e:
        logger.error(f"Error loading category data from {filepath}: {e}")
        return {}

def process_map(map_info, config):
    """Process a single map comparison using command line interface"""
    target_source, target_desc, target_space, target_den = map_info
    
    # Determine null space and density based on source space
    null_space = config['source_space']
    null_den = "164k" if config['source_space'] == "fsaverage" else "32k"
    
    # Check if target_source is a file path (external map)
    is_external = Path(target_source).exists() if target_source else False
    
    # Build the command - use dummy values for external maps
    if is_external:
        cmd = [
            "python", config['pipeline'],
            "--surface", config['left_hemi'], config['right_hemi'],
            config['source_space'],
            "dummy", "dummy",  # Use "dummy" as placeholder when external map is provided
            target_space, target_den,
            "--null_model", config['null_model'],
            null_space, null_den,
            "--n-perm", str(config['n_perm']),
            "--seed", str(config['seed']),
            "--external-map", target_source  # Pass actual file path here
        ]
    else:
        cmd = [
            "python", config['pipeline'],
            "--surface", config['left_hemi'], config['right_hemi'],
            config['source_space'],
            target_source, target_desc,
            target_space, target_den,
            "--null_model", config['null_model'],
            null_space, null_den,
            "--n-perm", str(config['n_perm']),
            "--seed", str(config['seed'])
        ]
    
    logger.info(f"Starting {target_source} - {target_desc}")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    result_base = {
        'target_source': target_source,
        'target_desc': target_desc,
        'target_space': target_space,
        'target_den': target_den,
        'null_model': config['null_model'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        runtime = time.time() - start_time
        
        # Parse the output
        correlation, p_value, mean_null, std_null = parse_results(result.stdout)
        
        logger.info(f"Completed {target_source} - {target_desc}, correlation: {correlation}, p-value: {p_value}, time: {runtime:.2f}s")
        
        return {
            'target_source': target_source,
            'target_desc': target_desc,
            'target_space': target_space,
            'target_den': target_den,
            'correlation': correlation,
            'p_value': p_value,
            'mean_null': mean_null,
            'std_null': std_null,
            'runtime': runtime
        }
        
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout processing {target_source} - {target_desc}")
        return {**result_base, 'correlation': None, 'p_value': None, 'mean_null': None, 
                'std_null': None, 'runtime': None, 'error': 'timeout'}
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing {target_source} - {target_desc}: {e}")
        logger.error(f"STDERR: {e.stderr}")  # Change to ERROR so you can see it
        logger.error(f"STDOUT: {e.stdout}")  # Also log stdout
        return {**result_base, 'correlation': None, 'p_value': None, 'mean_null': None, 
                'std_null': None, 'runtime': None, 'error': str(e)}
    except Exception as e:
        logger.error(f"Unexpected error processing {target_source} - {target_desc}: {e}")
        return {**result_base, 'correlation': None, 'p_value': None, 'mean_null': None, 
                'std_null': None, 'runtime': None, 'error': str(e)}

def load_existing_results(filename="neuromaps_surface_results.csv"):
    """Load existing results if available"""
    if os.path.exists(filename):
        return pd.read_csv(filename).to_dict('records')
    return []

def get_pending_maps(existing_results, all_maps):
    """Determine which maps still need to be processed"""
    processed = {(r['target_source'], r['target_desc'], r['target_space'], r['target_den']) 
                for r in existing_results if r['correlation'] is not None}
    
    return [m for m in all_maps if tuple(m) not in processed]

def enrich_results_with_categories(results, category_map):
    """Add category information to results"""
    enriched_results = []
    
    for result in results:
        # Create a copy of the result
        enriched_result = result.copy()
        
        # Get the key for category lookup
        key = (result['target_source'], result['target_desc'])
        
        # Add category information if available
        if key in category_map:
            categories = category_map[key]
            for category_key, category_value in categories.items():
                if category_value is not None:
                    enriched_result[category_key] = category_value
        
        enriched_results.append(enriched_result)
    
    return enriched_results

def main():
    args = parse_args()
    
    # Build config dict from args
    config = {
        'left_hemi': args.left_hemi,
        'right_hemi': args.right_hemi,
        'source_space': args.source_space,
        'n_perm': args.n_perm,
        'seed': args.seed,
        'null_model': args.null_model,
        'pipeline': args.pipeline
    }
    
    # Combine regular maps with external maps
    all_maps = maps + external_maps
    logger.info(f"Total maps to process: {len(all_maps)} (regular: {len(maps)}, external: {len(external_maps)})")
    
    # Load category information from ODS file
    category_map = load_category_data(args.category_file)
    
    # Load any existing results
    existing_results = load_existing_results()
    logger.info(f"Loaded {len(existing_results)} existing results")
    
    # Determine which maps still need to be processed
    pending_maps = get_pending_maps(existing_results, all_maps)
    logger.info(f"Found {len(pending_maps)} maps remaining to process")
    
    if not pending_maps:
        logger.info("All maps already processed. Generating final report.")
        results_df = pd.DataFrame(existing_results)
    else:
        # Store results
        results = existing_results.copy()
        
        # Group maps by space to prioritize processing
        fsaverage_maps = [m for m in pending_maps if m[2] == "fsaverage"]
        other_maps = [m for m in pending_maps if m not in fsaverage_maps]
        prioritized_maps = fsaverage_maps + other_maps
        
        # Process maps sequentially
        for map_info in tqdm(prioritized_maps, desc="Processing maps", unit="map"):
            result = process_map(map_info, config)
            results.append(result)
            
            # Save intermediate results
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"neuromaps_{args.type_study}_results{args.study}.csv", index=False)
            
        # Create final DataFrame
        results_df = pd.DataFrame(results)
    
    # Enrich results with category information
    enriched_results = enrich_results_with_categories(results_df.to_dict('records'), category_map)
    enriched_df = pd.DataFrame(enriched_results)
    
    # Display results
    print("\nAll Results:")
    print(enriched_df[['target_source', 'target_desc', 'correlation', 'p_value']].to_string(index=False))
    
    # Save final results with categories
    enriched_df.to_csv(f"neuromaps_{args.type_study}_results_final{args.study}.csv", index=False)
    logger.info(f"Results with categories saved to neuromaps_{args.type_study}_results_final{args.study}.csv")
    
    # Sort and display significant results
    significant_results = enriched_df[enriched_df['p_value'] <= 0.05].sort_values('correlation', ascending=False)
    logger.info(f"Found {len(significant_results)} significant correlations (p <= 0.05)")
    
    # Create a nicely formatted table for the significant results
    if not significant_results.empty:
        sig_columns = ['target_source', 'target_desc', 'correlation', 'p_value']
        category_columns = ['category', 'subcategory', 'target_type', 'modality']
        for col in category_columns:
            if col in significant_results.columns:
                sig_columns.append(col)
        
        sig_table = significant_results[sig_columns]
        sig_table.sort_values('correlation', ascending=False, inplace=True)
        sig_table.to_csv(f"significant_{args.type_study}_correlations{args.study}.csv", index=False)
        logger.info(f"Significant correlations saved to significant_{args.type_study}_correlations{args.study}.csv")
        print("\nSignificant correlations (p < 0.05):")
        print(sig_table.to_string(index=False))

if __name__ == "__main__":
    main()

# Example command to run the script:

# For surface study with vazquez_rodriguez null model
'''
python batch_processing_surface_script.py \
  --left-hemi lh.thickness.AdvsPARTwithoutLBStudy_Delta_eTIV_norm2_gamma.gii \
  --right-hemi rh.thickness.AdvsPARTwithoutLBStudy_Delta_eTIV_norm2_gamma.gii \
  --source-space fsaverage \
  --n-perm 1000 \
  --seed 1234 \
  --null-model vazquez_rodriguez \
  --pipeline pipeline_neuromaps_v5.py \
  --study "_AdvsPART_withoutLB_Study_gamma_vazquez_rodriguez_test2" \
  --type-study surface
'''

# For parcellated study with cornblath null model
'''
python batch_processing_surface_script.py \
  --left-hemi lh.thickness.AdvsPARTwithoutLBStudy_Delta_eTIV_norm2_gamma.gii \
  --right-hemi rh.thickness.AdvsPARTwithoutLBStudy_Delta_eTIV_norm2_gamma.gii \
  --source-space fsaverage \
  --n-perm 1000 \
  --seed 1234 \
  --null-model cornblath \
  --pipeline neuromaps_multi_parcellation.py \
  --study "_AdvsPART_withLB_Study_Age_Delta_eTIV_norm2_cornblath_test2" \
  --type-study parcellated
'''