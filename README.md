
...
# Thesis Development Steps: Structural and Functional Brain Map Analysis

**Author:** Luis Araújo
**Title:** Análise Estrutural e Funcional de Mapas Cerebrais
**Last Updated:** December 2025

This document describes the chronological development steps of the thesis project, including code references and commands used throughout the analysis pipeline.

---

## Overview

This thesis analyzes structural MRI data from the NACC dataset to identify brain regions affected in Alzheimer's Disease (AD) compared to Primary Age-Related Tauopathy (PART), and correlates these findings with molecular brain maps using spatial permutation testing.

**Main Pipeline:**
```
Raw MRI Data → Preprocessing → FreeSurfer Analysis → Statistical Testing →
Neuromaps Correlations → Network Mapping → Visualization
```

---

## Phase 1: Data Acquisition and Preprocessing

### 1.1 Automated MRI Data Extraction

**Goal:** Extract and filter T1-weighted MRI scans from NACC compressed archives.

**Scripts:**
- `automated_filtering/process_MRIs.py` - Extract subjects from ZIP files
- `automated_filtering/process_MRIs_modality.py` - Filter by modality using JSON metadata
- `automated_filtering/cross_MRIs.py` - Cross-reference with demographics
- `automated_filtering/process_MRIs_pipeline.sh` - Orchestration script

**Key Commands:**
```bash
# Full pipeline execution
bash process_MRIs_pipeline.sh \
  -s "/media/memad/Disk1/NACC_MRI_subjects" \
  -d "/media/memad/Disk1/NACC_MRI_subjects_filtered" \
  -c "subjects.csv" \
  -n "subjects_missing.csv" \
  -m "t1" \
  -t "mprage,t1,3d_ir"
```

**What it does:**
1. Extracts DICOM/NIfTI from ZIP archives
2. Runs `dcm2niix` for DICOM to NIfTI conversion (when needed)
3. Parses JSON sidecar files to identify T1-weighted sequences
4. Filters scans matching search terms (mprage, 3d_ir, etc.)
5. Creates CSV of successfully processed subjects

**Code Reference:** `automated_filtering/process_MRIs_modality.py:45-78`

---

## Phase 2: Cortical Reconstruction with FreeSurfer

### 2.1 Initial Surface Reconstruction

**Goal:** Perform full cortical reconstruction for all subjects.

**Script:** `FreeSurfer/freesurfer_process.sh`

**Key Command:**
```bash
# Parallel processing with 12 jobs
recon-all -i /path/to/scan.nii -s NACC145249 -all -qcache
```

**Code Reference:** `FreeSurfer/freesurfer_process.sh:54`

**Parameters:**
- `-all`: Complete reconstruction pipeline
- `-qcache`: Enables smoothing at different FWHM values for group analysis
- Parallel processing: 12 subjects simultaneously using GNU `parallel`

**Outputs:**
- Surface meshes: `surf/{lh,rh}.{pial,white,inflated}`
- Parcellations: `label/{lh,rh}.aparc.annot` (Desikan-Killiany atlas)
- Morphometry: `surf/{lh,rh}.{thickness,area,volume}`

**Visualization command:**
```bash
# View reconstruction results
freeview -v mri/T1.mgz \
         -f surf/lh.pial:edgecolor=red \
         -f surf/lh.white:edgecolor=yellow \
         -f surf/rh.pial:edgecolor=red \
         -f surf/rh.white:edgecolor=yellow
```

### 2.2 Local Gyrification Index (LGI)

**Goal:** Compute gyrification measures for each subject.

**Script:** `FreeSurfer/createLGI.sh`

**Key Commands:**
```bash
# Smooth LGI data (FWHM=10mm)
mris_fwhm --s NACC145249 --hemi lh \
  --cortex --smooth-only --fwhm 10 \
  --i surf/lh.pial_lgi \
  --o surf/lh.pial_lgi.fwhm10.mgh

# Resample to fsaverage space
mri_surf2surf --srcsubject NACC145249 \
  --srcsurfval surf/lh.pial_lgi.fwhm10.mgh \
  --trgsubject fsaverage \
  --trgsurfval surf/lh.pial_lgi.fwhm10.fsaverage.mgh \
  --hemi lh
```

**Code Reference:** `FreeSurfer/createLGI.sh:31-40`

**Why smoothing?** Reduces noise and improves statistical power in group comparisons.

**Visualization:**
```bash
# View LGI on subject's surface
freeview -f surf/lh.pial:overlay=surf/lh.pial_lgi
```

### 2.3 Subcortical Segmentation

**Goal:** Segment subcortical structures (amygdala, anterior nuclei).

**Script:** `FreeSurfer/segmentAAN.sh`

**Key Command:**
```bash
# Run advanced subcortical segmentation
segmentHA_T1.sh NACC145249
```

**Outputs:** High-resolution amygdala and hippocampal subfield segmentations

---

## Phase 3: Group-Level Statistical Analysis

### 3.1 Prepare Group Data

**Goal:** Aggregate individual subject data for group comparisons.

**Script:** `FreeSurfer/runMrisPreproc.sh`

**Key Command:**
```bash
# Concatenate subject data into group-level file
mris_preproc --fsgd FSGD/AdvsPARTStudy.fsgd \
  --cache-in thickness.fwhm10.fsaverage.mgh \
  --target fsaverage \
  --hemi lh \
  --out lh.thickness.AdvsPARTStudy.10.mgh
```

**Code Reference:** `FreeSurfer/runMrisPreproc.sh`

**What it does:** Creates a single file with all subjects' data aligned to fsaverage space.

### 3.2 General Linear Model (GLM) Analysis

**Goal:** Compare AD vs PART groups while controlling for covariates.

**Script:** `FreeSurfer/runGLMs.sh`

**Key Command:**
```bash
mri_glmfit \
  --y lh.thickness.AdvsPARTStudy.10.mgh \
  --fsgd FSGD/AdvsPARTStudy.fsgd doss \
  --C Contrasts/AD-PART_Age_Sex_Delta_eTIV.mtx \
  --surf fsaverage lh \
  --cortex \
  --glmdir lh.thickness.AdvsPARTStudy.10.glmdir
```

**Code Reference:** `FreeSurfer/runGLMs.sh:8-15`

**Model:** `Thickness ~ Group + Age + Sex + eTIV`

**Contrast Matrices:**
- `AD-PART_Age_Sex_Delta_eTIV.mtx`: Tests AD > PART
- `PART-AD_Age_Sex_Delta_eTIV.mtx`: Tests PART > AD

**FSGD File Structure:**
```
GroupDescriptorFile 1
Title AdvsPARTStudy
Class AD
Class PART
Variables Age Sex eTIV
Input NACC123456 AD 75.2 1 1520.3
Input NACC789012 PART 78.5 0 1450.1
...
```

### 3.3 Cluster-Wise Correction

**Goal:** Correct for multiple comparisons using permutation testing.

**Script:** `FreeSurfer/runClustSims.sh`

**Key Command:**
```bash
mri_glmfit-sim \
  --glmdir lh.thickness.AdvsPARTStudy.10.glmdir \
  --cache 1.3 pos \
  --cwp 0.05 \
  --2spaces
```

**Parameters:**
- `--cache 1.3 pos`: Cluster-forming threshold (p < 0.05, positive effects)
- `--cwp 0.05`: Cluster-wise p-value threshold
- `--2spaces`: Correct for both hemispheres

**Output:** Significant clusters in `cache.th13.pos.sig.cluster.mgh`

**Visualization:**
```bash
# View significant clusters
freeview -f $SUBJECTS_DIR/fsaverage/surf/lh.inflated:overlay=lh.thickness.AdvsPARTStudy.10.glmdir/cache.th13.pos.sig.cluster.mgh:overlay_threshold=1,5 \
         -f $SUBJECTS_DIR/fsaverage/surf/rh.inflated:overlay=rh.thickness.AdvsPARTStudy.10.glmdir/cache.th13.pos.sig.cluster.mgh:overlay_threshold=1,5
```

### 3.4 Export Statistics

**Goal:** Extract mean values per parcellation region.

**Scripts:**
- `FreeSurfer/export_desikan_killiany.sh` - Thickness, volume, surface area
- `FreeSurfer/export_desikan_killiany_LGI.sh` - Gyrification indices

**Key Commands:**
```bash
# Export cortical thickness
aparcstats2table --subjects NACC* \
  --hemi lh \
  --meas thickness \
  --tablefile lh_thickness_aparc.txt

# Export subcortical volumes
asegstats2table --subjects NACC* \
  --meas volume \
  --tablefile aseg_volume.txt
```

**Outputs:** Tab-separated files for statistical analysis in R/Python

---

## Phase 4: Neuromaps Correlation Analysis

### 4.1 Convert FreeSurfer Output to GIFTI

**Goal:** Prepare significant cluster maps for neuromaps analysis.

**Key Commands:**
```bash
# Convert MGH to GIFTI format
mri_convert \
  lh.thickness.AdvsPARTStudy.10.glmdir/cache.th13.pos.sig.cluster.mgh \
  lh.thickness.sig.cluster.gii

mri_convert \
  rh.thickness.AdvsPARTStudy.10.glmdir/cache.th13.pos.sig.cluster.mgh \
  rh.thickness.sig.cluster.gii
```

**Code Reference:** `FreeSurfer/freesurfer_process.sh:74-77`

**Why GIFTI?** Required format for `neuromaps` Python library.

### 4.2 Single Map Correlation

**Goal:** Test correlation between structural findings and a molecular map.

**Script:** `Neuromaps/pipeline_neuromaps_v5.py`

**Surface-based Example:**
```bash
python pipeline_neuromaps_v5.py \
  --surface lh.thickness.sig.cluster.gii rh.thickness.sig.cluster.gii \
  fsaverage raichle cbf fsLR 164k \
  --null_model vazquez_rodriguez fsaverage 164k \
  --n-perm 1000 \
  --seed 1234
```

**Code Reference:** `Neuromaps/pipeline_neuromaps_v5.py`

**Parameters:**
- `--surface`: Left and right hemisphere GIFTI files
- `fsaverage`: Source space of input data
- `raichle cbf`: Target map (cerebral blood flow)
- `fsLR 164k`: Target space/resolution
- `--null_model vazquez_rodriguez`: Spatial autocorrelation-preserving nulls
- `--n-perm 1000`: Number of permutations for p-value

**Volumetric Example:**
```bash
python pipeline_neuromaps_v5.py \
  --volumetric input.nii.gz \
  MNI152 aghourian2017 feobv MNI152 1mm \
  --null_model burt2020 MNI152 1mm \
  --n-perm 1000
```

### 4.3 Batch Processing Multiple Maps

**Goal:** Correlate structural findings with 70+ molecular brain maps.

**Script:** `Neuromaps/batch_processing_surface_script.py`

**Command:**
```bash
python batch_processing_surface_script.py \
  --left-hemi lh.thickness.sig.cluster.gii \
  --right-hemi rh.thickness.sig.cluster.gii \
  --source-space fsaverage \
  --n-perm 1000 \
  --null-model vazquez_rodriguez \
  --study "AdvsPARTStudy" \
  --type-study surface
```

**Code Reference:** `Neuromaps/batch_processing_surface_script.py:24-89`

**Map Categories Tested:**
1. **Metabolism:** CBF, CBV, CMRglc, CMRO2
2. **Mitochondrial:** MRC, CII, TRC, CI, CIV, MitoD
3. **Neurotransmitters:**
   - Serotonin: 5-HT1A (way100635), 5-HT2A (az10419369), SERT (dasb)
   - Dopamine: D1 (sch23390), D2 (raclopride, flb457), DAT (fpcit)
   - GABA: flumazenil
   - Glutamate: abp688
   - Acetylcholine: feobv
   - Opioids: carfentanil
4. **Other:** Neuroinflammation (pbr28), cognitive PC1

**Output Format:**
```
Results saved to: neuromaps_results_AdvsPARTStudy_surface_YYYYMMDD_HHMMSS.csv
Columns:
- source_map: Your structural map
- target_map: Molecular map tested
- correlation: Pearson r value
- p_value: Permutation-based p-value
- null_mean: Mean of null distribution
- null_std: Standard deviation of null distribution
```

### 4.4 Convert to fsaverage6 for NCT

**Goal:** Prepare maps for network control theory analysis.

**Key Command:**
```bash
# Resample from fsaverage to fsaverage6
mri_surf2surf \
  --srcsubject fsaverage \
  --trgsubject fsaverage6 \
  --hemi lh \
  --srcsurfval lh.thickness.sig.cluster.mgh \
  --trgsurfval lh.thickness.sig.cluster.fsaverage6.mgh
```

**Why fsaverage6?** NCT toolbox uses parcellations defined on fsaverage6 space.

---

## Phase 5: Network Control Theory Analysis

### 5.1 Map Clusters to Brain Networks

**Goal:** Identify which brain networks are affected in the structural findings.

**Script:** `NCT/nct.py`

**Key Command:**
```bash
python nct.py
```

**Code Reference:** `NCT/nct.py`

**What it does:**
1. Loads cluster maps (`.mgh` format)
2. Binarizes at threshold 0.5
3. Maps to parcellation atlases:
   - **AS400K17**: Schaefer 400 parcels + Kong 17 subcortical
   - **TY17**: Yeo 17 networks
   - **EG17**: Gordon 17 networks
   - **AS200K17**: Schaefer 200 parcels + Kong 17 subcortical

**Atlas Mapping Examples:**
```
Cluster in left superior temporal → Default Mode Network
Cluster in right lateral occipital → Visual Network
Cluster in medial prefrontal → Default Mode Network (Anterior)
```

**Visualization:**
```bash
# View parcellation overlay
freeview -f $SUBJECTS_DIR/fsaverage6/surf/lh.inflated:annot=lh.Schaefer2018_400Parcels_17Networks_order.annot
```

---

## Phase 6: Visualization

### 6.1 Brain Surface Plots

**Goal:** Create publication-quality figures.

**Script:** `Utils/surfplott.py`

**Key Command:**
```python
python surfplott.py
```

**Code Reference:** `Utils/surfplott.py`

**What it generates:**
- Cluster maps (binary significant regions)
- Gamma maps (effect sizes / t-statistics)
- Thickness/volume maps
- LGI maps

**Features:**
- Multiple colormaps: `cold_hot`, `viridis`, `rocket`, `turbo`
- Inflated surface visualization
- Both hemispheres (lateral and medial views)
- 300 DPI resolution for publications

**Example Output:**
```
results_without_LB/
├── lh_thickness_cluster.png
├── rh_thickness_cluster.png
├── lh_thickness_gamma.png
└── rh_thickness_gamma.png
```

### 6.2 Statistical Maps

**Viewing gamma (effect size) maps:**
```bash
freeview -f $SUBJECTS_DIR/fsaverage/surf/lh.inflated:overlay=lh.thickness.AdvsPARTStudy.10.glmdir/AD-PART_Age_Sex_Delta_eTIV/gamma.mgh:overlay_threshold=0.5,3.0
```

**Viewing significance maps:**
```bash
freeview -f $SUBJECTS_DIR/fsaverage/surf/lh.inflated:overlay=lh.thickness.AdvsPARTStudy.10.glmdir/cache.th13.pos.sig.cluster.mgh:overlay_threshold=1,5
```

---

## Phase 7: Additional Analyses

### 7.1 Normalize Volumetric Data to MNI152

**Goal:** Transform volumetric data to standard space for neuromaps.

**Script:** `Utils/niftiTOmni152.py`

**Key Command:**
```python
python niftiTOmni152.py
```

**Code Reference:** `Utils/niftiTOmni152.py`

**What it does:**
- Uses `nilearn` for pure Python normalization
- Resamples to MNI152 template (1mm, 2mm, or 3mm)
- Alternative to FSL's FLIRT/FNIRT (older scripts in `old/`)

**Example normalization:**
```python
from nilearn.image import resample_to_img
from nilearn.datasets import load_mni152_template

template = load_mni152_template(resolution=2)
normalized = resample_to_img(source_img, template)
```

---

## Study-Specific Analyses

### Study 1: AD vs PART (Without Lewy Bodies)

**Directory:** `results_without_LB/`

**Subjects:** AD patients and PART patients without Lewy body pathology

**Measures:** Cortical thickness, volume, LGI

**Key Findings Locations:**
- Cluster maps: `{lh,rh}.thickness.AdvsPARTStudy.10.glmdir/cache.th13.pos.sig.cluster.mgh`
- Effect sizes: `{lh,rh}.thickness.AdvsPARTStudy.10.glmdir/AD-PART_Age_Sex_Delta_eTIV/gamma.mgh`

### Study 2: AD vs PART (With Lewy Bodies)

**Directory:** `results_with_LB/`

**Same pipeline as Study 1, but including subjects with Lewy body pathology**

### Study 3: Quartile Analysis

**Directory:** `results_quartile/`

**Approach:** Divide subjects into quartiles based on pathology severity

**Comparisons:**
- Group 1 vs Group 0 (mild vs minimal pathology)
- Group 2 vs Group 0 (moderate vs minimal)
- Group 2 vs Group 1 (moderate vs mild)

### Study 4: Neocortical Lewy Body

**Directory:** `neocortical_LB/`

**Focus:** Specifically analyze neocortical Lewy body distribution patterns

---

## Common Commands Reference

### FreeSurfer Quality Check
```bash
# View reconstruction quality
freeview -v $SUBJECTS_DIR/NACC145249/mri/brainmask.mgz \
         -f $SUBJECTS_DIR/NACC145249/surf/lh.pial:edgecolor=red \
         -f $SUBJECTS_DIR/NACC145249/surf/lh.white:edgecolor=blue \
         -f $SUBJECTS_DIR/NACC145249/surf/rh.pial:edgecolor=red \
         -f $SUBJECTS_DIR/NACC145249/surf/rh.white:edgecolor=blue
```

### View Group Analysis Results
```bash
# Overlay statistical map on fsaverage
freeview -f $SUBJECTS_DIR/fsaverage/surf/lh.inflated:overlay=lh.thickness.AdvsPARTStudy.10.glmdir/cache.th13.pos.sig.cluster.mgh:overlay_threshold=1,5:overlay_custom=0,255,0 \
         -f $SUBJECTS_DIR/fsaverage/surf/rh.inflated:overlay=rh.thickness.AdvsPARTStudy.10.glmdir/cache.th13.pos.sig.cluster.mgh:overlay_threshold=1,5:overlay_custom=0,255,0 \
         -layout vertical -viewport 3d
```

### Extract ROI Statistics
```bash
# Get mean thickness in a specific region
mri_segstats --annot NACC145249 lh aparc \
             --i $SUBJECTS_DIR/NACC145249/surf/lh.thickness \
             --sum lh.thickness.stats.txt
```

### Format Conversions
```bash
# MGH to GIFTI (for neuromaps)
mri_convert input.mgh output.gii

# GIFTI to MGH (from neuromaps)
mri_convert input.gii output.mgh

# NIfTI to MGH
mri_convert input.nii.gz output.mgh

# Resample between surface spaces
mri_surf2surf --srcsubject fsaverage \
              --trgsubject fsaverage6 \
              --hemi lh \
              --srcsurfval input.mgh \
              --trgsurfval output.mgh
```

### Parallel Processing
```bash
# Process multiple subjects in parallel
ls -d NACC* | parallel -j 12 "recon-all -s {} -all -qcache"

# Run multiple GLMs in parallel
for hemi in lh rh; do
  for meas in thickness volume; do
    echo "Processing $hemi $meas" &
  done
done
wait
```

---

## Software Environment

### Required Software
- **FreeSurfer 7.x** - Cortical reconstruction and analysis
- **Python 3.8+** with packages:
  - `neuromaps` - Brain map correlations
  - `nibabel` - Neuroimaging file I/O
  - `nilearn` - MRI processing
  - `surfplot` - Brain surface visualization
  - `pandas`, `numpy`, `scipy` - Data analysis
  - `matplotlib`, `seaborn` - Plotting

### Environment Setup
```bash
# FreeSurfer
export FREESURFER_HOME=/usr/local/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Subjects directory
export SUBJECTS_DIR=/path/to/freesurfer_output

# Python environment
conda create -n thesis python=3.10
conda activate thesis
pip install neuromaps nibabel nilearn surfplot pandas matplotlib
```

---

## Key Design Decisions

1. **Why fsaverage space?**
   - Standard FreeSurfer template with 163,842 vertices per hemisphere
   - Enables group analysis and comparison with published atlases
   - Compatible with neuromaps library

2. **Why spatial permutation testing?**
   - Brain data exhibits spatial autocorrelation
   - Standard parametric tests inflate false positives
   - Spin-based permutations preserve spatial structure

3. **Why Desikan-Killiany atlas?**
   - Widely used parcellation (34 regions per hemisphere)
   - Good balance between anatomical detail and interpretability
   - Defined on cortical surface, avoiding partial volume effects

4. **Why control for eTIV?**
   - Brain volumes correlate with head size
   - eTIV (estimated total intracranial volume) proxy for premorbid brain size
   - Controls for individual differences in brain size

5. **Why FWHM=10mm smoothing?**
   - Improves signal-to-noise ratio
   - Appropriate for cortical thickness data
   - Matches recommendations for group-level analysis

---

## Troubleshooting

### FreeSurfer reconstruction fails
```bash
# Check for motion artifacts or poor contrast
freeview -v mri/orig.mgz

# Manually fix topology errors
recon-all -s NACC145249 -autorecon2-cp -autorecon3
```

### Neuromaps transformation errors
```bash
# Verify input data is in correct space
mris_info lh.thickness.gii

# Check for NaN values
mri_info --vox-only input.mgh | grep -i nan
```

### Memory issues with parallel processing
```bash
# Reduce number of parallel jobs
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
parallel --jobs 6  # instead of 12
```

---

## Future Directions

1. **Additional Analyses:**
   - Structural covariance networks
   - Longitudinal analysis (if follow-up data available)
   - Machine learning classification (AD vs PART)

2. **Extended Correlations:**
   - Gene expression maps (Allen Human Brain Atlas)
   - Functional connectivity networks
   - White matter tractography

3. **Validation:**
   - Cross-validation with independent dataset
   - Sensitivity analyses with different atlases
   - Comparison with volumetric approaches

---

## References

**Key Papers:**
- Markiewicz et al. (2022) - Neuromaps: Spatial correlation of brain maps
- Alexander-Bloch et al. (2018) - Spatial permutation testing
- Vázquez-Rodríguez et al. (2019) - Spatial null models
- Desikan et al. (2006) - Desikan-Killiany atlas

**Tools Documentation:**
- FreeSurfer: https://surfer.nmr.mgh.harvard.edu/
- Neuromaps: https://netneurolab.github.io/neuromaps/
- CBIG Network Correspondence: https://github.com/ThomasYeoLab/CBIG

---

## Appendix: File Structure

```
Thesis/
├── automated_filtering/          # Phase 1: Data preprocessing
│   ├── process_MRIs.py
│   ├── process_MRIs_modality.py
│   ├── cross_MRIs.py
│   └── process_MRIs_pipeline.sh
│
├── FreeSurfer/                   # Phase 2-3: Structural analysis
│   ├── freesurfer_process.sh     # recon-all wrapper
│   ├── createLGI.sh              # Gyrification
│   ├── segmentAAN.sh             # Subcortical
│   ├── runMrisPreproc.sh         # Group prep
│   ├── runGLMs.sh                # Statistics
│   ├── runClustSims.sh           # Correction
│   ├── export_desikan_killiany.sh
│   ├── export_desikan_killiany_LGI.sh
│   ├── FSGD/                     # Study designs
│   └── Contrasts/                # GLM contrasts
│
├── Neuromaps/                    # Phase 4: Molecular correlations
│   ├── pipeline_neuromaps_v5.py
│   ├── neuromaps_multi_parcellation.py
│   └── batch_processing_surface_script.py
│
├── NCT/                          # Phase 5: Network analysis
│   ├── nct.py
│   └── config
│
├── Utils/                        # Phase 6: Visualization
│   ├── surfplott.py
│   └── niftiTOmni152.py
│
├── results_without_LB/           # Study 1 outputs
├── results_with_LB/              # Study 2 outputs
├── results_quartile/             # Study 3 outputs
├── neocortical_LB/               # Study 4 outputs
│
└── web_app/                      # Optional web interface
    ├── app.py
    └── pipeline.py
```

---

**Document Status:** Living document - updated as analysis progresses
**Last Pipeline Run:** See git history for timestamps
**Questions?** Refer to code comments or FreeSurfer/neuromaps documentation
