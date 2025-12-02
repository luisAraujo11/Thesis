#!/bin/bash
# Script to export Desikan-Killiany atlas segmentation with LGI metric for all subjects

# Set the SUBJECTS_DIR if needed (uncomment and modify this line if necessary)
export SUBJECTS_DIR="/media/memad/Disk1/NACC_MRI_subjects_Neuromaps_Neocortical_LB/cross_subjects/freesurfer_output"

# Create and set output directory
BASE_DIR=$(pwd)
OUTPUT_DIR="$BASE_DIR/LGI_results_Neocortical_LB"
mkdir -p "$OUTPUT_DIR"

echo "Subject directory: $SUBJECTS_DIR"
echo "Output directory: $OUTPUT_DIR"

# Change to the SUBJECTS_DIR to find subjects
cd $SUBJECTS_DIR

# Create a temporary file with all subject IDs
find . -maxdepth 1 -type d -name "NACC*" | sed 's|./||' > $OUTPUT_DIR/subjects_list.txt

# Change back to the output directory
cd $OUTPUT_DIR

echo "Found $(wc -l < subjects_list.txt) subjects"

# LGI results from the parcellations (loop version)
echo "Extracting LGI parcellation stats..."

for subj in $(cat subjects_list.txt); do
  echo "  -> Processing $subj"

  # Left hemisphere adicionar aseg
  mri_segstats \
    --annot ${subj} lh aparc \
    --i ${SUBJECTS_DIR}/${subj}/surf/lh.pial_lgi \
    --sum ${OUTPUT_DIR}/${subj}_lh.aparc.pial_lgi.stats

  # Right hemisphere
  mri_segstats \
    --annot ${subj} rh aparc \
    --i ${SUBJECTS_DIR}/${subj}/surf/rh.pial_lgi \
    --sum ${OUTPUT_DIR}/${subj}_rh.aparc.pial_lgi.stats

  # Optionally append to combined files (uncomment if needed)
  #cat ${OUTPUT_DIR}/${subj}_lh.aparc.pial_lgi.stats >> lh.aparc.pial_lgi.stats
  #cat ${OUTPUT_DIR}/${subj}_rh.aparc.pial_lgi.stats >> rh.aparc.pial_lgi.stats
done

echo "These files can be opened directly in Excel or other spreadsheet software."

# Clean up
rm subjects_list.txt