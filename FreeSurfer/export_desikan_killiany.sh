#!/bin/bash
# Script to export Desikan-Killiany atlas segmentation for all subjects

# study name 
STUDY_NAME="without_LB"

# Set the SUBJECTS_DIR if needed (uncomment and modify this line if necessary)
export SUBJECTS_DIR="/media/memad/Disk1/NACC_MRI_subjects_Neuromaps_${STUDY_NAME}/cross_subjects/freesurfer_output"


# Output directory (current directory where the script is run)
OUTPUT_DIR=$(pwd)

echo "Subject directory: $SUBJECTS_DIR"
echo "Output directory: $OUTPUT_DIR"

# Change to the SUBJECTS_DIR to find subjects
cd $SUBJECTS_DIR

# Create a temporary file with all subject IDs
find . -maxdepth 1 -type d -name 'NACC[0-9]*' | sed 's|./||' > $OUTPUT_DIR/subjects_list.txt

# Change back to the output directory
cd $OUTPUT_DIR

echo "Found $(wc -l < subjects_list.txt) subjects"

# Extract measurements from left hemisphere with study name prefix
echo "Extracting left hemisphere thickness measurements..."
aparcstats2table --subjects $(cat subjects_list.txt) --hemi lh --meas thickness --parc=aparc --tablefile=lh_thickness_aparc_${STUDY_NAME}.txt

echo "Extracting left hemisphere volume measurements..."
aparcstats2table --subjects $(cat subjects_list.txt) --hemi lh --meas volume --parc=aparc --tablefile=lh_volume_aparc_${STUDY_NAME}.txt

echo "Extracting left hemisphere surface area measurements..."
aparcstats2table --subjects $(cat subjects_list.txt) --hemi lh --meas area --parc=aparc --tablefile=lh_area_aparc_${STUDY_NAME}.txt

# Extract measurements from right hemisphere
echo "Extracting right hemisphere thickness measurements..."
aparcstats2table --subjects $(cat subjects_list.txt) --hemi rh --meas thickness --parc=aparc --tablefile=rh_thickness_aparc_${STUDY_NAME}.txt

echo "Extracting right hemisphere volume measurements..."
aparcstats2table --subjects $(cat subjects_list.txt) --hemi rh --meas volume --parc=aparc --tablefile=rh_volume_aparc_${STUDY_NAME}.txt

echo "Extracting right hemisphere surface area measurements..."
aparcstats2table --subjects $(cat subjects_list.txt) --hemi rh --meas area --parc=aparc --tablefile=rh_area_aparc_${STUDY_NAME}.txt

# Also extract subcortical segmentation statistics
echo "Extracting subcortical volume measurements..."
asegstats2table --subjects $(cat subjects_list.txt) --meas volume --stats=aseg.stats --tablefile=aseg_volume_${STUDY_NAME}.txt

# subcortical mean measurements...
echo "Extracting subcortical mean measurements..."
asegstats2table --subjects $(cat subjects_list.txt) --meas mean --stats=aseg.stats --tablefile=aseg_mean_${STUDY_NAME}.txt

echo "Done! Results saved to *_aparc.txt and aseg_volume.txt files"
echo "These files can be opened directly in Excel or other spreadsheet software."

# Clean up
rm subjects_list.txt