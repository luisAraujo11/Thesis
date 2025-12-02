#!/bin/bash

# Path variables
BASE_DIR="/media/memad/Disk1/NACC_MRI_subjects_Neuromaps_quartile/cross_subjects" # change for different path
CSV_FILE="${BASE_DIR}/cross_subjects.csv"
OUTPUT_DIR="${BASE_DIR}/freesurfer_output"

# Uncomment this line if FreeSurfer is not already in your environment
# source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Set SUBJECTS_DIR to store FreeSurfer outputs
export SUBJECTS_DIR="$OUTPUT_DIR"
mkdir -p $SUBJECTS_DIR

#export SUBJECTS_DIR="/media/memad/Disk1/NACC_MRI_subjects_Neuromaps_Neocortical_LB/cross_subjects/freesurfer_output_NCT"
# 17.55 -> 
# NACC145249
# Log files
LOG_FILE="${BASE_DIR}/recon_all_processing.log"
echo "====================================================" > $LOG_FILE
echo "Starting FreeSurfer recon-all processing at $(date)" >> $LOG_FILE
echo "Output directory: $SUBJECTS_DIR" >> $LOG_FILE
echo "====================================================" >> $LOG_FILE

# Performance settings
N_JOBS=12

# Create a simple commands file
COMMANDS_FILE="${BASE_DIR}/recon_all_commands.txt"
> $COMMANDS_FILE

# Process the CSV file
awk -F, 'NR>1 {gsub(/"/, "", $2); gsub(/"/, "", $3); print $2","$3}' "$CSV_FILE" | while IFS=, read -r subject_id scan_name; do
    # Remove .zip extension if present in the scan name
    scan_dir_name=${scan_name%.zip}
    scan_dir="${BASE_DIR}/${scan_dir_name}"
    
    # Check if directory exists
    if [ ! -d "$scan_dir" ]; then
        echo "ERROR: Directory $scan_dir does not exist. Skipping subject $subject_id." >> $LOG_FILE
        continue
    fi
    
    # Find the first NIfTI file in the directory
    first_nifti=$(find "$scan_dir" -type f \( -name "*.nii" -o -name "*.nii.gz" \) | sort | head -n 1)
    
    # If no NIfTI files found
    if [ -z "$first_nifti" ]; then
        echo "ERROR: No NIfTI files found in $scan_dir for subject $subject_id." >> $LOG_FILE
        continue
    fi
    
    # Add the simple recon-all command to our commands file
    echo "recon-all -i \"$first_nifti\" -s \"$subject_id\" -all -qcache" >> $COMMANDS_FILE # qcache usefull for group analysis
    #echo "recon-all -s \"$subject_id\" -qcache" >> $COMMANDS_FILE

    # Log the subject and file selected
    echo "Added subject $subject_id using $(basename "$first_nifti")" >> $LOG_FILE
done

# Count how many commands were created
COMMAND_COUNT=$(wc -l < "$COMMANDS_FILE")
echo "Will process $COMMAND_COUNT subjects in parallel using $N_JOBS jobs" | tee -a $LOG_FILE
echo "====================================================" | tee -a $LOG_FILE

# Run the commands in parallel - using the simple progress display
cat "$COMMANDS_FILE" | parallel --progress --jobs $N_JOBS

echo "====================================================" | tee -a $LOG_FILE
echo "Processing complete at $(date)" | tee -a $LOG_FILE
echo "====================================================" | tee -a $LOG_FILE

# For left hemisphere significance map
# mri_convert lh.thickness.AdvsPARTStudy.10.glmdir/cache.th13.pos.sig.cluster.mgh lh.thickness.sig.cluster.gii

# For right hemisphere
# mri_convert rh.thickness.AdvsPARTStudy.10.glmdir/cache.th13.pos.sig.cluster.mgh rh.thickness.sig.cluster.gii