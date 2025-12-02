#!/bin/bash
SUBJECTS_DIR=/media/memad/Disk1/NACC_MRI_subjects_Neuromaps_Neocortical_LB/cross_subjects/freesurfer_output
export SUBJECTS_DIR
export PATH=/Applications/MATLAB_R2021b.app/bin:${PATH}

subjects=($(ls $SUBJECTS_DIR | grep "^NACC[0-9]*"))

# Run 12/32 subjects in parallel 
for subject in "${subjects[@]}"; do
    echo "Processing lGI for $subject"
    recon-all -s $subject -localGI &
    
    # Keep 12 jobs running
    if (( $(jobs -r | wc -l) >= 12 )); then # maybe change to 12 to not overload the workstation
        wait -n
    fi
done
wait