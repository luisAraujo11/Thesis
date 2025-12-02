# script to extract all the "arousalNetworkVolumes.v10.stats" files from each subject in the subjects directory

SUBJECTS_DIR=/media/memad/Disk1/NACC_MRI_subjects_Neuromaps_Neocortical_LB/cross_subjects/freesurfer_output
output_dir=~/luis/AAN_segmentation/Neocortical_LB
mkdir -p $output_dir

# get list of subjects
subjects=($(ls $SUBJECTS_DIR | grep "^NACC[0-9]*"))

# loop through each subject
for subject in "${subjects[@]}"; do
    echo "Processing subject: $subject"
    # define the path to the stats file
    stats_file="$SUBJECTS_DIR/$subject/stats/arousalNetworkVolumes.v10.stats"
    # check if the stats file exists
    if [[ -f $stats_file ]]; then
        echo "Found stats file for subject $subject"
        # copy the stats file to the output directory with a new name
        cp $stats_file "$output_dir/${subject}_arousalNetworkVolumes.v10.stats"
    else
        echo "Stats file not found for subject $subject"
    fi
done