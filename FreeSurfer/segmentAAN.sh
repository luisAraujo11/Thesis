# script to run the segmentAAN.sh script for each subject in the subjects directory

# change accordingly to the study
SUBJECTS_DIR=/data

# change accordingly to the number of cores
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=12

# get list of subjects
subjects=($(ls $SUBJECTS_DIR | grep "^NACC[0-9]*"))

# Run up to 12 subjects in parallel
max_jobs=12
job_count=0
for subject in "${subjects[@]}"; do
    echo "Processing subject: $subject"
    SegmentAAN.sh "$subject" &
    job_count=$((job_count+1))
    if (( job_count % max_jobs == 0 )); then
        wait  # Wait for all background jobs to finish before starting new ones
    fi
done
wait  # Wait for any remaining jobs to finish
