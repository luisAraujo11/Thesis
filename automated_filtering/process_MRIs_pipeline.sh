#!/bin/bash

: '
=========================================================================================================================================
This script is a pipeline to process MRI data from zipped files, extract specific modality images, and create a cross-subject dataset.
First in process_MRIs.py, it extracts subjects from zipped files based on a CSV file with subject IDs (subject name, which is the unique identifier, must be in the third column).
Second in process_MRIs_modality.py, it extracts specific modality images (e.g., T1s) based on the json (created by the tool dcm2niix) files from the extracted subjects.
Third in cross_MRIs.py, it creates a cross-subject dataset based on the extracted modality images and a second CSV file with subject IDs(subject name, which is the unique identifier, must be in the third column), also creates a CSV with the valid sujects.

Note: If you dont want to use all three steps, you can run can comment (#) the python execution line.
=========================================================================================================================================
Example usage:
1)  Make the script executable:  
        chmod +x process_MRIs_pipeline.sh

2.1)  Run the pipeline with simple arguments:
        ./process_MRIs_pipeline.sh -s /path/to/source -d /path/to/dest -c /path/to/csv

2.2)Run the pipeline with custom modality and terms:
        ./process_MRIs_pipeline.sh -s /path/to/source -d /path/to/dest -c /path/to/csv -m t1 -t 'mprage,t1,3d_ir'

2.3)Run the pipeline with a second CSV file:
        ./process_MRIs_pipeline.sh -s /path/to/source -d /path/to/dest -c /path/to/csv -n /path/to/scnd_csv

2.4)Run the pipeline with a second CSV file and custom modality:
        ./process_MRIs_pipeline.sh -s /path/to/source -d /path/to/dest -c /path/to/csv -n /path/to/scnd_csv -m t1 -t 'mprage,t1,3d_ir'

Examples:
./process_MRIs_pipeline.sh -s "/media/memad/Disk1/NACC_MRI_RawData_0324/all/nifti/" -d "/media/memad/Disk1/NACC_MRI_subjects_Neuromaps_test1" -c "subjects.csv"
./process_MRIs_pipeline.sh -s "/media/memad/Disk1/NACC_MRI_RawData_0324/all/nifti/" -d "/media/memad/Disk1/NACC_MRI_subjects_Neuromaps_test2" -c "subjects.csv" -n "subjects_last.csv"
./process_MRIs_pipeline.sh -s "/media/memad/Disk1/NACC_MRI_RawData_0324/all/nifti/" -d "/media/memad/Disk1/NACC_MRI_subjects_Neuromaps_test3" -c "subjects.csv" -n "subjects_last.csv" -m "mprage" -t "mprage,mp-rage,mp2rage"
=========================================================================================================================================
'

# Default values
SOURCE_DIR=""       # obligatory
DEST_DIR=""         # obligatory
CSV_FILE=""         # obligatory
SCND_CSV_FILE=""    # optional
MODALITY="t1"       # Default modality is T1
MODALITY_TERMS=""   # Custom modality terms (optional)

# Function to display usage
usage() {
    echo "Usage: $0 -s <source_dir> -d <dest_dir> -c <csv_file> [-n <scnd_csv_file>] [-m <modality>] [-t <modality_terms>]"
    echo "  -s: Source directory containing zipped MRI data"
    echo "  -d: Destination directory for processed data"
    echo "  -c: CSV file with subject IDs"
    echo "  -n: Second CSV file with subject IDs (optional)"
    echo "  -m: MRI modality (default: t1)"
    echo "  -t: Custom modality terms as comma-separated list (optional)"
    echo "      Example: -t 'mprage,t1,3d_ir'"
    echo "Read script for more details!"
    exit 1
}

# Parse command-line arguments
while getopts "s:d:c:n:m:t:" opt; do
    case $opt in
        s) SOURCE_DIR="$OPTARG" ;;
        d) DEST_DIR="$OPTARG" ;;
        c) CSV_FILE="$OPTARG" ;;
        n) SCND_CSV_FILE="$OPTARG" ;;
        m) MODALITY="$OPTARG" ;;
        t) MODALITY_TERMS="$OPTARG" ;;
        *) usage ;;
    esac
done

# Check if required arguments are provided
if [ -z "$SOURCE_DIR" ] || [ -z "$DEST_DIR" ] || [ -z "$CSV_FILE" ]; then
    echo "Error: Missing required arguments."
    usage
fi

# Check if directories and file exist
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file does not exist: $CSV_FILE"
    exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Define derived directories
MODALITY_DIR="$DEST_DIR/${MODALITY}s"
CROSS_DIR="$DEST_DIR/cross_subjects"

echo "=== MRI Processing Pipeline ==="
echo "Source directory: $SOURCE_DIR"
echo "Destination directory: $DEST_DIR"
echo "Modality directory: $MODALITY_DIR"
echo "Cross subjects directory: $CROSS_DIR"
if [ -n "$SCND_CSV_FILE" ]; then
    echo "Second CSV file: $SCND_CSV_FILE"
else
    echo "CSV file: $CSV_FILE"
fi
echo "Modality: $MODALITY"
if [ -n "$MODALITY_TERMS" ]; then
    echo "Modality terms: $MODALITY_TERMS"
fi
echo "================================"

# Step 1: Unzip and process MRIs
echo "Step 1: Extracting subjects from zipped files..."
python process_MRIs.py "$SOURCE_DIR" "$DEST_DIR" "$CSV_FILE"

# Step 2: Process specific modality (e.g., T1s)
echo "Step 2: Extracting $MODALITY images..."
# Check if MODALITY_TERMS is set
if [ -n "$MODALITY_TERMS" ]; then
    echo "Using custom modality terms: $MODALITY_TERMS"
    python process_MRIs_modality.py "$DEST_DIR" "$MODALITY_DIR" "$CSV_FILE" "$MODALITY" "$MODALITY_TERMS"
else
    python process_MRIs_modality.py "$DEST_DIR" "$MODALITY_DIR" "$CSV_FILE" "$MODALITY"
fi

# Step 3: Cross subjects with CSV and create final dataset
echo "Step 3: Creating cross-subject dataset..."
if [ -n "$SCND_CSV_FILE" ]; then
    python cross_MRIs.py "$MODALITY_DIR" "$CROSS_DIR" "$SCND_CSV_FILE"
else
    python cross_MRIs.py "$MODALITY_DIR" "$CROSS_DIR" "$CSV_FILE"
fi
echo "=== Pipeline Complete ==="
echo "Results are in: $CROSS_DIR"