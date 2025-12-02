#!/bin/bash
# Save as createLGIForGroupAnalysis.sh

SUBJECTS_DIR=/media/memad/Disk1/NACC_MRI_subjects_Neuromaps_without_LB/cross_subjects/freesurfer_output
# Get list of subjects from your FSGD file (or create manually)
subjects=($(ls $SUBJECTS_DIR | grep "^NACC[0-9]*"))

echo "Processing LGI for $(echo "$subjects" | wc -l) subjects..."

for subject in "${subjects[@]}"; do
    echo "Processing $subject..."
    
    for hemi in lh rh; do
        # Check if basic LGI file exists
        if [ -f "${subject}/surf/${hemi}.pial_lgi" ]; then
            
            # Create smoothed versions
            for fwhm in 10; do
                output_smooth="${subject}/surf/${hemi}.pial_lgi.fwhm${fwhm}.mgh"
                output_fsavg="${subject}/surf/${hemi}.pial_lgi.fwhm${fwhm}.fsaverage.mgh"
                
                # Skip if already exists
                if [ -f "$output_fsavg" ]; then
                    echo "  $output_fsavg already exists, skipping..."
                    continue
                fi
                
                echo "  Creating ${hemi} fwhm${fwhm} for $subject..."
                
                # Smooth the data
                mris_fwhm --s ${subject} --hemi ${hemi} \
                    --cortex --smooth-only --fwhm ${fwhm} \
                    --i ${subject}/surf/${hemi}.pial_lgi \
                    --o ${output_smooth}
                
                # Resample to fsaverage
                mri_surf2surf --srcsubject ${subject} \
                    --srcsurfval ${output_smooth} \
                    --trgsubject fsaverage --trgsurfval ${output_fsavg} \
                    --hemi ${hemi}
                    
            done
        else
            echo "  WARNING: ${subject}/surf/${hemi}.pial_lgi not found!"
        fi
    done
done

echo "Done creating smoothed LGI files!"
