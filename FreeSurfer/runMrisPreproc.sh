#!/bin/bash

# In Bash, we use simple assignment instead of setenv
study=$1

# Convert foreach loops to for loops in Bash
for hemi in lh rh; do

  for smoothing in 10; do
  
    for meas in pial_lgi; do
    
        # Variable syntax is ${var} in Bash
        mris_preproc --fsgd FSGD/${study}.fsgd \
          --cache-in ${meas}.fwhm${smoothing}.fsaverage \
          --target fsaverage \
          --hemi ${hemi} \
          --out ${hemi}.${meas}.${study}.${smoothing}.mgh
          
    done
      
  done
    
done
