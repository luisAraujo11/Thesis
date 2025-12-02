#!/bin/bash

study=$1

for meas in pial_lgi; do
  for hemi in lh rh; do
    for smoothness in 10; do
      for dir in ${hemi}.${meas}.${study}.${smoothness}.glmdir; do
        mri_glmfit-sim \
          --glmdir ${dir} \
          --cache 1.3 pos \
          --cwp 0.05 \
          --2spaces
      done
    done
  done
done
