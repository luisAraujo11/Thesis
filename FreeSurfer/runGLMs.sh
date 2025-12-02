#!/bin/bash

study=$1

for hemi in lh rh; do
  for smoothness in 10; do
    for meas in pial_lgi; do
        mri_glmfit \
        --y ${hemi}.${meas}.${study}.${smoothness}.mgh \
        --fsgd FSGD/${study}.fsgd doss \
        --C Contrasts/AD-PART_Age_Sex_Delta_eTIV.mtx \
        --C Contrasts/PART-AD_Age_Sex_Delta_eTIV.mtx \
        --surf fsaverage ${hemi} \
        --cortex \
        --glmdir ${hemi}.${meas}.${study}.${smoothness}.glmdir
    done
  done
done
