import numpy as np
import nibabel as nib
import cbig_network_correspondence as cnc

# Load mgh files
lh = nib.load("lh_pial_lgi_Neocortical_LB_sig_cluster_fsa6.mgh").get_fdata().squeeze()  # force shape: (40962,)
rh = nib.load("rh_pial_lgi_Neocortical_LB_sig_cluster_fsa6.mgh").get_fdata().squeeze()  # force shape: (40962,)

# Concatenate into full cortical map
data = np.concatenate([lh, rh])  # shape: (81924,)
print(f"Data shape: {data.shape}")
print("Array contents:", data[:100])  # print first 10 elements for verification

# Binary thresholding
threshold = 0.5
data = (data > threshold).astype(np.int32)  # binary thresholding at 0.5
print(f"Data after thresholding shape: {data.shape}")
print("Array contents after thresholding:", data[:100])  # print first 10 elements for verification

# Save in correct flat shape
np.save("pial_lgi_Neocortical_LB_Hard.npy", data.reshape(-1, 1))  # ensure it's flat
#print("Data saved to volume_ADvsPART_without_LB_Hard.npy")

# Just to verify shape again
data = np.load("pial_lgi_Neocortical_LB_Hard.npy")
print("Loaded shape:", data.shape)  # should print (81924, 1)
#print("Array contents:", data[:1000])  # print first 10 elements for verification

# Set up for CBIG
file_path = 'pial_lgi_Neocortical_LB_Hard.npy'
config = 'config'
atlas_names_list = [
    "AS400K17",    # Schaefer 400 + Kong 17 networks (comprehensive)
    "TY17",        # Yeo 17 networks (classic)
    "EG17",        # Power/Gordon 17 networks (task-based)
    "AS200K17"     # Schaefer 200 + Kong 17 (coarser resolution)
]

ref_params = cnc.compute_overlap_with_atlases.DataParams(config, file_path)
cnc.compute_overlap_with_atlases.network_correspondence(
    ref_params,
    atlas_names_list,
    "/mounts/disk2/projeto/NCT/output_pial_lgi_Neocortical_LB_Hard"  # output directory
)
