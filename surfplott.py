# ==============================================================================
# IMPORTS
# ==============================================================================
import numpy as np
import nibabel as nib
from surfplot import Plot
import matplotlib.pyplot as plt                 # type: ignore
from neuromaps.datasets import fetch_fsaverage  # type: ignore


# ==============================================================================
# FETCH SURFACES
# ==============================================================================
# Fetch fsaverage surfaces for visualization
surfaces = fetch_fsaverage(density='164k')
lh, rh = surfaces['inflated']
lh_sulc, rh_sulc = surfaces['sulc']  # Sulcal depth for shading
# lh, rh = surfaces['pial']  # Alternative: use pial surface


# ==============================================================================
# FILE PATHS - WITH LB STUDY
# ==============================================================================
# Volume analysis
lh_withLB_gamma = "results_with_LB/lh.volume.AdvsPARTwithLBStudy_Delta_eTIV_norm_gamma.gii"
rh_withLB_gamma = "results_with_LB/rh.volume.AdvsPARTwithLBStudy_Delta_eTIV_norm_gamma.gii"
lh_withLB_gamma2 = "results_with_LB/lh.volume.AdvsPARTwithLBStudy_Delta_eTIV_norm2_gamma.gii"
rh_withLB_gamma2 = "results_with_LB/rh.volume.AdvsPARTwithLBStudy_Delta_eTIV_norm2_gamma.gii"

# Significance maps
lh_withLB_sig = "results_with_LB/lh.volume.AdvsPARTwithLBStudy_Age_Delta_eTIV_norm_sig.gii"
rh_withLB_sig = "results_with_LB/rh.volume.AdvsPARTwithLBStudy_Age_Delta_eTIV_norm_sig.gii"

# Thickness analysis
lh_thickness_withLB_gamma = "results_with_LB/lh.thickness.AdvsPARTwithLBStudy_Delta_eTIV_norm2_gamma.gii"
rh_thickness_withLB_gamma = "results_with_LB/rh.thickness.AdvsPARTwithLBStudy_Delta_eTIV_norm2_gamma.gii"


# ==============================================================================
# FILE PATHS - WITHOUT LB STUDY
# ==============================================================================
# Volume analysis
lh_withoutLB_gamma = "results_without_LB/lh.volume.AdvsPARTwithoutLBStudy_Delta_eTIV_norm_gamma.gii"
rh_withoutLB_gamma = "results_without_LB/rh.volume.AdvsPARTwithoutLBStudy_Delta_eTIV_norm_gamma.gii"
lh_withoutLB_gamma2 = "results_without_LB/lh.volume.AdvsPARTwithoutLBStudy_Delta_eTIV_norm2_gamma.gii"
rh_withoutLB_gamma2 = "results_without_LB/rh.volume.AdvsPARTwithoutLBStudy_Delta_eTIV_norm2_gamma.gii"

# Cluster analysis
lh_withoutLB_cluster = "results_without_LB/lh.volume.AdvsPARTwithoutLBStudy_cluster.gii"
rh_withoutLB_cluster = "results_without_LB/rh.volume.AdvsPARTwithoutLBStudy_cluster.gii"

# Significance maps
lh_withoutLB_sig = "results_without_LB/lh.volume.AdvsPARTwithoutLBStudy_Age_Delta_eTIV_norm_sig.gii"
rh_withoutLB_sig = "results_without_LB/rh.volume.AdvsPARTwithoutLBStudy_Age_Delta_eTIV_norm_sig.gii"

# Thickness analysis
lh_thickness_withoutLB_gamma = "results_without_LB/lh.thickness.AdvsPARTwithoutLBStudy_Delta_eTIV_norm2_gamma.gii"
rh_thickness_withoutLB_gamma = "results_without_LB/rh.thickness.AdvsPARTwithoutLBStudy_Delta_eTIV_norm2_gamma.gii"
lh_thickness_withoutLB_cluster = "results_without_LB/lh.thickness.AdvsPARTwithoutLBStudy_cluster.gii"
rh_thickness_withoutLB_cluster = "results_without_LB/rh.thickness.AdvsPARTwithoutLBStudy_cluster.gii"
lh_thickness_withoutLB_gamma2 = "results_without_LB/lh.thickness.AdvsPARTwithoutLBStudy_Delta_eTIV_norm2_gamma_15.gii"
rh_thickness_withoutLB_gamma2 = "results_without_LB/rh.thickness.AdvsPARTwithoutLBStudy_Delta_eTIV_norm2_gamma_15.gii"


# ==============================================================================
# FILE PATHS - QUARTILE STUDY
# ==============================================================================
# Volume analysis - Group comparisons
lh_volume_quartile_group1vs0_gamma = "results_quartile/lh.volume.AdvsPARTquartileStudy_Age_Delta_eTIV_norm_group1vs0_gamma.gii"
rh_volume_quartile_group1vs0_gamma = "results_quartile/rh.volume.AdvsPARTquartileStudy_Age_Delta_eTIV_norm_group1vs0_gamma.gii"
lh_volume_quartile_group2vs0_gamma = "results_quartile/lh.volume.AdvsPARTquartileStudy_Age_Delta_eTIV_norm_group2vs0_gamma.gii"
rh_volume_quartile_group2vs0_gamma = "results_quartile/rh.volume.AdvsPARTquartileStudy_Age_Delta_eTIV_norm_group2vs0_gamma.gii"
lh_volume_quartile_group2vs1_gamma = "results_quartile/lh.volume.AdvsPARTquartileStudy_Age_Delta_eTIV_norm_group2vs1_gamma.gii"
rh_volume_quartile_group2vs1_gamma = "results_quartile/rh.volume.AdvsPARTquartileStudy_Age_Delta_eTIV_norm_group2vs1_gamma.gii"

# Thickness analysis - Group comparisons
lh_thickness_quartile_group1vs0_gamma = "results_quartile/lh.thickness.AdvsPARTquartileStudy_Age_Delta_eTIV_norm_group1vs0_gamma.gii"
rh_thickness_quartile_group1vs0_gamma = "results_quartile/rh.thickness.AdvsPARTquartileStudy_Age_Delta_eTIV_norm_group1vs0_gamma.gii"
lh_thickness_quartile_group2vs0_gamma = "results_quartile/lh.thickness.AdvsPARTquartileStudy_Age_Delta_eTIV_norm_group2vs0_gamma.gii"
rh_thickness_quartile_group2vs0_gamma = "results_quartile/rh.thickness.AdvsPARTquartileStudy_Age_Delta_eTIV_norm_group2vs0_gamma.gii"
lh_thickness_quartile_group2vs1_gamma = "results_quartile/lh.thickness.AdvsPARTquartileStudy_Age_Delta_eTIV_norm_group2vs1_gamma.gii"
rh_thickness_quartile_group2vs1_gamma = "results_quartile/rh.thickness.AdvsPARTquartileStudy_Age_Delta_eTIV_norm_group2vs1_gamma.gii"


# ==============================================================================
# FILE PATHS - NEOCORTICAL LB ANALYSIS
# ==============================================================================
# Volume analysis
lh_volume_NeocorticalLB_gamma = "neocortical_LB/lh.volume.NeocorticalLB_Analysis_gamma.gii"
rh_volume_NeocorticalLB_gamma = "neocortical_LB/rh.volume.NeocorticalLB_Analysis_gamma.gii"
lh_volume_NeocorticalLB_sig = "neocortical_LB/lh.volume.NeocorticalLB_Analysis_sig.gii"
rh_volume_NeocorticalLB_sig = "neocortical_LB/rh.volume.NeocorticalLB_Analysis_sig.gii"
lh_volume_NeocorticalLB_cluster = "neocortical_LB/lh.volume.NeocorticalLB_Analysis_cluster.gii"
rh_volume_NeocorticalLB_cluster = "neocortical_LB/rh.volume.NeocorticalLB_Analysis_cluster.gii"

# Thickness analysis
lh_thickness_NeocorticalLB_gamma = "neocortical_LB/lh.thickness.NeocorticalLB_Analysis_gamma.gii"
rh_thickness_NeocorticalLB_gamma = "neocortical_LB/rh.thickness.NeocorticalLB_Analysis_gamma.gii"
lh_thickness_NeocorticalLB_sig = "neocortical_LB/lh.thickness.NeocorticalLB_Analysis_sig.gii"
rh_thickness_NeocorticalLB_sig = "neocortical_LB/rh.thickness.NeocorticalLB_Analysis_sig.gii"
lh_thickness_NeocorticalLB_cluster = "neocortical_LB/lh.thickness.NeocorticalLB_Analysis_cluster.gii"
rh_thickness_NeocorticalLB_cluster = "neocortical_LB/rh.thickness.NeocorticalLB_Analysis_cluster.gii"


# ==============================================================================
# LOAD DATA
# ==============================================================================
# Load gamma (effect size) data
lh_gii_gamma = nib.load(lh_withoutLB_gamma2)
rh_gii_gamma = nib.load(rh_withoutLB_gamma2)
lh_data_gamma = lh_gii_gamma.darrays[0].data
rh_data_gamma = rh_gii_gamma.darrays[0].data

# Load cluster (significance) data
lh_gii_cluster = nib.load(lh_thickness_withoutLB_cluster)
rh_gii_cluster = nib.load(rh_thickness_withoutLB_cluster)
lh_data_cluster = lh_gii_cluster.darrays[0].data
rh_data_cluster = rh_gii_cluster.darrays[0].data


# ==============================================================================
# DATA MASKING AND THRESHOLDING
# ==============================================================================
# Create masked gamma data - only show values where significance map indicates significance
lh_data_masked = lh_data_gamma.copy()
rh_data_masked = rh_data_gamma.copy()

# Optional: Apply significance threshold
# lh_data_masked[lh_data_cluster < -np.log10(0.05)] = 0
# rh_data_masked[rh_data_cluster < -np.log10(0.05)] = 0

# Print data statistics for verification
print(f"Original gamma data range: {np.min(lh_data_gamma):.3f} to {np.max(lh_data_gamma):.3f}")
print(f"Significance data range: {np.min(lh_data_cluster):.3f} to {np.max(lh_data_cluster):.3f}")
print(f"Masked gamma data range: {np.min(lh_data_masked):.3f} to {np.max(lh_data_masked):.3f}")
print(f"Number of significant vertices: LH={np.sum(lh_data_masked != 0)}, RH={np.sum(rh_data_masked != 0)}")


# ==============================================================================
# COLORMAP CONFIGURATION
# ==============================================================================
# Calculate symmetric color range around zero
min_val = min(np.min(lh_data_masked), np.min(rh_data_masked))
max_val = max(np.max(lh_data_masked), np.max(rh_data_masked))
max_abs = max(abs(min_val), abs(max_val))

# Colormap options for different data types
colormap_options = {
    'microstructure': 'RdYlGn',     # Green-yellow-red (e.g., T1w/T2w ratio)
    'metabolism': 'viridis',        # Yellow-green-blue (e.g., CBF, CBV)
    'function': 'plasma',           # Yellow-red-purple (e.g., functional gradient)
    'expansion': 'copper',          # Brown-orange (e.g., evolutionary expansion)
    'electro': 'coolwarm',          # Blue-white-red (e.g., power maps)
    'receptors': 'magma',           # Purple-red-yellow (e.g., receptor maps)
    'genomics': 'RdPu'              # Red-purple (e.g., gene expression)
}

# Select colormap
selected_cmap = 'RdBu_r'


# ==============================================================================
# CREATE BRAIN SURFACE PLOT
# ==============================================================================
# Initialize plot with specified parameters
p = Plot(lh, rh, size=(600, 300), zoom=1.8, views=['lateral', 'medial'])

# Add sulcal depth for anatomical context
p.add_layer({'left': lh_sulc, 'right': rh_sulc}, 
            cmap='gray', 
            cbar=False, 
            alpha=0.2)

# Add main data layer (masked gamma values)
p.add_layer({'left': lh_data_masked, 'right': rh_data_masked}, 
            cmap=selected_cmap, 
            color_range=(-max_abs, max_abs),  # Symmetric range
            cbar_label='Volume') # Change label as needed

# Configure colorbar appearance
cbar_kws = dict(
    location='bottom', 
    n_ticks=3,      # Display min, 0, and max values
    decimals=2, 
    shrink=0.8
)

# Build the figure
fig = p.build(cbar_kws=cbar_kws)


# ==============================================================================
# FINALIZE AND SAVE FIGURE
# ==============================================================================
# Add title to figure
plt.figtext(0.5, 0.01, "PART vs AD without LB", ha='center', fontsize=7)

# Save figure to file
plt.savefig('brain_map_volume_ADvsPART_withoutLB_gamma.png', dpi=300, bbox_inches='tight')

# Display figure
plt.show()


# ==============================================================================
# HELPER FUNCTION - SAVE GIFTI FROM TEMPLATE
# ==============================================================================
def save_gifti_from_template(data, template_file, output_file):
    """
    Save data as GIFTI file using an existing GIFTI file as template.
    
    Parameters:
    -----------
    data : ndarray
        Data array to save
    template_file : str
        Path to template GIFTI file
    output_file : str
        Path for output GIFTI file
    """
    # Load template
    template_gii = nib.load(template_file)
    
    # Create new GIFTI with same structure as template
    new_gii = nib.gifti.GiftiImage()
    new_gii.meta = template_gii.meta
    
    # Create new data array with template structure
    darray = nib.gifti.GiftiDataArray(
        data=data.astype(np.float32),
        intent=template_gii.darrays[0].intent,
        datatype=template_gii.darrays[0].datatype,
        meta=template_gii.darrays[0].meta
    )
    
    new_gii.add_gifti_data_array(darray)
    nib.save(new_gii, output_file)
    print(f"Saved using template: {output_file}")


# ==============================================================================
# SAVE CORRECTED GIFTI FILES
# ==============================================================================
# Save left hemisphere
save_gifti_from_template(
    lh_data_masked, 
    "results_without_LB/lh.thickness.AdvsPARTwithoutLBStudy_cluster.gii",
    'lh.thickness.AdvsPARTwithoutLBStudy_cluster_corrected_v2.gii'
)

# Save right hemisphere
save_gifti_from_template(
    rh_data_masked,
    "results_without_LB/rh.thickness.AdvsPARTwithoutLBStudy_cluster.gii",
    'rh.thickness.AdvsPARTwithoutLBStudy_cluster_corrected_v2.gii'
)