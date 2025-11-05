"""
Calculate mean QUS values within ROI and create scatter plots vs eGFR
Compatible with MATLAB v7.3 (HDF5) files
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy import stats
from scipy.ndimage import zoom
import os
import h5py
import re
import glob
import pandas as pd

# -------------------------------------------------------------------------
# Global debug flag
# -------------------------------------------------------------------------
DEBUG = True
def debug(msg):
    if DEBUG:
        print(msg)

# -------------------------------------------------------------------------
# Helper: extract MATLAB string from cell array element
# -------------------------------------------------------------------------
def extract_matlab_string(cell_item):
    """Extract string from MATLAB cell array element"""
    if isinstance(cell_item, np.ndarray):
        if cell_item.size == 0:
            return ""
        if cell_item.dtype.kind in ['U', 'S']:
            return str(cell_item.flat[0])
        elif cell_item.dtype == object:
            return extract_matlab_string(cell_item.flat[0])
        else:
            return str(cell_item.flat[0])
    else:
        return str(cell_item)

# -------------------------------------------------------------------------
# Helper: Find and load kidney_mask from ROI file
# -------------------------------------------------------------------------
def find_and_load_kidney_mask(sample_id, raw_data_dir='data/Raw_data'):
    """
    Find and load kidney_mask from corresponding ROI file based on sample_id.
    
    Args:
        sample_id: String like 'P89_PTONR_01_Image_1_rf' or similar
        raw_data_dir: Path to Raw_data directory
    
    Returns:
        kidney_mask: numpy array (2D binary mask) or None if not found
    """
    # Parse sample_id to extract patient info
    # Expected format: P<number>_<code>_<number>_Image_<number>_rf
    # Example: P89_PTONR_01_Image_1_rf
    
    # Try different patterns to match sample_id
    patterns = [
        r'P(\d+)_([A-Z]+)_(\d+)_Image_(\d+)_rf',  # P89_PTONR_01_Image_1_rf
        r'P(\d+)_([A-Z]+)_(\d+)_Image_(\d+)',     # P89_PTONR_01_Image_1
        r'P(\d+).*Image_(\d+)',                   # Fallback: P89...Image_1
    ]
    
    patient_num = None
    image_num = None
    
    for pattern in patterns:
        match = re.search(pattern, sample_id)
        if match:
            patient_num = match.group(1)
            if len(match.groups()) >= 4:
                image_num = match.group(4)
            elif len(match.groups()) >= 2:
                image_num = match.group(2)
            break
    
    if patient_num is None or image_num is None:
        debug(f"⚠️ Could not parse sample_id: {sample_id}")
        return None
    
    # Construct possible ROI file paths
    # Format: P<num>_<code>_<num>_Image_<num>_rf_raw_kidney.mat
    # Search in: data/Raw_data/P<num>_*/P<num>_Image_<num>/ROIs/*rf_raw_kidney.mat
    
    # Search for patient directory
    patient_pattern = f"P{patient_num}_*"
    patient_dirs = glob.glob(os.path.join(raw_data_dir, patient_pattern))
    
    if not patient_dirs:
        debug(f"⚠️ Patient directory not found for P{patient_num}")
        return None
    
    # Search for ROI file in each patient directory
    for patient_dir in patient_dirs:
        # Look for Image directory
        image_pattern = f"P{patient_num}_Image_{image_num}"
        image_dirs = glob.glob(os.path.join(patient_dir, image_pattern))
        
        if not image_dirs:
            continue
        
        # Look for ROI file in ROIs subdirectory
        for image_dir in image_dirs:
            roi_dir = os.path.join(image_dir, 'ROIs')
            if not os.path.exists(roi_dir):
                continue
            
            # Search for rf_raw_kidney.mat file
            roi_pattern = os.path.join(roi_dir, '*rf_raw_kidney.mat')
            roi_files = glob.glob(roi_pattern)
            
            if roi_files:
                # Try to load kidney_mask from first matching file
                for roi_file in roi_files:
                    try:
                        debug(f"Loading kidney_mask from: {roi_file}")
                        roi_data = loadmat(roi_file, struct_as_record=False, squeeze_me=True)
                        
                        if 'kidney_mask' in roi_data:
                            kidney_mask = roi_data['kidney_mask']
                            # Ensure it's a 2D array
                            if isinstance(kidney_mask, np.ndarray):
                                if kidney_mask.ndim > 2:
                                    kidney_mask = kidney_mask.squeeze()
                                debug(f"✅ Loaded kidney_mask with shape {kidney_mask.shape}")
                                return kidney_mask
                        else:
                            debug(f"⚠️ 'kidney_mask' not found in {roi_file}")
                    except Exception as e:
                        debug(f"⚠️ Error loading {roi_file}: {e}")
                        continue
    
    debug(f"⚠️ Could not find ROI file for sample_id: {sample_id}")
    return None

# -------------------------------------------------------------------------
# File paths
# -------------------------------------------------------------------------
data_dir = 'data/QUS_combined'
maps_file = os.path.join(data_dir, 'maps_combined.mat')
sample_id_file = os.path.join(data_dir, 'sample_id_combined.mat')
egfr_csv_file = 'csv/patient_eGFR_at_pocus_2025_Jul_polynomial_estimation.csv'
raw_data_dir = 'data/Raw_data'

assert os.path.exists(maps_file), f"Error: maps_combined.mat not found at {maps_file}"
assert os.path.exists(sample_id_file), f"Error: sample_id_combined.mat not found at {sample_id_file}"
assert os.path.exists(egfr_csv_file), f"Error: eGFR CSV not found at {egfr_csv_file}"

# -------------------------------------------------------------------------
# Load maps_combined.mat (v7.3 / HDF5)
# -------------------------------------------------------------------------
debug("Loading maps_combined.mat (v7.3 HDF5 format)...")

with h5py.File(maps_file, 'r') as f:
    debug(f"Top-level keys in maps_combined.mat: {list(f.keys())}")

    def list_all_datasets(g, prefix=''):
        """Recursively list all datasets in the HDF5 structure."""
        for k, v in g.items():
            path = f"{prefix}/{k}" if prefix else k
            if isinstance(v, h5py.Dataset):
                debug(f"Dataset: {path} | shape={v.shape} | dtype={v.dtype}")
            elif isinstance(v, h5py.Group):
                list_all_datasets(v, path)

    debug("\n--- Listing all datasets in maps_combined.mat ---")
    list_all_datasets(f)
    debug("--- End of dataset list ---\n")

    # Try to automatically pick the first dataset
    def find_first_dataset(h5_obj):
        for key in h5_obj.keys():
            item = h5_obj[key]
            if isinstance(item, h5py.Dataset):
                return key, item
            elif isinstance(item, h5py.Group):
                result = find_first_dataset(item)
                if result is not None:
                    return f"{key}/{result[0]}", result[1]
        return None

    result = find_first_dataset(f)
    if "all_parametric_maps_combined" in f:
        dataset = f["all_parametric_maps_combined"]
        maps_var = dataset[()]
        maps_var = np.transpose(maps_var, (3, 2, 1, 0))
        print(f"✅ Transposed maps_var to {maps_var.shape} (expected: (2928, 192, 6, 550))")

        print(f"✅ Loaded 'all_parametric_maps_combined' with shape {maps_var.shape} and dtype {maps_var.dtype}")
    else:
        raise ValueError("Could not find 'all_parametric_maps_combined' in maps_combined.mat")


debug(f"maps_var.ndim = {maps_var.ndim}")
debug(f"maps_var.shape = {maps_var.shape}")

# -------------------------------------------------------------------------
# Load sample_id_combined.mat
# -------------------------------------------------------------------------
debug("\nLoading sample_id_combined.mat...")
sample_id_data = loadmat(sample_id_file, struct_as_record=False, squeeze_me=True)
sample_id_keys = [k for k in sample_id_data.keys() if not k.startswith('__')]
debug(f"Keys in sample_id_combined.mat: {sample_id_keys}")

if len(sample_id_keys) == 0:
    raise ValueError("No data found in sample_id_combined.mat")

for k in sample_id_keys:
    v = sample_id_data[k]
    debug(f"  {k}: type={type(v)}, shape={getattr(v, 'shape', None)}, dtype={getattr(v, 'dtype', None)}")

sample_ids_var = sample_id_data[sample_id_keys[0]]

# -------------------------------------------------------------------------
# Handle sample IDs
# -------------------------------------------------------------------------
if isinstance(sample_ids_var, np.ndarray) and sample_ids_var.dtype == object:
    sample_ids = []
    for i in range(sample_ids_var.shape[0] if sample_ids_var.ndim > 0 else 1):
        item = sample_ids_var[i] if sample_ids_var.ndim == 1 else sample_ids_var[i, 0]
        sample_ids.append(extract_matlab_string(item))
    debug(f"Extracted {len(sample_ids)} sample IDs from cell array")
elif isinstance(sample_ids_var, np.ndarray) and sample_ids_var.dtype.kind in ['U', 'S']:
    sample_ids = [str(x) for x in sample_ids_var]
    debug(f"Extracted {len(sample_ids)} sample IDs from string array")
elif isinstance(sample_ids_var, (list, tuple)):
    sample_ids = [str(s) for s in sample_ids_var]
else:
    sample_ids = [str(sample_ids_var)]

debug(f"Sample IDs (first 5): {sample_ids[:5]}")

# -------------------------------------------------------------------------
# Convert maps_var to list of cases
# -------------------------------------------------------------------------
maps_list = []
if maps_var.ndim == 4:
    for i in range(maps_var.shape[-1]):
        maps_list.append(maps_var[..., i])
    debug(f"Split 4D array into {len(maps_list)} cases")
else:
    maps_list = [maps_var]
    debug("Non-4D data detected; treating as single case")

# -------------------------------------------------------------------------
# Match sample IDs
# -------------------------------------------------------------------------
if len(sample_ids) != len(maps_list):
    debug(f"⚠️ Warning: sample_ids length ({len(sample_ids)}) != maps_list length ({len(maps_list)})")
    if len(sample_ids) < len(maps_list):
        sample_ids.extend([f"Case_{i+1}" for i in range(len(sample_ids), len(maps_list))])
    else:
        sample_ids = sample_ids[:len(maps_list)]

debug(f"After adjustment: {len(sample_ids)} sample IDs for {len(maps_list)} cases")

# -------------------------------------------------------------------------
# Determine QUS parameters structure
# -------------------------------------------------------------------------
first_map = maps_list[0]
if first_map.ndim == 3:
    n_qus_params = first_map.shape[2]
    debug(f"Detected {n_qus_params} QUS parameters")
else:
    n_qus_params = 1
    debug(f"Warning: Unexpected map structure, assuming 1 QUS parameter")

# QUS parameter names
# Expected order: SAS, ESD, EAC, MBF, SS, SI
qus_names_list = ['SAS', 'ESD', 'EAC', 'MBF', 'SS', 'SI']
if n_qus_params == len(qus_names_list):
    qus_names = qus_names_list
else:
    # Fallback if number of parameters doesn't match
    qus_names = qus_names_list[:n_qus_params] if n_qus_params <= len(qus_names_list) else \
                qus_names_list + [f'QUS{i+1}' for i in range(len(qus_names_list), n_qus_params)]
    debug(f"Warning: Expected {len(qus_names_list)} QUS parameters, found {n_qus_params}. Using: {qus_names}")

# -------------------------------------------------------------------------
# Load eGFR data
# -------------------------------------------------------------------------
debug("\nLoading eGFR data from CSV...")
egfr_df = pd.read_csv(egfr_csv_file)
debug(f"Loaded {len(egfr_df)} eGFR records")
debug(f"Columns: {egfr_df.columns.tolist()}")

# Create dictionary mapping Patient ID to eGFR
egfr_dict = {}
for _, row in egfr_df.iterrows():
    patient_id = int(row['Patient ID'])
    egfr_value = row['eGFR (abs/closest)']
    if not pd.isna(egfr_value):
        egfr_dict[patient_id] = float(egfr_value)

debug(f"Created eGFR dictionary with {len(egfr_dict)} valid entries")

# -------------------------------------------------------------------------
# Save QUS maps as separate files (resized to 224x224)
# -------------------------------------------------------------------------
debug("\n" + "="*60)
debug("Saving QUS maps as separate matrix files (resized to 224x224)...")
debug("="*60)

target_size = (224, 224)
output_dir = 'data/QUS_resized'
os.makedirs(output_dir, exist_ok=True)

# Initialize storage for each QUS parameter: shape will be (224, 224, n_cases)
qus_matrices = {qus_name: [] for qus_name in qus_names}

for case_idx in range(len(maps_list)):
    case_map = maps_list[case_idx]
    
    for qus_idx in range(n_qus_params):
        map_data = case_map[..., qus_idx] if case_map.ndim == 3 else case_map
        qus_name = qus_names[qus_idx]
        
        # Resize to 224x224 using interpolation
        if map_data.shape[:2] != target_size:
            zoom_factors = (target_size[0] / map_data.shape[0],
                          target_size[1] / map_data.shape[1])
            map_resized = zoom(map_data, zoom_factors, order=1)
        else:
            map_resized = map_data
        
        # Ensure exactly 224x224
        if map_resized.shape[:2] != target_size:
            # If slight mismatch due to rounding, crop or pad
            h, w = map_resized.shape[:2]
            if h > target_size[0]:
                map_resized = map_resized[:target_size[0], :]
            if w > target_size[1]:
                map_resized = map_resized[:, :target_size[1]]
            if h < target_size[0] or w < target_size[1]:
                # Pad with zeros if needed
                pad_h = max(0, target_size[0] - h)
                pad_w = max(0, target_size[1] - w)
                map_resized = np.pad(map_resized, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        
        qus_matrices[qus_name].append(map_resized)
    
    if (case_idx + 1) % 50 == 0:
        debug(f"  Processed {case_idx + 1}/{len(maps_list)} cases...")

# Convert lists to 3D arrays and save
debug("\nSaving QUS matrices to files...")
for qus_name in qus_names:
    # Stack all cases: shape will be (224, 224, n_cases)
    qus_matrix = np.stack(qus_matrices[qus_name], axis=2)
    debug(f"  {qus_name}: shape {qus_matrix.shape}")
    
    # Save as .mat file (HDF5 v7.3 format using h5py)
    output_file = os.path.join(output_dir, f'{qus_name}.mat')
    with h5py.File(output_file, 'w') as f:
        f.create_dataset(qus_name, data=qus_matrix)
    debug(f"  ✅ Saved {qus_name} to {output_file}")
    
    # Also save as .npy for easy Python loading
    output_file_npy = os.path.join(output_dir, f'{qus_name}.npy')
    np.save(output_file_npy, qus_matrix)
    debug(f"  ✅ Saved {qus_name} to {output_file_npy}")

debug(f"\n✅ All QUS matrices saved to {output_dir}/")

# -------------------------------------------------------------------------
# Extract patient ID from sample_id
# -------------------------------------------------------------------------
def extract_patient_id(sample_id):
    """Extract patient ID (integer) from sample_id string"""
    match = re.search(r'P(\d+)', sample_id)
    if match:
        return int(match.group(1))
    return None

# -------------------------------------------------------------------------
# Calculate mean QUS values within ROI for all cases
# -------------------------------------------------------------------------
debug("\nCalculating mean QUS values within ROI for all cases...")

# Storage for results: list of dicts with patient_id, egfr, and mean QUS values
results = []

for case_idx in range(len(maps_list)):
    case_map = maps_list[case_idx]
    case_label = sample_ids[case_idx]
    
    # Extract patient ID
    patient_id = extract_patient_id(case_label)
    if patient_id is None:
        debug(f"⚠️ Could not extract patient ID from: {case_label}")
        continue
    
    # Get eGFR for this patient
    if patient_id not in egfr_dict:
        debug(f"⚠️ No eGFR found for patient {patient_id}")
        continue
    
    egfr_value = egfr_dict[patient_id]
    
    # Load kidney_mask for this case
    kidney_mask = find_and_load_kidney_mask(case_label, raw_data_dir)
    
    if kidney_mask is None:
        debug(f"⚠️ No kidney_mask found for case {case_idx+1} ({case_label})")
        continue
    
    # Calculate mean QUS values within ROI for each parameter
    mean_qus_values = []
    
    for qus_idx in range(n_qus_params):
        map_data = case_map[..., qus_idx] if case_map.ndim == 3 else case_map
        
        # Resize mask if needed
        if map_data.shape != kidney_mask.shape:
            if abs(map_data.shape[0] - kidney_mask.shape[0]) < 50 and \
               abs(map_data.shape[1] - kidney_mask.shape[1]) < 50:
                zoom_factors = (map_data.shape[0] / kidney_mask.shape[0],
                              map_data.shape[1] / kidney_mask.shape[1])
                kidney_mask_resized = zoom(kidney_mask.astype(float), zoom_factors, order=1)
                kidney_mask_resized = (kidney_mask_resized > 0.5).astype(int)
            else:
                debug(f"⚠️ Size mismatch too large for case {case_idx+1}, QUS {qus_idx+1}")
                mean_qus_values.append(np.nan)
                continue
        else:
            kidney_mask_resized = kidney_mask
        
        # Calculate mean within ROI (where mask == 1)
        roi_pixels = map_data[kidney_mask_resized == 1]
        if len(roi_pixels) > 0:
            # Filter out NaN and inf values
            valid_pixels = roi_pixels[np.isfinite(roi_pixels)]
            if len(valid_pixels) > 0:
                mean_value = np.mean(valid_pixels)
                mean_qus_values.append(mean_value)
            else:
                mean_qus_values.append(np.nan)
        else:
            mean_qus_values.append(np.nan)
    
    # Store results
    result_dict = {
        'patient_id': patient_id,
        'egfr': egfr_value,
        'case_label': case_label
    }
    for qus_idx, mean_val in enumerate(mean_qus_values):
        result_dict[qus_names[qus_idx]] = mean_val
    
    results.append(result_dict)
    
    if (case_idx + 1) % 50 == 0:
        debug(f"Processed {case_idx + 1}/{len(maps_list)} cases...")

debug(f"\n✅ Processed {len(results)} cases with valid ROI and eGFR data")

# -------------------------------------------------------------------------
# Convert to DataFrame for plotting
# -------------------------------------------------------------------------
debug("\nPreparing data for plotting...")

# Convert results to DataFrame for easier plotting
results_df = pd.DataFrame(results)
debug(f"Total cases in DataFrame: {len(results_df)}")

# -------------------------------------------------------------------------
# Create scatter plots: eGFR vs Mean QUS for each parameter
# -------------------------------------------------------------------------
debug("\nCreating scatter plots...")

# Create figure with subplots: 2 rows x 3 columns for 6 QUS parameters
n_cols = 3
n_rows = (n_qus_params + n_cols - 1) // n_cols  # Ceiling division

if n_qus_params == 1:
    fig, axes = plt.subplots(1, 1, figsize=(6, 5))
    axes = np.array([[axes]])
else:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

fig.suptitle('eGFR vs Mean QUS Values (within ROI)', fontsize=16, fontweight='bold')

for qus_idx, qus_name in enumerate(qus_names):
    row = qus_idx // n_cols
    col = qus_idx % n_cols
    ax = axes[row, col]
    
    # Extract data for this QUS parameter and filter out zero values
    data_subset = results_df[[qus_name, 'egfr']].copy()
    
    # Remove cases where this QUS mean value is zero
    initial_count = len(data_subset)
    valid_data = data_subset[
        (data_subset[qus_name] != 0) & 
        data_subset[qus_name].notna()
    ].dropna()
    
    removed_count = initial_count - len(valid_data)
    if removed_count > 0:
        debug(f"  {qus_name}: Removed {removed_count} cases with zero mean value")
    
    if len(valid_data) > 0:
        x = valid_data['egfr'].values
        y = valid_data[qus_name].values
        
        # Create scatter plot
        ax.scatter(x, y, alpha=0.6, s=50)
        ax.set_xlabel('eGFR (ml/min/1.73m²)', fontsize=11)
        ax.set_ylabel(f'Mean {qus_name} (within ROI)', fontsize=11)
        ax.set_title(f'{qus_name} vs eGFR\n(n={len(valid_data)})', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient if enough points
        if len(valid_data) > 2:
            corr = np.corrcoef(x, y)[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, f'No valid data\nfor {qus_name}',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{qus_name} vs eGFR', fontsize=12)

# Hide unused subplots
for qus_idx in range(n_qus_params, n_rows * n_cols):
    row = qus_idx // n_cols
    col = qus_idx % n_cols
    axes[row, col].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.93)

output_file = 'egfr_vs_qus_scatter_plots.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
debug(f"\n✅ Scatter plots saved as '{output_file}'")

# Print summary statistics
debug("\n" + "="*60)
debug("Summary Statistics:")
debug("="*60)
for qus_name in qus_names:
    # Filter out zero values and NaN for statistics
    valid_data = results_df[
        (results_df[qus_name] != 0) & 
        results_df[qus_name].notna()
    ][[qus_name, 'egfr']].dropna()
    
    if len(valid_data) > 0:
        debug(f"\n{qus_name}:")
        debug(f"  Valid cases: {len(valid_data)}")
        debug(f"  Mean QUS range: [{np.min(valid_data[qus_name]):.4f}, {np.max(valid_data[qus_name]):.4f}]")
        debug(f"  eGFR range: [{np.min(valid_data['egfr']):.2f}, {np.max(valid_data['egfr']):.2f}]")
        if len(valid_data) > 2:
            corr = np.corrcoef(valid_data['egfr'], valid_data[qus_name])[0, 1]
            debug(f"  Correlation with eGFR: {corr:.4f}")

plt.show()

# -------------------------------------------------------------------------
# Create pixel-based histograms for two eGFR groups (< 60 and >= 60)
# -------------------------------------------------------------------------
debug("\n" + "="*60)
debug("Creating pixel-based histograms for eGFR groups (< 60 vs >= 60)...")
debug("="*60)

# Define eGFR threshold
egfr_threshold = 60

# Collect pixel value statistics for histogram bins (memory-efficient approach)
debug("Collecting pixel values from ROI for all cases (memory-efficient)...")

# First pass: determine min/max for each QUS parameter to set bin ranges
debug("  First pass: determining value ranges...")
qus_minmax = {qus_name: {'min': np.inf, 'max': -np.inf, 'low_count': 0, 'high_count': 0} 
              for qus_name in qus_names}

for case_idx in range(len(maps_list)):
    case_map = maps_list[case_idx]
    case_label = sample_ids[case_idx]
    
    patient_id = extract_patient_id(case_label)
    if patient_id is None or patient_id not in egfr_dict:
        continue
    
    egfr_value = egfr_dict[patient_id]
    kidney_mask = find_and_load_kidney_mask(case_label, raw_data_dir)
    if kidney_mask is None:
        continue
    
    for qus_idx in range(n_qus_params):
        map_data = case_map[..., qus_idx] if case_map.ndim == 3 else case_map
        qus_name = qus_names[qus_idx]
        
        # Resize mask if needed
        if map_data.shape != kidney_mask.shape:
            if abs(map_data.shape[0] - kidney_mask.shape[0]) < 50 and \
               abs(map_data.shape[1] - kidney_mask.shape[1]) < 50:
                zoom_factors = (map_data.shape[0] / kidney_mask.shape[0],
                              map_data.shape[1] / kidney_mask.shape[1])
                kidney_mask_resized = zoom(kidney_mask.astype(float), zoom_factors, order=1)
                kidney_mask_resized = (kidney_mask_resized > 0.5).astype(int)
            else:
                continue
        else:
            kidney_mask_resized = kidney_mask
        
        # Extract pixel values within ROI
        roi_pixels = map_data[kidney_mask_resized == 1]
        if len(roi_pixels) > 0:
            valid_pixels = roi_pixels[(np.isfinite(roi_pixels)) & (roi_pixels != 0)]
            if len(valid_pixels) > 0:
                qus_minmax[qus_name]['min'] = min(qus_minmax[qus_name]['min'], np.min(valid_pixels))
                qus_minmax[qus_name]['max'] = max(qus_minmax[qus_name]['max'], np.max(valid_pixels))
                if egfr_value < egfr_threshold:
                    qus_minmax[qus_name]['low_count'] += len(valid_pixels)
                else:
                    qus_minmax[qus_name]['high_count'] += len(valid_pixels)
    
    if (case_idx + 1) % 50 == 0:
        debug(f"  Processed {case_idx + 1}/{len(maps_list)} cases...")

# Define bins for each QUS parameter
qus_bins = {}
for qus_name in qus_names:
    if qus_minmax[qus_name]['min'] < np.inf:
        qus_bins[qus_name] = np.linspace(qus_minmax[qus_name]['min'], 
                                         qus_minmax[qus_name]['max'], 30)
    else:
        qus_bins[qus_name] = np.linspace(0, 1, 30)

# Second pass: compute histograms incrementally (memory-efficient)
debug("  Second pass: computing histograms...")
histograms = {qus_name: {'low': np.zeros(29), 'high': np.zeros(29)} for qus_name in qus_names}

for case_idx in range(len(maps_list)):
    case_map = maps_list[case_idx]
    case_label = sample_ids[case_idx]
    
    patient_id = extract_patient_id(case_label)
    if patient_id is None or patient_id not in egfr_dict:
        continue
    
    egfr_value = egfr_dict[patient_id]
    group = 'low' if egfr_value < egfr_threshold else 'high'
    
    kidney_mask = find_and_load_kidney_mask(case_label, raw_data_dir)
    if kidney_mask is None:
        continue
    
    for qus_idx in range(n_qus_params):
        map_data = case_map[..., qus_idx] if case_map.ndim == 3 else case_map
        qus_name = qus_names[qus_idx]
        
        # Resize mask if needed
        if map_data.shape != kidney_mask.shape:
            if abs(map_data.shape[0] - kidney_mask.shape[0]) < 50 and \
               abs(map_data.shape[1] - kidney_mask.shape[1]) < 50:
                zoom_factors = (map_data.shape[0] / kidney_mask.shape[0],
                              map_data.shape[1] / kidney_mask.shape[1])
                kidney_mask_resized = zoom(kidney_mask.astype(float), zoom_factors, order=1)
                kidney_mask_resized = (kidney_mask_resized > 0.5).astype(int)
            else:
                continue
        else:
            kidney_mask_resized = kidney_mask
        
        # Extract pixel values and compute histogram incrementally
        roi_pixels = map_data[kidney_mask_resized == 1]
        if len(roi_pixels) > 0:
            valid_pixels = roi_pixels[(np.isfinite(roi_pixels)) & (roi_pixels != 0)]
            if len(valid_pixels) > 0 and qus_name in qus_bins:
                hist, _ = np.histogram(valid_pixels, bins=qus_bins[qus_name])
                histograms[qus_name][group] += hist
    
    if (case_idx + 1) % 50 == 0:
        debug(f"  Processed {case_idx + 1}/{len(maps_list)} cases...")

# Create figure with subplots for histograms
fig_hist, axes_hist = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
fig_hist.suptitle(f'QUS Pixel Value Distributions: eGFR < {egfr_threshold} vs eGFR >= {egfr_threshold}', 
                  fontsize=16, fontweight='bold')

if n_rows == 1:
    axes_hist = axes_hist.reshape(1, -1) if n_qus_params > 1 else np.array([[axes_hist]])
elif n_qus_params == 1:
    axes_hist = np.array([[axes_hist]])

for qus_idx, qus_name in enumerate(qus_names):
    row = qus_idx // n_cols
    col = qus_idx % n_cols
    ax = axes_hist[row, col]
    
    # Get histogram counts and pixel counts for both groups
    hist_low = histograms[qus_name]['low']
    hist_high = histograms[qus_name]['high']
    total_low = qus_minmax[qus_name]['low_count']
    total_high = qus_minmax[qus_name]['high_count']
    
    if total_low == 0 and total_high == 0:
        ax.text(0.5, 0.5, f'No valid pixel data\nfor {qus_name}',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{qus_name}', fontsize=12)
        continue
    
    bins = qus_bins[qus_name]
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_widths = np.diff(bins)
    
    # Normalize histograms by total pixel count
    if total_low > 0:
        hist_low_norm = hist_low / total_low
        ax.bar(bin_centers, hist_low_norm, width=bin_widths, alpha=0.6, 
               label=f'eGFR < {egfr_threshold} (n={total_low} pixels)', 
               color='red', edgecolor='black', linewidth=0.5)
    
    if total_high > 0:
        hist_high_norm = hist_high / total_high
        ax.bar(bin_centers, hist_high_norm, width=bin_widths, alpha=0.6, 
               label=f'eGFR >= {egfr_threshold} (n={total_high} pixels)', 
               color='blue', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel(f'{qus_name} Pixel Values (within ROI)', fontsize=11)
    ax.set_ylabel('Proportion (normalized by group size)', fontsize=11)
    ax.set_title(f'{qus_name}\nPixel Distribution by eGFR Group', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Calculate approximate statistics from histogram (for display)
    # Note: This is approximate since we don't store raw values
    stats_text = []
    if total_low > 0:
        # Approximate mean from histogram
        mean_approx_low = np.average(bin_centers, weights=hist_low)
        stats_text.append(f'< {egfr_threshold}: μ≈{mean_approx_low:.3f}, n={total_low}')
    if total_high > 0:
        mean_approx_high = np.average(bin_centers, weights=hist_high)
        stats_text.append(f'>= {egfr_threshold}: μ≈{mean_approx_high:.3f}, n={total_high}')
    
    if stats_text:
        ax.text(0.02, 0.98, '\n'.join(stats_text), transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Hide unused subplots
for qus_idx in range(n_qus_params, n_rows * n_cols):
    row = qus_idx // n_cols
    col = qus_idx % n_cols
    axes_hist[row, col].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.93)

output_file_hist = 'qus_histograms_egfr_groups.png'
plt.savefig(output_file_hist, dpi=150, bbox_inches='tight')
debug(f"\n✅ Histograms saved as '{output_file_hist}'")

# Print group statistics (pixel-based, from histograms)
debug("\n" + "="*60)
debug(f"Pixel-based Group Statistics (eGFR < {egfr_threshold} vs >= {egfr_threshold}):")
debug("="*60)
for qus_name in qus_names:
    hist_low = histograms[qus_name]['low']
    hist_high = histograms[qus_name]['high']
    total_low = qus_minmax[qus_name]['low_count']
    total_high = qus_minmax[qus_name]['high_count']
    
    debug(f"\n{qus_name}:")
    if total_low > 0:
        bins = qus_bins[qus_name]
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mean_approx_low = np.average(bin_centers, weights=hist_low)
        debug(f"  eGFR < {egfr_threshold}: n={total_low} pixels, mean≈{mean_approx_low:.4f}")
    if total_high > 0:
        bins = qus_bins[qus_name]
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mean_approx_high = np.average(bin_centers, weights=hist_high)
        debug(f"  eGFR >= {egfr_threshold}: n={total_high} pixels, mean≈{mean_approx_high:.4f}")

plt.show()

debug("\n✅ Analysis complete!")