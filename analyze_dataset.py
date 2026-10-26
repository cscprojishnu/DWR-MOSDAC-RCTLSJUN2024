import os
import sys
import glob
import argparse
from datetime import datetime
import numpy as np
import netCDF4 as nc
from tqdm import tqdm

def analyze_dataset(data_dir):
    print("=" * 80)
    print("  ISRO MOSDAC DWR (RCTLS_L2C_VOL) DATASET COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    # 1. Discover Files
    search_pattern = os.path.join(data_dir, "*.nc")
    file_paths = sorted(glob.glob(search_pattern))
    total_files = len(file_paths)
    
    if total_files == 0:
        print(f"Error: No '.nc' files found in directory: {data_dir}")
        sys.exit(1)
        
    print(f"\n[1/5] FILE SYSTEM SUMMARY")
    print(f"  • Total Files Found       : {total_files}")
    
    file_sizes = [os.path.getsize(f) / (1024 * 1024) for f in file_paths] # MB
    total_size_gb = sum(file_sizes) / 1024
    print(f"  • Total Dataset Size      : {total_size_gb:.2f} GB")
    print(f"  • Single File Size (Mean) : {np.mean(file_sizes):.2f} MB")
    print(f"  • Single File Size (Min)  : {np.min(file_sizes):.2f} MB")
    print(f"  • Single File Size (Max)  : {np.max(file_sizes):.2f} MB")

    # 2. Inspect First File Structure
    print(f"\n[2/5] METADATA & STRUCTURE INSPECTION (First File: {os.path.basename(file_paths[0])})")
    sample_ds = nc.Dataset(file_paths[0])
    
    print("\n  --- Global Attributes ---")
    for attr in sample_ds.ncattrs():
        print(f"    • {attr}: {getattr(sample_ds, attr)}")
        
    print("\n  --- Dimensions ---")
    for dim_name, dim in sample_ds.dimensions.items():
        print(f"    • {dim_name}: size = {len(dim)}")
        
    print("\n  --- Variables & Data Types ---")
    var_names = list(sample_ds.variables.keys())
    for var_name in var_names:
        var = sample_ds.variables[var_name]
        print(f"    • {var_name:15s} | Shape: {str(var.shape):20s} | Type: {str(var.dtype):7s} | Units: {getattr(var, 'units', 'N/A')}")
    sample_ds.close()

    # 3. Filename Timestamp Parse & Temporal Continuity Check
    print(f"\n[3/5] TEMPORAL CONTINUITY ANALYSIS")
    timestamps = []
    corrupted_files = []
    
    for filepath in file_paths:
        fname = os.path.basename(filepath)
        # Expected format: RCTLS_DDMMMYYYY_HHMMSS_L2C_STD.nc
        try:
            parts = fname.split('_')
            date_str = parts[1] + "_" + parts[2]
            dt = datetime.strptime(date_str, "%d%b%Y_%H%M%S")
            timestamps.append(dt)
        except Exception:
            pass

    if len(timestamps) > 1:
        timestamps.sort()
        start_time = timestamps[0]
        end_time = timestamps[-1]
        time_diffs = [ (timestamps[i+1] - timestamps[i]).total_seconds() / 60.0 for i in range(len(timestamps)-1) ]
        
        print(f"  • Temporal Range Start   : {start_time}")
        print(f"  • Temporal Range End     : {end_time}")
        print(f"  • Mean Interval Between  : {np.mean(time_diffs):.2f} minutes")
        print(f"  • Median Interval        : {np.median(time_diffs):.2f} minutes")
        print(f"  • Min Interval           : {np.min(time_diffs):.2f} minutes")
        print(f"  • Max Gap Detected       : {np.max(time_diffs):.2f} minutes ({np.max(time_diffs)/60.0:.2f} hours)")

    # 4. Statistical Batch Sampling Across All Files
    print(f"\n[4/5] FULL DATASET STATISTICAL AGGREGATION")
    print("  Processing all files to aggregate reflectivity (DBZ) and velocity (VEL) statistics...")

    # Accumulators
    stats = {}
    for var in var_names:
        stats[var] = {
            'min': float('inf'),
            'max': float('-inf'),
            'sum': 0.0,
            'count': 0,
            'nan_count': 0,
            'valid_count': 0
        }

    for filepath in tqdm(file_paths, desc="  Scanning NetCDF Volumes", unit="file"):
        try:
            ds = nc.Dataset(filepath)
            for var in var_names:
                data = ds.variables[var][:]
                
                # Check NaNs and Masked arrays
                if np.ma.is_masked(data):
                    unmasked_data = data.compressed()
                    nans = data.size - unmasked_data.size
                else:
                    nans = np.isnan(data).sum()
                    unmasked_data = data[~np.isnan(data)]
                
                # Exclude standard radar fill values (-999.0 or similar)
                valid_data = unmasked_data[unmasked_data > -900.0] if unmasked_data.size > 0 else np.array([])

                stats[var]['nan_count'] += int(nans) + (unmasked_data.size - valid_data.size)
                stats[var]['valid_count'] += int(valid_data.size)
                
                if valid_data.size > 0:
                    stats[var]['min'] = min(stats[var]['min'], float(np.min(valid_data)))
                    stats[var]['max'] = max(stats[var]['max'], float(np.max(valid_data)))
                    stats[var]['sum'] += float(np.sum(valid_data))
            ds.close()
        except Exception as e:
            corrupted_files.append((filepath, str(e)))

    # 5. Output Summary Report
    print(f"\n[5/5] STATISTICAL REPORT SUMMARY")
    print("-" * 80)
    print(f"{'Variable':<12} | {'Min Val':<10} | {'Max Val':<10} | {'Mean Val':<10} | {'Valid %':<10} | {'Fill/NaN %':<10}")
    print("-" * 80)
    
    for var in var_names:
        v_stats = stats[var]
        tot_cells = v_stats['valid_count'] + v_stats['nan_count']
        valid_pct = (v_stats['valid_count'] / tot_cells * 100) if tot_cells > 0 else 0
        nan_pct = (v_stats['nan_count'] / tot_cells * 100) if tot_cells > 0 else 0
        mean_val = (v_stats['sum'] / v_stats['valid_count']) if v_stats['valid_count'] > 0 else 0.0
        
        min_str = f"{v_stats['min']:.2f}" if v_stats['min'] != float('inf') else "N/A"
        max_str = f"{v_stats['max']:.2f}" if v_stats['max'] != float('-inf') else "N/A"
        
        print(f"{var:<12} | {min_str:<10} | {max_str:<10} | {mean_val:<10.2f} | {valid_pct:<9.2f}% | {nan_pct:<9.2f}%")

    print("-" * 80)
    print(f"\n  • Corrupted / Unreadable Files: {len(corrupted_files)}")
    if len(corrupted_files) > 0:
        for cf, err in corrupted_files[:5]:
            print(f"    - {os.path.basename(cf)}: {err}")
    print("\n=" * 80)
    print("  ANALYSIS COMPLETE. Copy and paste this text output in your next message!")
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_dataset.py /path/to/nc_files_folder")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    analyze_dataset(target_dir)