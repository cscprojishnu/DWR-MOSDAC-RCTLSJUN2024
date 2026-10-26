import os
import glob
import numpy as np
import netCDF4 as nc
from tqdm import tqdm

def preprocess_nc_to_npy(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    nc_files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
    print(f"Converting {len(nc_files)} NetCDF files to uncompressed .npy format...")

    for fp in tqdm(nc_files, desc="Preprocessing Volumetric Radar Files"):
        fname = os.path.basename(fp).replace('.nc', '.npy')
        out_path = os.path.join(output_dir, fname)

        if os.path.exists(out_path):
            continue

        try:
            ds = nc.Dataset(fp, 'r')
            # Extract first 64 vertical levels directly
            dbz = np.asarray(ds.variables['DBZ'][0, :64, :, :], dtype=np.float32)
            vel = np.asarray(ds.variables['VEL'][0, :64, :, :], dtype=np.float32)
            ds.close()

            # Clean NaNs and invalid fill values
            dbz = np.nan_to_num(dbz, nan=-30.0)
            np.clip(dbz, -30.0, 75.0, out=dbz)
            vel = np.nan_to_num(vel, nan=0.0)

            # Standardized Normalization
            dbz_norm = (dbz + 30.0) / 105.0
            vel_norm = np.clip(vel / 100.0, -1.0, 1.0)

            # Stack to tensor [Channels=2, Z=64, H=481, W=481]
            stacked = np.stack([dbz_norm, vel_norm], axis=0)
            np.save(out_path, stacked)
        except Exception as e:
            print(f"Error processing {fp}: {e}")

    print("\n[Complete] Preprocessing finished! All files converted.")

if __name__ == "__main__":
    raw_dir = r"C:\Users\cscpr\Downloads\Radar\Dataset"
    prep_dir = r"C:\Users\cscpr\Downloads\Radar\Preprocessed"
    preprocess_nc_to_npy(raw_dir, prep_dir)