import os
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from dataset_loader import ISROVolumetricRadarDataset
from fno3d_model import FNO3D_Radar

def calculate_metrics_at_threshold(pred_dbz, target_dbz, threshold_dbz=20.0):
    # Denormalize [0, 1] -> [-30, 75] dBZ
    p_dbz = (pred_dbz * 105.0) - 30.0
    t_dbz = (target_dbz * 105.0) - 30.0

    hits = torch.sum((p_dbz >= threshold_dbz) & (t_dbz >= threshold_dbz)).item()
    misses = torch.sum((p_dbz < threshold_dbz) & (t_dbz >= threshold_dbz)).item()
    false_alarms = torch.sum((p_dbz >= threshold_dbz) & (t_dbz < threshold_dbz)).item()

    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0
    pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0

    return csi, pod, far

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prep_dir = r"C:\Users\cscpr\Downloads\Radar\Preprocessed"
    checkpoint_path = r"checkpoints/fno3d_isro_radar.pth"

    BATCH_SIZE = 2
    CROP_SHAPE = (64, 128, 128)

    dataset = ISROVolumetricRadarDataset(prep_dir, in_steps=2, out_steps=2, crop_shape=CROP_SHAPE)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FNO3D_Radar(in_steps=2, out_steps=2, width=24, modes1=8, modes2=12, modes3=12).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print("\n=========================================================================")
    print("  ISRO MOSDAC 3D FNO MODEL - FULL QUANTITATIVE EVALUATION REPORT")
    print("=========================================================================\n")

    thresholds = [10.0, 15.0, 20.0, 25.0, 30.0]
    metrics = {t: {'csi': [], 'pod': [], 'far': []} for t in thresholds}
    
    mae_dbz_list, rmse_dbz_list = [], []
    mae_vel_list, rmse_vel_list = [], []
    inference_times = []

    with torch.no_grad():
        for x_v, y_v in tqdm(val_loader, desc="Evaluating Test Sequences"):
            x_v, y_v = x_v.to(device), y_v.to(device)

            start = time.time()
            pred_v = model(x_v)
            torch.cuda.synchronize()
            inference_times.append((time.time() - start) / x_v.size(0))

            pred_dbz, target_dbz = pred_v[:, :, 0], y_v[:, :, 0]
            pred_vel, target_vel = pred_v[:, :, 1], y_v[:, :, 1]

            # Denormalize
            p_dbz_denorm = (pred_dbz * 105.0) - 30.0
            t_dbz_denorm = (target_dbz * 105.0) - 30.0
            p_vel_denorm = pred_vel * 100.0
            t_vel_denorm = target_vel * 100.0

            # Absolute & Root Squared Errors
            mae_dbz_list.append(torch.mean(torch.abs(p_dbz_denorm - t_dbz_denorm)).item())
            rmse_dbz_list.append(torch.sqrt(torch.mean((p_dbz_denorm - t_dbz_denorm) ** 2)).item())

            mae_vel_list.append(torch.mean(torch.abs(p_vel_denorm - t_vel_denorm)).item())
            rmse_vel_list.append(torch.sqrt(torch.mean((p_vel_denorm - t_vel_denorm) ** 2)).item())

            # Evaluate skills across thresholds
            for t in thresholds:
                c, p, f = calculate_metrics_at_threshold(pred_dbz, target_dbz, threshold_dbz=t)
                metrics[t]['csi'].append(c)
                metrics[t]['pod'].append(p)
                metrics[t]['far'].append(f)

    # Summary Display
    print("\n" + "="*75)
    print(" 1. METEOROLOGICAL FORECAST SKILL METRICS (VOLUMETRIC ECHO EXTRAPOLATION)")
    print("="*75)
    print(f"{'Threshold (dBZ)':<18} | {'CSI (Critical Success)':<22} | {'POD (Hit Rate)':<16} | {'FAR (False Alarm)':<16}")
    print("-" * 75)

    for t in thresholds:
        mean_csi = np.mean(metrics[t]['csi'])
        mean_pod = np.mean(metrics[t]['pod'])
        mean_far = np.mean(metrics[t]['far'])
        print(f"{t:<18.1f} | {mean_csi:<22.4f} | {mean_pod:<16.4f} | {mean_far:<16.4f}")

    print("-" * 75)

    print("\n" + "="*75)
    print(" 2. FIELD RECONSTRUCTION ACCURACY & INFERENCE BENCHMARK")
    print("="*75)
    print(f"  • Reflectivity Field MAE  : {np.mean(mae_dbz_list):.4f} dBZ")
    print(f"  • Reflectivity Field RMSE : {np.mean(rmse_dbz_list):.4f} dBZ")
    print(f"  • Radial Velocity MAE     : {np.mean(mae_vel_list):.4f} m/s")
    print(f"  • Radial Velocity RMSE    : {np.mean(rmse_vel_list):.4f} m/s")
    print(f"  • Single Scan Inference   : {np.mean(inference_times)*1000:.2f} ms (Target Edge GPU: RTX 4050)")
    print("="*75 + "\n")

if __name__ == "__main__":
    evaluate()