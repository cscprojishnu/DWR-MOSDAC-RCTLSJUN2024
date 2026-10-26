import os
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from fno3d_model import FNO3D_Radar

def calculate_metrics_at_threshold(pred_dbz, target_dbz, threshold_dbz=20.0):
    p_dbz = (pred_dbz * 105.0) - 30.0
    t_dbz = (target_dbz * 105.0) - 30.0

    hits = torch.sum((p_dbz >= threshold_dbz) & (t_dbz >= threshold_dbz)).item()
    misses = torch.sum((p_dbz < threshold_dbz) & (t_dbz >= threshold_dbz)).item()
    false_alarms = torch.sum((p_dbz >= threshold_dbz) & (t_dbz < threshold_dbz)).item()

    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0
    pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0

    return csi, pod, far

def evaluate_unseen_test_set():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = r"checkpoints/best_fno3d_isro.pth"
    test_split_path = r"checkpoints/test_dataset_split.pt"

    if not os.path.exists(test_split_path):
        print("Error: Test split not found! Run train_fno_q1.py first.")
        return

    test_ds = torch.load(test_split_path, weights_only=False)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)

    # Initialized with Option A parameters matching the 78.65M checkpoint
    model = FNO3D_Radar(in_steps=2, out_steps=2, width=32, modes1=12, modes2=20, modes3=20).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    print("\n=========================================================================")
    print("  ISRO MOSDAC 3D SHARP FNO - UNSEEN TEST SET FINAL REPORT (78.65M PARAMS)")
    print("=========================================================================\n")

    thresholds = [10.0, 15.0, 20.0, 25.0, 30.0]
    metrics = {t: {'csi': [], 'pod': [], 'far': []} for t in thresholds}
    
    mae_dbz_list, rmse_dbz_list = [], []
    mae_vel_list, rmse_vel_list = [], []
    inference_times = []

    with torch.no_grad():
        for x_v, y_v in tqdm(test_loader, desc="Testing Sharp Model on Unseen Scans"):
            x_v, y_v = x_v.to(device), y_v.to(device)

            start = time.time()
            pred_v = model(x_v)
            torch.cuda.synchronize()
            inference_times.append((time.time() - start) / x_v.size(0))

            pred_dbz, target_dbz = pred_v[:, :, 0], y_v[:, :, 0]
            pred_vel, target_vel = pred_v[:, :, 1], y_v[:, :, 1]

            p_dbz_denorm = (pred_dbz * 105.0) - 30.0
            t_dbz_denorm = (target_dbz * 105.0) - 30.0
            p_vel_denorm = pred_vel * 100.0
            t_vel_denorm = target_vel * 100.0

            mae_dbz_list.append(torch.mean(torch.abs(p_dbz_denorm - t_dbz_denorm)).item())
            rmse_dbz_list.append(torch.sqrt(torch.mean((p_dbz_denorm - t_dbz_denorm) ** 2)).item())

            mae_vel_list.append(torch.mean(torch.abs(p_vel_denorm - t_vel_denorm)).item())
            rmse_vel_list.append(torch.sqrt(torch.mean((p_vel_denorm - t_vel_denorm) ** 2)).item())

            for t in thresholds:
                c, p, f = calculate_metrics_at_threshold(pred_dbz, target_dbz, threshold_dbz=t)
                metrics[t]['csi'].append(c)
                metrics[t]['pod'].append(p)
                metrics[t]['far'].append(f)

    print("\n" + "="*75)
    print(" OFFICIAL PUBLICATION TABLE: SHARP 3D-FNO SKILL SCORES (TEST SET)")
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
    print(" OFFICIAL PUBLICATION TABLE: RECONSTRUCTION & HARDWARE BENCHMARK")
    print("="*75)
    print(f"  • Reflectivity Field MAE  : {np.mean(mae_dbz_list):.4f} dBZ")
    print(f"  • Reflectivity Field RMSE : {np.mean(rmse_dbz_list):.4f} dBZ")
    print(f"  • Radial Velocity MAE     : {np.mean(mae_vel_list):.4f} m/s")
    print(f"  • Radial Velocity RMSE    : {np.mean(rmse_vel_list):.4f} m/s")
    print(f"  • Edge Inference Speed    : {np.mean(inference_times)*1000:.2f} ms (NVIDIA RTX 4050 Laptop)")
    print("="*75 + "\n")

if __name__ == "__main__":
    evaluate_unseen_test_set()