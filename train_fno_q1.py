import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from dataset_loader import ISROVolumetricRadarDataset
from fno3d_model import FNO3D_Radar

def spatial_gradient_3d(tensor):
    """Calculates spatial gradients (dH, dW) to enforce sharp storm boundaries."""
    dh = torch.abs(tensor[:, :, :, 1:, :] - tensor[:, :, :, :-1, :])
    dw = torch.abs(tensor[:, :, :, :, 1:] - tensor[:, :, :, :, :-1])
    return torch.mean(dh) + torch.mean(dw)

def sharp_physics_focal_loss(pred, target, alpha_vel=0.5, lambda_phys=0.01, lambda_grad=2.0):
    dbz_pred, dbz_target = pred[:, :, 0], target[:, :, 0]
    vel_pred, vel_target = pred[:, :, 1], target[:, :, 1]

    # Focal Loss: Heavy penalty (50x) on intense convective cores (> 30 dBZ / norm > 0.57)
    weight_mask = torch.where(dbz_target > 0.57, 50.0, torch.where(dbz_target > 0.25, 15.0, 1.0))
    dbz_loss = torch.mean(weight_mask * torch.abs(dbz_pred - dbz_target))

    # Gradient Edge Loss: Penalizes blurriness & forces sharp storm edges
    grad_loss = torch.abs(spatial_gradient_3d(dbz_pred) - spatial_gradient_3d(dbz_target))

    vel_loss = torch.mean(torch.abs(vel_pred - vel_target))

    # Vertical Continuity
    dv_dz = torch.abs(vel_pred[:, :, 1:] - vel_pred[:, :, :-1])
    phys_loss = torch.mean(dv_dz)

    total_loss = dbz_loss + (lambda_grad * grad_loss) + (alpha_vel * vel_loss) + (lambda_phys * phys_loss)
    return total_loss, dbz_loss.item(), grad_loss.item()

def calculate_csi(pred_dbz, target_dbz, threshold_dbz=20.0):
    p_dbz = (pred_dbz * 105.0) - 30.0
    t_dbz = (target_dbz * 105.0) - 30.0

    hits = torch.sum((p_dbz >= threshold_dbz) & (t_dbz >= threshold_dbz)).item()
    misses = torch.sum((p_dbz < threshold_dbz) & (t_dbz >= threshold_dbz)).item()
    false_alarms = torch.sum((p_dbz >= threshold_dbz) & (t_dbz < threshold_dbz)).item()

    denominator = hits + misses + false_alarms
    return hits / denominator if denominator > 0 else 0.0

def train():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"=========================================================")
    print(f"  Option A Fine-Tuning: Sharp 3D-FNO with Gradient Loss")
    print(f"  Target GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"=========================================================")

    prep_dir = r"C:\Users\cscpr\Downloads\Radar\Preprocessed"

    BATCH_SIZE = 2
    EPOCHS = 30 # Focused 30-epoch training run for fast convergence
    LR = 1e-3
    CROP_SHAPE = (64, 128, 128)

    dataset = ISROVolumetricRadarDataset(prep_dir, in_steps=2, out_steps=2, crop_shape=CROP_SHAPE)
    total_len = len(dataset)
    train_size = int(0.70 * total_len)
    val_size = int(0.15 * total_len)
    test_size = total_len - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(test_ds, "checkpoints/test_dataset_split.pt")

    # Expanded 3D-FNO Model with high Fourier mode counts
    model = FNO3D_Radar(in_steps=2, out_steps=2, width=32, modes1=12, modes2=20, modes3=20).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    print(f"\nModel Initialized: {sum(p.numel() for p in model.parameters())/1e6:.2f} Million Parameters.\n")

    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        start_time = time.time()
        train_loss, train_dbz, train_grad = 0.0, 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{EPOCHS:02d}]", unit="batch")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred = model(x)
            loss, dbz_l, grad_l = sharp_physics_focal_loss(pred, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_dbz += dbz_l
            train_grad += grad_l

            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'DBZ': f"{dbz_l:.4f}", 'Grad': f"{grad_l:.4f}"})

        scheduler.step()
        train_loss /= len(train_loader)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_csi10, val_csi20, val_csi30 = 0.0, 0.0, 0.0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                pred_v = model(x_v)
                v_l, _, _ = sharp_physics_focal_loss(pred_v, y_v)
                val_loss += v_l.item()

                val_csi10 += calculate_csi(pred_v[:, :, 0], y_v[:, :, 0], threshold_dbz=10.0)
                val_csi20 += calculate_csi(pred_v[:, :, 0], y_v[:, :, 0], threshold_dbz=20.0)
                val_csi30 += calculate_csi(pred_v[:, :, 0], y_v[:, :, 0], threshold_dbz=30.0)

        val_loss /= len(val_loader)
        val_csi10 /= max(1, len(val_loader))
        val_csi20 /= max(1, len(val_loader))
        val_csi30 /= max(1, len(val_loader))
        elapsed = time.time() - start_time

        saved_str = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_fno3d_isro.pth")
            saved_str = " [BEST MODEL SAVED]"

        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Time: {elapsed:.1f}s | "
              f"Tr Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"CSI10: {val_csi10:.4f} | CSI20: {val_csi20:.4f} | CSI30: {val_csi30:.4f}{saved_str}\n")

    print("\nSharp 3D-FNO Training Complete! Best model saved to checkpoints/best_fno3d_isro.pth")

if __name__ == "__main__":
    train()