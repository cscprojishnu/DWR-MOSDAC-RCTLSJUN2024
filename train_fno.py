import os
import glob
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from dataset_loader import ISROVolumetricRadarDataset
from fno3d_model import FNO3D_Radar

def physics_informed_loss(pred, target, alpha_vel=0.5, lambda_phys=0.01):
    dbz_pred, dbz_target = pred[:, :, 0], target[:, :, 0]
    vel_pred, vel_target = pred[:, :, 1], target[:, :, 1]

    # Focal Loss Weight: Heavily penalize errors on actual echo voxels (> 0.2 norm ~ 10 dBZ)
    weight_mask = torch.where(dbz_target > 0.20, 20.0, 1.0)
    dbz_loss = torch.mean(weight_mask * torch.abs(dbz_pred - dbz_target))

    vel_loss = torch.mean(torch.abs(vel_pred - vel_target))

    dv_dz = torch.abs(vel_pred[:, :, 1:] - vel_pred[:, :, :-1])
    phys_loss = torch.mean(dv_dz)

    total_loss = dbz_loss + (alpha_vel * vel_loss) + (lambda_phys * phys_loss)
    return total_loss, dbz_loss.item(), vel_loss.item()

def calculate_csi(pred_dbz, target_dbz, threshold_dbz=10.0):
    p_dbz = (pred_dbz * 105.0) - 30.0
    t_dbz = (target_dbz * 105.0) - 30.0

    hits = torch.sum((p_dbz >= threshold_dbz) & (t_dbz >= threshold_dbz)).item()
    misses = torch.sum((p_dbz < threshold_dbz) & (t_dbz >= threshold_dbz)).item()
    false_alarms = torch.sum((p_dbz >= threshold_dbz) & (t_dbz < threshold_dbz)).item()

    denominator = hits + misses + false_alarms
    return hits / denominator if denominator > 0 else 0.0

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=========================================================")
    print(f"  Training 3D Fourier Neural Operator (FNO-3D)")
    print(f"  Target Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"=========================================================")

    prep_dir = r"C:\Users\cscpr\Downloads\Radar\Preprocessed"

    BATCH_SIZE = 2
    EPOCHS = 15
    LR = 2e-3
    CROP_SHAPE = (64, 128, 128)

    dataset = ISROVolumetricRadarDataset(prep_dir, in_steps=2, out_steps=2, crop_shape=CROP_SHAPE)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FNO3D_Radar(in_steps=2, out_steps=2, width=24, modes1=8, modes2=12, modes3=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\nModel Initialized: {sum(p.numel() for p in model.parameters())/1e6:.2f} Million Parameters.\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        start_time = time.time()
        train_loss, train_dbz, train_vel = 0.0, 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS:02d}", unit="batch")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred = model(x)
            loss, dbz_l, vel_l = physics_informed_loss(pred, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_dbz += dbz_l
            train_vel += vel_l

            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'DBZ': f"{dbz_l:.4f}"})

        scheduler.step()
        train_loss /= len(train_loader)
        train_dbz /= len(train_loader)
        train_vel /= len(train_loader)

        # Validation Loop evaluating 10, 20, and 30 dBZ thresholds
        model.eval()
        val_csi10, val_csi20, val_csi30 = 0.0, 0.0, 0.0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                pred_v = model(x_v)
                val_csi10 += calculate_csi(pred_v[:, :, 0], y_v[:, :, 0], threshold_dbz=10.0)
                val_csi20 += calculate_csi(pred_v[:, :, 0], y_v[:, :, 0], threshold_dbz=20.0)
                val_csi30 += calculate_csi(pred_v[:, :, 0], y_v[:, :, 0], threshold_dbz=30.0)

        val_csi10 /= max(1, len(val_loader))
        val_csi20 /= max(1, len(val_loader))
        val_csi30 /= max(1, len(val_loader))
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] Summary | Time: {elapsed:.1f}s | "
              f"Loss: {train_loss:.4f} (DBZ: {train_dbz:.4f}, VEL: {train_vel:.4f}) | "
              f"CSI10: {val_csi10:.4f} | CSI20: {val_csi20:.4f} | CSI30: {val_csi30:.4f}\n")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/fno3d_isro_radar.pth")
    print("\nTraining Complete! Checkpoint saved to checkpoints/fno3d_isro_radar.pth")

if __name__ == "__main__":
    train()