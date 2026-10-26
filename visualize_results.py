import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from fno3d_model import FNO3D_Radar

def generate_paper_figures():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = r"checkpoints/best_fno3d_isro.pth"
    test_split_path = r"checkpoints/test_dataset_split.pt"
    output_dir = r"figures"
    os.makedirs(output_dir, exist_ok=True)

    test_ds = torch.load(test_split_path, weights_only=False)
    
    # Model initialized with Option A shape matching the 78.65M checkpoint
    model = FNO3D_Radar(in_steps=2, out_steps=2, width=32, modes1=12, modes2=20, modes3=20).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    selected_x, selected_y, selected_pred = None, None, None
    with torch.no_grad():
        for i in range(len(test_ds)):
            x, y = test_ds[i]
            x_in = x.unsqueeze(0).to(device)
            pred = model(x_in).squeeze(0).cpu()

            target_dbz = (y[0, 0] * 105.0) - 30.0
            if target_dbz.max() > 30.0:
                selected_x, selected_y, selected_pred = x, y, pred
                break

    in_dbz_t0 = (selected_x[1, 0] * 105.0) - 30.0
    tgt_dbz_t1 = (selected_y[0, 0] * 105.0) - 30.0
    tgt_dbz_t2 = (selected_y[1, 0] * 105.0) - 30.0
    
    prd_dbz_t1 = (selected_pred[0, 0] * 105.0) - 30.0
    prd_dbz_t2 = (selected_pred[1, 0] * 105.0) - 30.0

    prd_dbz_t1 = torch.where(prd_dbz_t1 < 2.0, -30.0, prd_dbz_t1)
    prd_dbz_t2 = torch.where(prd_dbz_t2 < 2.0, -30.0, prd_dbz_t2)

    proj_in_t0 = torch.max(in_dbz_t0, dim=0)[0].numpy()
    proj_tgt_t1 = torch.max(tgt_dbz_t1, dim=0)[0].numpy()
    proj_tgt_t2 = torch.max(tgt_dbz_t2, dim=0)[0].numpy()
    proj_prd_t1 = torch.max(prd_dbz_t1, dim=0)[0].numpy()
    proj_prd_t2 = torch.max(prd_dbz_t2, dim=0)[0].numpy()

    active_mask = (proj_tgt_t2 > 5.0) | (proj_prd_t2 > 5.0)
    err_t2 = np.where(active_mask, np.abs(proj_tgt_t2 - proj_prd_t2), 0.0)

    # Figure 1: 2D Spatial Projections
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300)
    cmap = 'jet'
    vmin, vmax = -10, 50

    im0 = axes[0, 0].imshow(proj_in_t0, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("(a) Input Observation ($t_0$)", fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(proj_tgt_t1, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("(b) Ground Truth ($t+15$ min)", fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(proj_tgt_t2, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 2].set_title("(c) Ground Truth ($t+30$ min)", fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(proj_prd_t1, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("(d) Sharp 3D-FNO Forecast ($t+15$ min)", fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(proj_prd_t2, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("(e) Sharp 3D-FNO Forecast ($t+30$ min)", fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')

    im_err = axes[1, 2].imshow(err_t2, cmap='magma', vmin=0, vmax=12)
    axes[1, 2].set_title("(f) Localized Error ($t+30$ min)", fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')

    cbar_ax1 = fig.add_axes([0.15, 0.05, 0.4, 0.02])
    fig.colorbar(im0, cax=cbar_ax1, orientation='horizontal', label='Reflectivity $Z_e$ (dBZ)')

    cbar_ax2 = fig.add_axes([0.65, 0.05, 0.25, 0.02])
    fig.colorbar(im_err, cax=cbar_ax2, orientation='horizontal', label='Absolute Error (dBZ)')

    plt.suptitle("ISRO MOSDAC 3D Volumetric Radar Echo Extrapolation", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "fig1_reflectivity_comparison.png"), bbox_inches='tight')
    plt.close()

    # Figure 2: Vertical Profile
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    mid_lat = np.argmax(tgt_dbz_t1.sum(dim=(0, 2)).numpy())
    
    v_tgt = tgt_dbz_t1[:, mid_lat, :].numpy()
    v_prd = prd_dbz_t1[:, mid_lat, :].numpy()
    v_prd = np.where(v_prd < 2.0, -30.0, v_prd)
    
    v_active = (v_tgt > 5.0) | (v_prd > 5.0)
    v_err = np.where(v_active, np.abs(v_tgt - v_prd), 0.0)

    im1 = axes[0].imshow(v_tgt, cmap='jet', vmin=-10, vmax=50, origin='lower', aspect='auto', extent=[0, 128, 0, 16])
    axes[0].set_title("Ground Truth Vertical Slice", fontweight='bold')
    axes[0].set_xlabel("Longitude Index (W)")
    axes[0].set_ylabel("Altitude (km)")

    axes[1].imshow(v_prd, cmap='jet', vmin=-10, vmax=50, origin='lower', aspect='auto', extent=[0, 128, 0, 16])
    axes[1].set_title("Sharp 3D-FNO Predicted Vertical Slice", fontweight='bold')
    axes[1].set_xlabel("Longitude Index (W)")
    axes[1].set_ylabel("Altitude (km)")

    im2 = axes[2].imshow(v_err, cmap='magma', vmin=0, vmax=10, origin='lower', aspect='auto', extent=[0, 128, 0, 16])
    axes[2].set_title("Vertical Error Profile (dBZ)", fontweight='bold')
    axes[2].set_xlabel("Longitude Index (W)")
    axes[2].set_ylabel("Altitude (km)")

    fig.colorbar(im1, ax=axes[1], orientation='vertical', label='Reflectivity $Z_e$ (dBZ)')
    fig.colorbar(im2, ax=axes[2], orientation='vertical', label='Error (dBZ)')

    plt.suptitle("Vertical Altitude Structure Preservation (0-16 km Altitude)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_vertical_profile.png"), bbox_inches='tight')
    plt.close()

    print("Regenerated clean, publication-grade figures!")

if __name__ == "__main__":
    generate_paper_figures()