import os
import glob
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class ISROVolumetricRadarDataset(Dataset):
    def __init__(self, prep_dir, in_steps=2, out_steps=2, max_gap_minutes=30.0, crop_shape=(64, 128, 128)):
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.total_seq_len = in_steps + out_steps
        self.crop_shape = crop_shape

        file_paths = sorted(glob.glob(os.path.join(prep_dir, "*.npy")))
        file_info = []

        for fp in file_paths:
            fname = os.path.basename(fp)
            try:
                parts = fname.split('_')
                dt = datetime.strptime(f"{parts[1]}_{parts[2]}", "%d%b%Y_%H%M%S")
                file_info.append({'path': fp, 'time': dt})
            except Exception:
                continue

        file_info.sort(key=lambda x: x['time'])

        sequence_files = []
        current_group = []
        for i in range(len(file_info)):
            if not current_group:
                current_group.append(file_info[i])
            else:
                gap = (file_info[i]['time'] - file_info[i-1]['time']).total_seconds() / 60.0
                if gap <= max_gap_minutes:
                    current_group.append(file_info[i])
                else:
                    if len(current_group) >= self.total_seq_len:
                        for s in range(len(current_group) - self.total_seq_len + 1):
                            sequence_files.append([f['path'] for f in current_group[s : s + self.total_seq_len]])
                    current_group = [file_info[i]]

        if len(current_group) >= self.total_seq_len:
            for s in range(len(current_group) - self.total_seq_len + 1):
                sequence_files.append([f['path'] for f in current_group[s : s + self.total_seq_len]])

        print(f"[Dataset] Pre-loading {len(sequence_files)} sequences into RAM for instant epoch speeds...")
        self.samples = []

        for seq in tqdm(sequence_files, desc="Caching Data in RAM"):
            seq_data = [np.load(fp) for fp in seq] # Load 4 frames
            seq_array = np.stack(seq_data, axis=0) # [4, 2, 64, 481, 481]

            # Find active echo centroid
            dbz_intensity = seq_array[:, 0].sum(axis=(0, 1))
            max_pos = np.unravel_index(np.argmax(dbz_intensity), dbz_intensity.shape)

            center_h, center_w = max_pos[0], max_pos[1]
            half_h, half_w = self.crop_shape[1] // 2, self.crop_shape[2] // 2

            min_h = max(0, min(481 - self.crop_shape[1], center_h - half_h))
            min_w = max(0, min(481 - self.crop_shape[2], center_w - half_w))

            cropped = seq_array[
                :, 
                :, 
                :self.crop_shape[0], 
                min_h:min_h + self.crop_shape[1], 
                min_w:min_w + self.crop_shape[2]
            ].astype(np.float32)

            tensor = torch.from_numpy(cropped)
            x = tensor[:self.in_steps]
            y = tensor[self.in_steps:]
            self.samples.append((x, y))

        print(f"[Dataset] Ready! {len(self.samples)} sequences cached successfully.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]