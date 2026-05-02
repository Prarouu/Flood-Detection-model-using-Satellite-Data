import os
import cv2
import torch
from torch.utils.data import Dataset

class FloodDataset(Dataset):
    def __init__(self, root_dir, limit=None):
        self.samples = []
        
        for region in os.listdir(root_dir):
            region_path = os.path.join(root_dir, region, "tiles")
            
            vv_path = os.path.join(region_path, "vv")
            vh_path = os.path.join(region_path, "vh")
            label_path = os.path.join(region_path, "flood_label")
            
            if not os.path.exists(vv_path):
                continue
            
            for file in os.listdir(vv_path):
                label_file = file.replace("_vv", "")
                self.samples.append((
                    os.path.join(vv_path, file),
                    os.path.join(vh_path, file.replace("_vv", "_vh")),
                    os.path.join(label_path, label_file)
                ))
        
        # limit dataset for fast training
        if limit:
            self.samples = self.samples[:limit]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vv_path, vh_path, label_path = self.samples[idx]
        
        vv = cv2.imread(vv_path, 0) / 255.0
        vh = cv2.imread(vh_path, 0) / 255.0
        label = cv2.imread(label_path, 0) / 255.0
        
        image = torch.tensor([vv, vh], dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        
        return image, label
    