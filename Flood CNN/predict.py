import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import FloodCNN

# -------- CONFIG --------
MODEL_PATH = "checkpoints/flood_cnn_latest.pth"   # your saved checkpoint
VV_PATH = "Predicting input images/florence_20180510t231343_x-23_y-9_vv.png"
VH_PATH = "Predicting input images/florence_20180510t231343_x-23_y-9_vh.png"

THRESHOLD = 0.5
FLOOD_PIXEL_RATIO = 0.01   # 1% pixels rule

# -------- LOAD MODEL --------
model = FloodCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# -------- LOAD IMAGES --------
vv = cv2.imread(VV_PATH, 0) / 255.0
vh = cv2.imread(VH_PATH, 0) / 255.0

image = torch.tensor([vv, vh], dtype=torch.float32).unsqueeze(0)

# -------- PREDICTION --------
with torch.no_grad():
    pred = model(image)[0][0].numpy()

# -------- CONVERT TO BINARY MASK --------
binary_mask = (pred > THRESHOLD).astype(np.uint8)

# -------- FLOOD DECISION --------
flood_pixels = np.sum(binary_mask)
total_pixels = binary_mask.size

ratio = flood_pixels / total_pixels

if ratio > FLOOD_PIXEL_RATIO:
    result = "FLOODED"
else:
    result = "NON-FLOODED"

# -------- PRINT RESULT --------
print(f"Flood Pixel Ratio: {ratio:.4f}")
print(f"Final Prediction: {result}")

# -------- VISUALIZATION --------
plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.title("VV")
plt.imshow(vv, cmap='gray')

plt.subplot(1,3,2)
plt.title("Prediction Mask")
plt.imshow(binary_mask, cmap='gray')

plt.subplot(1,3,3)
plt.title(f"Result: {result}")
plt.imshow(pred, cmap='gray')

plt.show()