import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("best.pt")
val_dir = "drone_dataset/valid/images"
sample_img = [f for f in os.listdir(val_dir) if f.endswith(".jpg")][0]
img_path = os.path.join(val_dir, sample_img)

# Load and contrast-enhance
orig = cv2.imread(img_path)
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

# Inference
res_orig = model.predict(orig, conf=0.25, device="mps")[0].plot()
res_enh = model.predict(enhanced, conf=0.25, device="mps")[0].plot()

# Display comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(res_orig, cv2.COLOR_BGR2RGB))
axes[0].set_title("Standard YOLOv8x Detection")
axes[0].axis("off")
axes[1].imshow(cv2.cvtColor(res_enh, cv2.COLOR_BGR2RGB))
axes[1].set_title("CLAHE Contrast-Enhanced Detection")
axes[1].axis("off")
plt.tight_layout()
plt.savefig("detection_comparison.png")
plt.show()
