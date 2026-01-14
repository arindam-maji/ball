import torch
from ultralytics import YOLO
import coremltools as ct
import os

# ================= CONFIG =================
PT_PATH = r"best.pt"
TS_PATH = "best.torchscript.pt"  # intermediate TorchScript
MLMODEL_PATH = "paddle_fast_motion.mlmodel"

DEVICE = "cpu"
IMG_H, IMG_W = 960, 960  # match training image size

# ================= LOAD YOLO MODEL =================
print("Loading YOLO model...")
model = YOLO(PT_PATH)

# ================= TORCHSCRIPT EXPORT =================
print("Exporting to TorchScript...")
ts_model = model.model
example_input = torch.randn(1, 3, IMG_H, IMG_W)
traced = torch.jit.trace(ts_model, example_input)
traced.save(TS_PATH)
print(f"✅ TorchScript saved: {TS_PATH}")

# ================= COREML CONVERSION =================
print("Converting to CoreML (.mlmodel)...")
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(shape=example_input.shape)],
    minimum_deployment_target=ct.target.iOS15
)

mlmodel.save(MLMODEL_PATH)
print(f"✅ CoreML model saved: {MLMODEL_PATH}")
