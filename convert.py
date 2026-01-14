# convert_to_coreml.py
import torch
import coremltools as ct
from ultralytics import YOLO
import os

# ---------------- CONFIG ----------------
MODEL_PATH ="Court_Detection.pt"   # from GitHub Actions secret or default
OUTPUT_PATH ="court.mlmodel"
IMG_SIZE = 640  # input size

# ---------------- LOAD MODEL ----------------
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

# ---------------- TRACE MODEL ----------------
# Create a dummy input for tracing
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

print("Tracing the model for CoreML conversion...")
traced_model = model.model.model  # get the raw PyTorch nn.Module
traced_model.eval()
example_output = traced_model(dummy_input)

# ---------------- CONVERT TO COREML ----------------
print("Converting to CoreML...")
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="input_1", shape=dummy_input.shape)],
    convert_to="mlprogram", 
    minimum_deployment_target=ct.target.iOS15
# recommended for YOLOv8
)

# ---------------- SAVE ----------------
mlmodel.save(OUTPUT_PATH)
print(f"✅ Saved CoreML model at {OUTPUT_PATH}")
