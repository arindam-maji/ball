from ultralytics import YOLO
import torch
import coremltools as ct

# =========================================================
# CONFIG
# =========================================================
PT_PATH = "best.pt"
MLMODEL_PATH = "paddle_detection.mlmodel"
IMG_W, IMG_H = 960, 960  # match your training size
DEVICE = "cpu"            # CoreML conversion on CPU

# =========================================================
# LOAD YOLO MODEL
# =========================================================
print("Loading YOLO model...")
model = YOLO(PT_PATH)

# Make sure model is in eval mode
model.model.eval()

# =========================================================
# TRACE MODEL (TorchScript)
# =========================================================
print("Tracing model...")
example_input = torch.randn(1, 3, IMG_H, IMG_W)  # 3 channels RGB
traced = torch.jit.trace(model.model, example_input)

# =========================================================
# CONVERT TO COREML (.mlmodel)
# =========================================================
print("Converting to CoreML (.mlmodel for iOS14)...")
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="input", shape=example_input.shape)],
    minimum_deployment_target=ct.target.iOS14,  # .mlmodel only works for iOS14
    convert_to='neuralnetwork'
)

mlmodel.save(MLMODEL_PATH)
print(f"✅ Saved {MLMODEL_PATH}")


mlmodel.save(MLMODEL_PATH)
print(f"✅ Saved {MLMODEL_PATH}")

