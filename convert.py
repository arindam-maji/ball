import torch
import coremltools as ct

# ---------------- CONFIG ----------------
PTH_PATH = "tracknet_best_07.pth"
MLMODEL_PATH = "tracknet_ball_03.mlmodel"
INPUT_SHAPE = (1, 3, 288, 512)   # <-- adjust if your model differs
IOS_TARGET = ct.target.iOS15
# ----------------------------------------

print("Loading model...")

# 🔹 IMPORT YOUR MODEL CLASS
from model import TrackNet   # <-- must match your training code

model = TrackNet()
state_dict = torch.load(PTH_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

example_input = torch.randn(*INPUT_SHAPE)

print("Tracing TorchScript...")
traced_model = torch.jit.trace(model, example_input)

print("Converting to CoreML...")
mlmodel = ct.convert(
    traced_model,
    source="pytorch",
    inputs=[ct.ImageType(
        name="input",
        shape=INPUT_SHAPE,
        scale=1/255.0
    )],
    minimum_deployment_target=IOS_TARGET
)

mlmodel.save(MLMODEL_PATH)
print(f"✅ Saved {MLMODEL_PATH}")
