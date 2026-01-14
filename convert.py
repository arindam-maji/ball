import torch
import torch.nn as nn
import coremltools as ct

# =========================================================
# CONFIG
# =========================================================
PTH_PATH = "tracknet_best_07.pth"
MLMODEL_PATH = "tracknet_ball_03.mlmodel"  # .mlmodel format
IMG_W, IMG_H = 640, 360
DEVICE = "cpu"

# =========================================================
# MODEL DEFINITION
# =========================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class TrackNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(9, 64), ConvBlock(64, 64), nn.MaxPool2d(2),
            ConvBlock(64, 128), ConvBlock(128, 128), nn.MaxPool2d(2),
            ConvBlock(128, 256), ConvBlock(256, 256), nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(256, 128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(128, 64),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# =========================================================
# LOAD MODEL
# =========================================================
print("Loading PyTorch model...")
model = TrackNet().to(DEVICE)
model.load_state_dict(torch.load(PTH_PATH, map_location=DEVICE))
model.eval()

# =========================================================
# TRACE
# =========================================================
print("Tracing model...")
example_input = torch.randn(1, 9, IMG_H, IMG_W)
traced = torch.jit.trace(model, example_input)

# =========================================================
# CONVERT TO COREML (.mlmodel)
# =========================================================
print("Converting to CoreML (.mlmodel)...")
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="input", shape=example_input.shape)],
    minimum_deployment_target=ct.target.iOS15,
    convert_to='neuralnetwork'  # <- force .mlmodel output
)

mlmodel.save(MLMODEL_PATH)
print(f"✅ Saved {MLMODEL_PATH}")
