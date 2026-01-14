import onnx
from onnx_coreml import convert

ONNX_PATH = "tracknet_ball_03.onnx"
OUT_PATH = "tracknet_ball_03.mlmodel"

print("Loading ONNX model...")
onnx_model = onnx.load(ONNX_PATH)

print("Converting to CoreML...")
mlmodel = convert(
    onnx_model,
    minimum_ios_deployment_target="15"
)

mlmodel.save(OUT_PATH)
print(f"✅ Conversion successful: {OUT_PATH}")
