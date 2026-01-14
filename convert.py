import coremltools as ct
import onnx

ONNX_PATH = "tracknet_ball_03.onnx"
MLMODEL_PATH = "tracknet_ball_03.mlmodel"

print("Loading ONNX model...")
onnx_model = onnx.load(ONNX_PATH)

print("Converting ONNX → CoreML...")
mlmodel = ct.convert(
    onnx_model,
    source="onnx",
    minimum_deployment_target=ct.target.iOS15
)

print("Saving CoreML model...")
mlmodel.save(MLMODEL_PATH)

print("✅ Conversion successful:", MLMODEL_PATH)
