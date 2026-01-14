import coremltools as ct

print("coremltools version:", ct.__version__)

mlmodel = ct.converters.onnx.convert(
    model="tracknet_ball_03.onnx",
    minimum_deployment_target=ct.target.iOS15
)

mlmodel.save("tracknet_ball_03.mlmodel")
print("✅ Conversion successful")
