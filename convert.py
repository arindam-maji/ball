import coremltools as ct

mlmodel = ct.converters.onnx.convert(
    model="tracknet_ball_03.onnx",
    minimum_deployment_target=ct.target.iOS15
)

mlmodel.save("tracknet_ball_03.mlmodel")
print("✅ CoreML model generated")
