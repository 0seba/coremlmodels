import time
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

# Import from coremlmodels package
from coremlmodels import patch_model_linears, analyze_compute_plan, inspect_mil_program


def main():
    print("CoreML Linear-to-Conv2d Patching Example")
    print("========================================")

    # 1. Define a Huge MLP Model to target Neural Engine
    dim = 4096

    class HugeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Linear(dim // 2, 10),
            )

        def forward(self, x):
            return self.layers(x)

    model = HugeMLP()
    model.eval()
    print(f"\n[1] Created Huge MLP Model (Dim={dim})")

    # 2. Patch
    print("\n[2] Patching model (Linear -> Conv2d)...")
    patched_model = patch_model_linears(model, verbose=True)
    patched_model.eval()

    # 3. Inputs
    batch_size = 1
    input_features = dim
    example_input = torch.randn(batch_size, input_features)
    patched_input = example_input.view(batch_size, input_features, 1, 1)

    # 4. Convert
    print("\n[3] Converting to CoreML...")
    traced_model = torch.jit.trace(patched_model, patched_input)

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=(batch_size, input_features, 1, 1), name="input")],
        outputs=[ct.TensorType(name="output")],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS16,
    )

    model_path = "HugeMLP_Patched.mlpackage"
    mlmodel.save(model_path)
    print(f"Saved CoreML model to {model_path}")

    # 5. Verify
    print("\n[4] verifying Outputs...")
    with torch.no_grad():
        torch_output = patched_model(patched_input)

    prediction = mlmodel.predict({"input": patched_input.numpy()})

    max_diff = np.max(
        np.abs(torch_output.numpy().flatten() - prediction["output"].flatten())
    )
    print(f"Max absolute difference: {max_diff:.6f}")
    if max_diff < 1e-2:
        print("✅ Pytorch vs CoreML outputs match!")
    else:
        print("❌ Outputs mismatch significantly.")

    # 6. Timing
    print("\n[5] Timing Inference (Average of 20 runs)...")
    for _ in range(5):
        mlmodel.predict({"input": patched_input.numpy()})

    start_time = time.time()
    for _ in range(20):
        mlmodel.predict({"input": patched_input.numpy()})
    coreml_time_ms = (time.time() - start_time) * 1000 / 20
    print(f"CoreML Inference: {coreml_time_ms:.4f} ms")

    # 7. Analysis via Core Library Tools
    analyze_compute_plan(mlmodel)
    inspect_mil_program(mlmodel)


if __name__ == "__main__":
    main()
