# Conversion Guide

Replace `nn.Linear` with `nn.Conv2d(kernel_size=1)` for Neural Engine optimization.

## Key Points

- Input: 4D tensor `(batch, channels, H, W)`
- Output: 4D tensor `(batch, out_channels, H, W)`
- Weight: Reshape from `(out, in)` to `(out, in, 1, 1)`
- No clone - keep weights as views
- No auto-reshape - users provide correct input shapes

## Validation

After conversion, verify Neural Engine usage:

1. Use `coremlmodels.analyze_compute_plan(mlmodel)` to check that `conv` layers run on `NeuralEngine`.
2. Use `coremlmodels.inspect_mil_program(mlmodel)` to inspect exact input shapes and constant values.