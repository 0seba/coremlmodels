# Conversion Guide

Replace `nn.Linear` with `nn.Conv2d(kernel_size=1)` for Neural Engine optimization.

## Key Points

- Input: 4D tensor `(batch, channels, H, W)`
- Output: 4D tensor `(batch, out_channels, H, W)`
- Weight: Reshape from `(out, in)` to `(out, in, 1, 1)`
- No clone - keep weights as views
- No auto-reshape - users provide correct input shapes