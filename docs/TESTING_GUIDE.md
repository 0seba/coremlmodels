# Testing Guide

## Pattern

```python
def test_name():
    linear = nn.Linear(10, 20)
    patcher = LinearToConv2dPatcher(linear)

    x = torch.randn(2, 10)
    x_4d = x.view(2, 10, 1, 1)

    linear_out = linear(x)
    patcher_out = patcher(x_4d).squeeze(-1).squeeze(-1)

    torch.testing.assert_close(linear_out, patcher_out)
```

## Run Tests

```bash
uv run pytest tests/test_patch_linears.py -v
```
