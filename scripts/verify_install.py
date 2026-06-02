"""Verify imports and GPU access"""

from importlib.metadata import version

print("Import verification\n")

try:
    import torch

    print("[OK] torch imported")
except Exception as e:
    print(f"[FAIL] torch import failed: {e}")
    exit(1)

try:
    import transformer_lens

    print("[OK] transformer_lens imported")
except Exception as e:
    print(f"[FAIL] transformer_lens import failed: {e}")
    exit(1)

try:
    import sae_lens
    from sae_lens import SAE, SAETransformerBridge

    print("[OK] sae_lens imported")
except Exception as e:
    print(f"[FAIL] sae_lens import failed: {e}")
    exit(1)

try:
    import transformers

    print("[OK] transformers imported")
except Exception as e:
    print(f"[FAIL] transformers import failed: {e}")
    exit(1)

try:
    import datasets

    print("[OK] datasets imported")
except Exception as e:
    print(f"[FAIL] datasets import failed: {e}")
    exit(1)

try:
    import numpy as np

    print("[OK] numpy imported")
except Exception as e:
    print(f"[FAIL] numpy import failed: {e}")
    exit(1)

try:
    import pandas as pd

    print("[OK] pandas imported")
except Exception as e:
    print(f"[FAIL] pandas import failed: {e}")
    exit(1)

try:
    import sklearn

    print("[OK] sklearn imported")
except Exception as e:
    print(f"[FAIL] sklearn import failed: {e}")
    exit(1)

try:
    import matplotlib

    print("[OK] matplotlib imported")
except Exception as e:
    print(f"[FAIL] matplotlib import failed: {e}")
    exit(1)

try:
    import streamlit

    print("[OK] streamlit imported")
except Exception as e:
    print(f"[FAIL] streamlit import failed: {e}")
    exit(1)

print(f"\nPyTorch:          {version('torch')}")
print(f"TransformerLens:  {version('transformer-lens')}")
print(f"SAELens:          {version('sae-lens')}")
print(f"Transformers:     {version('transformers')}")
print(f"CUDA available:   {torch.cuda.is_available()}")
print(f"GPU:              {torch.cuda.get_device_name(0)}")
print(
    f"VRAM:             {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
)

# dtype test
x = torch.randn(16, 16, dtype=torch.float16, device="cuda")
y = x @ x.T
assert not torch.isnan(y).any(), "fp16 matmul produced NaN"
print("fp16 matmul:      OK")

print("\nAll checks passed")
