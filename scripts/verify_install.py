"""Verify imports and GPU access"""
from importlib.metadata import version
print("Import verification\n")

try:
    import torch
    print("✓ torch imported")
except Exception as e:
    print(f"✗ torch import failed: {e}")
    exit(1)

try:
    import transformer_lens
    print("✓ transformer_lens imported")
except Exception as e:
    print(f"✗ transformer_lens import failed: {e}")
    exit(1)

try:
    import sae_lens
    from sae_lens import SAE, HookedSAETransformer
    print("✓ sae_lens imported")
except Exception as e:
    print(f"✗ sae_lens import failed: {e}")
    exit(1)

try:
    import transformers
    print("✓ transformers imported")
except Exception as e:
    print(f"✗ transformers import failed: {e}")
    exit(1)

try:
    import datasets
    print("✓ datasets imported")
except Exception as e:
    print(f"✗ datasets import failed: {e}")
    exit(1)

try:
    import numpy as np
    print("✓ numpy imported")
except Exception as e:
    print(f"✗ numpy import failed: {e}")
    exit(1)

try:
    import pandas as pd
    print("✓ pandas imported")
except Exception as e:
    print(f"✗ pandas import failed: {e}")
    exit(1)

try:
    import sklearn
    print("✓ sklearn imported")
except Exception as e:
    print(f"✗ sklearn import failed: {e}")
    exit(1)

try:
    import matplotlib
    print("✓ matplotlib imported")
except Exception as e:
    print(f"✗ matplotlib import failed: {e}")
    exit(1)

try:
    import streamlit
    print("✓ streamlit imported")
except Exception as e:
    print(f"✗ streamlit import failed: {e}")
    exit(1)

print(f"\nPyTorch:          {version('torch')}")
print(f"TransformerLens:  {version('transformer-lens')}")
print(f"SAELens:          {version('sae-lens')}")
print(f"Transformers:     {version('transformers')}")
print(f"CUDA available:   {torch.cuda.is_available()}")
print(f"GPU:              {torch.cuda.get_device_name(0)}")
print(f"VRAM:             {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# dtype test
x = torch.randn(16, 16, dtype=torch.float16, device="cuda")
y = x @ x.T
assert not torch.isnan(y).any(), "fp16 matmul produced NaN"
print("fp16 matmul:      OK")

print("\nAll checks passed.")