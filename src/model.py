import gc
import torch
from sae_lens import SAE, SAETransformerBridge


def load_model(config: dict) -> SAETransformerBridge:
    torch.cuda.empty_cache()
    gc.collect()

    model = SAETransformerBridge.boot_transformers(
        config["model"]["name"],
        device=str(config["_device"]),
        dtype=config["_dtype"],
    )

    print("\n== Model Info ==")
    print(f"Model loaded: {config['model']['name']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print("================\n")
    return model


def load_sae(config: dict, layer: int | None = None, device: str | None = None) -> SAE:
    layer = layer or config["sae"]["primary_layer"]
    width = config["sae"]["width"]
    l0 = config["sae"]["l0"]
    sae_id = f"layer_{layer}_width_{width}_l0_{l0}"
    device = device or str(config["_device"])

    sae = SAE.from_pretrained(
        release=config["sae"]["release"],
        sae_id=sae_id,
        device=device,
        dtype=str(config["_dtype"]).replace("torch.", ""),
    )

    print("\n== SAE Info ==")
    print(f"SAE loaded: {sae_id}")
    print(f"Dictionary size: {sae.cfg.d_sae}")
    print(f"Input dimension: {sae.cfg.d_in}")
    print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print("==============\n")
    return sae


def load_multiple_saes(config: dict, layers: list[int] | None = None) -> dict[int, SAE]:
    layers = layers or config["sae"]["layers"]
    return {layer: load_sae(config, layer) for layer in layers}


def print_memory_report():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(
            f"\nVRAM: {alloc:.2f}GB allocated / {reserved:.2f}GB reserved / {total:.2f}GB total"
        )
