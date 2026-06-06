import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm


def collect_activations(
    model,
    sae,
    prompts: list[str],
    layer: int,
    token_positions: list[str],
    batch_size: int = 1,
    max_seq_len: int = 128,
    save_path: Path | None = None,
    dtype=torch.float16,
) -> dict:
    hook_name = model.get_sae_hook_name(sae, internal="hook_sae_acts_post")
    n_features = sae.cfg.d_sae

    results = {}
    for position in token_positions:
        results[position] = np.zeros((len(prompts), n_features), dtype=np.float16)

    for start_index in tqdm(range(0, len(prompts), batch_size), desc=f"Layer {layer}"):
        end_index = min(start_index + batch_size, len(prompts))
        batch_prompts = prompts[start_index:end_index]

        tokens = model.to_tokens(batch_prompts)
        if tokens.shape[1] > max_seq_len:
            tokens = tokens[:, :max_seq_len]

        seq_len = tokens.shape[1]

        with torch.no_grad():
            _, cache = model.run_with_cache_with_saes(tokens, saes=[sae])

        sae_acts = cache[hook_name]  # shape: [batch, seq, n_features]

        for position_name in token_positions:
            match position_name:
                case "last":
                    position_index = -1
                case "last-1":
                    position_index = -2 if seq_len >= 2 else -1
                case "last-2":
                    position_index = -3 if seq_len >= 3 else -1
                case "mean":
                    acts_np = sae_acts.mean(dim=1).cpu().numpy().astype(np.float16)
                    results[position_name][start_index:end_index] = acts_np
                    continue
                case _:
                    raise ValueError(f"Unknown position: {position_name}")

            acts_np = sae_acts[:, position_index, :].cpu().numpy().astype(np.float16)
            results[position_name][start_index:end_index] = acts_np

        del cache, sae_acts
        torch.cuda.empty_cache()

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        for position_name, array in results.items():
            np.save(save_path / f"layer_{layer}_{position_name}.npy", array)
        print(f"Saved activations to {save_path}")
