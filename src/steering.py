import torch
from functools import partial
from tqdm import tqdm


def get_steering_direction(sae, feature_id: int) -> torch.Tensor:
    return sae.W_dec[feature_id]  # shape: [d_model]


def _steering_hook(residual, hook, direction, coeff):
    return residual + coeff * direction.to(residual.device, residual.dtype)


def _ablation_hook(sae_acts, hook, feature_ids):
    for fid in feature_ids:
        sae_acts[:, :, fid] = 0.0
    return sae_acts


def steer_and_generate(
    model,
    sae,
    prompt: str,
    feature_id: int,
    coeff: float,
    layer: int,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
) -> dict:

    direction = get_steering_direction(sae, feature_id)
    hook_name = f"blocks.{layer}.hook_out"

    hook_fn = partial(_steering_hook, direction=direction, coeff=coeff)
    tokens = model.to_tokens(prompt)

    with torch.no_grad():
        model.add_hook(hook_name, hook_fn)
        try:
            output = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=1 if temperature == 0 else 50,
            )
        finally:
            model.reset_hooks()

    generated_tokens = output[0, tokens.shape[1] :]
    response = model.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {
        "prompt": prompt,
        "feature_id": feature_id,
        "coeff": coeff,
        "layer": layer,
        "response": response,
    }


def ablate_and_generate(
    model,
    sae,
    prompt: str,
    feature_ids: list[int],
    layer: int,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
) -> dict:
    hook_name = model.get_sae_hook_name(sae, internal="hook_sae_acts_post")
    hook_fn = partial(_ablation_hook, feature_ids=feature_ids)

    tokens = model.to_tokens(prompt)

    with torch.no_grad():
        model.add_sae(sae)
        model.add_hook(hook_name, hook_fn)
        try:
            output = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=1 if temperature == 0 else 50,
            )
        finally:
            model.reset_saes()
            model.reset_hooks()

    generated_tokens = output[0, tokens.shape[1] :]
    response = model.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {
        "prompt": prompt,
        "ablated_features": feature_ids,
        "layer": layer,
        "response": response,
    }


def batch_steer_experiment(
    model,
    sae,
    prompts: list[str],
    feature_id: int,
    coefficients: list[float],
    layer: int,
    max_new_tokens: int = 64,
) -> list[dict]:
    responses = []
    for coeff in coefficients:
        print(f"\nCoefficient {coeff:+.0f}")
        for prompt in tqdm(prompts, desc=f"coeff={coeff}", leave=False):
            result = steer_and_generate(
                model, sae, prompt, feature_id, coeff, layer, max_new_tokens
            )
            responses.append(result)
        print(f" Generated {len(prompts)} responses")
    return responses
