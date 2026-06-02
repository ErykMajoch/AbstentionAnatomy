import csv
import json
import time
import urllib.request
from pathlib import Path


def fetch_feature(model_id: str, sae_id: str, feature_id: int) -> dict | None:
    url = f"https://www.neuronpedia.org/api/feature/{model_id}/{sae_id}/{feature_id}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "AbstentionAnatomy/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read())
    except Exception as e:
        print(f"\n  Failed to fetch feature {feature_id}: {e}")
        return None


def _extract_context(tokens: list[str], values: list[float], window: int = 3) -> str:
    if not tokens or not values:
        return ""
    max_idx = max(range(len(values)), key=lambda i: values[i])
    start = max(0, max_idx - window)
    end = min(len(tokens), max_idx + window + 1)
    snippet_tokens = tokens[start:end]
    clean = "".join(snippet_tokens).strip()
    clean = (
        clean.replace("<bos>", "")
        .replace("<end_of_turn>", "")
        .replace("<start_of_turn>", "")
        .strip()
    )
    return clean


def summarise_feature(data: dict) -> tuple[str, str, float]:
    position_tokens = data.get("pos_str", [])
    top_tokens = [t.strip() for t in position_tokens[:8] if t.strip()]
    token_summary = ", ".join(top_tokens) if top_tokens else "no data"
    sparsity = data.get("frac_nonzero", 0.0)

    snippets = []
    for act_example in data.get("activations", [])[:5]:
        tokens = act_example.get("tokens", [])
        values = act_example.get("values", [])
        ctx = _extract_context(tokens, values)
        if ctx and ctx not in snippets:
            snippets.append(ctx)

    explanation = "; ".join(snippets) if snippets else token_summary
    return token_summary, explanation, sparsity


def fetch_neuronpedia_explanations(
    model_id: str,
    sae_id: str,
    feature_ids: list[int],
    activations: list[float],
    save_path: str | Path | None = None,
    delay: float = 0.5,
) -> list[dict]:
    rows = []
    for rank, (fidelity, activation) in enumerate(zip(feature_ids, activations)):
        print(f"  [{rank + 1}/{len(feature_ids)}] Feature {fidelity}", end="")
        data = fetch_feature(model_id, sae_id, fidelity)

        if data:
            token_summary, explanation, sparsity = summarise_feature(data)
            print(f" sparsity={sparsity:.2%}")
        else:
            token_summary, explanation, sparsity = "fetch failed", "fetch failed", 0.0
            print()

        rows.append(
            {
                "Rank": rank + 1,
                "Feature ID": fidelity,
                "Activation": round(activation, 4),
                "Explanation": explanation,
                "Top Positive Tokens": token_summary,
                "Sparsity": round(sparsity * 100, 2),
                "Neuronpedia URL": f"https://www.neuronpedia.org/{model_id}/{sae_id}/{fidelity}",
            }
        )

        if rank < len(feature_ids) - 1:
            time.sleep(delay)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"Saved to '{save_path}'")

    return rows
