"""文本生成：加载 TransformerLM checkpoint，并支持 temperature 与 top-p 采样。"""

from __future__ import annotations

import argparse
import pickle

import torch

from ..fullLM import TransformerLM
from ..tokenizer import Tokenizer


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> int:
    """Sample one token from logits using temperature and optional top-p sampling."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    probabilities = torch.softmax(logits / temperature, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_ids = torch.sort(probabilities, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Keep the first token that makes the cumulative probability reach top_p.
        remove = cumulative_probs - sorted_probs > top_p
        sorted_probs[remove] = 0

        probabilities = torch.zeros_like(probabilities)
        probabilities.scatter_(0, sorted_ids, sorted_probs)
        probabilities = probabilities / probabilities.sum()

    return int(torch.multinomial(probabilities, num_samples=1).item())


@torch.no_grad()
def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cpu",
) -> str:
    """Generate up to max_new_tokens after prompt."""
    token_ids = tokenizer.encode(prompt)
    if len(token_ids) == 0:
        raise ValueError("prompt must contain at least one token")

    end_token = tokenizer.bytes_to_id.get(b"<|endoftext|>")
    model.eval()

    for _ in range(max_new_tokens):
        # The model has a fixed context window; keep the newest tokens.
        input_ids = torch.tensor([token_ids[-context_length:]], dtype=torch.long, device=device)
        logits = model(input_ids)[0, -1]
        next_token = sample_next_token(logits, temperature, top_p)
        token_ids.append(next_token)

        if end_token is not None and next_token == end_token:
            break

    return tokenizer.decode(token_ids)


def load_tokenizer(path: str) -> Tokenizer:
    with open(path, "rb") as file:
        saved = pickle.load(file)
    if isinstance(saved, Tokenizer):
        return saved
    return Tokenizer(
        vocab=saved["vocab"],
        merges=saved["merges"],
        special_tokens=saved.get("special_tokens", []),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a TransformerLM checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.device == "gpu":
        args.device = "cuda"
    d_ff = args.d_ff if args.d_ff is not None else int(8 * args.d_model / 3)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    tokenizer = load_tokenizer(args.tokenizer)
    print(
        generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            context_length=args.context_length,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
