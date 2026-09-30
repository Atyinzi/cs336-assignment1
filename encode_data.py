"""数据预处理：使用 BPE tokenizer 将文本流式编码为二进制 token ID 文件。"""

from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def encode_file(input_path: Path, output_path: Path, tokenizer_path: Path) -> None:
    with tokenizer_path.open("rb") as file:
        tokenizer = pickle.load(file)
    # run_tinystories_bpe.py stores the tokenizer parts in a dictionary.
    if isinstance(tokenizer, dict):
        tokenizer = Tokenizer(
            vocab=tokenizer["vocab"],
            merges=tokenizer["merges"],
            special_tokens=tokenizer.get("special_tokens", []),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending_tokens: list[int] = []
    total_tokens = 0

    with open_text(input_path) as source, output_path.open("wb") as output:
        for line in source:
            pending_tokens.extend(tokenizer.encode(line))
            if len(pending_tokens) >= 1_000_000:
                tokens = np.asarray(pending_tokens, dtype=np.uint16)
                tokens.tofile(output)
                total_tokens += len(tokens)
                pending_tokens.clear()
                print(f"encoded {total_tokens:,} tokens", flush=True)

        if pending_tokens:
            tokens = np.asarray(pending_tokens, dtype=np.uint16)
            tokens.tofile(output)
            total_tokens += len(tokens)

    print(f"saved {total_tokens:,} tokens to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert text into uint16 token IDs")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    encode_file(args.input, args.output, args.tokenizer)


if __name__ == "__main__":
    main()
