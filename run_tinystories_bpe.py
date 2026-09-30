from __future__ import annotations

import json
import os
import pickle
import threading
import time
from pathlib import Path

import psutil

from cs336_basics.train_bpe import train_bpe


INPUT_PATH = Path("data/TinyStoriesV2-GPT4-train.txt")
OUTPUT_DIR = Path("artifacts/tinystories_bpe_10000")

VOCAB_SIZE = 10_000
SPECIAL_TOKEN = "<|endoftext|>"


def display_token(token: bytes) -> str:
    """将 token 转成便于查看的文本，无法解码的字节显示为转义形式。"""

    return token.decode("utf-8", errors="backslashreplace")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"找不到训练集：{INPUT_PATH.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    process = psutil.Process(os.getpid())
    baseline_rss = process.memory_info().rss
    peak_rss = baseline_rss

    stop_monitor = threading.Event()

    def monitor_memory() -> None:
        """定期读取当前进程 RSS，以近似统计峰值物理内存。"""

        nonlocal peak_rss

        while not stop_monitor.wait(0.05):
            current_rss = process.memory_info().rss
            peak_rss = max(peak_rss, current_rss)

    monitor_thread = threading.Thread(
        target=monitor_memory,
        daemon=True,
    )
    monitor_thread.start()

    input_size_gib = INPUT_PATH.stat().st_size / 1024**3

    print("=" * 60)
    print("TinyStories byte-level BPE training")
    print(f"Input: {INPUT_PATH.resolve()}")
    print(f"Input size: {input_size_gib:.2f} GiB")
    print(f"Maximum vocabulary size: {VOCAB_SIZE:,}")
    print(f"Special token: {SPECIAL_TOKEN!r}")
    print("=" * 60)
    print("Training started. The current implementation has no progress bar.")
    print("You can monitor system RAM in Windows Task Manager.")
    print(flush=True)

    start_time = time.perf_counter()

    try:
        vocab, merges = train_bpe(
            input_path=INPUT_PATH,
            vocab_size=VOCAB_SIZE,
            special_tokens=[SPECIAL_TOKEN],
        )
    finally:
        elapsed_seconds = time.perf_counter() - start_time

        stop_monitor.set()
        monitor_thread.join()

        current_rss = process.memory_info().rss
        peak_rss = max(peak_rss, current_rss)

        print()
        print("Training process ended.")
        print(f"Elapsed time: {elapsed_seconds:.2f} seconds")
        print(f"Elapsed time: {elapsed_seconds / 60:.2f} minutes")
        print(f"Peak RSS: {peak_rss / 1024**3:.2f} GiB")
        print(
            "Peak RSS above baseline: "
            f"{(peak_rss - baseline_rss) / 1024**3:.2f} GiB"
        )
        print(flush=True)

    # 基本正确性检查。
    expected_merges = VOCAB_SIZE - 256 - 1

    assert len(vocab) == VOCAB_SIZE, (
        f"词表大小错误：期望 {VOCAB_SIZE}，实际 {len(vocab)}"
    )
    assert len(merges) == expected_merges, (
        f"merge 数量错误：期望 {expected_merges}，实际 {len(merges)}"
    )
    assert SPECIAL_TOKEN.encode("utf-8") in vocab.values()

    # 1. Pickle 可以无损保存 Python bytes，适合之后重新加载 tokenizer。
    pickle_path = OUTPUT_DIR / "tokenizer.pkl"

    with open(pickle_path, "wb") as file:
        pickle.dump(
            {
                "vocab": vocab,
                "merges": merges,
                "special_tokens": [SPECIAL_TOKEN],
            },
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # 2. JSON 版本便于人工检查。
    # hex 是 bytes 的无损文本表示，text 是方便阅读的解码结果。
    vocab_json = {
        str(token_id): {
            "hex": token.hex(),
            "text": display_token(token),
            "n_bytes": len(token),
        }
        for token_id, token in vocab.items()
    }

    vocab_json_path = OUTPUT_DIR / "vocab.json"

    with open(vocab_json_path, "w", encoding="utf-8") as file:
        json.dump(
            vocab_json,
            file,
            ensure_ascii=False,
            indent=2,
        )

    merges_json = [
        {
            "rank": rank,
            "left_hex": left.hex(),
            "right_hex": right.hex(),
            "left_text": display_token(left),
            "right_text": display_token(right),
            "merged_hex": (left + right).hex(),
            "merged_text": display_token(left + right),
        }
        for rank, (left, right) in enumerate(merges)
    ]

    merges_json_path = OUTPUT_DIR / "merges.json"

    with open(merges_json_path, "w", encoding="utf-8") as file:
        json.dump(
            merges_json,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # 整个词表中的最长 token，包括特殊 token。
    longest_length = max(len(token) for token in vocab.values())

    longest_tokens = [
        (token_id, token)
        for token_id, token in vocab.items()
        if len(token) == longest_length
    ]

    # ID 0～255 是基础字节，ID 256 是特殊 token，
    # 所以 257 之后才是通过 BPE merge 学到的 token。
    learned_tokens = [
        (token_id, token)
        for token_id, token in vocab.items()
        if token_id >= 257
    ]

    longest_learned_length = max(
        len(token)
        for _, token in learned_tokens
    )

    longest_learned_tokens = [
        (token_id, token)
        for token_id, token in learned_tokens
        if len(token) == longest_learned_length
    ]

    report = {
        "input_path": str(INPUT_PATH.resolve()),
        "input_size_gib": input_size_gib,
        "vocab_size": len(vocab),
        "number_of_merges": len(merges),
        "special_tokens": [SPECIAL_TOKEN],
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_seconds / 60,
        "baseline_rss_gib": baseline_rss / 1024**3,
        "peak_rss_gib": peak_rss / 1024**3,
        "peak_rss_above_baseline_gib": (
            peak_rss - baseline_rss
        ) / 1024**3,
        "longest_token_length_bytes": longest_length,
        "longest_tokens": [
            {
                "id": token_id,
                "bytes_repr": repr(token),
                "hex": token.hex(),
                "text": display_token(token),
            }
            for token_id, token in longest_tokens
        ],
        "longest_learned_token_length_bytes": longest_learned_length,
        "longest_learned_tokens": [
            {
                "id": token_id,
                "bytes_repr": repr(token),
                "hex": token.hex(),
                "text": display_token(token),
            }
            for token_id, token in longest_learned_tokens
        ],
    }

    report_path = OUTPUT_DIR / "training_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 60)
    print("Training succeeded")
    print(f"Vocabulary size: {len(vocab):,}")
    print(f"Number of merges: {len(merges):,}")
    print(f"Training time: {elapsed_seconds:.2f} seconds")
    print(f"Peak RSS: {peak_rss / 1024**3:.2f} GiB")

    print()
    print(f"Longest token length: {longest_length} bytes")
    for token_id, token in longest_tokens:
        print(
            f"  id={token_id}, "
            f"bytes={token!r}, "
            f"text={display_token(token)!r}"
        )

    print()
    print(
        "Longest learned BPE token length: "
        f"{longest_learned_length} bytes"
    )
    for token_id, token in longest_learned_tokens:
        print(
            f"  id={token_id}, "
            f"bytes={token!r}, "
            f"text={display_token(token)!r}"
        )

    print()
    print("Saved files:")
    print(f"  {pickle_path.resolve()}")
    print(f"  {vocab_json_path.resolve()}")
    print(f"  {merges_json_path.resolve()}")
    print(f"  {report_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
