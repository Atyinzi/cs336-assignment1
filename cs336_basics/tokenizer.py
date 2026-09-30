"""字节级 BPE tokenizer：完成特殊 token、编码、解码和流式编码。"""

from collections.abc import Iterable, Iterator

import regex


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []

        # merge 越早出现，优先级越高（rank 越小）。
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.bytes_to_id = {token: token_id for token_id, token in self.vocab.items()}

        # PDF 要求：特殊 token 不在词表中时，将它追加到词表末尾。
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in self.bytes_to_id:
                token_id = len(self.vocab)
                self.vocab[token_id] = token_bytes
                self.bytes_to_id[token_bytes] = token_id

    def _get_bpe_merges(self, piece: bytes) -> list[bytes]:
        """对一个 pre-token 应用 BPE merges，返回合并后的字节片段。"""

        # BPE 从单字节 token 开始。
        parts = [bytes([byte]) for byte in piece]

        while len(parts) > 1:
            # 找出当前相邻、并且存在于 merge 表中的 pair。
            pairs = {
                (parts[index], parts[index + 1])
                for index in range(len(parts) - 1)
                if (parts[index], parts[index + 1]) in self.merge_ranks
            }
            if not pairs:
                break

            # 每次应用训练过程中最早产生的 merge。
            best_pair = min(pairs, key=self.merge_ranks.get)
            new_parts: list[bytes] = []
            index = 0

            # 从左到右合并 best_pair 的所有不重叠出现位置。
            while index < len(parts):
                if index + 1 < len(parts) and (parts[index], parts[index + 1]) == best_pair:
                    new_parts.append(parts[index] + parts[index + 1])
                    index += 2
                else:
                    new_parts.append(parts[index])
                    index += 1

            parts = new_parts

        return parts

    def encode(self, text: str) -> list[int]:
        """将字符串编码为 token ID。"""

        if not text:
            return []

        # 特殊 token 按长度降序排列，保证重叠时优先匹配较长者。
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        if sorted_special_tokens:
            special_pattern = "|".join(regex.escape(token) for token in sorted_special_tokens)
            chunks = regex.split(f"({special_pattern})", text)
        else:
            chunks = [text]

        token_ids: list[int] = []
        special_token_set = set(self.special_tokens)

        for chunk in chunks:
            if not chunk:
                continue

            if chunk in special_token_set:
                token_ids.append(self.bytes_to_id[chunk.encode("utf-8")])
                continue

            # 普通文本先按 GPT-2 正则预分词，再独立执行 BPE；merge 不跨
            # pre-token 边界。
            for match in regex.finditer(PAT, chunk):
                piece = match.group().encode("utf-8")
                for merged_piece in self._get_bpe_merges(piece):
                    token_ids.append(self.bytes_to_id[merged_piece])

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """逐块编码字符串 iterable，惰性地产生 token ID。"""

        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """将 token ID 解码为字符串。"""

        # 单个 token 可能不是合法 UTF-8，所以先拼接全部 bytes，再统一解码。
        all_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return all_bytes.decode("utf-8", errors="replace")
