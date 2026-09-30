"""BPE 词表训练：完成预分词、pair 统计、频率合并和词表输出。"""

from collections import Counter, defaultdict
from pathlib import Path
import heapq

import regex


PAT = regex.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")


def _count_pretokens(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    """统计每种 pre-token 出现次数，特殊 token 只充当边界。"""

    if special_tokens:
        specials = sorted(set(special_tokens), key=len, reverse=True)
        pattern = "|".join(regex.escape(token) for token in specials)
        chunks = regex.split(pattern, text)
    else:
        chunks = (text,)

    byte_tokens = [bytes([value]) for value in range(256)]
    counts: Counter[tuple[bytes, ...]] = Counter()
    for chunk in chunks:
        for match in PAT.finditer(chunk):
            word = tuple(byte_tokens[value] for value in match.group().encode("utf-8"))
            counts[word] += 1
    return counts


def _merge(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """从左到右合并一个 pre-token 内所有不重叠的目标 pair。"""

    result = []
    index = 0
    while index < len(word):
        if index + 1 < len(word) and (word[index], word[index + 1]) == pair:
            result.append(pair[0] + pair[1])
            index += 2
        else:
            result.append(word[index])
            index += 1
    return tuple(result)


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """返回 ID→字节串的词表，以及按学习顺序排列的 merge 列表。"""

    vocab = {value: bytes([value]) for value in range(256)}
    for special in dict.fromkeys(special_tokens):
        vocab[len(vocab)] = special.encode("utf-8")
    if vocab_size < len(vocab):
        raise ValueError("vocab_size is smaller than the initial vocabulary")

    with open(input_path, encoding="utf-8") as file:
        word_counts = _count_pretokens(file.read(), special_tokens)

    # 相同的 pre-token 只存一次；其频次是它对 pair 计数的权重。
    words = list(word_counts)
    frequencies = [word_counts[word] for word in words]
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for word_id, (word, frequency) in enumerate(zip(words, frequencies)):
        for pair, occurrences in Counter(zip(word, word[1:])).items():
            pair_counts[pair] += occurrences * frequency
            pair_to_words[pair].add(word_id)

    # 每个频次对应一个 pair 集合：先取最大频次，平局取字典序最大 pair。
    buckets: defaultdict[int, set[tuple[bytes, bytes]]] = defaultdict(set)
    heap: list[int] = []
    for pair, count in pair_counts.items():
        buckets[count].add(pair)
        heapq.heappush(heap, -count)

    def update_count(pair: tuple[bytes, bytes], count: int) -> None:
        old_count = pair_counts.get(pair, 0)
        if old_count:
            buckets[old_count].discard(pair)
        if count:
            pair_counts[pair] = count
            buckets[count].add(pair)
            heapq.heappush(heap, -count)
        else:
            pair_counts.pop(pair, None)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        # 堆中旧频次不必每次删除，取最大值时再丢弃空桶。
        while heap and not buckets[-heap[0]]:
            heapq.heappop(heap)
        if not heap:
            break

        best_pair = max(buckets[-heap[0]])
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        # 只有包含 best_pair 的词形会变化：减去旧贡献，合并，再加新贡献。
        for word_id in list(pair_to_words[best_pair]):
            frequency = frequencies[word_id]
            old_pairs = Counter(zip(words[word_id], words[word_id][1:]))
            for pair, occurrences in old_pairs.items():
                update_count(pair, pair_counts[pair] - occurrences * frequency)
                pair_to_words[pair].discard(word_id)

            words[word_id] = _merge(words[word_id], best_pair)
            new_pairs = Counter(zip(words[word_id], words[word_id][1:]))
            for pair, occurrences in new_pairs.items():
                update_count(pair, pair_counts.get(pair, 0) + occurrences * frequency)
                pair_to_words[pair].add(word_id)

        pair_to_words.pop(best_pair, None)

    return vocab, merges
