from __future__ import annotations

from typing import Iterable
import regex as re
from collections import defaultdict, Counter
import json
import logging
import time
from tqdm import tqdm
from multiprocessing import Pool
from pretokenization_example import find_chunk_boundaries
import os
import functools
import heapq
import pickle

NUM_PROCESSES = os.cpu_count() or 1
DISIRED_NUM_CHUNKS = 256
SPLIT_TOKEN = b"<|endoftext|>"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

@functools.lru_cache()
def gpt_byte_encoder() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256+n)
            n += 1
    return {b: chr(c) for b, c in zip(bs,cs)}

def encode_token(token: bytes) -> str:
    enc = gpt_byte_encoder()
    return "".join(enc[b] for b in token)

class HeapItem:
    __slots__ = ('count', 'pair')
    
    def __init__(self, count: int, pair: tuple[bytes, bytes]):
        self.count = count 
        self.pair = pair

    def __lt__(self, other: HeapItem) -> bool:
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair

def process_chunk(args):
    """
    每一个worker自己读文件并切分
    """
    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end-start).decode("utf-8", errors="ignore")
        if special_tokens:
            split_pat = "|".join(re.escape(i) for i in special_tokens)
            segments = re.split(split_pat, chunk)
        else:
            segments = [chunk]
    
    word_freqs = defaultdict(int)
    for seg in segments:
        for match in re.finditer(PAT, seg):
            encoded_word = tuple(bytes([i]) for i in match.group().encode('utf-8'))
            word_freqs[encoded_word] += 1
    return word_freqs

def merge_word(word: list[bytes], a: bytes, b: bytes, comb: bytes, delta: dict[tuple[bytes, bytes], int], f: int) -> tuple[list[bytes], bool]:
    out: list[bytes] = []
    i = 0
    n = len(word)
    changed = False
    while i < n:
        if i < n-1 and word[i] == a and word[i+1] == b:
            changed = True
            if i > 0:
                # 紧邻上一个合并点时 out[-1] 才是 comb，否则用原词左邻居
                left = comb if (out and out[-1] == comb and word[i-1] == b) else word[i-1]
                delta[(left, a)] -= f
                delta[(left, comb)] += f
            if i + 2 < n:
                right = word[i+2]
                delta[(b, right)] -= f
                delta[(comb, right)] += f
            delta[(a, b)] -= f
            out.append(comb)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out, changed


class BPETokenizer:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}        
        self.merges: list[tuple[bytes, bytes]] = []
        self.pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
        self.pair_to_word: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    def _pretokenize(self):
        t0 = time.perf_counter()
        # ---- 阶段 1: 多进程预分词 ----
        t_pre = time.perf_counter()
        logger.info("Pre-tokenizing with multiprocessing Pool ...")
        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, DISIRED_NUM_CHUNKS, b"<|endoftext|>")
        tasks = [(self.input_path, start, end, self.special_tokens) for
        start, end in zip(boundaries[:-1], boundaries[1:])]
        logger.info(
            "Pre-tokenize: %d chunk(s), %d worker(s)", len(tasks), NUM_PROCESSES
        )
        word_freqs = defaultdict(int)
        try:
            with Pool(processes=NUM_PROCESSES) as pool:
                for freq in tqdm(
                    pool.imap_unordered(process_chunk, tasks),
                    total=len(tasks),
                    desc="Pre-tokenize",
                    unit="chunk",
                ):
                    for w, c in freq.items():
                        word_freqs[w] += c
        except Exception:
            logger.exception("多进程预分词失败")
            raise

        logger.info(
            "Pre-tokenize done: %d unique words, elapsed=%.1fs",
            len(word_freqs), time.perf_counter() - t_pre,
        )
        return word_freqs

    def _init_pairs(self, words: list[list[bytes]], freqs: list[int]):
        t = time.perf_counter()
        for i, word in enumerate(tqdm(words, desc="Count pairs", unit="word")):
            for x, y in zip(word[:-1], word[1:]):
                self.pair_counts[(x,y)] += freqs[i]
                self.pair_to_word[(x,y)].add(i)
        logger.info("Initial pairs: %d distinct", len(self.pair_counts))

    def merge(self):
        t0 = time.perf_counter()
        logger.info(
            "Start BPE: input=%s, vocab_size=%d, special_tokens=%s",
            self.input_path, self.vocab_size, self.special_tokens,
        )
 
        word_freqs = self._pretokenize()
        if not word_freqs:
            logger.warning("空输入, 直接返回基础 vocab")
            for tok in self.special_tokens:
                self.vocab[len(self.vocab)] = tok.encode("utf-8")
            return self.vocab, self.merges

        words = [list(w) for w in word_freqs]
        freqs = list(word_freqs.values())
        self._init_pairs(words, freqs)

        heap = [HeapItem(c, p) for p, c in self.pair_counts.items()]
        heapq.heapify(heap)

        def pop_best():
            while heap:
                item = heapq.heappop(heap)
                if self.pair_counts.get(item.pair) == item.count:
                    return item.pair
            return None

        num_merges = self.vocab_size - 256 - len(self.special_tokens)
        logger.info("Target vocab_size=%d -> %d merges", self.vocab_size, num_merges)
        t_merge = time.perf_counter()
        pbar = tqdm(range(num_merges), desc="BPE merges", unit="merge")
        done = 0
        for step in pbar:
            best = pop_best()
            if best is None:
                logger.warning("No pairs left, stop at merge %d", step)
                break
            a, b = best
            comb = a + b
            self.vocab[len(self.vocab)] = comb
            self.merges.append((a,b))
            done += 1
            # 每 500 步在进度条上挂一点当前状态
            if step % 500 == 0:
                pbar.set_postfix(
                    best=comb.decode("utf-8", errors="replace")[:12],
                    count=self.pair_counts.get(best, 0),
                    pairs=len(self.pair_counts),
                )
            # pop 取出 best 的倒排，同时把它从索引里删掉
            affected = self.pair_to_word.pop(best, set())
            delta: dict[tuple[bytes, bytes], int] = defaultdict(int)

            for idx in affected:
                word = words[idx]
                f = freqs[idx]
                new_word, changed = merge_word(word, a, b, comb, delta, f)
                # 倒排里残留的脏 idx（实际已不含 best），跳过
                if not changed:
                    continue
                words[idx] = new_word

                # 维护倒排 pair_to_word：只登记 comb 相关的新 pair
                for x, y in zip(new_word, new_word[1:]):
                    if x == comb or y == comb:
                        self.pair_to_word[(x, y)].add(idx)

            # 一次性应用 delta 到 pair_counts + heap
            for pair, d in delta.items():
                if d == 0:
                    continue
                new_count = self.pair_counts.get(pair, 0) + d
                if new_count <= 0:
                    self.pair_counts.pop(pair, None)
                    self.pair_to_word.pop(pair, None)
                else:
                    self.pair_counts[pair] = new_count
                    heapq.heappush(heap, HeapItem(new_count, pair))
                
            # best 的 pair_to_word 已在上面 pop，这里只清 count
            self.pair_counts.pop(best, None)
 
        dt_merge = time.perf_counter() - t_merge
        rate = done / dt_merge if dt_merge > 0 else 0.0
        logger.info(
            "Merge loop: %d merges, elapsed=%.1fs, %.0f merges/s",
            done, dt_merge, rate,
        )
 
        for tok in self.special_tokens:
            self.vocab[len(self.vocab)] = tok.encode("utf-8")
 
        logger.info("BPE done, total elapsed=%.1fs", time.perf_counter() - t0)
        return self.vocab, self.merges

def save_outputs(vocab, merges, out_dir="."):
    # 1) 无损二进制, 保证完美 round-trip
    with open(os.path.join(out_dir, "bpe.pkl"), "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)
 
    # 2) 可读且可逆的 vocab.json
    readable = {str(k): encode_token(v) for k, v in vocab.items()}
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(readable, f, ensure_ascii=False, indent=2)
 
    # 3) 可读且可逆的 merges.txt
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{encode_token(a)} {encode_token(b)}\n")


class TqdmLoggingHandler(logging.Handler):
    """让 logging 通过 tqdm.write 输出, 不会冲掉进度条。"""
 
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)



if __name__ == "__main__":
    import tracemalloc, sys
    logger = logging.getLogger("bpe")
 
    handler = TqdmLoggingHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    )
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(), handlers=[handler]
    )
    tracemalloc.start()   
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    tokenizer = BPETokenizer(input_path, vocab_size, special_tokens)
    vocab, merges = tokenizer.merge()
 
    current, peak = tracemalloc.get_traced_memory()
    logger.info("内存: 当前 %.1f MB, 峰值 %.1f MB", current / 1e6, peak / 1e6)
    tracemalloc.stop()
 
    save_outputs(vocab, merges)
    logger.info(
        "Saved bpe.pkl / vocab.json (%d) / merges.txt (%d)",
        len(vocab), len(merges),
    )