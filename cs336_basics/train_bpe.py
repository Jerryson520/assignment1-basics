from typing import Iterable
import regex as re
from collections import defaultdict
import json
import logging
import time
from tqdm import tqdm
from multiprocessing import Pool
from pretokenization_example import find_chunk_boundaries
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bpe")

num_processes = os.cpu_count()
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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

    def _pre_tokenize(self, input) -> dict[tuple[bytes, ...], int]:
        word_freqs = defaultdict(int) 
        if self.special_tokens:
            split_pat = "|".join(re.escape(i) for i in self.special_tokens)
            chunks = re.split(split_pat, input)
        else:
            chunks = [input]
        
        logger.info("Pre-tokenizing: %d chunk(s), %d chars total", len(chunks), len(input))
        for chunk in tqdm(chunks, desc="Pre-tokenize", unit="chunk"):
            for match in re.finditer(self.PAT, chunk):
                encode_word = tuple(bytes([b]) for b in match.group().encode('utf-8'))
                word_freqs[encode_word] += 1
        logger.info("Pre-tokenize done: %d unique words", len(word_freqs))
        return word_freqs

    def merge(self):
        t0 = time.perf_counter()
        # with open(self.input_path, 'r', encoding='utf-8') as f:
        #     input = f.read()
        # word_freqs = self._pre_tokenize(input)
        logger.info(
            "Start BPE training: input=%s, vocab_size=%d, special_tokens=%s",
            self.input_path, self.vocab_size, self.special_tokens,
        )
 
        # ---- 阶段 1: 多进程预分词 (耗时大头, 之前完全没日志) ----
        t_pre = time.perf_counter()
        logger.info("Pre-tokenizing with multiprocessing Pool ...")
        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        tasks = [(self.input_path, start, end, self.special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
        
        word_freqs = defaultdict(int)
        with Pool() as pool:
            local_freqs = pool.imap_unordered(process_chunk, tasks)
            for freq in local_freqs:
                for word, ff in freq.items():
                    word_freqs[word] += ff
        logger.info(
            "Pre-tokenize done: %d docs, %d unique words, elapsed=%.1fs",
            len(word_freqs), time.perf_counter() - t_pre,
        )

        words = [word for word in word_freqs]
        freqs = list(word_freqs.values())
        for i, word in enumerate(tqdm(words, desc="Count pairs", unit="word")):
            for x, y in zip(word[:-1], word[1:]):
                self.pair_counts[(x,y)] += freqs[i]
                self.pair_to_word[(x,y)].add(i)
        logger.info("Initial pairs: %d distinct", len(self.pair_counts))

        num_merges = self.vocab_size - 256 - len(self.special_tokens)
        logger.info("Target vocab_size=%d -> %d merges", self.vocab_size, num_merges)
        pbar = tqdm(range(num_merges), desc="BPE merges", unit="merge")
        for step in pbar:
            if not self.pair_counts:
                logger.warning("No pairs left, stopping at merge %d", step)
                break
            best = max(self.pair_counts, key=lambda x: [self.pair_counts[x], x])
            a, b = best
            comb = a + b
            self.vocab[len(self.vocab)] = comb
            self.merges.append((a,b))

            # 每 500 步在进度条上挂一点当前状态,方便扫一眼
            if step % 500 == 0:
                pbar.set_postfix(
                    best=comb.decode("utf-8", errors="replace")[:12],
                    count=self.pair_counts.get(best, 0),
                    pairs=len(self.pair_counts),
                )

            for idx in list(self.pair_to_word[best]):
                word = words[idx]
                f = freqs[idx]
                for x, y in zip(word[:-1], word[1:]):
                    self.pair_counts[(x,y)] -= f
                    if self.pair_counts[(x,y)] <= 0:
                        del self.pair_counts[(x,y)]
                        self.pair_to_word[(x,y)].discard(idx)
            
                new_word, i = [], 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i+1] == b:
                        new_word.append(comb)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                
                words[idx] = new_word

                for x, y in zip(new_word[:-1], new_word[1:]):
                    self.pair_counts[(x,y)] += f
                    self.pair_to_word[(x,y)].add(idx)
          
        for t in self.special_tokens:
            self.vocab[len(self.vocab)] = t.encode('utf-8')
        
        logger.info(
            "Done: vocab=%d, merges=%d, elapsed=%.1fs",
            len(self.vocab), len(self.merges), time.perf_counter() - t0,
        )

        return self.vocab, self.merges
        

if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    tokenizer = BPETokenizer(input_path, vocab_size, special_tokens)
    vocab, merges = tokenizer.merge()

    vocab_serializable = {str(k): v.decode("utf-8", errors="replace") for k, v in vocab.items()}
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)

    with open("merges.txt", "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(a.decode("utf-8", errors="replace") + " " + b.decode("utf-8", errors="replace") + "\n")

    logger.info("Saved vocab.json (%d entries) and merges.txt (%d lines)", len(vocab), len(merges))