from collections.abc import Iterable, Iterator
import json
import functools
import re
import regex
import multiprocessing
from multiprocessing import Pool


NUM_PROCESSES = 8

@functools.lru_cache()
def gpt_byte_decoder() -> dict[int, str]:
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
    return {chr(c): b for b, c in zip(bs,cs)}

def decode_token(s: str) -> bytes:
    dec = gpt_byte_decoder()
    return bytes(dec[ch] for ch in s)

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _apply_merges(word: list[bytes], merge_ranks: dict[tuple[bytes, bytes], int]) -> list[bytes]:
    while len(word) >= 2:
        best, best_rank = None, None
        for pair in zip(word[:-1], word[1:]):
            r = merge_ranks.get(pair)
            if r is not None and (best_rank is None or r < best_rank):
                best, best_rank = pair, r
        if best is None:
            break
        a, b = best
        new, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new.append(a + b)
                i += 2
            else:
                new.append(word[i])
                i += 1
        word = new
    return word

def process_chunk(args):
    chunk, inv_vocab, merge_ranks, special_tokens = args
    if not chunk: return []
    if special_tokens and chunk in special_tokens: 
        return [inv_vocab[chunk.encode("utf-8")]]
    out = []
    for match in regex.finditer(PAT, chunk):
        word = [bytes([x]) for x in match.group().encode("utf-8")]
        word = _apply_merges(word, merge_ranks)
        out.extend([inv_vocab[b] for b in word])
    return out

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]=None):
        self.vocab = vocab
        self.inv_vocab = {v: idx for idx, v in self.vocab.items()}
        self.merges = merges
        self.merge_ranks = {merges[i]: i for i in range(len(merges))}
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merge_filepath: str, special_tokens: list[str]=None):
        with open(vocab_filepath, "r") as f:
            readable = json.load(f)
            vocab = {int(idx): decode_token(v) for idx, v in readable.items()}
        merges = []
        with open(merge_filepath, "r") as f:
            for line in f:
                line = line.strip()
                s1, s2 = line.split(" ", 1)
                merges.append((decode_token(s1), decode_token(s2)))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            pat = "(" + "|".join(re.escape(token) for token in self.special_tokens) + ")"
            chunks = re.split(pat, text)
        else:
            chunks = [text]
        out = []
        for chunk in chunks:
            args = (chunk, self.inv_vocab, self.merge_ranks, self.special_tokens)
            out.extend(process_chunk(args))
        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for iter_ in iterable:
            if self.special_tokens:
                pat = "(" + "|".join(re.escape(token) for token in self.special_tokens) + ")"
                chunks = re.split(pat, iter_)
            else:
                chunks = [iter_]
            for chunk in chunks:
                for token_id in process_chunk((chunk, self.inv_vocab, self.merge_ranks, self.special_tokens)):
                    yield token_id
                

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[id] for id in ids).decode("utf-8", errors="replace")

