from collections.abc import Iterable, Iterator
import json
import functools
import os
import re
import shutil
import tempfile
import regex
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from cs336_basics.pretokenization_example import find_chunk_boundaries


NUM_PROCESSES = 8

@functools.lru_cache()
def gpt_byte_decoder() -> dict[int, str]:
    """构建 GPT-2 风格的「可见字符 -> 原始字节」映射，用于把 vocab.json 里的字符串还原成 bytes。"""
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
    """把 vocab/merges 文件里的可见字符串解码回它代表的原始字节序列。"""
    dec = gpt_byte_decoder()
    return bytes(dec[ch] for ch in s)

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _apply_merges(word: list[bytes], merge_ranks: dict[tuple[bytes, bytes], int]) -> list[bytes]:
    """对单个词（字节列表）反复应用 rank 最低的相邻 merge，直到没有可合并的 pair。"""
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
    """把一段文本预分词后逐词应用 merge，转成 token id 列表；整段恰为特殊 token 时直接映射。"""
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

# 每个 worker 进程的全局词表槽：由 Pool 的 initializer 在 worker 启动时填一次，
# 之后该 worker 处理的所有 task 都复用，避免把 32K 词表随每个 task 反复 pickle。
_WORKER = {}

def _init_worker(inv_vocab, merge_ranks, special_tokens):
    """Pool initializer：每个 worker 启动时只接收一次词表/merges，存进进程全局 _WORKER。"""
    _WORKER["inv_vocab"] = inv_vocab
    _WORKER["merge_ranks"] = merge_ranks
    _WORKER["special_tokens"] = special_tokens

def _encode_byte_range_core(file_path, start, end, inv_vocab, merge_ranks, special_tokens):
    """读取文件的 [start, end) 字节段，按特殊 token 切分后逐段编码成 token id 列表。"""
    with open(file_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    if special_tokens:
        pat = "(" + "|".join(re.escape(t) for t in special_tokens) + ")"
        sub_chunks = re.split(pat, text)
    else:
        sub_chunks = [text]
    out = []
    for sc in sub_chunks:
        out.extend(process_chunk((sc, inv_vocab, merge_ranks, special_tokens)))
    return out

def _encode_byte_range(args):
    """多进程 worker：args=(file_path, start, end)，词表从进程全局 _WORKER 取，编码成 token id 列表。"""
    file_path, start, end = args
    return _encode_byte_range_core(
        file_path, start, end,
        _WORKER["inv_vocab"], _WORKER["merge_ranks"], _WORKER["special_tokens"],
    )

def _encode_range_to_shard(args):
    """多进程 worker：args=(file_path, start, end, shard_path, dtype_str)，词表从 _WORKER 取；结果以原始字节写入 shard_path，返回 token 数。"""
    file_path, start, end, shard_path, dtype_str = args
    ids = _encode_byte_range_core(
        file_path, start, end,
        _WORKER["inv_vocab"], _WORKER["merge_ranks"], _WORKER["special_tokens"],
    )
    np.asarray(ids, dtype=dtype_str).tofile(shard_path)
    return shard_path, len(ids)

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]=None):
        """从 vocab、merges、special_tokens 构造分词器；特殊 token 按长度降序排，保证最长匹配优先。"""
        self.vocab = vocab
        self.inv_vocab = {v: idx for idx, v in self.vocab.items()}
        self.merges = merges
        self.merge_ranks = {merges[i]: i for i in range(len(merges))}
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merge_filepath: str, special_tokens: list[str]=None):
        """从磁盘上的 vocab.json 和 merges.txt 加载并构造分词器。"""
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

    def encode_file(self, file_path: str, num_processes: int = NUM_PROCESSES,
                    num_chunks: int = None) -> list[int]:
        """用 find_chunk_boundaries 在特殊 token 处切块，多进程并行编码整个文件，返回内存中的 token id 列表。

        num_chunks 与 num_processes 解耦：默认切 num_processes*16 块（远多于进程数），
        让 Pool 动态分配以负载均衡；块小、token 不均时不会被最慢的块拖住。
        """
        num_chunks = num_chunks or num_processes * 16
        split_token = self.special_tokens[0].encode("utf-8") if self.special_tokens else b"\n"
        with open(file_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_chunks, split_token)
        tasks = [
            (file_path, start, end)  # 瘦身：词表经 initializer 只发一次，task 不再携带
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        init_args = (self.inv_vocab, self.merge_ranks, self.special_tokens)
        with Pool(processes=num_processes, initializer=_init_worker, initargs=init_args) as pool:
            # imap 保序且能边完成边返回，配合 tqdm 显示已完成块数进度
            results = list(tqdm(pool.imap(_encode_byte_range, tasks),
                                total=len(tasks), desc="encode_file", unit="chunk"))
        return [tid for chunk_ids in results for tid in chunk_ids]

    def encode_file_to_npy(self, in_path: str, out_path: str,
                           num_processes: int = NUM_PROCESSES,
                           num_chunks: int = None) -> int:
        """并行编码整个文件并落盘为 .npy。各 worker 写自己分片，主进程按序合并，低内存。返回 token 总数。

        num_chunks 与 num_processes 解耦：默认切 num_processes*16 块（远多于进程数），
        让 Pool 动态分配以负载均衡；分片更多但单片更小，主进程峰值内存也更低。
        """
        num_chunks = num_chunks or num_processes * 16
        dtype = np.uint16 if max(self.vocab) < 2**16 else np.uint32
        dtype_str = np.dtype(dtype).str
        split_token = self.special_tokens[0].encode("utf-8") if self.special_tokens else b"\n"
        with open(in_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_chunks, split_token)

        tmp_dir = tempfile.mkdtemp(prefix="bpe_shards_")
        try:
            tasks = [
                (in_path, start, end, os.path.join(tmp_dir, f"shard_{i}.bin"), dtype_str)
                for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
            ]
            init_args = (self.inv_vocab, self.merge_ranks, self.special_tokens)
            with Pool(processes=num_processes, initializer=_init_worker, initargs=init_args) as pool:
                # imap 保序：results[i] 对应第 i 段；配合 tqdm 显示已完成分片进度
                results = list(tqdm(pool.imap(_encode_range_to_shard, tasks),
                                    total=len(tasks), desc="encode_to_npy", unit="chunk"))

            total = sum(count for _, count in results)
            out = np.lib.format.open_memmap(out_path, mode="w+", dtype=dtype, shape=(total,))
            offset = 0
            for shard_path, count in results:  # 按段顺序写入，等价于串行编码顺序
                out[offset:offset + count] = np.fromfile(shard_path, dtype=dtype)
                offset += count
            out.flush()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return total

    def encode(self, text: str) -> list[int]:
        """把一段内存中的字符串编码成 token id 列表（串行，先按特殊 token 切分再逐段编码）。"""
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
        """惰性编码一个字符串可迭代对象（如逐行读取的文件），按需 yield token id，内存恒定。"""
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
        """把 token id 列表还原成字符串（拼接各 token 的字节后按 UTF-8 解码，非法字节用替换符）。"""
        return b"".join(self.vocab[id] for id in ids).decode("utf-8", errors="replace")