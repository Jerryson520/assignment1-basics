from bpe_tokenizer import BPETokenizer
import random
import logging
import time
import os
import tempfile
from multiprocessing import Pool
logger = logging.getLogger("tokenizer_experiments")


def shuffle_samples(file_input_path: str, k: int = 10, seed: int = 42, sep: str="<|endoftext|>"):
    """
    蓄水池算法：单遍扫描，从文件中等概率抽取 k 行（非空行）。
    """
    logger.info("Sampling %d docs from %s (seed=%d, sep=%r) ...", k, file_input_path, seed, sep)
    t0 = time.perf_counter()
    samples = []
    rng = random.Random(seed)
    n = 0  # 已处理的有效行数
    buff = []

    def flush():
        nonlocal n, buff
        doc = "".join(buff).strip()
        buff.clear()
        if not doc: return
        if len(samples) < k:
            samples.append(doc)
        else:
            j = rng.randint(0, n)
            if j < k:
                samples[j] = doc
        n += 1

    with open(file_input_path, "r", encoding="utf-8") as f:
        for line in f:
            # sep 可能内联在行内（如 OWT），按 sep 切分；每个 sep 标记一篇文档结束
            parts = line.split(sep)
            for i, part in enumerate(parts):
                buff.append(part)
                if i < len(parts) - 1:
                    flush()
        flush()
    logger.info(
        "Sampled %d/%d docs from %s in %.2fs",
        len(samples), n, file_input_path, time.perf_counter() - t0,
    )
    return samples


def compression_ratio(args):
    text, encoder = args
    total_bytes = len(text.encode("utf-8"))
    tokens = len(encoder.encode(text))
    ratio = total_bytes / tokens if tokens else 0.0
    logger.debug(
        "bytes=%d tokens=%d ratio=%.3f bytes/token", total_bytes, tokens, ratio
    )
    return ratio


def mean_compression_ratio(name: str, samples: list[str], encoder, use_mp: bool = False) -> float:
    logger.info(
        "Computing compression ratio for %s over %d samples (use_mp=%s) ...",
        name, len(samples), use_mp,
    )
    t0 = time.perf_counter()
    tasks = [(s, encoder) for s in samples]
    if use_mp:
        with Pool() as pool:
            ratios = pool.map(compression_ratio, tasks)
    else:
        ratios = [compression_ratio(t) for t in tasks]
    # 子进程里的 logger.debug 不会输出，这里在主进程逐条记录明细
    total_bytes = 0
    for i, (s, r) in enumerate(zip(samples, ratios)):
        nbytes = len(s.encode("utf-8"))
        total_bytes += nbytes
        logger.debug("%s[%d]: bytes=%d ratio=%.3f bytes/token", name, i, nbytes, r)
    mean = sum(ratios) / len(ratios) if ratios else 0.0
    elapsed = time.perf_counter() - t0
    logger.info(
        "%s: mean=%.3f min=%.3f max=%.3f bytes/token over %d samples, %d bytes total (%.2fs)",
        name, mean, min(ratios), max(ratios), len(ratios), total_bytes, elapsed,
    )
    return mean


PILE_BYTES = 825 * 1024**3  # 825 GB

def measure_throughput(name: str, file_path: str, encoder, nchars: int = 10_000_000,
                       num_processes: int = 8) -> float:
    """多进程吞吐：截取 nchars 字符到临时文件，用 encode_file 并行编码计时。"""
    # encode_file 接收的是文件路径，所以先把样本截到临时文件
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read(nchars)
    nbytes = len(text.encode("utf-8"))
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tf:
        tf.write(text)
        tmp_path = tf.name

    try:
        t0 = time.perf_counter()
        n_tokens = len(encoder.encode_file(tmp_path, num_processes=num_processes))
        elapsed = time.perf_counter() - t0
    finally:
        os.remove(tmp_path)

    bps = nbytes / elapsed
    pile_seconds = PILE_BYTES / bps
    print(f"{name}: {bps/1e6:.2f} MB/s "
          f"({nbytes} bytes, {n_tokens} tokens, {elapsed:.2f}s, {num_processes} procs) | "
          f"Pile(825GB) ≈ {pile_seconds/3600:.1f} h ({pile_seconds/86400:.1f} days)")
    return bps

def encode_to_npy(name: str, tokenizer, in_path: str, out_path: str,
                  num_processes: int = 8) -> int:
    """带日志地把整个文件并行编码并落盘为 .npy，记录耗时、token 数、吞吐与文件大小。"""
    in_bytes = os.path.getsize(in_path)
    logger.info("[%s] Encoding %s (%.2f GB) -> %s with %d procs ...",
                name, in_path, in_bytes / 1024**3, out_path, num_processes)
    t0 = time.perf_counter()
    total = tokenizer.encode_file_to_npy(in_path, out_path, num_processes=num_processes)
    elapsed = time.perf_counter() - t0
    out_bytes = os.path.getsize(out_path)
    logger.info(
        "[%s] Done: %d tokens in %.1fs (%.2f MB/s) | %s = %.2f GB (%.3f bytes/token)",
        name, total, elapsed, in_bytes / 1e6 / elapsed,
        out_path, out_bytes / 1024**3, in_bytes / total if total else 0.0,
    )
    return total


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    owt_data = "data/owt_train.txt"
    owt_valid = "data/owt_valid.txt"
    owt_vocab, owt_merges = "results/owt/vocab.json", "results/owt/merges.txt"
    tinystory_data = "data/TinyStoriesV2-GPT4-train.txt"
    tinystory_valid = "data/TinyStoriesV2-GPT4-valid.txt"
    tinystory_vocab, tinystory_merges = "results/tinystories/vocab.json", "results/tinystories/merges.txt"
    special_tokens = ["<|endoftext|>"]

    # stage1/2 才需要采样；只跑 stage4 时跳过，省去几十秒采样耗时
    # logger.info("===== Stage 1: compression ratio =====")
    # owt_10 = shuffle_samples(owt_data)
    # tinystory_10 = shuffle_samples(tinystory_data)

    logger.info("Loading OpenWebText tokenizer from %s / %s", owt_vocab, owt_merges)
    owt_tokenizer = BPETokenizer.from_files(owt_vocab, owt_merges, special_tokens)
    logger.info("Loading TinyStories tokenizer from %s / %s", tinystory_vocab, tinystory_merges)
    tinystory_tokenizer = BPETokenizer.from_files(tinystory_vocab, tinystory_merges, special_tokens)

    # # ====== stage1: bytes/token ratio ======
    # owt_compression_ratio = mean_compression_ratio("OpenWebText", owt_10, owt_tokenizer)
    # tinystory_compression_ratio = mean_compression_ratio("TinyStories", tinystory_10, tinystory_tokenizer)

    # logger.info("OpenWebText compression ratio: %.3f bytes/token", owt_compression_ratio)
    # logger.info("TinyStories compression ratio: %.3f bytes/token", tinystory_compression_ratio)

    # # ====== stage2: cross-tokenizer ======
    # owt_compression_ratio = mean_compression_ratio("OpenWebText", owt_10, tinystory_tokenizer)
    # tinystory_compression_ratio = mean_compression_ratio("TinyStories", tinystory_10, owt_tokenizer)

    # logger.info("OpenWebText compression ratio: %.3f bytes/token", owt_compression_ratio)
    # logger.info("TinyStories compression ratio: %.3f bytes/token", tinystory_compression_ratio)

    # # ====== stage3: throughput & Pile estimate ======
    # logger.info("===== Stage 3: throughput =====")
    # measure_throughput("OpenWebText", owt_data, owt_tokenizer)
    # measure_throughput("TinyStories", tinystory_data, tinystory_tokenizer)

    # ====== stage4: 编码训练/验证集为 token id 序列并落盘 ======
    logger.info("===== Stage 4: encode datasets to .npy =====")
    encode_to_npy("TinyStories/valid", tinystory_tokenizer, tinystory_valid, "results/ts_valid_ids.npy", num_processes=8)
    encode_to_npy("OpenWebText/valid", owt_tokenizer, owt_valid, "results/owt_valid_ids.npy", num_processes=8)

    encode_to_npy("OpenWebText/train", owt_tokenizer, owt_data, "results/owt_train_ids.npy", num_processes=8)
    encode_to_npy("TinyStories/train", tinystory_tokenizer, tinystory_data, "results/ts_train_ids.npy", num_processes=8)
