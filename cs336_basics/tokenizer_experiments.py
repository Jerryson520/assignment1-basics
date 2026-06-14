from bpe_tokenizer import BPETokenizer
import random
import logging
import time
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
            if not line.strip():
                continue
            if line.strip() == sep:
                flush()
            else:
                buff.append(line)
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    owt_data = "data/owt_train.txt"
    owt_vocab, owt_merges = "results/owt/vocab.json", "results/owt/merges.txt"
    tinystory_data = "data/TinyStoriesV2-GPT4-train.txt"
    tinystory_vocab, tinystory_merges = "results/tinystories/vocab.json", "results/tinystories/merges.txt"
    special_tokens = ["<|endoftext|>"]

    # ====== stage1: bytes/token ratio ======
    logger.info("===== Stage 1: compression ratio =====")
    owt_10 = shuffle_samples(owt_data)
    tinystory_10 = shuffle_samples(tinystory_data)

    logger.info("Loading OpenWebText tokenizer from %s / %s", owt_vocab, owt_merges)
    owt_tokenizer = BPETokenizer.from_files(owt_vocab, owt_merges, special_tokens)
    logger.info("Loading TinyStories tokenizer from %s / %s", tinystory_vocab, tinystory_merges)
    tinystory_tokenizer = BPETokenizer.from_files(tinystory_vocab, tinystory_merges, special_tokens)

    owt_compression_ratio = mean_compression_ratio("OpenWebText", owt_10, owt_tokenizer)
    tinystory_compression_ratio = mean_compression_ratio("TinyStories", tinystory_10, tinystory_tokenizer)

    logger.info("OpenWebText compression ratio: %.3f bytes/token", owt_compression_ratio)
    logger.info("TinyStories compression ratio: %.3f bytes/token", tinystory_compression_ratio)

    # ====== stage2: cross-tokenizer ======
    owt_compression_ratio = mean_compression_ratio("OpenWebText", owt_10, tinystory_tokenizer)
    tinystory_compression_ratio = mean_compression_ratio("TinyStories", tinystory_10, owt_tokenizer)

    logger.info("OpenWebText compression ratio: %.3f bytes/token", owt_compression_ratio)
    logger.info("TinyStories compression ratio: %.3f bytes/token", tinystory_compression_ratio)