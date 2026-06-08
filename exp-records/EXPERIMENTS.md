# BPE Training Experiments

## 2026-06-08 — multiprocessing imap pre-tokenize

- **Data**: `data/TinyStoriesV2-GPT4-valid.txt`
- **vocab_size**: 10000
- **special_tokens**: `["<|endoftext|>"]`
- **Implementation**: `multiprocessing.Pool` + `imap` over `doc_generate` generator; worker = `pre_tokenize`
- **Commit**: a158843 (working tree dirty: cs336_basics/train_bpe.py)

### Results
- Initial pairs: 932 distinct
- Target merges: 9743
- **Total elapsed: 40.1s**
  - Count pairs: ~negligible (13110 words)
  - BPE merges loop: ~33s

### Notes
- Well under the 2-minute target.
- No single-process baseline recorded yet — consider running once with the old `_pre_tokenize` for a clean comparison.
