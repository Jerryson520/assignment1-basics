#!/bin/bash
# §7.4 OWT 生成：owt-baseline checkpoint + OWT tokenizer（vocab 32000）
uv run python -m cs336_basics.decoding \
  --checkpoint /root/autodl-tmp/cs336/data/checkpoints/owt-baseline/ckpt_final.pt \
  --vocab   /root/autodl-tmp/cs336/tokenizer/owt/vocab.json \
  --merges  /root/autodl-tmp/cs336/tokenizer/owt/merges.txt \
  --device cuda \
  --vocab-size 32000 \
  --max-new-tokens 256 \
  --temperature 0.8 --top-p 0.9
