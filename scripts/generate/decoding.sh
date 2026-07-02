# !/bin/bash
uv run python -m cs336_basics.decoding \
  --checkpoint checkpoints/baseline/ckpt_final.pt --device cuda \
  --temperature 0.8 --top-p 0.9 --max-new-tokens 256
