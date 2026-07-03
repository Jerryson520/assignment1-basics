uv run python -m cs336_basics.decoding \
  --checkpoint /root/autodl-tmp/cs336/data/checkpoints/lr_best/ckpt_final.pt \
  --vocab   /root/autodl-tmp/cs336/tokenizer/tinystories/vocab.json \
  --merges  /root/autodl-tmp/cs336/tokenizer/tinystories/merges.txt \
  --device cuda \
  --max-new-tokens 300 \
  --temperature 0.5 --top-p 0.9