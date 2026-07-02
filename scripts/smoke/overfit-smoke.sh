#!/bin/bash
uv run python -m cs336_basics.train \
  --train-data /root/autodl-tmp/cs336/tokenizer/tinystories/ts_train_ids.npy \
  --valid-data /root/autodl-tmp/cs336/tokenizer/tinystories/ts_valid_ids.npy \
  --device cuda \
  --vocab-size 10000 --context-length 256 \
  --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 --theta 10000 \
  --batch-size 64 \
  --overfit-one-batch \
  --max-iters 500 --log-interval 10 \
  --eval-interval 999999 \
  --checkpoint-dir checkpoints/smoke \
  --wandb-project cs336-assignment1 --wandb-run-name overfit-smoke
