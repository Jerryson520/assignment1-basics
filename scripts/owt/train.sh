#!/bin/bash
# §7.4 OpenWebText 主实验：vocab=32000，其余沿用 TinyStories baseline 配置。
lr=2e-3
echo "===== OWT baseline  LR = $lr ====="
uv run python -m cs336_basics.train \
    --train-data /root/autodl-tmp/cs336/tokenizer/owt/owt_train_ids.npy \
    --valid-data /root/autodl-tmp/cs336/tokenizer/owt/owt_valid_ids.npy \
    --device cuda \
    --vocab-size 32000 --context-length 256 \
    --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 --theta 10000 \
    --batch-size 64 --max-iters 20000 \
    --lr-max $lr --warmup-iters 200 \
    --eval-interval 500 --log-interval 50 \
    --checkpoint-interval 1000000 \
    --checkpoint-dir /root/autodl-tmp/cs336/data/checkpoints/owt-baseline \
    --wandb-project cs336-assignment1 --wandb-run-name owt-baseline
