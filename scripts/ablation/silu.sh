#!/bin/bash
# §7.3 SwiGLU vs SiLU 消融：无门控 SiLU FFN（--no-swiglu），d_ff=4*d_model=2048 匹配参数量。
# 对照组复用 lr_best（SwiGLU, d_ff=1344）。
uv run python -m cs336_basics.train \
    --train-data /root/autodl-tmp/cs336/tokenizer/tinystories/ts_train_ids.npy \
    --valid-data /root/autodl-tmp/cs336/tokenizer/tinystories/ts_valid_ids.npy \
    --device cuda \
    --vocab-size 10000 --context-length 256 \
    --d-model 512 --num-layers 4 --num-heads 16 --d-ff 2048 --theta 10000 \
    --batch-size 64 --max-iters 20000 \
    --lr-max 2e-3 --warmup-iters 200 \
    --eval-interval 500 --log-interval 50 \
    --checkpoint-interval 1000000 \
    --checkpoint-dir /root/autodl-tmp/cs336/data/checkpoints/silu \
    --wandb-project cs336-assignment1 --wandb-run-name silu \
    --no-swiglu
