#!/bin/bash
# §7.3 RMSNorm 消融（对照组复用 lr_best；lr-min 用默认 3e-5 与其一致）
uv run python -m cs336_basics.train \
    --train-data /root/autodl-tmp/cs336/tokenizer/tinystories/ts_train_ids.npy \
    --valid-data /root/autodl-tmp/cs336/tokenizer/tinystories/ts_valid_ids.npy \
    --device cuda \
    --vocab-size 10000 --context-length 256 \
    --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 --theta 10000 \
    --batch-size 64 --max-iters 20000 \
    --lr-max 2e-3 --warmup-iters 200 \
    --eval-interval 5 --log-interval 50 \
    --checkpoint-interval 1000000 \
    --checkpoint-dir /root/autodl-tmp/cs336/data/checkpoints/norm_off \
    --wandb-project cs336-assignment1 --wandb-run-name no-rmsnorm-diverge \
    --no-rmsnorm