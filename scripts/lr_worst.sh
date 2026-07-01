#!/bin/bash
lr=1
echo "===== LR = $lr ====="
uv run python -m cs336_basics.train \
    --train-data /root/autodl-tmp/cs336/tokenizer/tinystories/ts_train_ids.npy \
    --valid-data /root/autodl-tmp/cs336/tokenizer/tinystories/ts_valid_ids.npy \
    --device cuda \
    --vocab-size 10000 --context-length 256 \
    --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 --theta 10000 \
    --batch-size 64 --max-iters 2000 \
    --lr-max $lr --warmup-iters 200 \
    --eval-interval 500 --log-interval 50 \
    --checkpoint-dir checkpoints/lr_$lr \
    --wandb-project cs336-assignment1 --wandb-run-name lr-$lr
