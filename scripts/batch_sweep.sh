#!/bin/bash
TOKENS=327680000   # 全量 token 预算
CTX=256
for batch in 16 64 128 256 512; do
	lr=$(awk -v b=$batch 'BEGIN{printf "%.4e", 2e-3*sqrt(b/64)}')
	lrmin=$(awk -v l=$lr 'BEGIN{printf "%.4e", l/10}')
	iters=$(awk -v t=$TOKENS -v b=$batch -v c=$CTX 'BEGIN{printf "%d", t/(b*c)}')
	echo "===== Batch Size = $batch  (lr=$lr, lr_min=$lrmin, max-iters=$iters) ====="
	uv run python -m cs336_basics.train \
		--train-data /root/autodl-tmp/cs336/tokenizer/tinystories/ts_train_ids.npy \
		--valid-data /root/autodl-tmp/cs336/tokenizer/tinystories/ts_valid_ids.npy \
		--device cuda \
		--vocab-size 10000 --context-length 256 \
		--d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 --theta 10000 \
		--batch-size $batch --max-iters $iters \
		--lr-max $lr --lr-min $lrmin --warmup-iters 200 \
		--eval-interval 500 --log-interval 50 \
        --checkpoint-interval 1000000 \
		--checkpoint-dir /root/autodl-tmp/cs336/data/checkpoints/bs_$batch \
		--wandb-project cs336-assignment1 --wandb-run-name bs-$batch
done