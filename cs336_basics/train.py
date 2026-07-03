import argparse
import math
import os
import time
import numpy as np
import torch
from cs336_basics.adamw import AdamW
from cs336_basics.full_transformer import TransformerLM
from einops import rearrange, einsum

def get_batch(dataset, batch_size, context_length, device):
    start = np.random.randint(0, len(dataset)-context_length, batch_size)
    inputs = np.stack([dataset[s:s+context_length] for s in start])
    labels = np.stack([dataset[s+1:s+context_length+1] for s in start])
    return torch.from_numpy(inputs).long().to(device), torch.from_numpy(labels).long().to(device)

def cross_entropy(logits, targets):
    logits = rearrange(logits, "b s v -> (b s) v")
    targets = rearrange(targets, "b s -> (b s)")
    max_val = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_val

    logsumexp = torch.log(torch.exp(shifted).sum(dim=-1))
    correct = torch.gather(shifted, -1, targets.unsqueeze(-1)).squeeze(-1)
    return (logsumexp - correct).mean()

def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    total = torch.sqrt(sum(g.pow(2).sum() for g in grads))

    if total > max_l2_norm:
        factor = max_l2_norm / (total + eps)
        for g in grads:
            g.mul_(factor)

def lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_iters):
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    elif warmup_iters <= it <= cosine_iters:
        frac = (it - warmup_iters) / (cosine_iters - warmup_iters)
        return min_lr + 0.5 * (1 + math.cos(frac * math.pi)) * (max_lr - min_lr)
    else:
        return min_lr

def save_checkpoint(model, optimizer, iteration, out):
    outputs = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(outputs, out)

def load_checkpoint(src, model, optimizer):
    ckpt = torch.load(src, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt['iteration']

@torch.no_grad()
def evaluate(model, data, batch_size, context_length, device, n_batches):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, context_length, device)
        losses.append(cross_entropy(model(x), y).item())
    model.train()
    return sum(losses) / len(losses)

def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--valid-data", type=str, required=True)
    # LLM
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--theta", type=int, default=10000)
    # optimizer
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lr-max", type=float, default=3e-4)
    parser.add_argument("--lr-min", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=200)

    # training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--overfit-one-batch", action="store_true")
    parser.add_argument("--no-rmsnorm", action="store_true")
    parser.add_argument("--no-prenorm", action="store_true")
    parser.add_argument("--no-rope", action="store_true")

    # logging / checkpoint
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoint")
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None)

    # wandb
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()

    train_data = np.load(args.train_data, mmap_mode="r")
    valid_data = np.load(args.valid_data, mmap_mode="r")

    model = TransformerLM(
        vocab_size = args.vocab_size,
        context_length = args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff = args.d_ff,
        theta=args.theta,
        use_rmsnorm=not args.no_rmsnorm,
        use_prenorm=not args.no_prenorm,
        use_rope=not args.no_rope,
    )
    model = model.to(args.device)
    optimizer = AdamW(
        model.parameters(), 
        lr=args.lr_max, 
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    start_iter = 0 if args.resume is None else load_checkpoint(args.resume, model, optimizer)
    if args.overfit_one_batch:
        fixed_batch = get_batch(train_data, args.batch_size, args.context_length, args.device)
    start_time = time.time()

    for it in range(start_iter, args.max_iters):
        lr = lr_cosine_schedule(it, args.lr_max, args.lr_min, args.warmup_iters, args.max_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr
        
        if args.overfit_one_batch:
            x, y = fixed_batch
        else:
            x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)
        loss = cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        gradient_clipping(model.parameters(), args.grad_clip)

        optimizer.step()

        if it % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"iter {it} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.1f}s")
            if use_wandb:
                wandb.log({
                    "train/loss": loss.item(), 
                    "lr": lr,
                    "wallclock_sec": elapsed
                }, step=it)

        if it > 0 and it % args.eval_interval == 0:
            elapsed = time.time() - start_time
            vloss = evaluate(model, valid_data, args.batch_size, args.context_length, args.device, args.eval_batches)
            print(f"iter {it} | valid loss {vloss:.4f} | {elapsed:.1f}s")
            if use_wandb:
                wandb.log({
                    "valid/loss": vloss, 
                    "lr": lr,
                    "wallclock_sec": elapsed
                }, step=it)
        
        if it > 0 and it % args.checkpoint_interval == 0:
            path = os.path.join(args.checkpoint_dir, f"ckpt_{it}.pt")
            save_checkpoint(model, optimizer, it, path)
            print(f"saved checkpoint -> {path}")
    
    final = os.path.join(args.checkpoint_dir, "ckpt_final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final)
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()