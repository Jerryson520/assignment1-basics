import torch
import torch.nn as nn
from collections.abc import Callable, Iterable
from typing import Optional
import math
import matplotlib.pyplot as plt
import pandas as pd

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[callable]=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t+1) * grad
                state["t"] = t + 1

        return loss

if __name__ == "__main__":
    lrs = [1e1, 1e2, 1e3]
    n_steps = 10
 
    # 用同一个初始权重，保证不同 lr 之间公平对比
    torch.manual_seed(0)
    init_weights = 5 * torch.randn((10, 10))
 
    history = {}  # lr -> 每步的 loss 列表
    for lr in lrs:
        print(f"learning rate is {lr}")
        weights = torch.nn.Parameter(init_weights.clone())
        opt = SGD([weights], lr=lr)
 
        losses = []
        for t in range(n_steps):
            opt.zero_grad()
            loss = (weights ** 2).mean()
            losses.append(loss.cpu().item())
            print(loss.cpu().item())
            loss.backward()
            opt.step()
        history[lr] = losses
 
    # ---- 画图 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for lr in lrs:
        ax.plot(range(n_steps), history[lr], marker="o", label=f"lr = {lr:g}")
 
    ax.set_yscale("log")  # 对数轴，否则 lr=1e3 的发散会压扁其他曲线
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss  (log scale)")
    ax.set_title("SGD loss vs. iteration for different learning rates")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)
 
    fig.tight_layout()
    fig.savefig("lr_tuning.png", dpi=150)
    print("\nSaved figure to lr_tuning.png")
    plt.show()
    