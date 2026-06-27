import torch
import argparse
from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.full_transformer import TransformerLM

@torch.no_grad()
def topp_sampling(
    model: TransformerLM, 
    tokenizer: BPETokenizer, 
    prompt: str, 
    max_new_tokens:int, 
    temperature:float, 
    top_p: float, 
    eot_token_id: int, 
    device: torch.device, 
    eps=10e-6
):
    init_input = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    x = init_input
    while x[0, -1] != eot_token_id and x.shape[-1] <= max_new_tokens + init_input.shape[-1]:
        logits = model(x)[0, -1]
        vocab_size = logits.shape[-1]
        max_val = logits.max(dim=-1, keepdim=True).values
        y = (logits - max_val) / (temperature + eps)
        denominator = torch.sum(torch.exp(y))
        probs = torch.exp(y) / denominator
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        probs = torch.cumsum(sorted_probs, dim=-1) - sorted_probs

        i = 0
        while i < len(probs):
            if probs[i] > top_p:
                break
            i += 1
        
        sorted_indices = sorted_indices[:i]
        sorted_probs = sorted_probs[:i] / torch.sum(sorted_probs[:i])

        choice = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices[choice]

        x = torch.cat([x, next_token.unsqueeze(0)], dim=-1)

    return tokenizer.decode(x[0].tolist())


# ============================================================================
# 测试入口：加载 tokenizer + checkpoint，对几个示例 prompt 生成文本
# 运行： uv run python -m cs336_basics.decoding --checkpoint <ckpt.pt>
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="训练好的 ckpt 路径")
    parser.add_argument("--vocab", type=str, default="results/tinystories/vocab.json")
    parser.add_argument("--merges", type=str, default="results/tinystories/merges.txt")
    parser.add_argument("--device", type=str, default="cpu")
    # 模型结构需和训练时一致
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--theta", type=int, default=10000)
    # 采样超参
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    # 1) tokenizer（带特殊 token，才能识别 <|endoftext|>）
    tokenizer = BPETokenizer.from_files(args.vocab, args.merges, ["<|endoftext|>"])
    eot_token_id = tokenizer.encode("<|endoftext|>")[0]  # TinyStories 里是 9999

    # 2) 模型 + 载入权重
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
    ).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 3) 示例 prompt
    prompts = [
        "Once upon a time",
        "The little dog",
        "Lily went to the park and",
    ]

    for prompt in prompts:
        out = topp_sampling(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eot_token_id=eot_token_id,
            device=args.device,
        )
        print("=" * 60)
        print(f"[prompt] {prompt}")
        print(f"[output] {out}")