import json
from tests.adapters import run_train_bpe

if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = run_train_bpe(
        input_path,
        vocab_size,
        special_tokens=special_tokens
    )

    vocab_serializable = {str(k): v.decode("utf-8", errors="replace") for k, v in vocab.items()}
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)

    with open("merges.txt", "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(a.decode("utf-8", errors="replace") + " " + b.decode("utf-8", errors="replace") + "\n")
