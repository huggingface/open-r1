import argparse
from transformers import AutoConfig
from math import gcd

def get_tensor_parallel_size(model_name: str, revision: str = None, default_tp: int = 8) -> int:
    try:
        config = AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=True)
        num_heads = getattr(config, 'num_attention_heads', None)

        if num_heads is not None and num_heads % default_tp != 0:
            tp = gcd(num_heads, default_tp)
            return max(tp, 1)
        else:
            return default_tp
    except Exception as e:
        print(f"Warning: Failed to fetch config for {model_name}@{revision}: {e}")
        return default_tp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name or path")
    parser.add_argument("--revision", type=str, default=None, help="Model revision if applicable")
    parser.add_argument("--default_tp", type=int, default=8, help="Default TP size (usually GPUs per node)")

    args = parser.parse_args()

    tp = get_tensor_parallel_size(args.model_name, args.revision, args.default_tp)
    print(tp)
