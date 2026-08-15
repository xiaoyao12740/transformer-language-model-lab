import argparse

from src.generation.sampling import generate
from src.training.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", default="First Citizen:")
    parser.add_argument("--length", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    model, tokenizer, _ = load_checkpoint(args.checkpoint)
    ids = generate(model, tokenizer.encode(args.prompt), args.length, args.temperature, args.top_k)
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()

