"""Training entry point for the manual Transformer on bilingual word pairs."""
import argparse

from .train_multilingual import main
from .translation import CHECKPOINT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the French word translator")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--max-vocab", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default=str(CHECKPOINT))
    parser.add_argument("--data-dir", default="data/muse/processed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=0.005)
    args = parser.parse_args()
    main(args.epochs, args.max_vocab, args.batch_size, args.output, args.data_dir,
         args.device, args.learning_rate)
