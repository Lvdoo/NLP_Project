"""Train the sentence-level hand-written Transformer on OPUS-100."""
from .train_seq2seq import main

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from .translation import CHECKPOINT, ROOT

    parser = argparse.ArgumentParser(description="Entraîner le Transformer manuel sur OPUS-100")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-vocab", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "opus100" / "processed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    args = parser.parse_args()
    main(args.epochs, args.max_vocab, args.batch_size, args.output, args.data_dir,
         args.device, args.learning_rate)
