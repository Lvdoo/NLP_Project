"""Train the hand-written Transformer on OPUS-100 parallel sentences."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from .model import Transformer
from .bpe_tokenizer import load as load_bpe
from .translation import CHECKPOINT, ROOT

DATA = ROOT / "data" / "opus100" / "processed"
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>", "<lang_en>", "<lang_fr>", "<lang_de>", "<lang_es>"]
PAD, BOS, EOS, UNK = range(4)
LANG_ID = {lang: i + 4 for i, lang in enumerate(("en", "fr", "de", "es"))}
MAX_LENGTH = 96


def read_rows(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_vocab(path, max_vocab, enc):
    counts = Counter()
    for row in read_rows(path):
        counts.update(enc.encode(row["source_text"], disallowed_special=()))
        counts.update(enc.encode(row["target_text"], disallowed_special=()))
    return [int(t) for t, _ in counts.most_common(max_vocab - len(SPECIALS))]


def encode_row(row, bpe_to_id):
    src_pieces = [bpe_to_id.get(x, UNK) for x in TOKENIZER.encode(row["source_text"], disallowed_special=())]
    tgt_pieces = [bpe_to_id.get(x, UNK) for x in TOKENIZER.encode(row["target_text"], disallowed_special=())]
    if len(src_pieces) > MAX_LENGTH - 3 or len(tgt_pieces) > MAX_LENGTH - 3:
        return None
    return ([BOS, LANG_ID[row["source_lang"]], *src_pieces, EOS],
            [BOS, LANG_ID[row["target_lang"]], *tgt_pieces, EOS])


def make_batches(encoded, batch_size, shuffle=False):
    indexes = list(range(len(encoded)))
    if shuffle:
        random.shuffle(indexes)
    for start in range(0, len(indexes), batch_size):
        batch = [encoded[i] for i in indexes[start:start + batch_size]]
        src_len = max(len(x[0]) for x in batch)
        tgt_len = max(len(x[1]) for x in batch)
        src = torch.full((len(batch), src_len), PAD, dtype=torch.long)
        tgt = torch.full((len(batch), tgt_len), PAD, dtype=torch.long)
        for i, (s, t) in enumerate(batch):
            src[i, :len(s)] = torch.tensor(s)
            tgt[i, :len(t)] = torch.tensor(t)
        yield src, tgt


def encode_split(path, bpe_to_id):
    result = []
    for row in read_rows(path):
        encoded = encode_row(row, bpe_to_id)
        if encoded:
            result.append(encoded)
    return result


def evaluate(model, rows, batch_size, device):
    model.eval()
    loss_sum = token_count = correct = exact = sentence_count = 0
    with torch.inference_mode():
        for src, tgt in make_batches(rows, batch_size):
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])
            gold = tgt[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), gold.reshape(-1),
                                   ignore_index=PAD, reduction="sum")
            mask = gold.ne(PAD)
            pred = logits.argmax(-1)
            loss_sum += loss.item()
            token_count += mask.sum().item()
            correct += ((pred == gold) & mask).sum().item()
            exact += ((pred.eq(gold) | ~mask).all(1)).sum().item()
            sentence_count += len(src)
    return {"loss": loss_sum / max(1, token_count), "token_accuracy": correct / max(1, token_count),
            "exact_match": exact / max(1, sentence_count)}


def main(epochs=10, max_vocab=16000, batch_size=32, output=CHECKPOINT,
         data_dir=DATA, device_name="auto", learning_rate=0.0005):
    global TOKENIZER
    try:
        TOKENIZER = load_bpe(ROOT / "data" / "opus100" / "raw" / "cl100k_base.tiktoken")
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    random.seed(17)
    torch.manual_seed(17)
    torch.set_num_threads(1)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA demandé mais aucun GPU CUDA n’est visible par PyTorch.")
    paths = {s: Path(data_dir) / f"{s}.jsonl" for s in ("train", "validation", "test")}
    if any(not p.is_file() for p in paths.values()):
        raise SystemExit("Données OPUS-100 absentes : lancez python -m src.prepare_opus100.")
    bpe_ids = build_vocab(paths["train"], max_vocab, TOKENIZER)
    bpe_to_id = {token: i + len(SPECIALS) for i, token in enumerate(bpe_ids)}
    vocab_size = len(SPECIALS) + len(bpe_ids)
    encoded = {split: encode_split(path, bpe_to_id) for split, path in paths.items()}
    if any(not encoded[s] for s in ("train", "validation", "test")):
        raise SystemExit("Une partition est vide après le filtrage des phrases trop longues.")
    config = {"d_model": 128, "heads": 4, "d_ff": 256, "layers": 2}
    model = Transformer(vocab_size, vocab_size, pad_id=PAD, **config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(f"device={device} vocab={max_vocab} train={len(encoded['train'])} "
          f"validation={len(encoded['validation'])} test={len(encoded['test'])}")
    best_loss, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        loss_sum = tokens = 0
        for step, (src, tgt) in enumerate(make_batches(encoded["train"], batch_size, shuffle=True), 1):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(src, tgt[:, :-1])
            gold = tgt[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), gold.reshape(-1),
                                   ignore_index=PAD)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = gold.ne(PAD).sum().item()
            loss_sum += loss.item() * n
            tokens += n
            if step % 200 == 0:
                print(f"epoch {epoch+1}/{epochs} batch={step} train_loss_so_far={loss_sum/max(1,tokens):.4f}",
                      flush=True)
        val = evaluate(model, encoded["validation"], batch_size, device)
        print(f"epoch {epoch+1}/{epochs} train_loss={loss_sum/max(1,tokens):.4f} "
              f"val_loss={val['loss']:.4f} val_token_acc={val['token_accuracy']:.4f} "
              f"val_exact={val['exact_match']:.4f}")
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(device)
    test = evaluate(model, encoded["test"], batch_size, device)
    print(f"test_loss={test['loss']:.4f} test_token_acc={test['token_accuracy']:.4f} "
          f"test_exact={test['exact_match']:.4f}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "bpe_ids": bpe_ids, "vocab_size": vocab_size,
                "config": config, "languages": ["en", "fr", "de", "es"],
                "targets_from_french": ["en", "de", "es"], "pairs": ["fr-en", "de-en", "en-es"],
                "tokenizer_file": "data/opus100/raw/cl100k_base.tiktoken",
                "max_length": MAX_LENGTH, "seed": 17, "best_validation_loss": best_loss,
                "test_metrics": test, "train_rows": len(encoded["train"])}, output)
    print(f"checkpoint={output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-vocab", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    args = parser.parse_args()
    main(args.epochs, args.max_vocab, args.batch_size, args.output, args.data_dir,
         args.device, args.learning_rate)
