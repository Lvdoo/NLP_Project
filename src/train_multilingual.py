"""Train the manual encoder-decoder Transformer on MUSE word pairs."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .model import Transformer
from .translation import BOS, EOS, PAD, UNK, CHECKPOINT, ROOT, tokenize

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>", "<lang_en>", "<lang_fr>",
            "<lang_de>", "<lang_es>"]
DATA = ROOT / "data" / "muse" / "processed"


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def make_vocab(train_path, max_vocab):
    counts = Counter()
    for row in read_jsonl(train_path):
        counts.update(tokenize(row["source_text"]))
        counts.update(tokenize(row["target_text"]))
    return SPECIALS + [t for t, _ in counts.most_common(max(0, max_vocab - len(SPECIALS)))
                        if t not in SPECIALS]


def encode_rows(rows, stoi):
    encoded = []
    for row in rows:
        src_lang = f"<lang_{row['source_lang']}>"
        tgt_lang = f"<lang_{row['target_lang']}>"
        src = [BOS, stoi[src_lang], *[stoi.get(t, UNK) for t in tokenize(row["source_text"])], EOS]
        tgt = [BOS, stoi[tgt_lang], *[stoi.get(t, UNK) for t in tokenize(row["target_text"])], EOS]
        encoded.append((src, tgt))
    return encoded


def batchify(rows, indexes, batch_size, shuffle=False):
    indexes = list(indexes)
    if shuffle:
        random.shuffle(indexes)
    for start in range(0, len(indexes), batch_size):
        batch = [rows[i] for i in indexes[start:start + batch_size]]
        src_len = max(len(x[0]) for x in batch)
        tgt_len = max(len(x[1]) for x in batch)
        src = torch.full((len(batch), src_len), PAD, dtype=torch.long)
        tgt = torch.full((len(batch), tgt_len), PAD, dtype=torch.long)
        for i, (s, t) in enumerate(batch):
            src[i, :len(s)] = torch.tensor(s)
            tgt[i, :len(t)] = torch.tensor(t)
        yield src, tgt, batch


def word_logits(logits, tgt, target_vocab_ids, target_lang_ids):
    language_tokens = tgt[:, 1]
    language_indexes = torch.full_like(language_tokens, -1)
    language_ids = list(target_vocab_ids)
    for index, lang in enumerate(language_ids):
        language_indexes[language_tokens.eq(target_lang_ids[lang])] = index
    allowed = torch.zeros((len(language_ids), logits.size(-1)), dtype=torch.bool, device=logits.device)
    for index, lang in enumerate(language_ids):
        allowed[index, torch.tensor(target_vocab_ids[lang], device=logits.device)] = True
    mask = allowed[language_indexes.clamp_min(0)]
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def lexical_loss(logits, tgt, target_vocab_ids, target_lang_ids):
    predicted = word_logits(logits[:, 1, :], tgt, target_vocab_ids, target_lang_ids)
    return F.cross_entropy(predicted, tgt[:, 2])


def evaluate(model, rows, batch_size, stoi, device, target_vocab_ids, target_lang_ids):
    model.eval()
    total_loss = total_tokens = correct = exact = examples = word_correct = word_count = 0
    copy_tokens = copy_correct = copy_exact = 0
    with torch.inference_mode():
        for src, tgt, raw_batch in batchify(rows, range(len(rows)), batch_size):
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])
            word_targets = tgt[:, 2]
            constrained = word_logits(logits[:, 1, :], tgt, target_vocab_ids, target_lang_ids)
            predictions = constrained.argmax(-1)
            loss = F.cross_entropy(constrained, word_targets)
            total_loss += loss.item() * len(raw_batch)
            total_tokens += len(raw_batch)
            correct += (predictions == word_targets).sum().item()
            for i, (src_ids, tgt_ids) in enumerate(raw_batch):
                exact += int(predictions[i].item() == tgt_ids[2])
                examples += 1
                word_correct += int(predictions[i].item() == tgt_ids[2])
                word_count += 1
                # Identity/copy baseline; the target-language token is known.
                source_words = src_ids[2:-1]
                target_words = tgt_ids[2:-1]
                matches = int(len(source_words) == 1 and len(target_words) == 1 and source_words[0] == target_words[0])
                copy_tokens += 1
                copy_correct += matches
                copy_exact += matches
    return {"loss": total_loss / max(1, total_tokens),
            "token_accuracy": correct / max(1, total_tokens),
            "exact_match": exact / max(1, examples),
            "word_accuracy": word_correct / max(1, word_count),
            "copy_token_accuracy": copy_correct / max(1, copy_tokens),
            "copy_exact_match": copy_exact / max(1, examples)}


def main(epochs=15, max_vocab=100000, batch_size=64, output=CHECKPOINT,
         data_dir=DATA, device_name="auto", learning_rate=0.005):
    random.seed(17)
    torch.manual_seed(17)
    torch.set_num_threads(1)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA demandé mais aucun GPU CUDA n’est visible par PyTorch.")
    paths = {name: Path(data_dir) / f"{name}.jsonl" for name in ("train", "validation", "test")}
    if any(not p.is_file() for p in paths.values()):
        raise SystemExit("Données absentes : exécute d’abord python -m src.prepare_muse_words.")
    rows = {k: list(read_jsonl(p)) for k, p in paths.items()}
    if not rows["train"] or not rows["validation"] or not rows["test"]:
        raise SystemExit("Train/validation/test doit contenir des exemples.")
    vocab = make_vocab(paths["train"], max_vocab)
    known_tokens = set(vocab)
    train_before = len(rows["train"])
    rows["train"] = [r for r in rows["train"]
                     if r["source_text"] in known_tokens and r["target_text"] in known_tokens]
    print(f"train_vocab_covered={len(rows['train'])}/{train_before}")
    known_targets = {lang: {r["target_text"] for r in rows["train"] if r["target_lang"] == lang}
                     for lang in {r["target_lang"] for r in rows["train"]}}
    known_sources = {lang: {r["source_text"] for r in rows["train"] if r["source_lang"] == lang}
                     for lang in {r["source_lang"] for r in rows["train"]}}
    for split in ("validation", "test"):
        before = len(rows[split])
        rows[split] = [r for r in rows[split]
                       if all(t in known_tokens for t in tokenize(r["target_text"]))
                       and r["target_text"] in known_targets.get(r["target_lang"], set())]
        known = sum(r["source_text"] in known_sources.get(r["source_lang"], set()) for r in rows[split])
        print(f"{split}_evaluable={len(rows[split])}/{before} source_in_vocab={known}/{len(rows[split])}")
        if not rows[split]:
            raise SystemExit(f"Aucun exemple {split} avec cible dans le vocabulaire.")
    stoi = {t: i for i, t in enumerate(vocab)}
    encoded = {k: encode_rows(v, stoi) for k, v in rows.items()}
    languages = sorted({r["source_lang"] for r in rows["train"]} |
                       {r["target_lang"] for r in rows["train"]})
    cfg = {"d_model": 64, "heads": 4, "d_ff": 128, "layers": 2}
    model = Transformer(len(vocab), len(vocab), pad_id=PAD, **cfg).to(device)
    target_languages = sorted({r["target_lang"] for r in rows["train"]})
    target_vocab = {lang: sorted({r["target_text"] for r in rows["train"] if r["target_lang"] == lang})
                    for lang in target_languages}
    target_vocab_ids = {lang: [stoi[word] for word in words] for lang, words in target_vocab.items()}
    target_lang_ids = {lang: stoi[f"<lang_{lang}>"] for lang in target_languages}
    print(f"device={device} train={len(rows['train'])} val={len(rows['validation'])} test={len(rows['test'])}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val, best_state, best_epoch, best_val_metrics = float("inf"), None, 0, None
    for epoch in range(epochs):
        model.train()
        total, tokens = 0.0, 0
        for src, tgt, _ in batchify(encoded["train"], range(len(encoded["train"])), batch_size, shuffle=True):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            logits = model(src, tgt[:, :-1])
            gold = tgt[:, 1:]
            loss = lexical_loss(logits, tgt, target_vocab_ids, target_lang_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = gold.ne(PAD).sum().item()
            total += loss.item() * n
            tokens += n
        val = evaluate(model, encoded["validation"], batch_size, stoi, device,
                       target_vocab_ids, target_lang_ids)
        print(f"epoch {epoch+1}/{epochs} train_loss={total/max(1,tokens):.4f} "
              f"val_loss={val['loss']:.4f} val_token_acc={val['token_accuracy']:.4f} "
              f"val_word_acc={val['word_accuracy']:.4f} val_exact={val['exact_match']:.4f}")
        if val["loss"] < best_val:
            best_val, best_epoch = val["loss"], epoch + 1
            best_val_metrics = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test = evaluate(model, encoded["test"], batch_size, stoi, device,
                    target_vocab_ids, target_lang_ids)
    print(f"test token_acc={test['token_accuracy']:.4f} exact={test['exact_match']:.4f} "
          f"word_acc={test['word_accuracy']:.4f} "
          f"copy_baseline_token_acc={test['copy_token_accuracy']:.4f} "
          f"copy_baseline_exact={test['copy_exact_match']:.4f}")
    targets_from_french = sorted({r["target_lang"] for r in rows["train"] if r["source_lang"] == "fr"}
                                 | {r["target_lang"] for r in rows["train"]
                                    if r["source_lang"] == "en" and r["target_lang"] in ("de", "es")})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "vocab": vocab, "config": cfg,
                "languages": languages, "targets_from_french": targets_from_french,
                "source_vocab": known_sources,
                "target_vocab": target_vocab,
                "pivot_language": "en", "best_epoch": best_epoch, "seed": 17,
                "max_vocab": max_vocab, "max_tokens_per_sentence": 64,
                "learning_rate": learning_rate,
                "loss_strategy": "lexical token classification",
                "train_rows": len(rows["train"]), "validation_rows": len(rows["validation"]),
                "test_rows": len(rows["test"]), "validation_metrics": best_val_metrics,
                "test_metrics": test,
                "pairs": sorted({f"{r['source_lang']}-{r['target_lang']}" for r in rows["train"]})}, output)
    print(f"checkpoint={output.resolve()} best_epoch={best_epoch} vocab={len(vocab)} "
          f"targets_from_french={','.join(targets_from_french)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--max-vocab", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=0.005)
    args = parser.parse_args()
    main(args.epochs, args.max_vocab, args.batch_size, args.output, args.data_dir,
         args.device, args.learning_rate)
