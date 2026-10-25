"""Download a bounded OPUS-100 subset and write parallel sentence JSONL splits."""
import argparse
import json
import re
import tarfile
import urllib.request
from pathlib import Path

try:
    from .bpe_tokenizer import download as download_bpe
except ImportError:  # Also allow: python src/prepare_opus100.py
    from bpe_tokenizer import download as download_bpe

ROOT = Path(__file__).resolve().parents[1]
# OPUS-100 names supervised archives using its canonical pair order: en-fr,
# de-en, en-es. Keep source/target order separate from the archive identifier.
PAIRS = (("fr", "en", "en-fr"), ("de", "en", "de-en"), ("en", "es", "en-es"))
BASE = "https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-{}-v1.0.tar.gz"


def member_for(members, split, lang):
    aliases = {"validation": ("dev", "valid", "validation")}.get(split, (split,))
    for alias in aliases:
        pattern = re.compile(rf"(?:^|[./_-]){alias}\.{lang}$")
        found = [m for m in members if m.isfile() and pattern.search(m.name)]
        if found:
            return found[0]
    candidates = [m.name for m in members if m.isfile()]
    raise RuntimeError(f"Fichier {split}.{lang} absent de l’archive; exemples: {candidates[:12]}")


def append_pair(archive_path, source_lang, target_lang, split, out, limit=None):
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        left, right = member_for(members, split, source_lang), member_for(members, split, target_lang)
        a, b = archive.extractfile(left), archive.extractfile(right)
        count = 0
        for x, y in zip(a, b):
            if limit is not None and count >= limit:
                break
            source, target = x.decode("utf-8", "replace").strip(), y.decode("utf-8", "replace").strip()
            if not source or not target:
                continue
            out.write(json.dumps({"source_lang": source_lang, "source_text": source,
                                  "target_lang": target_lang, "target_text": target}, ensure_ascii=False) + "\n")
            out.write(json.dumps({"source_lang": target_lang, "source_text": target,
                                  "target_lang": source_lang, "target_text": source}, ensure_ascii=False) + "\n")
            count += 1
        return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", type=Path, default=ROOT / "data" / "opus100" / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "opus100" / "processed")
    parser.add_argument("--max-train-per-pair", type=int, default=50000,
                        help="Bound each source pair; set 0 to use the full OPUS-100 train split.")
    args = parser.parse_args()
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    download_bpe(args.archive_dir / "cl100k_base.tiktoken")
    archives = {}
    for _, _, pair in PAIRS:
        path = args.archive_dir / f"opus-100-corpus-{pair}-v1.0.tar.gz"
        if not path.is_file():
            url = BASE.format(pair)
            print(f"Téléchargement {pair}: {url}")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as exc:
                raise SystemExit(f"Téléchargement échoué ({pair}): {exc}") from exc
        archives[pair] = path

    counts = {}
    for split in ("train", "validation", "test"):
        path = args.output / f"{split}.jsonl"
        total = 0
        with path.open("w", encoding="utf-8", newline="\n") as out:
            for src, tgt, pair in PAIRS:
                limit = args.max_train_per_pair or None
                if split != "train":
                    limit = None
                n = append_pair(archives[pair], src, tgt, split, out, limit)
                print(f"{split} {pair}: {n} phrases alignées")
                total += n * 2
        counts[split] = total
    (args.output / "manifest.json").write_text(json.dumps({
        "dataset": "OPUS-100", "pairs": [pair for _, _, pair in PAIRS],
        "counts_including_reverse": counts, "max_train_per_pair": args.max_train_per_pair,
        "note": "English-centric: French to German/Spanish uses French→English→target pivot."
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSONL prêts: {args.output}")


if __name__ == "__main__":
    main()
