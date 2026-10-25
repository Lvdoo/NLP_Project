"""Prepare French-source lexical pairs from MUSE's full bilingual dictionaries."""
import json
import random
from collections import Counter
from pathlib import Path

from .translation import ROOT, tokenize

RAW = ROOT / "data" / "muse" / "raw"
OUT = ROOT / "data" / "muse" / "processed"
PAIRS = {"fr-en": ("fr", "en"), "fr-de": ("fr", "de"), "fr-es": ("fr", "es")}
BASE_URL = "https://dl.fbaipublicfiles.com/arrival/dictionaries/{}.txt"


def read_pairs(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().casefold().split()
        if len(parts) == 2 and len(tokenize(parts[0])) == len(tokenize(parts[1])) == 1:
            yield parts[0], parts[1]


def dedupe(pairs):
    grouped = {}
    for source, target in pairs:
        grouped.setdefault(source, []).append(target)
    return [(source, next((target for target in targets if target != source), targets[0]))
            for source, targets in grouped.items()]


def main():
    try:
        import urllib.request
        RAW.mkdir(parents=True, exist_ok=True)
        for pair in PAIRS:
            path = RAW / f"{pair}.full.txt"
            if not path.is_file():
                print(f"Téléchargement MUSE {pair} ({BASE_URL.format(pair)})")
                urllib.request.urlretrieve(BASE_URL.format(pair), path)
    except Exception as exc:
        raise SystemExit(f"Dictionnaire MUSE complet inaccessible: {exc}") from exc

    grouped_rows = {}
    for pair in PAIRS:
        grouped_rows[pair] = dedupe(list(read_pairs(RAW / f"{pair}.full.txt")))

    # One global split keeps a French lemma in one split across all target pairs.
    rng = random.Random(17)
    all_sources = sorted({source for pairs in grouped_rows.values() for source, _ in pairs})
    rng.shuffle(all_sources)
    n_test, n_val = max(1, len(all_sources) // 20), max(1, len(all_sources) // 20)
    test_words = set(all_sources[:n_test])
    val_words = set(all_sources[n_test:n_test + n_val])
    train, validation, test = [], [], []
    for pair, langs in PAIRS.items():
        pairs = grouped_rows[pair]
        for source, target in pairs:
            row = {"source_lang": langs[0], "source_text": source,
                   "target_lang": langs[1], "target_text": target}
            (test if source in test_words else validation if source in val_words else train).append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation), ("test", test)):
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {name: len(rows) for name, rows in (("train", train), ("validation", validation), ("test", test))}
    counts["source_words_by_language_pair"] = {p: len(rows) for p, rows in grouped_rows.items()}
    counts["rows_by_direction"] = {name: dict(Counter(f"{r['source_lang']}-{r['target_lang']}" for r in rows))
                                   for name, rows in (("train", train), ("validation", validation), ("test", test))}
    (OUT / "manifest.json").write_text(json.dumps({
        "dataset": "MUSE full bilingual dictionaries, French-source directions",
        "counts": counts,
        "license": "CC BY-NC 4.0; non-commercial use only; retain attribution.",
        "notes": ["Full public dictionaries (up to 100k entries per pair); one listed target retained per French lemma.",
                  "Words are split by French source lemma (80/10/10), preventing source-lemma overlap.",
                  "This is lexical, context-free word translation. Held-out French lemmas are OOV to the closed-vocabulary model; their scores measure that limitation, not phrase translation."]
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("splits:", counts)


if __name__ == "__main__":
    main()
