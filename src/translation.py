"""Inference utilities for the manually implemented multilingual Transformer."""
import json
import re
import secrets
import unicodedata
from pathlib import Path

import torch

from .model import Transformer

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "word_muse_full_transformer.pt"
LANG_NAMES = {"en": "anglais", "de": "allemand", "es": "espagnol", "fr": "français"}
SUPPORTED_TARGETS = ("en", "de", "es")
PAD, BOS, EOS, UNK = range(4)
TOKEN_RE = re.compile(r"[^\W_]+|[^\w\s]", re.UNICODE)


def normalize(text):
    return unicodedata.normalize("NFC", text).casefold().strip()


def tokenize(text):
    return TOKEN_RE.findall(normalize(text))


def detokenize(tokens):
    text = " ".join(tokens)
    return re.sub(r"\s+([,.;:!?…%)\]])", r"\1", re.sub(r"([(\[])\s+", r"\1", text))


def choose_target(source_lang="fr", candidates=SUPPORTED_TARGETS, chooser=secrets.choice):
    eligible = [lang for lang in candidates if lang != source_lang and lang in LANG_NAMES]
    return chooser(eligible) if eligible else None


class MultilingualTranslator:
    def __init__(self, checkpoint=CHECKPOINT):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.vocab = saved["vocab"]
        self.stoi = {token: i for i, token in enumerate(self.vocab)}
        self.itos = self.vocab
        self.config = saved["config"]
        self.pairs = set(saved.get("pairs", []))
        self.languages = tuple(saved["languages"])
        self.targets_from_french = tuple(saved["targets_from_french"])
        self.source_vocab = saved.get("source_vocab", {})
        self.target_vocab = saved.get("target_vocab", {})
        self.lexicon = {}
        for target_lang in ("en", "de", "es"):
            full_file = ROOT / "data" / "muse" / "raw" / f"fr-{target_lang}.full.txt"
            if full_file.is_file():
                with full_file.open(encoding="utf-8") as stream:
                    for line in stream:
                        pair = line.casefold().split()
                        if len(pair) == 2 and len(tokenize(pair[0])) == len(tokenize(pair[1])) == 1:
                            self.lexicon.setdefault(("fr", pair[0], target_lang), pair[1])
        train_words = ROOT / "data" / "muse" / "processed" / "train.jsonl"
        if not self.lexicon and train_words.is_file():
            with train_words.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    key = (row["source_lang"], row["source_text"], row["target_lang"])
                    self.lexicon.setdefault(key, row["target_text"])
            if not self.target_vocab:
                self.target_vocab = {}
                for (src_lang, _, tgt_lang), word in self.lexicon.items():
                    self.target_vocab.setdefault(tgt_lang, set()).add(word)
                self.target_vocab = {lang: sorted(words) for lang, words in self.target_vocab.items()}
        self.model = Transformer(len(self.vocab), len(self.vocab), pad_id=PAD, **self.config)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()

    def _translate_pair(self, text, source_lang, target_lang, max_tokens=96):
        lang_src, lang_tgt = f"<lang_{source_lang}>", f"<lang_{target_lang}>"
        if lang_src not in self.stoi or lang_tgt not in self.stoi:
            raise ValueError("Paire de langues absente du vocabulaire du modèle.")
        words = tokenize(text)
        if not words:
            raise ValueError("Le texte est vide après normalisation.")
        if len(words) > max_tokens:
            raise ValueError(f"Texte trop long : limite {max_tokens} tokens par étape.")
        src = [BOS, self.stoi[lang_src], *[self.stoi.get(w, UNK) for w in words], EOS]
        source = torch.tensor([src], dtype=torch.long)
        target = torch.tensor([[BOS, self.stoi[lang_tgt]]], dtype=torch.long)
        candidates = self.target_vocab.get(target_lang, [])
        candidate_ids = [self.stoi[word] for word in candidates if word in self.stoi]
        if not candidate_ids:
            raise ValueError(f"Aucun mot cible entraîné pour {LANG_NAMES.get(target_lang, target_lang)}.")
        with torch.inference_mode():
            memory, src_mask = self.model.encode(source)
            logits = self.model.decode(target, memory, src_mask)[0, -1]
            allowed = torch.tensor(candidate_ids, dtype=torch.long)
            return self.itos[allowed[logits[allowed].argmax().item()]]

    def translate(self, text, target_lang):
        if target_lang not in self.targets_from_french or target_lang == "fr":
            raise ValueError("Langue cible non prise en charge par les données entraînées.")
        if ("fr", target_lang) not in self.pairs and target_lang != "en":
            pivot = self._translate_pair(text, "fr", "en")
            return self._translate_pair(pivot, "en", target_lang), pivot
        return self._translate_pair(text, "fr", target_lang), None

    def random_translation(self, text, chooser=secrets.choice):
        words = tokenize(text)
        if not words or not any(any(char.isalnum() for char in token) for token in words):
            raise ValueError("Saisis au moins un mot français.")
        target = choose_target("fr", self.targets_from_french, chooser)
        if target is None:
            raise ValueError("Aucune langue cible différente du français n’est disponible.")
        model_tokens, reference_tokens, pivot_tokens = [], [], []
        missing_words = []
        for word in words:
            if not any(char.isalnum() for char in word):
                model_tokens.append({"source": word, "translation": word, "available": True})
                reference_tokens.append({"source": word, "translation": word, "available": True})
                continue
            source_known = word in self.source_vocab.get("fr", self.stoi)
            model_word = pivot = None
            if source_known:
                model_word, pivot = self.translate(word, target)
            reference_word = self.lexicon.get(("fr", word, target))
            model_tokens.append({"source": word, "translation": model_word or None,
                                 "available": bool(model_word)})
            reference_tokens.append({"source": word, "translation": reference_word,
                                     "available": bool(reference_word)})
            if not reference_word:
                missing_words.append(word)
            if pivot:
                pivot_tokens.append(pivot)
        translated = detokenize([item["translation"] or f"⟦{item['source']}: non prédit⟧"
                                 for item in model_tokens])
        reference = detokenize([item["translation"] or f"⟦{item['source']}: absent⟧"
                                for item in reference_tokens])
        pivot = detokenize(pivot_tokens) if pivot_tokens else None
        return {"source_text": text, "target_lang": target,
                "target_name": LANG_NAMES[target], "translation": translated,
                "model_tokens": model_tokens, "reference_tokens": reference_tokens,
                "reference_translation": reference, "missing_words": missing_words,
                "pivot_text": pivot, "route": ["fr", "en", target] if pivot else ["fr", target]}
