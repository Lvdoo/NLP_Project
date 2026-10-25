"""Sentence-level inference for the hand-written Transformer trained on OPUS-100."""
import secrets
import unicodedata
from pathlib import Path

import torch

from .model import Transformer
from .bpe_tokenizer import load as load_bpe

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "sentence_transformer.pt"
LANG_NAMES = {"en": "anglais", "de": "allemand", "es": "espagnol", "fr": "français"}
SUPPORTED_TARGETS = ("en", "de", "es")
PAD, BOS, EOS, UNK = range(4)
LANG_ID = {"en": 4, "fr": 5, "de": 6, "es": 7}


def normalize(text):
    return unicodedata.normalize("NFC", text).strip()


def choose_target(source_lang="fr", candidates=SUPPORTED_TARGETS, chooser=secrets.choice):
    eligible = [lang for lang in candidates if lang != source_lang and lang in LANG_NAMES]
    return chooser(eligible) if eligible else None


class MultilingualTranslator:
    def __init__(self, checkpoint=CHECKPOINT):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.config = saved["config"]
        self.vocab_size = saved["vocab_size"]
        self.bpe_ids = [int(x) for x in saved["bpe_ids"]]
        self.bpe_to_id = {token: i + 8 for i, token in enumerate(self.bpe_ids)}
        self.id_to_bpe = self.bpe_ids
        self.languages = tuple(saved["languages"])
        self.targets_from_french = tuple(saved["targets_from_french"])
        self.max_length = int(saved.get("max_length", 96))
        tokenizer_path = ROOT / saved.get("tokenizer_file", "data/opus100/raw/cl100k_base.tiktoken")
        self.tokenizer = load_bpe(tokenizer_path)
        self.model = Transformer(self.vocab_size, self.vocab_size, pad_id=PAD, **self.config)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.device = torch.device("cpu")

    def _translate_pair(self, text, source_lang, target_lang):
        if source_lang not in self.languages or target_lang not in self.languages:
            raise ValueError("Paire de langues absente du modèle.")
        pieces = [self.bpe_to_id.get(x, UNK)
                  for x in self.tokenizer.encode(normalize(text), disallowed_special=())]
        if not pieces:
            raise ValueError("Le texte est vide après normalisation.")
        if len(pieces) > self.max_length - 3:
            raise ValueError(f"Texte trop long pour le modèle ({self.max_length - 3} sous-mots maximum).")
        source = torch.tensor([[BOS, LANG_ID[source_lang], *pieces, EOS]], dtype=torch.long)
        target = torch.tensor([[BOS, LANG_ID[target_lang]]], dtype=torch.long)
        forbidden = [PAD, BOS, *LANG_ID.values()]
        with torch.inference_mode():
            memory, src_mask = self.model.encode(source)
            for _ in range(self.max_length - 2):
                logits = self.model.decode(target, memory, src_mask)[0, -1].clone()
                logits[forbidden] = torch.finfo(logits.dtype).min
                token = int(logits.argmax().item())
                if token == EOS:
                    break
                target = torch.cat((target, torch.tensor([[token]])), dim=1)
        ids = [self.id_to_bpe[t - 8] for t in target[0, 2:].tolist() if 8 <= t < self.vocab_size]
        return self.tokenizer.decode(ids).strip()

    def translate(self, text, target_lang):
        if target_lang not in self.targets_from_french:
            raise ValueError("Langue cible non prise en charge par les données entraînées.")
        if target_lang in ("en", "fr"):
            return self._translate_pair(text, "fr", target_lang), None
        pivot = self._translate_pair(text, "fr", "en")
        return self._translate_pair(pivot, "en", target_lang), pivot

    def random_translation(self, text, chooser=secrets.choice):
        text = normalize(text)
        if not text:
            raise ValueError("Saisis un texte français à traduire.")
        target = choose_target("fr", self.targets_from_french, chooser)
        if target is None:
            raise ValueError("Aucune langue cible différente du français n’est disponible.")
        translation, pivot = self.translate(text, target)
        return {"source_text": text, "target_lang": target, "target_name": LANG_NAMES[target],
                "translation": translation, "pivot_text": pivot,
                "route": ["fr", "en", target] if target != "en" else ["fr", "en"]}
