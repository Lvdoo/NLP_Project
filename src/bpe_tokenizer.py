"""Local cl100k byte-pair tokenizer loader; tokenizer only, no pretrained translation model."""
import urllib.request
from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe

URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
PATTERN = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"
SPECIAL_TOKENS = {"<|endoftext|>": 100257, "<|fim_prefix|>": 100258, "<|fim_middle|>": 100259,
                  "<|fim_suffix|>": 100260, "<|endofprompt|>": 100276}


def load(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer BPE absent: {path}; relancez src.prepare_opus100.")
    ranks = load_tiktoken_bpe(str(path), expected_hash=SHA256)
    return tiktoken.Encoding("cl100k_base_local", pat_str=PATTERN,
                             mergeable_ranks=ranks, special_tokens=SPECIAL_TOKENS)


def download(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        print(f"Téléchargement tokenizer BPE (1,6 Mo): {URL}")
        urllib.request.urlretrieve(URL, path)
    load(path)  # integrity check
    return path
