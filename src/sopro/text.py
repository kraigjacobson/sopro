from __future__ import annotations

import re
from typing import List, Optional

import sentencepiece as spm
import torch

LANGUAGE_TAGS = {
    "en": "<|lang_en|>",
    "pt": "<|lang_pt|>",
    "fr": "<|lang_fr|>",
    "de": "<|lang_de|>",
}

_TERMINALS = {".", "!", "?", "-", ",", ";", ":"}
_REPLACEMENTS = (
    (" ,", ","),
    (" .", "."),
    (" !", "!"),
    (" ?", "?"),
    (" ;", ";"),
    (" :", ":"),
    ("“", '"'),
    ("”", '"'),
    ("‘", "'"),
    ("’", "'"),
)


def language_tag(lang: Optional[str]) -> str:
    if not lang:
        return ""
    key = str(lang).strip().lower()
    if key not in LANGUAGE_TAGS:
        raise ValueError(f"unsupported language {lang!r}; expected one of {sorted(LANGUAGE_TAGS)}")
    return LANGUAGE_TAGS[key]


_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")


def _pack(parts: List[str], max_chars: int) -> List[str]:
    out: List[str] = []
    cur = ""
    for part in parts:
        if not cur:
            cur = part
        elif len(cur) + 1 + len(part) <= max_chars:
            cur = f"{cur} {part}"
        else:
            out.append(cur)
            cur = part
    if cur:
        out.append(cur)
    return out


def split_text(text: str, max_chars: int) -> List[str]:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return [text] if text else []
    segments: List[str] = []
    for sentence in _pack(_SENTENCE_END.split(text), max_chars):
        if len(sentence) <= max_chars:
            segments.append(sentence)
            continue
        for clause in _pack(_CLAUSE_END.split(sentence), max_chars):
            if len(clause) <= max_chars:
                segments.append(clause)
            else:
                segments.extend(_pack(clause.split(" "), max_chars))
    return segments


def normalize_text(text: str) -> str:
    text = str(text).strip()
    if not text:
        return "You need to add some text for me to talk."
    special = re.match(r"^((?:<\|[^|\s]+?\|>\s*)+)(.*)$", text)
    if special is not None:
        prefix = re.sub(r"\s+", " ", special.group(1)).strip()
        body = special.group(2).strip()
        return f"{prefix} {normalize_text(body)}" if body else prefix
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split()).replace("…", "...")
    for old, new in _REPLACEMENTS:
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text).strip()
    if not any(text.endswith(p) for p in _TERMINALS):
        text += "."
    return text


class TextTokenizer:
    def __init__(self, model_path: str, max_length: int = 512) -> None:
        self._sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.bos_id = int(self._sp.bos_id()) if self._sp.bos_id() >= 0 else 1
        self.eos_id = int(self._sp.eos_id()) if self._sp.eos_id() >= 0 else 2
        self.unk_id = int(self._sp.unk_id()) if self._sp.unk_id() >= 0 else 0
        self.vocab_size = int(self._sp.get_piece_size())
        self.max_length = int(max_length)

    def encode(self, text: str, lang: Optional[str] = None) -> List[int]:
        tag = language_tag(lang)
        text = normalize_text(f"{tag} {text}" if tag else text)
        ids = [self.bos_id] + [int(x) for x in self._sp.encode(text, out_type=int)] + [self.eos_id]
        ids = ids[: self.max_length]
        return ids if ids else [self.unk_id]

    def encode_tensor(self, text: str, lang: Optional[str], device: torch.device) -> torch.Tensor:
        return torch.tensor([self.encode(text, lang)], dtype=torch.long, device=device)
