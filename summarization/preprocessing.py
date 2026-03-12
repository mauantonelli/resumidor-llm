import re
import unicodedata
from typing import Optional


class TextPreprocessor:
    def __init__(self, max_length: int = 1024):
        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def remove_references(self, text: str) -> str:
        text = re.sub(r"\[[\d,;\s]+\]", "", text)
        text = re.sub(
            r"(?i)refer[eê]ncias\s*bibliogr[aá]ficas.*$", "", text, flags=re.DOTALL
        )
        text = re.sub(r"(?i)references\s*$.*$", "", text, flags=re.DOTALL)
        return text.strip()

    def truncate_text(self, text: str, max_words: Optional[int] = None) -> str:
        if max_words is None:
            max_words = self.max_length
        words = text.split()
        if len(words) > max_words:
            words = words[:max_words]
        return " ".join(words)

    def prepare_for_summarization(self, text: str) -> str:
        text = self.clean_text(text)
        text = self.remove_references(text)
        text = self.truncate_text(text)
        return text
