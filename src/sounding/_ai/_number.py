# 摘自 inspect_ai 0.3.287.dev30+g7340817a6：scorer/_common.py 的数字归一化、_util/pattern.py 的 ANSWER_PATTERN_WORD（MIT）。
# 原样复制，同步上游时重新摘取。

import math

from ._unicode import unicode_number_to_float

ANSWER_PATTERN_WORD = (
    r"(?is)ANSWER\s*:\s*(\S+?)(?=[.,;:!?]?\s*(?:\n|$))(?!.*ANSWER\s*:)"
)


_NUMERIC_PUNCTUATION_TRIM = "!?:;()[]{}'\"`"


def _clean_numeric_word(s: str) -> str:
    return s.strip(_NUMERIC_PUNCTUATION_TRIM)


def _parse_number(s: str, *, trim_punctuation: bool = True) -> float | None:
    """Parse `s` as a finite number, or None.

    Recognises plain float syntax (signs, decimals, exponents) and the
    full set of inputs handled by ``unicode_number_to_float`` (unicode
    minus, vulgar fractions, superscripts, Chinese numerals, fullwidth
    digits, locale grouping). Rejects strings with trailing non-numeric
    content (so ``"5 some text"`` returns None). Sentence/enclosing
    punctuation attached to the number ("42!", "(42)") is trimmed unless
    ``trim_punctuation=False`` — the ``location="exact"`` path passes False
    so that exact stays actually exact. ``nan`` and ``inf`` also return None.
    """
    s_cleaned = _clean_numeric_word(s) if trim_punctuation else s
    for parse in (float, unicode_number_to_float):
        try:
            num = parse(s_cleaned)
        except ValueError:
            continue
        if math.isfinite(num):
            return num
    return None


def normalize_number(
    number: str, precision: int = 5, *, trim_punctuation: bool = True
) -> str:
    num = _parse_number(number, trim_punctuation=trim_punctuation)
    if num is None:
        return number
    return format(num, f".{precision}g")
