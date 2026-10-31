"""Text normalisation for name matching and script profiling.

``normalize_for_matching`` folds free-text mentions (payee names typed by the user, directory
display names and aliases) into a comparable form. It is a lexical convenience for the payee
directory and nothing else: it is never applied to a payee identifier, to an amount or to a
currency code. Identifiers are compared verbatim by ``tasdiq.payees.PayeeDirectory.by_id``;
amounts and currencies are produced by ``tasdiq.money`` from the raw utterance, and the
consent record stores integers and ISO 4217 codes that this module never touches.

``script_profile`` reports which scripts and digit sets an utterance uses. It is recorded as
provenance for language and code-switching analyses and takes no part in any decision.
"""

from __future__ import annotations

import re
import unicodedata

TATWEEL = "ـ"

_HARAKAT = re.compile("[ً-ْٰؐ-ؚ]")
_ALEF_VARIANTS = str.maketrans({"آ": "ا", "أ": "ا", "إ": "ا"})
_LETTER_FOLDS = str.maketrans({"ة": "ه", "ى": "ي"})
_WHITESPACE = re.compile(r"\s+")

_ARABIC_BLOCKS: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
_EASTERN_DIGITS = frozenset("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹")
_ASCII_DIGITS = frozenset("0123456789")


def _is_arabic_letter(ch: str) -> bool:
    if not unicodedata.category(ch).startswith("L"):
        return False
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _ARABIC_BLOCKS)


def _is_latin_letter(ch: str) -> bool:
    if not unicodedata.category(ch).startswith("L"):
        return False
    if ch.isascii():
        return ch.isalpha()
    return "LATIN" in unicodedata.name(ch, "")


def _strip_punctuation(text: str) -> str:
    """Replace punctuation and symbols by spaces, keeping a hyphen that joins two word characters."""
    out: list[str] = []
    n = len(text)
    for i, ch in enumerate(text):
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            if ch == "-" and 0 < i < n - 1 and _is_word_char(text[i - 1]) and _is_word_char(text[i + 1]):
                out.append(ch)
            else:
                out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def _is_word_char(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat.startswith("L") or cat.startswith("N")


def normalize_for_matching(s: str) -> str:
    """Fold a free-text mention into the form used for exact directory matching.

    The steps are: removal of Unicode format characters (category Cf: bidi marks, zero-width
    joiners and non-joiners); NFKC normalisation (which also folds the Arabic presentation
    forms and fullwidth letters to their base characters); removal of tatweel (U+0640);
    removal of the Arabic harakat (U+064B to U+0652, U+0670, U+0610 to U+061A); unification of the alef variants
    (U+0622, U+0623, U+0625 to U+0627); taa marbuta (U+0629) to haa (U+0647); alef maqsura
    (U+0649) to yaa (U+064A); replacement of punctuation and symbols by spaces, except a hyphen
    that joins two word characters inside an identifier such as "al-ain"; lowercasing of
    Latin letters; and collapsing of whitespace runs to single spaces with the ends stripped.

    The function is applied to mentions, display names and aliases only. Payee identifiers,
    amounts and currency codes are never passed through it.
    """
    text = "".join(ch for ch in s if unicodedata.category(ch) != "Cf")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(TATWEEL, "")
    text = _HARAKAT.sub("", text)
    text = text.translate(_ALEF_VARIANTS).translate(_LETTER_FOLDS)
    text = _strip_punctuation(text)
    text = text.lower()
    return _WHITESPACE.sub(" ", text).strip()


def script_profile(s: str) -> dict[str, float | int | str]:
    """Return the share of each script and digit set among the letters and digits of a text.

    The result has the keys "arabic_letters", "latin_letters", "eastern_digits" and
    "ascii_digits", each a fraction of the counted characters (letters and digits only, so
    spaces and punctuation are ignored), "n_chars" with the number of counted characters, and
    "dominant", which is "ar" when Arabic-script letters outnumber Latin letters, "en" in
    the opposite case, "mix" when both scripts occur equally often, and "none" when the text
    has no letters. An empty text yields zero fractions and "none". The profile is provenance
    for analysis and is never used in a decision.
    """
    arabic = latin = eastern = ascii_digits = 0
    for ch in unicodedata.normalize("NFC", s):
        if ch in _EASTERN_DIGITS:
            eastern += 1
        elif ch in _ASCII_DIGITS:
            ascii_digits += 1
        elif _is_arabic_letter(ch):
            arabic += 1
        elif _is_latin_letter(ch):
            latin += 1
    n = arabic + latin + eastern + ascii_digits
    if arabic == 0 and latin == 0:
        dominant = "none"
    elif arabic > latin:
        dominant = "ar"
    elif latin > arabic:
        dominant = "en"
    else:
        dominant = "mix"

    def frac(k: int) -> float:
        return k / n if n else 0.0

    return {
        "arabic_letters": frac(arabic),
        "latin_letters": frac(latin),
        "eastern_digits": frac(eastern),
        "ascii_digits": frac(ascii_digits),
        "n_chars": n,
        "dominant": dominant,
    }
