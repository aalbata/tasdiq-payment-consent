"""Money representation, amount parsing and amount formatting.

An amount is always an integer number of minor units together with an ISO 4217 code. The
parser reads amounts written with ASCII, Arabic-Indic (U+0660 to U+0669) or Extended
Arabic-Indic (U+06F0 to U+06F9) digits (digits of any other script are not numbers), with
Arabic or Western separators, or with English or Modern Standard Arabic number words, and
reads the currency from a name, code, symbol or abbreviation adjacent to the number. A
number phrase preceded by a spelled-out sign (minus, negative, ناقص, سالب), or a word
phrase interrupted by an unrecognised word after "and", yields no amount and is flagged;
a bare multiplier or hundreds word without a currency label is not an amount; an integer
of seven or more digits with no adjacent currency label is an identifier, not an amount.
The formatter renders an amount for the confirmation surface in English ("250.00 AED")
or in Arabic with Eastern digits ("٢٥٠٫٠٠ درهم").

Exact arithmetic uses ``fractions.Fraction``; no float enters an amount.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from fractions import Fraction

FLAGS: tuple[str, ...] = (
    "multiple_amounts",
    "no_currency",
    "unrecognised_fraction",
    "number_words",
    "digit_script_ar",
    "thousands_separator",
    "decimal",
    "unrecognised_number_word",
    "negative",
    "mixed_digit_scripts",
    "secondary_unlabelled_amount",
)
INVALIDATING_FLAGS: frozenset[str] = frozenset({"unrecognised_fraction", "unrecognised_number_word", "negative"})
"""Flags that leave the entry they mark without an amount."""

DEFAULT_MINOR_DIGITS: int = 2
"""Scale assumed for an amount whose currency is unknown."""


@dataclass(frozen=True)
class CurrencyInfo:
    """Static description of one currency.

    ``names_en`` and ``names_ar`` are the words that name the currency in each language
    (the first Arabic name is the one written by ``format_amount``); ``symbols`` are signs
    and abbreviations such as "$" or "د.إ"; ``minor_en`` and ``minor_ar`` name the minor
    unit; ``dual_ar`` are Arabic dual noun forms that denote two units on their own.
    """

    code: str
    minor_digits: int
    names_en: tuple[str, ...]
    names_ar: tuple[str, ...]
    symbols: tuple[str, ...] = ()
    minor_en: tuple[str, ...] = ()
    minor_ar: tuple[str, ...] = ()
    dual_ar: tuple[str, ...] = ()

    @property
    def scale(self) -> int:
        """Number of minor units in one major unit."""
        return 10**self.minor_digits


def _ci(
    code: str,
    minor_digits: int,
    names_en: Iterable[str],
    names_ar: Iterable[str],
    symbols: Iterable[str] = (),
    minor_en: Iterable[str] = (),
    minor_ar: Iterable[str] = (),
    dual_ar: Iterable[str] = (),
) -> CurrencyInfo:
    return CurrencyInfo(
        code,
        minor_digits,
        tuple(names_en),
        tuple(names_ar),
        tuple(symbols),
        tuple(minor_en),
        tuple(minor_ar),
        tuple(dual_ar),
    )


CURRENCIES: dict[str, CurrencyInfo] = {
    "AED": _ci(
        "AED",
        2,
        ["dirham", "dirhams", "dh", "dhs", "aed", "uae dirham", "uae dirhams", "emirati dirham", "emirati dirhams"],
        ["درهم", "دراهم", "درهماً", "درهمًا", "درهما", "درهم إماراتي", "درهم اماراتي", "دراهم إماراتية"],
        ["د.إ", "د.إ.", "AED"],
        ["fils"],
        ["فلس", "فلوس", "فلساً", "فلسا"],
        ["درهمان", "درهمين"],
    ),
    "SAR": _ci(
        "SAR",
        2,
        ["riyal", "riyals", "sar", "saudi riyal", "saudi riyals"],
        ["ريال", "ريالات", "ريالاً", "ريالا", "ريال سعودي"],
        ["ر.س", "ر.س.", "SAR", "﷼"],
        ["halala", "halalas"],
        ["هللة", "هللات"],
        ["ريالان", "ريالين"],
    ),
    "USD": _ci(
        "USD",
        2,
        ["dollar", "dollars", "usd", "us dollar", "us dollars", "american dollar", "american dollars"],
        ["دولار", "دولارات", "دولاراً", "دولارا", "دولار أمريكي", "دولار امريكي"],
        ["$", "US$", "USD"],
        ["cent", "cents"],
        ["سنت", "سنتات"],
        ["دولاران", "دولارين"],
    ),
    "EUR": _ci(
        "EUR",
        2,
        ["euro", "euros", "eur"],
        ["يورو"],
        ["€", "EUR"],
        ["cent", "cents"],
        ["سنت", "سنتات"],
    ),
    "GBP": _ci(
        "GBP",
        2,
        ["pound", "pounds", "gbp", "sterling", "pound sterling", "pounds sterling", "british pound", "british pounds"],
        ["جنيه", "جنيهات", "جنيهاً", "جنيها", "جنيه استرليني", "جنيه إسترليني", "جنيه استرلينى"],
        ["£", "GBP"],
        ["penny", "pence"],
        ["بنس"],
        ["جنيهان", "جنيهين"],
    ),
    "KWD": _ci(
        "KWD",
        3,
        ["dinar", "dinars", "kwd", "kd", "kuwaiti dinar", "kuwaiti dinars"],
        ["دينار كويتي", "دينار", "دنانير", "ديناراً", "دينارا", "دنانير كويتية"],
        ["د.ك", "د.ك.", "KWD"],
        ["fils"],
        ["فلس", "فلوس", "فلساً", "فلسا"],
        ["ديناران", "دينارين"],
    ),
    "BHD": _ci(
        "BHD",
        3,
        ["bhd", "bd", "bahraini dinar", "bahraini dinars"],
        ["دينار بحريني", "دنانير بحرينية"],
        ["د.ب", "د.ب.", "BHD"],
        ["fils"],
        ["فلس", "فلوس", "فلساً", "فلسا"],
    ),
    "OMR": _ci(
        "OMR",
        3,
        ["rial", "rials", "omr", "omani rial", "omani rials"],
        ["ريال عماني", "ريال عُماني", "ريالات عمانية"],
        ["ر.ع", "ر.ع.", "OMR"],
        ["baisa", "baisas", "baiza"],
        ["بيسة", "بيسات"],
    ),
    "QAR": _ci(
        "QAR",
        2,
        ["qar", "qr", "qatari riyal", "qatari riyals"],
        ["ريال قطري", "ريالات قطرية"],
        ["ر.ق", "ر.ق.", "QAR"],
    ),
    "JOD": _ci(
        "JOD",
        3,
        ["jod", "jd", "jordanian dinar", "jordanian dinars"],
        ["دينار أردني", "دينار اردني", "دنانير أردنية"],
        ["د.أ", "د.أ.", "JOD"],
        ["fils", "qirsh", "piastre", "piastres"],
        ["فلس", "فلوس", "قرش", "قروش"],
    ),
    "EGP": _ci(
        "EGP",
        2,
        ["egp", "egyptian pound", "egyptian pounds"],
        ["جنيه مصري", "جنيهات مصرية", "جنيه مصرى"],
        ["ج.م", "ج.م.", "EGP"],
        ["piastre", "piastres"],
        ["قرش", "قروش"],
    ),
    "INR": _ci(
        "INR",
        2,
        ["rupee", "rupees", "inr", "indian rupee", "indian rupees"],
        ["روبية", "روبيات", "روبية هندية"],
        ["₹", "INR"],
        ["paisa", "paise"],
        ["بيسة هندية"],
        ["روبيتان", "روبيتين"],
    ),
    "PKR": _ci(
        "PKR",
        2,
        ["pkr", "pakistani rupee", "pakistani rupees"],
        ["روبية باكستانية", "روبيات باكستانية"],
        ["PKR"],
        ["paisa", "paise"],
        [],
    ),
    "PHP": _ci(
        "PHP",
        2,
        ["peso", "pesos", "php", "philippine peso", "philippine pesos"],
        ["بيزو", "بيزو فلبيني"],
        ["₱", "PHP"],
        ["centavo", "centavos", "sentimo"],
        ["سنتافو"],
    ),
}


@dataclass
class AmountParse:
    """Result of ``parse_amount``.

    ``amount_minor`` and ``currency`` are the resolved amount and code, or None when no
    single amount could be resolved; ``matched_text`` is the text span (with digits
    normalised) that produced the amount; ``candidates`` lists every distinct amount found as
    (amount_minor, currency) in order of appearance; ``flags`` is a subset of ``FLAGS``.
    """

    amount_minor: int | None
    currency: str | None
    matched_text: str
    candidates: list[tuple[int, str | None]] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)


_EASTERN = "٠١٢٣٤٥٦٧٨٩"
_EXTENDED = "۰۱۲۳۴۵۶۷۸۹"
_ASCII = "0123456789"
_DIGIT_MAP = str.maketrans(_EASTERN + _EXTENDED + "٫٬", _ASCII + _ASCII + ".,")
_TO_EASTERN = str.maketrans(_ASCII, _EASTERN)
_AR_DIGIT_RE = re.compile("[" + _EASTERN + _EXTENDED + "]")
_NUMBER_TOKEN_RE = re.compile("[0-9" + _EASTERN + _EXTENDED + "]+(?:[,.،٫٬][0-9" + _EASTERN + _EXTENDED + "]+)*")
ARABIC_DECIMAL_SEPARATOR = "٫"
ARABIC_THOUSANDS_SEPARATOR = "٬"


def strip_format_characters(text: str) -> str:
    """Remove Unicode format characters (category Cf: bidi marks, zero-width joiners and the like)."""
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def normalize_digits(text: str) -> str:
    """Map Arabic-Indic and Extended Arabic-Indic digits to ASCII digits.

    U+066B (Arabic decimal separator) becomes "." and U+066C (Arabic thousands separator)
    becomes ","; Unicode format characters (category Cf, such as the bidi marks and the
    zero-width joiners) are removed; every other character is kept. The mapping is
    idempotent.
    """
    return strip_format_characters(text).translate(_DIGIT_MAP)


def mixed_digit_scripts(text: str) -> bool:
    """True when one number token of the text mixes ASCII digits with Arabic-Indic or Extended Arabic-Indic digits."""
    for match in _NUMBER_TOKEN_RE.finditer(text):
        token = match.group(0)
        if any(ch in _ASCII for ch in token) and _AR_DIGIT_RE.search(token):
            return True
    return False


def eastern_digits(text: str) -> str:
    """Map ASCII digits to Arabic-Indic digits (U+0660 to U+0669)."""
    return text.translate(_TO_EASTERN)


_HARAKAT_RE = re.compile("[ً-ْٰؐ-ؚـ]")
_ALEF = str.maketrans({"آ": "ا", "أ": "ا", "إ": "ا", "ة": "ه", "ى": "ي"})


def _fold(word: str) -> str:
    """Lexical fold of one word for lexicon lookup (marks removed, alef unified, lowercase).

    A dotted abbreviation such as "د.إ" or "د.أ" keeps its hamza forms, because the alef
    unification that makes number words robust to spelling would merge the dirham and the
    Jordanian dinar abbreviations.
    """
    w = unicodedata.normalize("NFC", word)
    w = _HARAKAT_RE.sub("", w).lower().strip(".'’،,")
    if "." not in w:
        w = w.translate(_ALEF)
    return w


# Number-word lexicons. Arabic entries are written in their usual spelling and folded at import.

_EN_UNITS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_EN_TENS: dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_EN_MULT: dict[str, int] = {"thousand": 1000, "million": 1_000_000}
_EN_FRACTIONS: dict[str, Fraction] = {"half": Fraction(1, 2), "quarter": Fraction(1, 4)}

_AR_UNITS_RAW: dict[str, int] = {
    "صفر": 0, "واحد": 1, "واحدة": 1, "اثنان": 2, "اثنين": 2, "اثنتان": 2, "اثنتين": 2,
    "ثلاثة": 3, "ثلاث": 3, "أربعة": 4, "أربع": 4, "خمسة": 5, "خمس": 5, "ستة": 6, "ست": 6,
    "سبعة": 7, "سبع": 7, "ثمانية": 8, "ثماني": 8, "ثمان": 8, "تسعة": 9, "تسع": 9,
}
_AR_TEEN_UNITS_RAW: dict[str, int] = {"أحد": 1, "إحدى": 1, "اثنا": 2, "اثني": 2, "اثنتا": 2, "اثنتي": 2}
_AR_TEN_RAW: tuple[str, ...] = ("عشرة", "عشر")
_AR_TENS_RAW: dict[str, int] = {
    "عشرون": 20, "عشرين": 20, "ثلاثون": 30, "ثلاثين": 30, "أربعون": 40, "أربعين": 40,
    "خمسون": 50, "خمسين": 50, "ستون": 60, "ستين": 60, "سبعون": 70, "سبعين": 70,
    "ثمانون": 80, "ثمانين": 80, "تسعون": 90, "تسعين": 90,
}
_AR_HUNDRED_RAW: tuple[str, ...] = ("مئة", "مائة")
_AR_HUNDREDS_RAW: dict[str, int] = {
    "مئتان": 200, "مئتين": 200, "مئتي": 200, "مائتان": 200, "مائتين": 200, "مائتي": 200,
    "ثلاثمئة": 300, "ثلاثمائة": 300, "أربعمئة": 400, "أربعمائة": 400, "خمسمئة": 500, "خمسمائة": 500,
    "ستمئة": 600, "ستمائة": 600, "سبعمئة": 700, "سبعمائة": 700, "ثمانمئة": 800, "ثمانمائة": 800,
    "ثمانيمئة": 800, "ثمانيمائة": 800, "تسعمئة": 900, "تسعمائة": 900,
}
_AR_MULT_RAW: dict[str, int] = {
    "ألف": 1000, "ألفا": 1000, "ألفاً": 1000, "آلاف": 1000, "الآف": 1000,
    "مليون": 1_000_000, "مليونا": 1_000_000, "مليوناً": 1_000_000, "ملايين": 1_000_000,
}
_AR_DUAL_MULT_RAW: dict[str, int] = {
    "ألفان": 2000, "ألفين": 2000, "ألفي": 2000, "مليونان": 2_000_000, "مليونين": 2_000_000,
}
_AR_FRACTIONS_RAW: dict[str, Fraction] = {"نصف": Fraction(1, 2), "ربع": Fraction(1, 4)}

_UNITS: dict[str, int] = {**_EN_UNITS, **{_fold(k): v for k, v in _AR_UNITS_RAW.items()}}
_TEEN_UNITS: dict[str, int] = {_fold(k): v for k, v in _AR_TEEN_UNITS_RAW.items()}
_TEN_WORDS: frozenset[str] = frozenset(_fold(w) for w in _AR_TEN_RAW)
_TENS: dict[str, int] = {**_EN_TENS, **{_fold(k): v for k, v in _AR_TENS_RAW.items()}}
_HUNDRED_WORDS: frozenset[str] = frozenset({"hundred", *(_fold(w) for w in _AR_HUNDRED_RAW)})
_HUNDREDS: dict[str, int] = {_fold(k): v for k, v in _AR_HUNDREDS_RAW.items()}
_MULT: dict[str, int] = {**_EN_MULT, **{_fold(k): v for k, v in _AR_MULT_RAW.items()}}
_DUAL_MULT: dict[str, int] = {_fold(k): v for k, v in _AR_DUAL_MULT_RAW.items()}
_FRACTIONS: dict[str, Fraction] = {**_EN_FRACTIONS, **{_fold(k): v for k, v in _AR_FRACTIONS_RAW.items()}}
_AND_WORDS: frozenset[str] = frozenset({"and", "و"})
_ARTICLES: frozenset[str] = frozenset({"a", "an"})
_OF_WORDS: frozenset[str] = frozenset({"of"})
_AR_PREFIXES: tuple[str, ...] = ("و", "ب", "ل", "ف", "ك")
_NEGATIVE_WORDS: frozenset[str] = frozenset({"minus", "negative", _fold("ناقص"), _fold("سالب")})
LONG_INTEGER_DIGITS: int = 7
"""Digit-only integers of at least this length are identifiers unless a currency label is adjacent."""

_NUMBER_WORDS: frozenset[str] = frozenset(
    set(_UNITS) | set(_TEEN_UNITS) | _TEN_WORDS | set(_TENS) | _HUNDRED_WORDS | set(_HUNDREDS)
    | set(_MULT) | set(_DUAL_MULT) | set(_FRACTIONS)
)


def _build_label_index() -> tuple[dict[tuple[str, ...], str], dict[tuple[str, ...], frozenset[str]], dict[str, str], int]:
    major: dict[tuple[str, ...], str] = {}
    minor: dict[tuple[str, ...], set[str]] = {}
    dual: dict[str, str] = {}
    longest = 1
    for code, info in CURRENCIES.items():
        for label in (*info.names_en, *info.names_ar, *info.symbols, info.code):
            key = tuple(_fold(w) for w in label.split())
            if not key or any(not k for k in key):
                continue
            longest = max(longest, len(key))
            existing = major.get(key)
            if existing is not None and existing != code:
                raise ValueError(f"currency label {label!r} is shared by {existing} and {code}")
            major[key] = code
        for label in (*info.minor_en, *info.minor_ar):
            key = tuple(_fold(w) for w in label.split())
            longest = max(longest, len(key))
            minor.setdefault(key, set()).add(code)
        for label in info.dual_ar:
            dual[_fold(label)] = code
    return major, {k: frozenset(v) for k, v in minor.items()}, dual, longest


_MAJOR_LABELS, _MINOR_LABELS, _DUAL_LABELS, _MAX_LABEL = _build_label_index()
_LABEL_FIRST: frozenset[str] = frozenset(k[0] for k in (*_MAJOR_LABELS, *_MINOR_LABELS)) | frozenset(_DUAL_LABELS)
_SYMBOL_CHARS: frozenset[str] = frozenset("$€£₹₱¥﷼")


def _build_singular_nouns() -> dict[str, str]:
    """Single-word singular currency nouns (the first English and Arabic names) that denote one unit on their own."""
    nouns: dict[str, str] = {}
    for code, info in CURRENCIES.items():
        for label in (info.names_en[0], info.names_ar[0]):
            key = _fold(label)
            if " " in label or key == code.lower() or (key,) not in _MAJOR_LABELS:
                continue
            if _MAJOR_LABELS[(key,)] == code:
                nouns[key] = code
    return nouns


_SINGULAR_NOUNS: dict[str, str] = _build_singular_nouns()


def lookup_minor_unit(label: str) -> frozenset[str]:
    """ISO 4217 codes whose minor unit the label names ("fils", "فلس", "cents"); empty when it names none."""
    key = tuple(_fold(w) for w in str(label).split())
    if not key:
        return frozenset()
    return _MINOR_LABELS.get(key, frozenset())


def lookup_currency(label: str) -> str | None:
    """ISO 4217 code for a currency name, code, symbol or abbreviation, or None."""
    key = tuple(_fold(w) for w in str(label).split())
    if not key:
        return None
    if key in _MAJOR_LABELS:
        return _MAJOR_LABELS[key]
    if len(key) == 1 and key[0] in _DUAL_LABELS:
        return _DUAL_LABELS[key[0]]
    return None


# Tokeniser

@dataclass
class _Token:
    kind: str  # "num", "word", "sym", "other"
    text: str
    fold: str
    start: int
    end: int
    value: Fraction | None = None
    num_flags: frozenset[str] = frozenset()


_TOKEN_RE = re.compile(
    r"(?P<num>[0-9]+(?:[,.،][0-9]+)*)"
    r"|(?P<word>[^\W\d_]+(?:[.'’\-][^\W\d_]+)*\.?)"
    r"|(?P<sym>[$€£₹₱¥﷼])"
    r"|(?P<other>\S)",
)


def _parse_numeric(raw: str) -> tuple[Fraction, frozenset[str]] | None:
    """Value and separator flags of a digit string, or None when the separators are inconsistent.

    Groups of exactly three digits after a comma are thousands ("2,500"); a comma followed by
    one or two digits is a decimal mark ("250,50"); a single dot is a decimal mark ("250.50",
    "1.000"); dots between three-digit groups with a comma decimal are European thousands
    ("2.500,50").
    """
    s = raw.replace("،", ",")
    flags: set[str] = set()

    def groups_ok(parts: list[str]) -> bool:
        return 1 <= len(parts[0]) <= 3 and all(len(p) == 3 for p in parts[1:]) and all(p.isdigit() for p in parts)

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            int_part, dec = s.rsplit(",", 1)
            parts = int_part.split(".")
        else:
            int_part, dec = s.rsplit(".", 1)
            parts = int_part.split(",")
        if not groups_ok(parts) or not dec.isdigit() or "," in dec or "." in dec:
            return None
        flags |= {"thousands_separator", "decimal"}
        return Fraction(int("".join(parts))) + Fraction(int(dec), 10 ** len(dec)), frozenset(flags)
    if "," in s:
        parts = s.split(",")
        if len(parts) >= 2 and groups_ok(parts):
            flags.add("thousands_separator")
            return Fraction(int("".join(parts))), frozenset(flags)
        if len(parts) == 2 and parts[1].isdigit() and 1 <= len(parts[1]) <= 2 and parts[0].isdigit():
            flags.add("decimal")
            return Fraction(int(parts[0])) + Fraction(int(parts[1]), 10 ** len(parts[1])), frozenset(flags)
        return None
    if "." in s:
        parts = s.split(".")
        if len(parts) == 2:
            flags.add("decimal")
            return Fraction(int(parts[0])) + Fraction(int(parts[1]), 10 ** len(parts[1])), frozenset(flags)
        if groups_ok(parts):
            flags.add("thousands_separator")
            return Fraction(int("".join(parts))), frozenset(flags)
        return None
    return Fraction(int(s)), frozenset()


def _split_prefix(fold: str) -> list[str] | None:
    """Split an Arabic clitic prefix from a number or currency word, or return None."""
    rest = fold
    stripped: list[str] = []
    for _ in range(2):
        if len(rest) < 2 or rest[0] not in _AR_PREFIXES:
            break
        candidate = rest[1:]
        if candidate in _NUMBER_WORDS or candidate in _LABEL_FIRST:
            stripped.append(rest[0])
            rest = candidate
            break
        if candidate[0] in _AR_PREFIXES:
            stripped.append(rest[0])
            rest = candidate
            continue
        break
    if not stripped or rest == fold:
        return None
    if rest not in _NUMBER_WORDS and rest not in _LABEL_FIRST:
        return None
    return [*stripped, rest]


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup or "other"
        raw = m.group(0)
        if kind == "num":
            parsed = _parse_numeric(raw)
            if parsed is None:
                tokens.append(_Token("other", raw, raw, m.start(), m.end()))
            else:
                tokens.append(_Token("num", raw, raw, m.start(), m.end(), parsed[0], parsed[1]))
            continue
        if kind == "word":
            pos = m.start()
            for part in raw.split("-"):
                if part:
                    fold = _fold(part)
                    pieces = None if fold in _NUMBER_WORDS or fold in _LABEL_FIRST else _split_prefix(fold)
                    if pieces:
                        offset = pos
                        for piece in pieces:
                            tokens.append(_Token("word", piece, piece, offset, offset + len(piece)))
                            offset += len(piece)
                        tokens[-1].end = pos + len(part)
                    else:
                        tokens.append(_Token("word", part, fold, pos, pos + len(part)))
                pos += len(part) + 1
            continue
        tokens.append(_Token(kind, raw, raw, m.start(), m.end()))
    return tokens


# Number phrase parser

@dataclass
class _Group:
    hundreds: int | None = None
    tens: int | None = None
    unit: int | None = None
    teen_unit: int | None = None
    frac: Fraction | None = None
    literal: Fraction | None = None

    def empty(self) -> bool:
        return (
            self.hundreds is None and self.tens is None and self.unit is None
            and self.teen_unit is None and self.frac is None and self.literal is None
        )

    def value(self) -> Fraction:
        if self.literal is not None:
            return self.literal + (self.frac or 0)
        return Fraction((self.hundreds or 0) + (self.tens or 0) + (self.unit or 0) + (self.teen_unit or 0)) + (self.frac or 0)


@dataclass
class _Phrase:
    value: Fraction
    start: int  # token index
    end: int  # token index, exclusive
    words: bool
    digits: bool
    flags: set[str]
    currency: str | None = None
    unit_kind: str | None = None  # "major" | "minor"
    minor_set: frozenset[str] = frozenset()
    label_start: int | None = None
    label_end: int | None = None
    word_tokens: int = 0


def _is_number_start(tokens: list[_Token], i: int) -> bool:
    tok = tokens[i]
    if tok.kind == "num":
        return True
    if tok.kind != "word":
        return False
    f = tok.fold
    if (
        f in _UNITS or f in _TEN_WORDS or f in _TENS or f in _HUNDRED_WORDS or f in _HUNDREDS
        or f in _MULT or f in _DUAL_MULT or f in _FRACTIONS
    ):
        return True
    if f in _TEEN_UNITS and i + 1 < len(tokens) and tokens[i + 1].fold in _TEN_WORDS:
        return True
    return False


def _parse_phrase(tokens: list[_Token], i: int) -> tuple[Fraction, int, bool, bool, int, set[str]] | None:
    """Parse the longest number phrase starting at token i.

    Returns (value, end_index, used_words, used_digits, word_token_count, flags) or None.
    """
    total = Fraction(0)
    group = _Group()
    last_mult: int | None = None
    j = i
    n = len(tokens)
    used_words = False
    used_digits = False
    word_tokens = 0
    flags: set[str] = set()
    consumed_any = False

    def group_value_or_one() -> Fraction:
        return group.value() if not group.empty() else Fraction(1)

    while j < n:
        tok = tokens[j]
        if tok.kind == "num":
            if not group.empty() or (consumed_any and last_mult is None):
                break
            group.literal = tok.value
            flags |= set(tok.num_flags)
            used_digits = True
            consumed_any = True
            j += 1
            continue
        if tok.kind != "word":
            break
        f = tok.fold
        if group.literal is not None and f not in _MULT and f not in _FRACTIONS and f not in _AND_WORDS and f not in _ARTICLES:
            break
        if f in _AND_WORDS:
            k = j + 1
            if k < n and tokens[k].fold in _ARTICLES:
                k += 1
            if k < n and (tokens[k].kind == "num" or (tokens[k].kind == "word" and (tokens[k].fold in _NUMBER_WORDS))):
                if tokens[k].kind == "num" and (not group.empty() or last_mult is None):
                    break
                j += 1
                continue
            break
        if f in _ARTICLES:
            if j + 1 < n and (tokens[j + 1].fold in _HUNDRED_WORDS or tokens[j + 1].fold in _MULT or tokens[j + 1].fold in _FRACTIONS):
                j += 1
                continue
            break
        if f in _OF_WORDS:
            if j + 2 < n and tokens[j + 1].fold in _ARTICLES and tokens[j + 2].fold in _MULT:
                j += 1
                continue
            break
        if f in _TEEN_UNITS and j + 1 < n and tokens[j + 1].fold in _TEN_WORDS:
            if not group.empty() and (group.unit is not None or group.tens is not None or group.teen_unit is not None):
                break
            group.teen_unit = 10 + _TEEN_UNITS[f]
            j += 2
            used_words = True
            word_tokens += 2
            consumed_any = True
            continue
        if f in _TEN_WORDS:
            if group.unit is not None and 1 <= group.unit <= 9 and group.tens is None and group.teen_unit is None:
                group.teen_unit = 10 + group.unit
                group.unit = None
            elif group.unit is None and group.teen_unit is None and group.tens is None:
                group.unit = 10
            else:
                break
            j += 1
            used_words = True
            word_tokens += 1
            consumed_any = True
            continue
        if f in _UNITS:
            v = _UNITS[f]
            if group.unit is not None or group.teen_unit is not None or group.frac is not None:
                break
            if group.tens is not None and v >= 10:
                break
            group.unit = v
            j += 1
            used_words = True
            word_tokens += 1
            consumed_any = True
            continue
        if f in _TENS:
            if group.tens is not None or group.teen_unit is not None or group.frac is not None:
                break
            if group.unit is not None and group.unit >= 10:
                break
            group.tens = _TENS[f]
            j += 1
            used_words = True
            word_tokens += 1
            consumed_any = True
            continue
        if f in _HUNDRED_WORDS:
            if group.hundreds is not None or group.frac is not None:
                break
            factor = 1
            if group.unit is not None and group.tens is None and group.teen_unit is None:
                factor = group.unit
                group.unit = None
            elif group.teen_unit is not None and group.tens is None and group.unit is None:
                factor = group.teen_unit
                group.teen_unit = None
            elif not group.empty():
                break
            group.hundreds = factor * 100
            j += 1
            used_words = True
            word_tokens += 1
            consumed_any = True
            continue
        if f in _HUNDREDS:
            if not group.empty():
                break
            group.hundreds = _HUNDREDS[f]
            j += 1
            used_words = True
            word_tokens += 1
            consumed_any = True
            continue
        if f in _MULT:
            mult = _MULT[f]
            if last_mult is not None and mult >= last_mult:
                break
            total += group_value_or_one() * mult
            group = _Group()
            last_mult = mult
            j += 1
            used_words = True
            word_tokens += 1
            consumed_any = True
            continue
        if f in _DUAL_MULT:
            value = _DUAL_MULT[f]
            base = 1000 if value < 1_000_000 else 1_000_000
            if not group.empty() or (last_mult is not None and base >= last_mult):
                break
            total += value
            group = _Group()
            last_mult = base
            j += 1
            used_words = True
            word_tokens += 1
            consumed_any = True
            continue
        if f in _FRACTIONS:
            frac = _FRACTIONS[f]
            if not group.empty():
                if group.frac is not None:
                    break
                group.frac = frac
            elif last_mult is not None:
                total += frac * last_mult
            else:
                group.frac = frac
            j += 1
            used_words = True
            word_tokens += 1
            consumed_any = True
            continue
        break

    if not consumed_any:
        return None
    value = total + (group.value() if not group.empty() else Fraction(0))
    return value, j, used_words, used_digits, word_tokens, flags


def _match_label(tokens: list[_Token], p: int) -> tuple[str, str | None, frozenset[str], int] | None:
    """Match a currency label at token p; returns (kind, code, minor_set, length)."""
    for length in range(min(_MAX_LABEL, len(tokens) - p), 0, -1):
        span = tokens[p : p + length]
        if any(t.kind not in ("word", "sym") for t in span):
            continue
        key = tuple(t.fold if t.kind == "word" else t.text for t in span)
        if key in _MAJOR_LABELS:
            return "major", _MAJOR_LABELS[key], frozenset(), length
        if key in _MINOR_LABELS:
            return "minor", None, _MINOR_LABELS[key], length
        if length == 1 and key[0] in _DUAL_LABELS:
            return "dual", _DUAL_LABELS[key[0]], frozenset(), 1
    return None


def _glued_before_ok(text: str, tokens: list[_Token], idx: int) -> bool:
    tok = tokens[idx]
    if tok.start == 0:
        return True
    prev = text[tok.start - 1]
    if prev.isspace() or prev in "([{\"'«“" or prev in _SYMBOL_CHARS:
        return True
    if prev in "-:/#+.%":
        return False
    if idx > 0 and tokens[idx - 1].end == tok.start:
        before = tokens[idx - 1]
        if before.kind == "word" and (before.fold in _AR_PREFIXES or _match_label(tokens, idx - 1) is not None):
            return True
        if before.kind == "sym":
            return True
    return False


def _glued_after_ok(text: str, tokens: list[_Token], idx: int) -> bool:
    tok = tokens[idx]
    if tok.end >= len(text):
        return True
    nxt = text[tok.end]
    if nxt.isspace() or nxt in ")]}\"'»”,.;!?،؛؟" or nxt in _SYMBOL_CHARS:
        return True
    if nxt in "-:/%" or nxt.isdigit():
        return False
    if idx + 1 < len(tokens) and tokens[idx + 1].start == tok.end:
        after = tokens[idx + 1]
        return after.kind == "word" and _match_label(tokens, idx + 1) is not None
    return False


def _label_before(tokens: list[_Token], i: int) -> tuple[str, str | None, frozenset[str], int] | None:
    """The longest currency label ending directly before token i, or None."""
    for length in range(min(_MAX_LABEL, i), 0, -1):
        before = _match_label(tokens, i - length)
        if before is not None and before[3] == length and before[0] != "dual":
            return before
    return None


def _label_after(tokens: list[_Token], j: int) -> tuple[int, tuple[str, str | None, frozenset[str], int] | None]:
    """The currency label at token j (one article may intervene); returns (label position, label)."""
    n = len(tokens)
    p = j
    if p < n and tokens[p].kind == "word" and tokens[p].fold in _ARTICLES:
        p += 1
    after = _match_label(tokens, p) if p < n else None
    if after is not None and after[0] == "dual":
        after = None
    return p, after


def _is_long_identifier(tokens: list[_Token], i: int) -> bool:
    """True for a digit-only integer of LONG_INTEGER_DIGITS or more digits with no adjacent currency label."""
    tok = tokens[i]
    if tok.kind != "num" or not tok.text.isdigit() or len(tok.text) < LONG_INTEGER_DIGITS:
        return False
    return _label_before(tokens, i) is None and _label_after(tokens, i + 1)[1] is None


def _interrupted_by_unknown_word(tokens: list[_Token], j: int) -> bool:
    """True when a word phrase ending at token j continues with "and" (or a glued و) and a word the lexicon lacks."""
    n = len(tokens)
    if j >= n or tokens[j].kind != "word":
        return False
    fold = tokens[j].fold
    if fold in _AND_WORDS:
        k = j + 1
        if k < n and tokens[k].kind == "word" and tokens[k].fold in _ARTICLES:
            k += 1
        if k >= n or tokens[k].kind != "word":
            return False
        return tokens[k].fold not in _NUMBER_WORDS and _match_label(tokens, k) is None
    if len(fold) > 1 and fold[0] == "و" and fold not in _NUMBER_WORDS and _match_label(tokens, j) is None:
        return fold[1:] not in _NUMBER_WORDS and (fold[1:],) not in _MAJOR_LABELS and (fold[1:],) not in _MINOR_LABELS
    return False


def _collect_phrases(text: str, tokens: list[_Token]) -> list[_Phrase]:
    phrases: list[_Phrase] = []
    i = 0
    n = len(tokens)
    while i < n:
        if not _is_number_start(tokens, i):
            i += 1
            continue
        if tokens[i].kind == "num" and not (_glued_before_ok(text, tokens, i) and _glued_after_ok(text, tokens, i)):
            i += 1
            continue
        if _is_long_identifier(tokens, i):
            i += 1
            continue
        parsed = _parse_phrase(tokens, i)
        if parsed is None:
            i += 1
            continue
        value, j, used_words, used_digits, word_tokens, flags = parsed
        phrase = _Phrase(value, i, j, used_words, used_digits, set(flags), word_tokens=word_tokens)
        if used_words:
            phrase.flags.add("number_words")
        # Currency after the phrase, allowing one article in between.
        p, after = _label_after(tokens, j)
        if after is not None:
            phrase.unit_kind, phrase.currency, phrase.minor_set = after[0], after[1], after[2]
            phrase.label_start, phrase.label_end = p, p + after[3]
            j_consumed = p + after[3]
        else:
            j_consumed = j
            # Currency before the phrase (symbol, code or name ending directly before the number),
            # longest label first.
            before = _label_before(tokens, i)
            if before is not None:
                phrase.unit_kind, phrase.currency, phrase.minor_set = before[0], before[1], before[2]
                phrase.label_start, phrase.label_end = i - before[3], i
            if used_words and _interrupted_by_unknown_word(tokens, j):
                phrase.flags.add("unrecognised_number_word")
        first = i if phrase.label_start is None else min(i, phrase.label_start)
        if first > 0 and tokens[first - 1].kind == "word" and tokens[first - 1].fold in _NEGATIVE_WORDS:
            phrase.flags.add("negative")
        phrases.append(phrase)
        i = j_consumed
    # Arabic dual nouns standing alone denote two units of the currency.
    covered = set()
    for ph in phrases:
        covered.update(range(ph.start, ph.end))
        if ph.label_start is not None:
            covered.update(range(ph.label_start, ph.label_end or ph.label_start))
    for idx, tok in enumerate(tokens):
        if idx in covered or tok.kind != "word" or tok.fold not in _DUAL_LABELS:
            continue
        if idx > 0 and (idx - 1) in covered:
            continue
        ph = _Phrase(Fraction(2), idx, idx + 1, True, False, {"number_words"}, word_tokens=1)
        ph.unit_kind, ph.currency = "major", _DUAL_LABELS[tok.fold]
        ph.label_start, ph.label_end = idx, idx + 1
        phrases.append(ph)
        covered.add(idx)
    # A bare singular currency noun directly before "and" and a minor-unit phrase denotes one unit.
    minor_starts = {ph.start: ph for ph in phrases if ph.unit_kind == "minor"}
    for idx, tok in enumerate(tokens):
        if idx in covered or tok.kind != "word" or tok.fold not in _SINGULAR_NOUNS:
            continue
        if idx + 2 >= len(tokens) or tokens[idx + 1].kind != "word" or tokens[idx + 1].fold not in _AND_WORDS:
            continue
        minor = minor_starts.get(idx + 2)
        if minor is None:
            continue
        code = _SINGULAR_NOUNS[tok.fold]
        if code not in minor.minor_set:
            continue
        ph = _Phrase(Fraction(1), idx, idx + 1, True, False, {"number_words"}, word_tokens=1)
        ph.unit_kind, ph.currency = "major", code
        ph.label_start, ph.label_end = idx, idx + 1
        phrases.append(ph)
    phrases.sort(key=lambda ph: ph.start)
    return phrases


def _resolve_minor_currency(minor_set: frozenset[str], major: str | None, default: str | None) -> str | None:
    if major is not None and major in minor_set:
        return major
    if default is not None and default in minor_set:
        return default
    if len(minor_set) == 1:
        return next(iter(minor_set))
    return None


def _span_text(text: str, tokens: list[_Token], ph: _Phrase) -> str:
    first = ph.start if ph.label_start is None else min(ph.start, ph.label_start)
    last = ph.end if ph.label_end is None else max(ph.end, ph.label_end)
    return text[tokens[first].start : tokens[last - 1].end]


def parse_amount(text: str, default_currency: str | None = None) -> AmountParse:
    """Parse a monetary amount from free text.

    Digits in any supported script with Arabic or Western separators, English and Modern
    Standard Arabic number words (including compound forms such as "ألفان وخمسمئة" and the
    fractions half, quarter, نصف, ربع), and minor-unit phrases such as "250 dirhams and 50
    fils" are recognised. The currency is read from a name, code, symbol or abbreviation
    next to the number; when none is present the ``default_currency`` is used and
    "no_currency" is flagged. The amount is returned in minor units of the resolved currency;
    when the currency is unknown a scale of two decimals is assumed. A fractional part that
    does not fit the currency's minor digits sets "unrecognised_fraction" and leaves the
    amount None, as does the presence of two or more distinct amounts ("multiple_amounts").
    A bare single number word without an adjacent currency ("one", "half", "million") is not
    an amount. A phrase preceded by a spelled-out sign ("negative"), or a word phrase whose
    continuation after "and" is not a number word, sets the flag of its kind and leaves the
    amount None. "mixed_digit_scripts" records a number written in two digit scripts, and
    "secondary_unlabelled_amount" records a labelled amount accompanied by a bare integer
    (the amount stays unresolved as "multiple_amounts").
    """
    flags: set[str] = set()
    source = unicodedata.normalize("NFC", text or "")
    if _AR_DIGIT_RE.search(source):
        flags.add("digit_script_ar")
    if mixed_digit_scripts(source):
        flags.add("mixed_digit_scripts")
    processed = _HARAKAT_RE.sub("", normalize_digits(source))
    tokens = _tokenize(processed)
    phrases = _collect_phrases(processed, tokens)

    default = default_currency.upper() if default_currency else None
    if default is not None and default not in CURRENCIES:
        default = None

    # Join "major and minor" pairs: a major phrase followed by a minor phrase of a compatible currency.
    entries: list[tuple[int | None, str | None, str, bool, set[str], bool]] = []
    # (amount_minor or None, currency, span_text, explicit_currency, flags, valid)
    k = 0
    while k < len(phrases):
        ph = phrases[k]
        nxt = phrases[k + 1] if k + 1 < len(phrases) else None
        joined = False
        if nxt is not None and ph.unit_kind != "minor" and nxt.unit_kind == "minor":
            gap_tokens = tokens[(ph.label_end if ph.label_end is not None and ph.label_end > ph.end else ph.end) : nxt.start]
            gap_ok = all(
                (t.kind == "other" and t.text in ",،") or (t.kind == "word" and t.fold in _AND_WORDS)
                for t in gap_tokens
            )
            minor_code = _resolve_minor_currency(nxt.minor_set, ph.currency, default)
            if gap_ok and minor_code is not None and (ph.currency is None or minor_code == ph.currency):
                code = ph.currency or minor_code
                info = CURRENCIES[code]
                total_minor = ph.value * info.scale + nxt.value
                ph_flags = set(ph.flags) | set(nxt.flags)
                first = ph.start if ph.label_start is None else min(ph.start, ph.label_start)
                last = nxt.label_end if nxt.label_end is not None else nxt.end
                span = processed[tokens[first].start : tokens[last - 1].end]
                explicit = ph.currency is not None or len(nxt.minor_set) == 1
                if total_minor.denominator != 1:
                    ph_flags.add("unrecognised_fraction")
                invalid = bool(ph_flags & INVALIDATING_FLAGS)
                entries.append((None if invalid else int(total_minor), code, span, explicit, ph_flags, not invalid))
                joined = True
                k += 2
        if joined:
            continue
        ph_flags = set(ph.flags)
        if ph.unit_kind == "minor":
            code = _resolve_minor_currency(ph.minor_set, None, default)
            explicit = code is not None and (len(ph.minor_set) == 1 or (default is not None and code == default))
            value = ph.value
            span = _span_text(processed, tokens, ph)
            if value.denominator != 1:
                ph_flags.add("unrecognised_fraction")
            invalid = bool(ph_flags & INVALIDATING_FLAGS)
            entries.append((None if invalid else int(value), code, span, explicit, ph_flags, not invalid))
            k += 1
            continue
        # Major phrase (or phrase without any currency label): a single bare number word is not an
        # amount, unless it is the recognised fragment of an interrupted or negated phrase.
        if ph.words and not ph.digits and ph.currency is None and ph.word_tokens == 1 and not (ph.flags & INVALIDATING_FLAGS):
            k += 1
            continue
        code = ph.currency if ph.currency is not None else default
        explicit = ph.currency is not None
        digits = CURRENCIES[code].minor_digits if code is not None else DEFAULT_MINOR_DIGITS
        minor_value = ph.value * (10**digits)
        span = _span_text(processed, tokens, ph)
        if minor_value.denominator != 1:
            ph_flags.add("unrecognised_fraction")
        invalid = bool(ph_flags & INVALIDATING_FLAGS)
        entries.append((None if invalid else int(minor_value), code, span, explicit, ph_flags, not invalid))
        k += 1

    for entry in entries:
        flags |= entry[4]

    candidates: list[tuple[int, str | None]] = []
    spans: dict[tuple[int, str | None], str] = {}
    labelled: set[tuple[int, str | None]] = set()
    unlabelled: set[tuple[int, str | None]] = set()
    for amount, code, span, explicit, _f, valid in entries:
        if valid and amount is not None:
            (labelled if explicit else unlabelled).add((amount, code))
            if (amount, code) not in candidates:
                candidates.append((amount, code))
                spans[(amount, code)] = span

    if not entries:
        return AmountParse(None, None, "", [], flags)

    if any(not explicit for _a, _c, _s, explicit, _f, _v in entries):
        flags.add("no_currency")

    bad = next((e for e in entries if not e[5]), None)
    if bad is not None:
        return AmountParse(None, bad[1], bad[2], candidates, flags)

    if len(candidates) >= 2:
        flags.add("multiple_amounts")
        if labelled and unlabelled - labelled:
            flags.add("secondary_unlabelled_amount")
        codes = {c for _, c in candidates}
        currency = codes.pop() if len(codes) == 1 else None
        matched = " | ".join(spans[c] for c in candidates)
        return AmountParse(None, currency, matched, candidates, flags)

    amount, code = candidates[0]
    return AmountParse(amount, code, spans[(amount, code)], candidates, flags)


def format_amount(amount_minor: int, currency: str, lang: str) -> str:
    """Render an amount for the confirmation surface.

    English (and any language other than "ar") gives the decimal amount followed by the
    ISO 4217 code, for example "250.00 AED" or "1.000 KWD". Arabic gives Eastern digits
    with the Arabic decimal separator U+066B followed by the first Arabic currency name,
    for example "٢٥٠٫٠٠ درهم". No thousands separators are written in either language.
    """
    info = CURRENCIES.get(str(currency).upper())
    if info is None:
        raise ValueError(f"unknown currency {currency!r}")
    if not isinstance(amount_minor, int) or isinstance(amount_minor, bool):
        raise TypeError("amount_minor must be an integer number of minor units")
    sign = "-" if amount_minor < 0 else ""
    major, minor = divmod(abs(amount_minor), info.scale)
    number = f"{major}.{minor:0{info.minor_digits}d}" if info.minor_digits else str(major)
    if lang == "ar":
        return f"{sign}{eastern_digits(number).replace('.', ARABIC_DECIMAL_SEPARATOR)} {info.names_ar[0]}"
    return f"{sign}{number} {info.code}"
