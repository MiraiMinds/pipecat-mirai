#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""The evaluation's arithmetic and its sentences: plain Python, no network.

``evaluate.py`` speaks each sentence, times it, transcribes the audio back
and hands the rows to :func:`summarize`; :func:`table` and :func:`report`
print and write what came out. Kept apart so it can be tested on its own.
"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections.abc import Iterable, Sequence

# Phone-agent lines: greetings, numbers, dates, money, brand names and the
# English words Indian callers and agents mix in. The same lines go to every
# voice under test, so the comparison is like for like.
#
# Each line is (what is sent, how it is heard). Speech-to-text writes what it
# hears in the language's own script, with numbers as words ("Sharma Motors"
# comes back as "शर्मा मोटर्स", "₹1,499" as "वन थाउज़ेंड ..."), so the error
# rates are measured against the second form: a correct reading scores 0.
SENTENCES: dict[str, list[tuple[str, str]]] = {
    "hi": [
        ("नमस्ते, मैं Sharma Motors की तरफ़ से बोल रही हूँ।", "नमस्ते मैं शर्मा मोटर्स की तरफ़ से बोल रही हूँ"),
        (
            "आपकी गाड़ी की service 15 नवंबर को due है, क्या मैं slot book कर दूँ?",
            "आपकी गाड़ी की सर्विस फिफ्टीन नवंबर को ड्यू है क्या मैं स्लॉट बुक कर दूँ",
        ),
        (
            "आपका order नंबर 48213 कल शाम तक deliver हो जाएगा।",
            "आपका ऑर्डर नंबर फोर एट टू वन थ्री कल शाम तक डिलीवर हो जाएगा",
        ),
        (
            "कुल amount ₹1,499 है, आप UPI से भी pay कर सकते हैं।",
            "कुल अमाउंट वन थाउज़ेंड फोर हंड्रेड नाइनटी नाइन रुपीज़ है आप यू पी आई से भी पे कर सकते हैं",
        ),
        ("क्या आप मुझे अपना pin code बता सकते हैं?", "क्या आप मुझे अपना पिन कोड बता सकते हैं"),
        ("ठीक है, मैंने आपकी request note कर ली है।", "ठीक है मैंने आपकी रिक्वेस्ट नोट कर ली है"),
        (
            "हमारी team आपको 24 घंटे के अंदर call back करेगी।",
            "हमारी टीम आपको ट्वेंटी फोर घंटे के अंदर कॉल बैक करेगी",
        ),
        (
            "आपका appointment सोमवार सुबह 10:30 बजे confirm हो गया है।",
            "आपका अपॉइंटमेंट सोमवार सुबह टेन थर्टी बजे कन्फर्म हो गया है",
        ),
        (
            "माफ़ कीजिए, मैं आपकी बात ठीक से सुन नहीं पाई, क्या आप दोहरा सकते हैं?",
            "माफ़ कीजिए मैं आपकी बात ठीक से सुन नहीं पाई क्या आप दोहरा सकते हैं",
        ),
        (
            "आपकी EMI ₹3,250 हर महीने की 5 तारीख़ को कटेगी।",
            "आपकी ई एम आई थ्री थाउज़ेंड टू हंड्रेड फिफ्टी रुपीज़ हर महीने की फाइव तारीख़ को कटेगी",
        ),
        (
            "Amazon और Flipkart दोनों पर यह offer 31 अक्टूबर तक valid है।",
            "ऐमज़ॉन और फ़्लिपकार्ट दोनों पर ये ऑफ़र थर्टी वन अक्टूबर तक वैलिड है",
        ),
        ("थैंक यू, आपका दिन शुभ हो!", "थैंक यू आपका दिन शुभ हो"),
    ],
    "en": [
        ("Hello, this is Asha calling from Sharma Motors.", "hello this is asha calling from sharma motors"),
        (
            "Your car service is due on the 15th of November. Shall I book a slot for you?",
            "your car service is due on the fifteenth of november shall i book a slot for you",
        ),
        (
            "Your order number 48213 will be delivered by tomorrow evening.",
            "your order number four eight two one three will be delivered by tomorrow evening",
        ),
        (
            "The total comes to ₹1,499, and you can pay by UPI as well.",
            "the total comes to one thousand four hundred ninety nine rupees and you can pay by upi as well",
        ),
        ("Could you please tell me your PIN code?", "could you please tell me your pin code"),
        ("Sure, I have noted down your request.", "sure i have noted down your request"),
        (
            "Our team will call you back within 24 hours.",
            "our team will call you back within twenty four hours",
        ),
        (
            "Your appointment is confirmed for Monday at 10:30 in the morning.",
            "your appointment is confirmed for monday at ten thirty in the morning",
        ),
        (
            "Sorry, I couldn't hear you clearly. Could you please repeat that?",
            "sorry i couldn't hear you clearly could you please repeat that",
        ),
        (
            "Your EMI of ₹3,250 will be debited on the 5th of every month.",
            "your emi of three thousand two hundred fifty rupees will be debited on the fifth of every month",
        ),
        (
            "This offer is valid on Amazon and Flipkart until the 31st of October.",
            "this offer is valid on amazon and flipkart until the thirty first of october",
        ),
        ("Thank you for your time, and have a great day!", "thank you for your time and have a great day"),
    ],
    "gu": [
        ("નમસ્તે, હું Sharma Motors તરફથી વાત કરું છું.", "નમસ્તે હું શર્મા મોટર્સ તરફથી વાત કરું છું"),
        (
            "તમારી ગાડીની service 15 નવેમ્બરે due છે, શું હું slot book કરી દઉં?",
            "તમારી ગાડીની સર્વિસ ફિફ્ટીન નવેમ્બરે ડ્યુ છે શું હું સ્લોટ બુક કરી દઉં",
        ),
        (
            "તમારો order નંબર 48213 કાલે સાંજ સુધીમાં deliver થઈ જશે.",
            "તમારો ઓર્ડર નંબર ફોર એટ ટુ વન થ્રી કાલે સાંજ સુધીમાં ડિલિવર થઈ જશે",
        ),
        (
            "કુલ રકમ ₹1,499 છે, તમે UPI થી પણ pay કરી શકો છો.",
            "કુલ રકમ વન થાઉઝન્ડ ફોર હન્ડ્રેડ નાઈન્ટી નાઈન રૂપીસ છે તમે યુપીઆઈથી પણ પે કરી શકો છો",
        ),
        ("શું તમે મને તમારો pin code કહી શકો?", "શું તમે મને તમારો પિન કોડ કહી શકો"),
        ("બરાબર, મેં તમારી request નોંધી લીધી છે.", "બરાબર મેં તમારી રિક્વેસ્ટ નોંધી લીધી છે"),
        ("અમારી team તમને 24 કલાકમાં call back કરશે.", "અમારી ટીમ તમને ટ્વેન્ટી ફોર કલાકમાં કોલ બેક કરશે"),
        (
            "તમારી appointment સોમવારે સવારે 10:30 વાગ્યે confirm થઈ ગઈ છે.",
            "તમારી અપોઇન્ટમેન્ટ સોમવારે સવારે ટેન થર્ટી વાગ્યે કન્ફર્મ થઈ ગઈ છે",
        ),
        (
            "માફ કરશો, હું તમારી વાત બરાબર સાંભળી ન શકી, ફરીથી કહેશો?",
            "માફ કરશો હું તમારી વાત બરાબર સાંભળી ન શકી ફરીથી કહેશો",
        ),
        ("થેન્ક યુ, તમારો દિવસ શુભ રહે!", "થેન્ક યુ તમારો દિવસ શુભ રહે"),
    ],
}

# Indian-script digits read as ASCII ones, so "१५" and "15" compare equal.
_DIGITS = {
    chr(base + i): str(i)
    for base in (0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66)
    for i in range(10)
}


# Spellings speech-to-text and writers vary on that don't change the word:
# nukta dropped (ज़ = ज), chandrabindu as anusvara (हूँ = हूं).
_SPELLING = str.maketrans({"\u093c": None, "\u0abc": None, "\u0901": "\u0902", "\u0a81": "\u0a82"})


def sentences_for(language: str) -> tuple[list[tuple[str, str]], str]:
    """The built-in lines for ``language`` and the language they are in (English when there are none)."""
    base = (language or "en").split("-", 1)[0].lower()
    if base in SENTENCES:
        return list(SENTENCES[base]), base
    return list(SENTENCES["en"]), "en"


def read_sentences(text: str) -> list[tuple[str, str]]:
    """Lines of your own: one per line, optionally ``what is sent<TAB>how it is heard``."""
    out = []
    for line in text.splitlines():
        sent, _, heard = line.partition("\t")
        if sent.strip():
            out.append((sent.strip(), heard.strip() or sent.strip()))
    return out


def normalize(text: str) -> str:
    """Lower case, no punctuation or symbols, ASCII digits, single spaces, nukta and chandrabindu evened out.

    Light on purpose: what is left to differ is the words.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.translate(_SPELLING)
    text = re.sub(r"(?<=\d),(?=\d)", "", "".join(_DIGITS.get(ch, ch) for ch in text))  # 1,499 -> 1499
    out = []
    for ch in text:
        category = unicodedata.category(ch)
        if category[0] in "PS":
            out.append(" " if category in ("Pd", "Ps", "Pe", "Pi", "Pf", "Po") else "")
        elif category[0] == "Z" or ch in "\t\n\r":
            out.append(" ")
        else:
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def edit_distance(reference: Sequence, hypothesis: Sequence) -> int:
    """Levenshtein distance: substitutions, insertions and deletions, each 1."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        current = [i]
        for j, h in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate after :func:`normalize`, spaces left out (they are a writing choice)."""
    ref = normalize(reference).replace(" ", "")
    hyp = normalize(hypothesis).replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate after :func:`normalize`."""
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def percentile(values: Iterable[float], q: float) -> float | None:
    """Nearest-rank percentile (``q`` in 0..100), as the repo's benchmarks report; None for no values."""
    ordered = sorted(v for v in values if v is not None)
    if not ordered:
        return None
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _mean(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.fmean(present) if present else None


def summarize(rows: list[dict]) -> list[dict]:
    """One summary per voice, in the order the voices first appear in ``rows``."""
    order: list[str] = []
    by_voice: dict[str, list[dict]] = {}
    for row in rows:
        if row["voice"] not in by_voice:
            order.append(row["voice"])
            by_voice[row["voice"]] = []
        by_voice[row["voice"]].append(row)
    out = []
    for voice in order:
        voice_rows = by_voice[voice]
        spoken = [r for r in voice_rows if not r.get("error")]
        first = [r["first_audio_ms"] for r in spoken if r.get("first_audio_ms") is not None]
        out.append(
            {
                "voice": voice,
                "role": voice_rows[0].get("role", ""),
                "sentences": len(voice_rows),
                "errors": len(voice_rows) - len(spoken),
                "first_audio_p50_ms": percentile(first, 50),
                "first_audio_p95_ms": percentile(first, 95),
                "total_p50_ms": percentile([r.get("total_ms") for r in spoken], 50),
                "rtf_p50": percentile([r.get("rtf") for r in spoken], 50),
                "audio_secs": sum(r.get("audio_secs") or 0.0 for r in spoken),
                "chars_billed": sum(r.get("chars_billed") or 0 for r in spoken),
                "cer_mean": _mean(r.get("cer") for r in spoken),
                "wer_mean": _mean(r.get("wer") for r in spoken),
                "similarity_mean": _mean(r.get("similarity") for r in spoken),
                "sockets": sorted({r["socket"] for r in voice_rows if r.get("socket")}),
            }
        )
    return out


def _fmt(value, kind: str = "") -> str:
    if value is None:
        return "–"
    if kind == "ms":
        return f"{value:.0f}"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "ratio":
        return f"{value:.2f}"
    if kind == "secs":
        return f"{value:.1f}"
    return str(value)


COLUMNS = [
    ("voice", "voice", ""),
    ("role", "role", ""),
    ("sentences", "n", ""),
    ("errors", "errors", ""),
    ("first_audio_p50_ms", "first audio p50 ms", "ms"),
    ("first_audio_p95_ms", "p95 ms", "ms"),
    ("total_p50_ms", "total p50 ms", "ms"),
    ("rtf_p50", "RTF p50", "ratio"),
    ("cer_mean", "CER", "pct"),
    ("wer_mean", "WER", "pct"),
    ("similarity_mean", "similarity", "ratio"),
]


def table(summary: list[dict]) -> str:
    """The summary as an aligned plain-text table."""
    header = [title for _, title, _ in COLUMNS]
    body = [[_fmt(s.get(key), kind) for key, _, kind in COLUMNS] for s in summary]
    widths = [max(len(row[i]) for row in [header, *body]) for i in range(len(header))]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in [header, *body]
    ]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(lines)


def markdown_table(summary: list[dict]) -> str:
    header = "| " + " | ".join(title for _, title, _ in COLUMNS) + " |"
    rule = "|" + "|".join(" --- " for _ in COLUMNS) + "|"
    body = ["| " + " | ".join(_fmt(s.get(key), kind) for key, _, kind in COLUMNS) + " |" for s in summary]
    return "\n".join([header, rule, *body])


def report(summary: list[dict], rows: list[dict], meta: dict) -> str:
    """``report.md``: the settings, the summary, and every sentence's numbers and transcript."""
    lines = [
        "# Voice evaluation",
        "",
        f"- When: {meta.get('when', '')}",
        f"- Transport: {meta.get('transport', '')}, {meta.get('sample_rate', '')} Hz, "
        f"{meta.get('runs', '')} run(s) of {meta.get('sentence_count', '')} sentences "
        f"({meta.get('language', '')})",
        f"- Speech-to-text check: {meta.get('asr', '')}",
        f"- Speaker similarity: {meta.get('similarity', '')}",
        "",
        markdown_table(summary),
        "",
        "First audio is the time from sending a sentence to its first audio byte; total is to its last.",
        "RTF is total time over audio length (below 1: the audio arrived faster than it plays). CER and",
        "WER compare Mira STT's transcript of the audio with how the line should be heard (numbers as",
        "words, English words in the line's script), after removing punctuation and case. Compare voices",
        "with each other on the same lines: a reading that differs but is still right counts as an error.",
        "",
        "## Sentences",
        "",
        "| voice | run | first audio ms | RTF | CER | sent | should be heard | Mira STT heard |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        heard = (r.get("transcript") or r.get("error") or "").replace("|", "/")
        lines.append(
            f"| {r['voice']} | {r['run']} | {_fmt(r.get('first_audio_ms'), 'ms')} "
            f"| {_fmt(r.get('rtf'), 'ratio')} | {_fmt(r.get('cer'), 'pct')} "
            f"| {r['text'].replace('|', '/')} | {(r.get('heard') or '').replace('|', '/')} | {heard} |"
        )
    return "\n".join(lines) + "\n"
