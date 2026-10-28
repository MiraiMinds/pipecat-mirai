"""The instant-voice cookbook's evaluation arithmetic: normalising, error rates, percentiles, the summary."""

import sys
from pathlib import Path

import pytest

COOKBOOK = Path(__file__).resolve().parents[2] / "packages/fourier-frequency/cookbook/instant-voice"
sys.path.insert(0, str(COOKBOOK))

import evalkit  # noqa: E402


def test_normalize_drops_case_punctuation_and_symbols():
    assert evalkit.normalize("Hello, World!  It's ₹1,499.") == "hello world it s 1499"
    assert evalkit.normalize("आपका order, कल deliver होगा।") == "आपका order कल deliver होगा"
    assert evalkit.normalize("१५ नवंबर") == "15 नवंबर"  # Devanagari digits
    assert evalkit.normalize("૧૦:૩૦") == "10 30"  # Gujarati digits, and the colon is a break


def test_normalize_keeps_vowel_signs():
    # Combining marks are part of the letters: dropping them would hide real differences.
    assert evalkit.normalize("नमस्ते") == "नमस्ते" and evalkit.normalize("નમસ્તે") == "નમસ્તે"


def test_normalize_evens_out_nukta_and_chandrabindu():
    assert evalkit.normalize("ज़रूर हूँ") == evalkit.normalize("जरूर हूं")
    assert evalkit.cer("तारीख़ को", "तारीख को") == 0.0


def test_a_correct_reading_scores_zero_against_how_it_is_heard():
    for language in ("hi", "en", "gu"):
        for sent, heard in evalkit.SENTENCES[language]:
            assert evalkit.cer(heard, heard) == 0.0 and evalkit.normalize(heard) == evalkit.normalize(heard)
            if any(ch.isdigit() for ch in sent):
                assert sent != heard  # digits are heard as words


def test_your_own_lines():
    lines = evalkit.read_sentences("Hello there.\n\nOrder 42 ships.\torder forty two ships\n")
    assert lines == [("Hello there.", "Hello there."), ("Order 42 ships.", "order forty two ships")]


def test_edit_distance():
    assert evalkit.edit_distance("kitten", "sitting") == 3
    assert evalkit.edit_distance("", "abc") == 3 and evalkit.edit_distance("abc", "") == 3
    assert evalkit.edit_distance(["a", "b", "c"], ["a", "c"]) == 1


def test_error_rates():
    assert evalkit.cer("नमस्ते जी", "नमस्ते जी।") == 0.0
    assert evalkit.cer("abcd", "abed") == 0.25
    assert evalkit.cer("ab cd", "abcd") == 0.0  # spaces are a writing choice
    assert evalkit.wer("your order ships today", "your order ship today") == 0.25
    assert evalkit.wer("one two", "one two three four") == 1.0
    assert evalkit.cer("", "") == 0.0 and evalkit.cer("", "x") == 1.0
    assert evalkit.wer("", "") == 0.0


def test_percentile_is_nearest_rank():
    values = [100, 300, 200, 500, 400]
    assert evalkit.percentile(values, 50) == 300
    assert evalkit.percentile(values, 95) == 500
    assert evalkit.percentile(values, 0) == 100
    assert evalkit.percentile([7], 95) == 7
    assert evalkit.percentile([], 50) is None and evalkit.percentile([None, 4], 50) == 4


def test_sentences_for_a_language():
    hindi, language = evalkit.sentences_for("hi-IN")
    assert language == "hi" and len(hindi) >= 10 and all(sent and heard for sent, heard in hindi)
    gujarati, language = evalkit.sentences_for("gu")
    assert language == "gu" and len(gujarati) >= 10
    tamil, language = evalkit.sentences_for("ta")
    assert language == "en" and tamil == evalkit.SENTENCES["en"]


def row(voice, role, first, total, audio, cer=None, error=None, socket="api", **extra):
    return {
        "voice": voice,
        "role": role,
        "run": 1,
        "text": "x",
        "first_audio_ms": first,
        "total_ms": total,
        "audio_secs": audio,
        "rtf": total / 1000 / audio if total and audio else None,
        "chars_billed": 10,
        "cer": cer,
        "wer": cer,
        "error": error,
        "socket": socket,
        **extra,
    }


def test_summary_per_voice_in_order():
    rows = [
        row("cv_1", "voice under test", 200, 1000, 2.0, cer=0.1),
        row("shruti", "baseline", 150, 800, 2.0, cer=0.0, socket="edge"),
        row("cv_1", "voice under test", 400, 1500, 3.0, cer=0.3),
        row("cv_1", "voice under test", None, None, 0.0, error="refused"),
    ]
    (clone, base) = evalkit.summarize(rows)
    assert clone["voice"] == "cv_1" and base["voice"] == "shruti"
    assert clone["sentences"] == 3 and clone["errors"] == 1
    assert clone["first_audio_p50_ms"] == 200 and clone["first_audio_p95_ms"] == 400
    assert clone["rtf_p50"] == pytest.approx(0.5)
    assert clone["cer_mean"] == pytest.approx(0.2) and clone["audio_secs"] == 5.0
    assert clone["chars_billed"] == 20 and clone["similarity_mean"] is None
    assert clone["sockets"] == ["api"] and base["sockets"] == ["edge"]


def test_table_and_report_render_every_voice():
    rows = [row("cv_1", "voice under test", 210, 900, 2.0, cer=0.05, similarity=0.81)]
    summary = evalkit.summarize(rows)
    text = evalkit.table(summary)
    assert "first audio p50 ms" in text and "cv_1" in text and "5.0%" in text and "0.81" in text
    md = evalkit.report(summary, rows, {"transport": "websocket", "runs": 1})
    assert md.startswith("# Voice evaluation") and "| cv_1 | voice under test |" in md
