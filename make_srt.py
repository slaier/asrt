#!/usr/bin/env python3
"""Generate an SRT subtitle file from transcript text and word-level timestamps (JSON)."""

import argparse
import json
from pathlib import Path

SAMPLE_RATE = 16000  # Audio sample rate (Hz), matching the silero_vad 16k model
SENTENCE_END = set("。？")  # Sentence-ending punctuation; split cues after these


def format_time(samples):
    """Convert sample counts to SRT time format HH:MM:SS,mmm."""
    ms = int(round(samples * 1000 / SAMPLE_RATE))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    seconds, ms = divmod(ms, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def parse(text: str, words):
    text = text.replace(" ", "")

    search_index = 0
    start_index = 0
    start_time = words[0]["start_sample"]

    for index, word in enumerate(words):
        offset = text.index(word["word"][0], search_index)
        if offset > 0 and text[offset - 1] in SENTENCE_END:
            end_time = words[index - 1]["end_sample"]
            yield (start_time, end_time, text[start_index:offset])
            start_index = offset
            start_time = word["start_sample"]
        search_index = offset + 1
        for char in word["word"][1:]:
            offset = text.index(char, search_index)
            search_index = offset + 1

    end_time = words[-1]["end_sample"]
    yield (start_time, end_time, text[start_index:])

def main():
    """Read transcript and word-level timestamps, then generate an SRT subtitle file."""
    parser = argparse.ArgumentParser(description="Generate SRT from transcript and word timestamps.")
    parser.add_argument("--transcript", type=Path, default=Path("workflow_outputs/transcript.txt"))
    parser.add_argument("--words", type=Path, default=Path("workflow_outputs/words.json"))
    parser.add_argument("-o", "--output", type=Path, default=Path("workflow_outputs/subtitles.srt"))
    args = parser.parse_args()

    text = args.transcript.read_text(encoding="utf-8")
    words = json.loads(args.words.read_text(encoding="utf-8"))

    result = "\n\n".join(
        f"{i + 1}\n{format_time(start)} --> {format_time(end)}\n{item}"
        for i, (start, end, item) in enumerate(parse(text, words)))
    args.output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
