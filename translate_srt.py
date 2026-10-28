#!/usr/bin/env python3
"""Translate an SRT subtitle file with a local llama.cpp server."""

import argparse
import json
import math
import os
import re
import sys
import textwrap
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

KANA_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f]")
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
NUMBERED_RE = re.compile(r"^\s*\[(\d+)\]\s*(.+?)\s*$")
LANGUAGE_SUFFIX_RE = re.compile(r"\.([A-Za-z]{2,3}(?:-[A-Za-z]{2,4})?)$")
LANGUAGE_CODES = {"ja", "en", "ko", "zh", "fr", "de", "es", "ru", "pt", "it"}
TARGET_NAMES = {
    "zh": "简体中文",
    "zh-hans": "简体中文",
    "zh-cn": "简体中文",
    "zh-hant": "繁體中文",
    "zh-tw": "繁體中文",
    "en": "English",
    "ja": "日本語",
    "ko": "한국어",
}


@dataclass
class Cue:
    index: str
    timing: str
    text: str
    raw: str | None = None
    translated: str | None = None


def parse_srt(content: str) -> list[Cue]:
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", content.strip()):
        if not block.strip():
            continue
        lines = block.splitlines()
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            cues.append(Cue(index="", timing="", text="", raw=block))
            continue
        index = lines[timing_index - 1].strip() if timing_index > 0 else ""
        timing = lines[timing_index].strip()
        text = "\n".join(lines[timing_index + 1 :]).strip()
        cues.append(Cue(index=index, timing=timing, text=text))
    return cues


def render_srt(cues: list[Cue]) -> str:
    blocks = []
    for cue in cues:
        if cue.raw is not None:
            blocks.append(cue.raw)
            continue
        text = cue.translated if cue.translated is not None else cue.text
        lines = ([cue.index] if cue.index else []) + [cue.timing]
        if text:
            lines.append(text)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def infer_source(path: Path) -> str | None:
    match = LANGUAGE_SUFFIX_RE.search(path.stem)
    if match and match.group(1).lower() in LANGUAGE_CODES:
        return match.group(1).lower()
    return None


def default_output(path: Path, target: str) -> Path:
    tag = re.sub(r"[^A-Za-z0-9-]", "", target) or "zh"
    stem = path.stem
    match = LANGUAGE_SUFFIX_RE.search(stem)
    if match:
        stem = stem[: match.start()]
    return path.with_name(f"{stem}.{tag}.srt")


def detect_japanese(cues: list[Cue]) -> bool:
    letter_cues = 0
    kana_cues = 0
    for cue in cues:
        if cue.raw is not None or not LETTER_RE.search(cue.text):
            continue
        letter_cues += 1
        if KANA_RE.search(cue.text):
            kana_cues += 1
    return kana_cues >= max(3, math.ceil(letter_cues * 0.1))


def needs_translation(cue: Cue, file_is_japanese: bool) -> bool:
    if cue.raw is not None or not cue.text.strip():
        return False
    letters = LETTER_RE.findall(cue.text)
    if not letters:
        return False
    if KANA_RE.search(cue.text):
        return True
    if file_is_japanese:
        return True
    han = len(HAN_RE.findall(cue.text))
    return han / len(letters) < 0.6


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def clean_translation(text: str) -> str:
    text = normalize_text(text)
    for opening, closing in (('"', '"'), ("'", "'"), ("“", "”"), ("「", "」"), ("『", "』")):
        if len(text) >= 2 and text.startswith(opening) and text.endswith(closing):
            text = text[1:-1].strip()
    return text


def wrap_text(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False)) or text


def wrap_width(target: str) -> int:
    return 20 if target.lower().startswith("zh") or "中文" in target else 42


def load_glossary(path: Path) -> list[tuple[str, str]]:
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        source, target = line.split("=", 1)
        source = source.strip()
        target = target.strip()
        if source and target:
            terms.append((source, target))
    return terms


def build_batch_prompt(segments: list[str], target_name: str, glossary: list[tuple[str, str]]) -> str:
    header = [
        f"Translate the following subtitle lines into {target_name}, without additional explanation.",
        "Keep the [number] markers and output exactly one line per input line.",
    ]
    if glossary:
        header.append("Use the following terminology exactly:")
        header.extend(f"{source} = {target}" for source, target in glossary)
    body = "\n".join(f"[{number}] {text}" for number, text in enumerate(segments, 1))
    return "\n".join(header) + "\n\n" + body


def build_single_prompt(text: str, target_name: str, glossary: list[tuple[str, str]]) -> str:
    header = [f"Translate the following segment into {target_name}, without additional explanation."]
    if glossary:
        header.append("Use the following terminology exactly:")
        header.extend(f"{source} = {target}" for source, target in glossary)
    return "\n".join(header) + "\n\n" + text


def chat_completion(args: argparse.Namespace, prompt: str) -> str:
    payload = json.dumps(
        {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 8192,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("response has no choices")
    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("response is empty")
    return content


def request_with_retry(args: argparse.Namespace, prompt: str, attempts: int = 3) -> str:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return chat_completion(args, prompt)
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
            last_error = error
            if attempt < attempts:
                time.sleep(attempt)
    raise RuntimeError(str(last_error))


def parse_numbered(content: str, expected: int) -> list[str]:
    found = {}
    for line in content.splitlines():
        match = NUMBERED_RE.match(line)
        if not match:
            continue
        number = int(match.group(1))
        if 1 <= number <= expected and number not in found:
            found[number] = clean_translation(match.group(2))
    if len(found) != expected or any(not found[number] for number in range(1, expected + 1)):
        raise ValueError(f"expected {expected} numbered lines, got {len(found)}")
    return [found[number] for number in range(1, expected + 1)]


def translate_batch(
    texts: list[str], args: argparse.Namespace, target_name: str, glossary: list[tuple[str, str]]
) -> list[str]:
    prompt = build_batch_prompt(texts, target_name, glossary)
    return parse_numbered(request_with_retry(args, prompt), len(texts))


def translate_single(text: str, args: argparse.Namespace, target_name: str, glossary: list[tuple[str, str]]) -> str:
    prompt = build_single_prompt(text, target_name, glossary)
    translation = clean_translation(request_with_retry(args, prompt, attempts=2))
    if not translation:
        raise ValueError("empty translation")
    return translation


def translate_group(
    items: list[tuple[int, str]], args: argparse.Namespace, target_name: str, glossary: list[tuple[str, str]]
) -> list[tuple[int, str | None]]:
    try:
        results = translate_batch([text for _, text in items], args, target_name, glossary)
    except Exception as error:
        print(f"batch failed ({error}); retrying {len(items)} cues individually", file=sys.stderr)
        results = []
        for index, text in items:
            try:
                results.append(translate_single(text, args, target_name, glossary))
            except Exception as single_error:
                print(f"cue {index} failed: {single_error}", file=sys.stderr)
                results.append(None)
    return [(index, translation) for (index, _), translation in zip(items, results)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate an SRT subtitle file with a local llama.cpp server.")
    parser.add_argument("input", type=Path, help="Input SRT file")
    parser.add_argument("-o", "--output", type=Path, help="Output SRT file (default: <input>.<target>.srt)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="llama.cpp OpenAI-compatible base URL")
    parser.add_argument("--model", default="tencent/Hy-MT2-30B-A3B", help="Model name or alias")
    parser.add_argument("--target", default="zh", help="Target language code or name")
    parser.add_argument("--batch-size", type=int, default=20, help="Cues per translation request")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent translation requests")
    parser.add_argument("--glossary", type=Path, help="Terminology file with source=target per line")
    parser.add_argument("--limit", type=int, help="Only translate the first N cues")
    parser.add_argument("--timeout", type=float, default=600, help="Request timeout in seconds")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    cues = parse_srt(args.input.read_text(encoding="utf-8-sig"))
    output = args.output or default_output(args.input, args.target)
    target_name = TARGET_NAMES.get(args.target.lower(), args.target)
    glossary = load_glossary(args.glossary) if args.glossary else []
    file_is_japanese = infer_source(args.input) == "ja" or detect_japanese(cues)
    scope = cues[: args.limit] if args.limit else cues
    candidates = [(position, cue) for position, cue in enumerate(scope) if needs_translation(cue, file_is_japanese)]

    groups = []
    for start in range(0, len(candidates), args.batch_size):
        chunk = candidates[start : start + args.batch_size]
        groups.append([(position, normalize_text(cue.text)) for position, cue in chunk])

    translated_count = 0
    completed_count = 0
    failed_positions = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(translate_group, group, args, target_name, glossary): len(group) for group in groups
        }
        for future in as_completed(futures):
            for position, translation in future.result():
                if translation is None:
                    failed_positions.append(position)
                else:
                    cues[position].translated = wrap_text(translation, wrap_width(args.target))
                    translated_count += 1
            completed_count += futures[future]
            write_atomic(output, render_srt(cues))
            print(f"progress: {completed_count}/{len(candidates)} cues processed", file=sys.stderr)

    write_atomic(output, render_srt(cues))
    skipped = len(scope) - len(candidates)
    print(
        f"done: total={len(scope)} translated={translated_count} skipped={skipped} failed={len(failed_positions)}",
        file=sys.stderr,
    )
    if failed_positions:
        labels = ", ".join(str(cues[position].index or position + 1) for position in sorted(failed_positions))
        print(f"failed cues: {labels}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
