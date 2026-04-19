#!/usr/bin/env python3
"""
School Helper Podcast module: generate long podcast scripts and synthesize audio with Kokoro TTS.
"""

import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from openai import OpenAI
from scripts.indexer import get_chroma_collection

# ── Config ──────────────────────────────────────────────────────────────────

RAG_MODEL = os.environ.get("RAG_MODEL", "gpt-4o-mini")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PODCAST_DIR = DATA_DIR / "podcasts"

VOICE_HOST1 = "af_heart"    # Alex — American female
VOICE_HOST2 = "am_michael"  # Jordan — American male

# ── OpenAI client ────────────────────────────────────────────────────────────

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ── Script generation ────────────────────────────────────────────────────────

SCRIPT_SYSTEM = (
    "You are an expert podcast script writer. "
    "Write long, detailed, engaging two-host podcast scripts that feel natural and conversational. "
    "Every single line of dialogue must start with exactly [ALEX]: or [JORDAN]:"
)

SCRIPT_PROMPT = """Write a full-length podcast episode between two hosts:

- **[ALEX]**: Enthusiastic, curious, asks great questions, connects ideas to everyday life
- **[JORDAN]**: Deep thinker, explains concepts clearly, loves finding surprising insights

{topic_line}

Documents to cover:
{summaries}

Content from the documents:
{content}

Write a COMPLETE, LONG podcast episode that:
1. Opens with a compelling hook that grabs attention (1-2 min of dialogue)
2. Introduces the topic and why listeners should care (2-3 min)
3. Does a thorough DEEP DIVE into every major concept, idea, and finding from the documents (15-20 min)
4. Highlights surprising connections between documents when multiple are selected
5. Weaves in specific facts, examples, and direct references from the content
6. Includes natural back-and-forth, follow-up questions, "wow" moments, and occasional light humour
7. Ends with clear key takeaways and a call to reflection (2-3 min)

FORMAT RULES — strictly follow these:
- Every line must start with [ALEX]: or [JORDAN]:
- No stage directions, no scene headers, no empty lines between dialogue
- Aim for 3500-4500 words of actual dialogue
- Keep each speaking turn to 2-5 sentences so it feels conversational
"""


def generate_podcast_script(doc_ids: list[str], kb: dict, course_id: str = "", topic: str = "") -> dict:
    """
    Generate a long, engaging two-host podcast script from selected documents.
    Returns {script, doc_count, word_count}.
    """
    # Get the course's documents
    if course_id and course_id in kb.get("courses", {}):
        course_docs = kb["courses"][course_id]["documents"]
    else:
        course_docs = {}

    collection = get_chroma_collection(course_id)

    summaries_lines = []
    content_blocks = []

    for doc_id in doc_ids:
        doc = course_docs.get(doc_id)
        if not doc:
            continue

        summary = doc.get("summary", "No summary available.")
        summaries_lines.append(f"- {doc['title']}: {summary}")

        results = collection.get(
            where={"doc_id": doc_id},
            include=["documents"],
        )
        chunks = results.get("documents", [])[:20]
        content_blocks.append(f"=== {doc['title']} ===\n" + "\n\n".join(chunks))

    if not summaries_lines:
        raise ValueError("No valid documents found for selected IDs.")

    content_text = "\n\n".join(content_blocks)
    words = content_text.split()
    if len(words) > 14000:
        content_text = " ".join(words[:14000]) + "\n\n[...content truncated...]"

    topic_line = f"The episode's central focus is: {topic}" if topic.strip() else "Cover all major themes across the selected documents."

    prompt = SCRIPT_PROMPT.format(
        topic_line=topic_line,
        summaries="\n".join(summaries_lines),
        content=content_text,
    )

    client = _get_client()
    resp = client.chat.completions.create(
        model=RAG_MODEL,
        messages=[
            {"role": "system", "content": SCRIPT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.72,
        max_tokens=6000,
    )

    script = resp.choices[0].message.content.strip()
    word_count = len(script.split())

    return {
        "script": script,
        "doc_count": len(summaries_lines),
        "word_count": word_count,
    }


# ── Audio synthesis ──────────────────────────────────────────────────────────

def synthesize_speech(script: str) -> str:
    """
    Convert a two-host podcast script to a WAV file using Kokoro TTS.
    Returns the filename (relative to PODCAST_DIR).
    Raises ImportError if kokoro/soundfile are not installed.
    """
    try:
        import numpy as np
        import soundfile as sf
        from kokoro import KPipeline
    except ImportError as e:
        raise ImportError(
            f"Kokoro TTS dependencies missing: {e}. "
            "Run: pip install kokoro soundfile misaki[en]"
        ) from e

    PODCAST_DIR.mkdir(parents=True, exist_ok=True)

    # Parse [ALEX]: / [JORDAN]: lines
    lines = []
    for raw in script.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        m = re.match(r'\[(ALEX|JORDAN)\]:\s*(.*)', raw, re.IGNORECASE)
        if m:
            speaker = m.group(1).upper()
            text = m.group(2).strip()
            if text:
                lines.append((speaker, text))

    if not lines:
        raise ValueError("No valid [ALEX]: / [JORDAN]: dialogue lines found in script.")

    pipeline = KPipeline(lang_code="a")  # American English
    sample_rate = 24000

    silence_turn = np.zeros(int(sample_rate * 0.45), dtype=np.float32)   # between speakers
    silence_same = np.zeros(int(sample_rate * 0.18), dtype=np.float32)   # same speaker continues

    segments = []
    prev_speaker = None

    for speaker, text in lines:
        voice = VOICE_HOST1 if speaker == "ALEX" else VOICE_HOST2

        if prev_speaker is not None:
            segments.append(silence_turn if speaker != prev_speaker else silence_same)

        try:
            for _, _, audio in pipeline(text, voice=voice, speed=1.0):
                if audio is not None and len(audio) > 0:
                    segments.append(audio)
        except Exception:
            pass  # skip lines that fail synthesis

        prev_speaker = speaker

    if not segments:
        raise ValueError("Audio synthesis produced no output.")

    audio_data = np.concatenate(segments)

    filename = f"podcast_{uuid.uuid4().hex[:10]}.wav"
    output_path = PODCAST_DIR / filename
    sf.write(str(output_path), audio_data, sample_rate)

    return filename
