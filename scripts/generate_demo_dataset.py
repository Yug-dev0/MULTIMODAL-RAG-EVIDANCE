"""
Generates a small, fully local, purpose-built multimodal demo dataset.

Produces (all under test_data/):
  tech_talk.mp4              - narrated 5-slide "recorded tech talk" video
  slides.pdf                 - the same 5 slides as a standalone PDF
  architecture_screenshot.png- a standalone whiteboard-style diagram image
  design_notes.pdf           - a short related design-notes document
  addendum_voice_note.wav    - a standalone follow-up audio clip
  tech_talk.ground_truth.json- slide timing/text for eval authoring ONLY
                                (never read by the ingestion pipeline itself)

Content is deliberately engineered around one narrative (scaling a checkout
database with a cache + read replicas) so that a single query — "what
architecture was discussed for reducing database load, who explained it, and
where was the diagram shown" — requires connecting evidence across video,
audio, image, and PDF, and across four separate files.

Requires macOS `say` and `ffmpeg`/`ffprobe` on PATH. No network access.

Run: .venv/bin/python scripts/generate_demo_dataset.py
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent.parent
RAW = ROOT / "test_data"
BUILD = ROOT / "data" / "processed" / "_dataset_build"
RAW.mkdir(parents=True, exist_ok=True)
if BUILD.exists():
    shutil.rmtree(BUILD)
BUILD.mkdir(parents=True, exist_ok=True)

W, H = 1280, 720
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
F_TITLE = ImageFont.truetype(str(FONT_DIR / "Arial Bold.ttf"), 48)
F_TITLE_BIG = ImageFont.truetype(str(FONT_DIR / "Arial Bold.ttf"), 64)
F_BODY = ImageFont.truetype(str(FONT_DIR / "Arial.ttf"), 30)
F_SMALL = ImageFont.truetype(str(FONT_DIR / "Arial.ttf"), 22)
F_LABEL = ImageFont.truetype(str(FONT_DIR / "Arial Bold.ttf"), 20)

NAVY = (23, 42, 71)
WHITE = (255, 255, 255)
ACCENT = (0, 122, 204)
GRAY = (90, 90, 90)
BOX_FILL = (235, 241, 250)
BOX_EDGE = (23, 42, 71)


def draw_box(d, xy, text, fill=BOX_FILL, edge=BOX_EDGE, font=F_LABEL):
    d.rectangle(xy, fill=fill, outline=edge, width=3)
    lines = text.split("\n")
    cx = (xy[0] + xy[2]) // 2
    total_h = len(lines) * (font.size + 6)
    cy = (xy[1] + xy[3]) // 2 - total_h // 2
    for line in lines:
        tw = d.textlength(line, font=font)
        d.text((cx - tw / 2, cy), line, fill=edge, font=font)
        cy += font.size + 6


def draw_arrow(d, p1, p2, width=3, color=BOX_EDGE):
    d.line([p1, p2], fill=color, width=width)
    ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    for da in (0.45, -0.45):
        a = ang + math.pi - da
        x = p2[0] + 14 * math.cos(a)
        y = p2[1] + 14 * math.sin(a)
        d.line([p2, (x, y)], fill=color, width=width)


def base_slide(header_label: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 90], fill=NAVY)
    d.text((40, 30), header_label, fill=WHITE, font=F_SMALL)
    label = "Scaling the DB Layer"
    tw = d.textlength(label, font=F_SMALL)
    d.text((W - tw - 40, 30), label, fill=(180, 195, 220), font=F_SMALL)
    return im, d


def bullets(d, items, x=70, y=190, gap=56, font=F_BODY):
    for i, t in enumerate(items):
        yy = y + i * gap
        d.ellipse([x, yy + 12, x + 12, yy + 24], fill=ACCENT)
        d.text((x + 30, yy), t, fill=(30, 30, 30), font=font)


def wrap_text(d, text, font, max_width):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if d.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# ---------------------------------------------------------------------------
# Slide content: (image builder) — each returns a PIL Image
# ---------------------------------------------------------------------------

def slide_1():
    im, d = base_slide("1 / 5")
    d.text((70, 260), "Scaling Our Database Layer", fill=NAVY, font=F_TITLE_BIG)
    d.text((70, 350), "Maya Chen — Staff Engineer, Platform Team", fill=GRAY, font=F_BODY)
    return im


def slide_2():
    im, d = base_slide("2 / 5")
    d.text((70, 110), "The Problem: Database Overload", fill=NAVY, font=F_TITLE)
    bullets(d, [
        "Checkout API p99 latency up 4x during peak",
        "Primary Postgres DB at 95% CPU at peak hours",
        "Every request hits the database directly (no cache)",
    ])
    draw_box(d, (140, 440, 380, 540), "App Servers")
    draw_arrow(d, (380, 490), (560, 490))
    draw_box(d, (560, 430, 830, 550), "Primary DB\n(Postgres)")
    return im


def slide_3():
    """The key cross-modal slide: the proposed cache + read replica architecture."""
    im, d = base_slide("3 / 5")
    d.text((70, 110), "Proposed Architecture: Cache + Read Replicas", fill=NAVY,
           font=ImageFont.truetype(str(FONT_DIR / "Arial Bold.ttf"), 40))
    bullets(d, [
        "Redis cache in front of DB absorbs hot reads",
        "Two read replicas handle reporting queries",
        "Writes still go to primary; async replication",
    ], y=190, gap=48)
    draw_box(d, (50, 440, 230, 540), "App Servers")
    draw_arrow(d, (230, 490), (360, 490))
    draw_box(d, (360, 440, 540, 540), "Redis Cache", fill=(255, 235, 205))
    draw_arrow(d, (540, 490), (690, 490))
    draw_box(d, (690, 430, 900, 550), "Primary DB\n(Postgres)")
    draw_arrow(d, (900, 465), (1080, 400))
    draw_arrow(d, (900, 505), (1080, 580))
    draw_box(d, (1080, 350, 1240, 430), "Read\nReplica 1", font=F_SMALL)
    draw_box(d, (1080, 540, 1240, 620), "Read\nReplica 2", font=F_SMALL)
    return im


def slide_4():
    im, d = base_slide("4 / 5")
    d.text((70, 110), "Impact", fill=NAVY, font=F_TITLE)
    bullets(d, [
        "Primary DB load down 70%",
        "p99 latency back to baseline",
        "Read replicas absorb nearly all reporting traffic",
    ])
    return im


def slide_5():
    im, d = base_slide("5 / 5")
    d.text((70, 110), "Next Steps", fill=NAVY, font=F_TITLE)
    bullets(d, [
        "Finalize cache invalidation strategy",
        "Add cache hit-rate monitoring dashboards",
        "Document on-call runbook for new architecture",
    ])
    return im


NARRATION = [
    ("Hi everyone, I'm Maya Chen, staff engineer on the platform team. Today I want to "
     "walk you through how we scaled our database layer to handle peak checkout traffic."),
    ("Let's start with the problem. Over the last quarter, our checkout API's p99 latency "
     "increased four times over. We traced this back to database overload. Our primary "
     "Postgres database was hitting ninety five percent CPU during peak hours, because "
     "every single request was hitting the database directly, with no caching in front of "
     "it at all."),
    ("To fix this, we proposed a new architecture. We added a Redis cache in front of the "
     "database to absorb hot reads, and we added two read replicas to handle reporting and "
     "analytics queries. Writes still go to the primary database, and changes replicate "
     "asynchronously to the replicas. This diagram shows the full flow, from the app "
     "servers through the cache, into the primary database, and out to the replicas."),
    ("After rolling this out, the results were significant. Load on the primary database "
     "dropped by seventy percent, p99 latency returned to baseline, and the read replicas "
     "are now absorbing almost all of our reporting traffic."),
    ("Looking ahead, our next steps are to finalize the cache invalidation strategy, add "
     "monitoring dashboards for cache hit rate, and document an on-call runbook for this "
     "new architecture. Thanks everyone."),
]

SLIDE_BUILDERS = [slide_1, slide_2, slide_3, slide_4, slide_5]
VOICE = "Samantha"


def run(cmd: list[str]):
    subprocess.run(cmd, check=True, capture_output=True)


def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def say_to_wav(text: str, out_wav: Path):
    aiff = out_wav.with_suffix(".aiff")
    run(["say", "-v", VOICE, "-o", str(aiff), text])
    run(["ffmpeg", "-y", "-i", str(aiff), "-ar", "44100", "-ac", "1", str(out_wav)])
    aiff.unlink(missing_ok=True)


def build_video():
    segments = []
    cursor = 0.0
    slide_timing = []
    for i, (builder, text) in enumerate(zip(SLIDE_BUILDERS, NARRATION), start=1):
        png = BUILD / f"slide{i}.png"
        builder().save(png)

        wav = BUILD / f"narration{i}.wav"
        say_to_wav(text, wav)
        duration = ffprobe_duration(wav)

        seg = BUILD / f"segment{i}.mp4"
        run([
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(png),
            "-i", str(wav),
            "-c:v", "libx264", "-tune", "stillimage",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={W}:{H}",
            "-shortest",
            str(seg),
        ])
        segments.append(seg)
        slide_timing.append({
            "slide_index": i,
            "start_time": round(cursor, 2),
            "end_time": round(cursor + duration, 2),
            "narration_text": text,
        })
        cursor += duration

    concat_list = BUILD / "concat.txt"
    concat_list.write_text("\n".join(f"file '{s.resolve()}'" for s in segments))
    out_video = RAW / "tech_talk.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(out_video)])

    (RAW / "tech_talk.ground_truth.json").write_text(json.dumps({
        "note": "Ground truth for eval authoring/debugging ONLY — never read by the ingestion pipeline.",
        "presenter": "Maya Chen",
        "total_duration_seconds": round(cursor, 2),
        "slides": slide_timing,
    }, indent=2))
    print(f"[dataset] wrote {out_video} ({cursor:.1f}s)")


def build_slides_pdf():
    images = [b() for b in SLIDE_BUILDERS]
    out = RAW / "slides.pdf"
    images[0].save(out, "PDF", save_all=True, append_images=images[1:])
    print(f"[dataset] wrote {out}")


def build_architecture_screenshot():
    """A standalone 'whiteboard capture' of the same architecture, styled
    differently from the slide deck, plus a sticky-note style open question
    that the standalone audio addendum resolves — for cross-file/cross-modal
    entity linking tests."""
    im = Image.new("RGB", (1400, 900), (250, 250, 245))
    d = ImageDraw.Draw(im)
    d.text((40, 30), "whiteboard-capture-2026-08-22.png", fill=(120, 120, 120), font=F_SMALL)
    d.text((40, 70), "Checkout DB scaling — architecture sketch", fill=(20, 20, 20),
           font=ImageFont.truetype(str(FONT_DIR / "Arial Bold.ttf"), 34))

    draw_box(d, (80, 300, 300, 400), "app servers", fill=(255, 255, 255), edge=(20, 20, 20))
    draw_arrow(d, (300, 350), (450, 350), color=(20, 20, 20))
    draw_box(d, (450, 300, 670, 400), "cache (redis)", fill=(255, 244, 214), edge=(20, 20, 20))
    draw_arrow(d, (670, 350), (900, 350), color=(20, 20, 20))
    draw_box(d, (900, 290, 1150, 410), "primary DB", fill=(255, 255, 255), edge=(20, 20, 20))
    draw_arrow(d, (1150, 320), (1330, 250), color=(20, 20, 20))
    draw_arrow(d, (1150, 380), (1330, 450), color=(20, 20, 20))
    draw_box(d, (1150, 190, 1350, 270), "replica 1", fill=(255, 255, 255), edge=(20, 20, 20), font=F_SMALL)
    draw_box(d, (1150, 430, 1350, 510), "replica 2", fill=(255, 255, 255), edge=(20, 20, 20), font=F_SMALL)

    # sticky note
    d.rectangle([950, 620, 1330, 780], fill=(255, 247, 158), outline=(200, 190, 90), width=3)
    for i, line in enumerate(["TODO: confirm whether the cache", "layer runs in HA / cluster mode", "before rollout - ask Maya"]):
        d.text((980, 650 + i * 34), line, fill=(60, 55, 10), font=F_SMALL)

    out = RAW / "architecture_screenshot.png"
    im.save(out)
    print(f"[dataset] wrote {out}")


DESIGN_NOTES_TEXT = (
    "Design Notes: Checkout DB Scaling (Draft)\n\n"
    "Context: The checkout service's DB has been overloaded during peak hours, causing "
    "elevated API latency. Maya (platform team) proposed introducing a caching layer in "
    "front of the DB, plus additional read replicas to absorb reporting load.\n\n"
    "Decision: Approved for next sprint. See the slide deck for the full architecture "
    "diagram of the app servers, cache, primary database, and replicas.\n\n"
    "Open question: need to confirm whether the cache layer will run in a clustered / "
    "high-availability configuration before rollout."
)


def build_design_notes_pdf():
    im = Image.new("RGB", (1000, 1300), WHITE)
    d = ImageDraw.Draw(im)
    y = 60
    for para in DESIGN_NOTES_TEXT.split("\n\n"):
        font = F_TITLE if para.startswith("Design Notes") else F_BODY
        max_w = 860
        for line in wrap_text(d, para, font, max_w):
            d.text((70, y), line, fill=(20, 20, 20), font=font)
            y += font.size + 14
        y += 20
    out = RAW / "design_notes.pdf"
    im.save(out, "PDF")
    print(f"[dataset] wrote {out}")


def build_addendum_audio():
    text = (
        "Hey, quick addendum on the checkout DB scaling design doc. To answer the open "
        "question, yes, we're going with Redis Cluster mode for the cache layer, "
        "specifically to get automatic failover and high availability. That should close "
        "out the last open item before we start the rollout. Thanks."
    )
    out = RAW / "addendum_voice_note.wav"
    say_to_wav(text, out)
    print(f"[dataset] wrote {out}")


if __name__ == "__main__":
    build_video()
    build_slides_pdf()
    build_architecture_screenshot()
    build_design_notes_pdf()
    build_addendum_audio()
    shutil.rmtree(BUILD)
    print("[dataset] done. Files in test_data/:")
    for p in sorted(RAW.iterdir()):
        print(" -", p.name)
