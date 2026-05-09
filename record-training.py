#!/usr/bin/env python3
"""Record the live training page as a webm via Playwright, then convert
to an X-friendly mp4 and a small gif using ffmpeg.

The page itself runs the training in JavaScript. We just observe.
"""
import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from playwright.async_api import async_playwright

URL = os.environ.get("FLAPPY_URL", "https://pndshch.github.io/flappy-rl/train.html")
WORK_DIR = Path("/tmp/flappy-rl-train-work")
MP4_OUT = Path("/tmp/flappy-rl-train.mp4")
GIF_OUT = Path("/tmp/flappy-rl-train.gif")

# Recording shape — 16:9 HD lands well on X. Wide enough that the
# stats panel renders side-by-side with the game (the train.css media
# query stacks below 980px).
VIEWPORT = {"width": 1280, "height": 720}
RECORD_SECONDS = 60


async def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    for p in WORK_DIR.glob("*"):
        if p.is_file():
            p.unlink()

    print(f"recording {URL} for {RECORD_SECONDS}s")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(WORK_DIR),
            record_video_size=VIEWPORT,
        )
        page = await ctx.new_page()
        await page.goto(URL, wait_until="networkidle")
        await page.wait_for_timeout(400)
        await page.evaluate("FlappyTrain.start()")

        # Live progress prints once per 5s.
        for i in range(RECORD_SECONDS // 5):
            await page.wait_for_timeout(5000)
            stats = await page.evaluate("FlappyTrain.getStats()")
            print(
                f"  t={(i+1)*5:>3}s  gen={stats['generation']:>3} "
                f"best_ever={stats['bestEver']:>3} eps={stats['totalEpisodes']:>4}"
            )

        # Stop training before tearing down so the last visible frame is calm.
        await page.evaluate("FlappyTrain.pause()")
        final = await page.evaluate("FlappyTrain.getStats()")
        print(f"final: gen={final['generation']} best_ever={final['bestEver']} eps={final['totalEpisodes']}")

        await page.close()
        await ctx.close()  # finalises the webm
        await browser.close()

    webms = list(WORK_DIR.glob("*.webm"))
    if not webms:
        raise SystemExit("no webm produced")
    webm = webms[0]
    kb = webm.stat().st_size / 1024
    print(f"webm: {webm}  ({kb:.0f} KB)")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found")

    # MP4 — H.264 yuv420p, X-compatible
    print("encoding mp4")
    cmd = [
        ffmpeg, "-y", "-i", str(webm),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-crf", "20", "-preset", "veryfast",
        "-movflags", "+faststart",
        str(MP4_OUT),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg mp4 stderr:", r.stderr[-600:])
        raise SystemExit(r.returncode)
    print(f"wrote {MP4_OUT}  ({MP4_OUT.stat().st_size/1024:.0f} KB)")

    # GIF — palette + dither for clean colour reproduction
    print("encoding gif")
    palette = WORK_DIR / "palette.png"
    subprocess.run([
        ffmpeg, "-y", "-i", str(webm),
        "-vf", "fps=12,scale=576:-1:flags=lanczos,palettegen=max_colors=64",
        str(palette),
    ], check=True, capture_output=True)
    subprocess.run([
        ffmpeg, "-y", "-i", str(webm), "-i", str(palette),
        "-lavfi", "fps=12,scale=576:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4",
        str(GIF_OUT),
    ], check=True, capture_output=True)
    print(f"wrote {GIF_OUT}  ({GIF_OUT.stat().st_size/1024:.0f} KB)")
    if shutil.which("gifsicle"):
        subprocess.run(["gifsicle", "-O3", str(GIF_OUT), "-o", str(GIF_OUT)], check=False)
        print(f"optimized: {GIF_OUT.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    asyncio.run(main())
