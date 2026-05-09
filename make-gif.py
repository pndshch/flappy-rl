#!/usr/bin/env python3
"""Capture Flappy RL gameplay -> GIF via playwright + Pillow.

Drives the game through window.FlappyEnv with a hand-tuned heuristic policy
so the GIF shows smooth gap-clearing rather than random flailing.
"""
import asyncio
import io
from pathlib import Path
from PIL import Image
from playwright.async_api import async_playwright

URL = "http://127.0.0.1:8765/index.html"
OUT = Path("/tmp/pndshch-hub/assets/flappy-rl.gif")
W, H = 480, 640
SCALE = 0.5
COLORS = 48
FRAME_STRIDE = 3          # capture every Nth physics frame (= ~50ms / frame)
FRAMES_TO_PLAY = 540      # ~9 seconds of in-game time at 60fps
SKIP_INITIAL = 90         # don't capture the dead air before the first pipe is close
SEED = 4                  # tuned to a long, photogenic run for policy F

frames: list[Image.Image] = []
durations: list[int] = []


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Pad ~40px to avoid right-edge clipping per past Chrome headless quirk.
        ctx = await browser.new_context(
            viewport={"width": W + 40, "height": H + 40},
            device_scale_factor=2,
        )
        page = await ctx.new_page()
        await page.goto(URL, wait_until="networkidle")
        await page.evaluate(f"FlappyEnv.resetGame({SEED})")
        await page.wait_for_timeout(80)

        canvas = page.locator("#game")

        # Hand-tuned policy F: jump iff the bird is below (gap center + 10) and not
        # already rising too fast. Tuned by sweeping policies in a benchmark run —
        # this one survives 30+ pipes with seed=4.
        TICK_JS = """(strideArg) => {
            const N = strideArg;
            for (let i = 0; i < N; i++) {
                if (FlappyEnv.isDone()) {
                    FlappyEnv.resetGame(Math.floor(Math.random() * 1e9));
                }
                const s = FlappyEnv.getState();
                const a = (s.playerY > s.gapCenterY + 10 && s.playerVelocity > -3) ? 1 : 0;
                FlappyEnv.step(a);
            }
        }"""

        total_frames = 0
        while total_frames < FRAMES_TO_PLAY:
            await page.evaluate(TICK_JS, FRAME_STRIDE)
            total_frames += FRAME_STRIDE
            if total_frames < SKIP_INITIAL:
                continue
            # Force a render frame before screenshotting so the visual matches state.
            await page.evaluate("new Promise(r => requestAnimationFrame(r))")
            png = await canvas.screenshot(type="png")
            img = Image.open(io.BytesIO(png)).convert("RGB")
            frames.append(img)
            # 60fps physics; each captured frame represents FRAME_STRIDE ticks.
            durations.append(int(1000 / 60 * FRAME_STRIDE))

        await browser.close()


asyncio.run(main())

out_w = int(W * SCALE)
out_h = int(H * SCALE)
small = [
    f.resize((out_w, out_h), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=COLORS)
    for f in frames
]
print(f"frames: {len(small)}  total ms: {sum(durations)}")
OUT.parent.mkdir(parents=True, exist_ok=True)
small[0].save(
    OUT,
    save_all=True,
    append_images=small[1:],
    duration=durations,
    loop=0,
    optimize=True,
    disposal=2,
)
print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")
