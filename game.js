/*
 * Flappy RL — a tiny Flappy Bird clone with a clean RL interface.
 *
 * The game logic lives entirely in `world` and is advanced by `step(action)`,
 * which executes one fixed timestep. Rendering is a separate function that
 * reads `world` and never mutates it.
 *
 * RL interface (exposed as `window.FlappyEnv`):
 *   resetGame(seed?)  -> initial state object
 *   step(action)      -> next state object (action: 0=noop, 1=jump)
 *   getState()        -> current state object
 *   isDone()          -> boolean
 *   releaseControl()  -> hand control back to the human play loop
 *   config            -> { width, height, ... } for normalisation
 *
 * State object shape:
 *   { playerY, playerVelocity, nextObstacleX, gapCenterY, gapSize, score, done }
 */
(() => {
  'use strict';

  // -------------------------------------------------------------------
  // Config
  // -------------------------------------------------------------------
  const W = 480;
  const H = 640;
  const GROUND_H = 60;
  const PLAYER_X = 120;
  const PLAYER_R = 14;
  const GRAVITY = 0.5;
  const JUMP_VY = -8;
  const PIPE_W = 60;
  const PIPE_GAP = 150;
  const PIPE_SPEED = 2.5;
  const PIPE_SPAWN_INTERVAL = 90; // frames between pipes
  const GAP_MIN_Y = 80 + PIPE_GAP / 2;
  const GAP_MAX_Y = (H - GROUND_H) - 80 - PIPE_GAP / 2;

  // -------------------------------------------------------------------
  // Seedable RNG (mulberry32) — keeps episodes reproducible for RL.
  // -------------------------------------------------------------------
  let rngState = 1;
  function seedRng(seed) { rngState = ((seed >>> 0) || 1); }
  function rand() {
    rngState = (rngState + 0x6D2B79F5) | 0;
    let t = rngState;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  // -------------------------------------------------------------------
  // World — pure logic, no DOM access.
  // -------------------------------------------------------------------
  const world = {
    playerY: H / 2,
    playerVelocity: 0,
    pipes: [],          // each: { x, gapCenterY, scored }
    spawnTimer: 0,
    score: 0,
    done: false,
    frame: 0,
  };

  function resetGame(seed) {
    seedRng(typeof seed === 'number' ? seed : ((Math.random() * 0xFFFFFFFF) >>> 0));
    world.playerY = H / 2;
    world.playerVelocity = 0;
    world.pipes = [];
    // Spawn the first pipe almost immediately so the player has something to react to.
    world.spawnTimer = PIPE_SPAWN_INTERVAL;
    world.score = 0;
    world.done = false;
    world.frame = 0;
    return getState();
  }

  function spawnPipe() {
    const gapCenterY = GAP_MIN_Y + rand() * (GAP_MAX_Y - GAP_MIN_Y);
    world.pipes.push({ x: W, gapCenterY, scored: false });
  }

  function circleHitsRect(cx, cy, cr, rx, ry, rw, rh) {
    const closestX = Math.max(rx, Math.min(cx, rx + rw));
    const closestY = Math.max(ry, Math.min(cy, ry + rh));
    const dx = cx - closestX;
    const dy = cy - closestY;
    return (dx * dx + dy * dy) < (cr * cr);
  }

  function step(action) {
    if (world.done) return getState();

    if (action === 1) {
      world.playerVelocity = JUMP_VY;
    }

    // physics
    world.playerVelocity += GRAVITY;
    world.playerY += world.playerVelocity;

    // pipe spawning
    world.spawnTimer++;
    if (world.spawnTimer >= PIPE_SPAWN_INTERVAL) {
      spawnPipe();
      world.spawnTimer = 0;
    }

    // pipe motion + scoring
    for (const p of world.pipes) {
      p.x -= PIPE_SPEED;
      if (!p.scored && (p.x + PIPE_W) < PLAYER_X) {
        p.scored = true;
        world.score += 1;
      }
    }
    // remove off-screen pipes
    world.pipes = world.pipes.filter(p => p.x + PIPE_W > -10);

    // ceiling
    if (world.playerY - PLAYER_R <= 0) {
      world.playerY = PLAYER_R;
      world.playerVelocity = 0;
      world.done = true;
    }
    // ground
    if (world.playerY + PLAYER_R >= H - GROUND_H) {
      world.playerY = H - GROUND_H - PLAYER_R;
      world.done = true;
    }
    // pipes
    for (const p of world.pipes) {
      // broad-phase x overlap check
      if (p.x < PLAYER_X + PLAYER_R && p.x + PIPE_W > PLAYER_X - PLAYER_R) {
        const topY2 = p.gapCenterY - PIPE_GAP / 2;
        const botY1 = p.gapCenterY + PIPE_GAP / 2;
        const playLeft = world.playerY;
        const cy = playLeft;
        if (
          circleHitsRect(PLAYER_X, cy, PLAYER_R, p.x, 0, PIPE_W, topY2) ||
          circleHitsRect(PLAYER_X, cy, PLAYER_R, p.x, botY1, PIPE_W, (H - GROUND_H) - botY1)
        ) {
          world.done = true;
        }
      }
    }

    world.frame++;
    return getState();
  }

  function getState() {
    // The "next" obstacle is the closest pipe whose right edge is still
    // ahead of (or at) the player. If none exists, fall back to canvas-edge
    // defaults so the agent always has finite numbers to consume.
    let nextObstacleX = W;
    let gapCenterY = H / 2;
    let bestDist = Infinity;
    for (const p of world.pipes) {
      if (p.x + PIPE_W >= PLAYER_X) {
        const d = p.x - PLAYER_X;
        if (d < bestDist) {
          bestDist = d;
          nextObstacleX = p.x;
          gapCenterY = p.gapCenterY;
        }
      }
    }
    return {
      playerY: world.playerY,
      playerVelocity: world.playerVelocity,
      nextObstacleX: nextObstacleX,
      gapCenterY: gapCenterY,
      gapSize: PIPE_GAP,
      score: world.score,
      done: world.done,
    };
  }

  function isDone() { return world.done; }

  // -------------------------------------------------------------------
  // Rendering — reads `world`, never mutates.
  // -------------------------------------------------------------------
  const canvas = document.getElementById('game');
  const ctx = canvas.getContext('2d');

  function drawPipe(x, y, w, h, isTop) {
    // body
    ctx.fillStyle = '#5fb84a';
    ctx.fillRect(x, y, w, h);
    // highlight stripe
    ctx.fillStyle = '#82d36a';
    ctx.fillRect(x + 4, y, 8, h);
    // dark edge
    ctx.fillStyle = '#3a8a2e';
    ctx.fillRect(x + w - 4, y, 4, h);
    // lip (cap)
    if (isTop) {
      ctx.fillStyle = '#5fb84a';
      ctx.fillRect(x - 4, h - 18, w + 8, 18);
      ctx.fillStyle = '#82d36a';
      ctx.fillRect(x - 2, h - 16, 8, 12);
      ctx.fillStyle = '#3a8a2e';
      ctx.fillRect(x - 4, h - 4, w + 8, 4);
    } else {
      ctx.fillStyle = '#5fb84a';
      ctx.fillRect(x - 4, y, w + 8, 18);
      ctx.fillStyle = '#82d36a';
      ctx.fillRect(x - 2, y + 4, 8, 12);
      ctx.fillStyle = '#3a8a2e';
      ctx.fillRect(x - 4, y, w + 8, 4);
    }
  }

  function render() {
    // sky gradient
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, '#7ec8ff');
    grad.addColorStop(1, '#bde7ff');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);

    // distant clouds (decorative; positions tied to frame for slow drift)
    const drift = (world.frame * 0.3) % (W + 200);
    drawCloud(((80 - drift) % (W + 200) + (W + 200)) % (W + 200) - 100, 90, 1);
    drawCloud(((300 - drift) % (W + 200) + (W + 200)) % (W + 200) - 100, 170, 0.8);
    drawCloud(((180 - drift) % (W + 200) + (W + 200)) % (W + 200) - 100, 260, 0.6);

    // pipes
    for (const p of world.pipes) {
      const topH = p.gapCenterY - PIPE_GAP / 2;
      const botY = p.gapCenterY + PIPE_GAP / 2;
      const botH = (H - GROUND_H) - botY;
      drawPipe(p.x, 0, PIPE_W, topH, true);
      drawPipe(p.x, botY, PIPE_W, botH, false);
    }

    // ground
    ctx.fillStyle = '#ded895';
    ctx.fillRect(0, H - GROUND_H, W, GROUND_H);
    ctx.fillStyle = '#c0b46c';
    ctx.fillRect(0, H - GROUND_H, W, 6);
    // ground texture (moving stripes)
    ctx.fillStyle = '#9c9054';
    const stripeOffset = (world.frame * PIPE_SPEED) % 24;
    for (let x = -stripeOffset; x < W; x += 24) {
      ctx.fillRect(x, H - GROUND_H + 12, 12, 4);
    }

    // player (yellow circle with eye + small beak hint)
    const px = PLAYER_X;
    const py = world.playerY;
    // tilt based on velocity (visual only — does not affect logic)
    const tilt = Math.max(-0.4, Math.min(1.2, world.playerVelocity / 10));
    ctx.save();
    ctx.translate(px, py);
    ctx.rotate(tilt);
    // body
    ctx.fillStyle = '#ffd23f';
    ctx.beginPath();
    ctx.arc(0, 0, PLAYER_R, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#7a5a1a';
    ctx.stroke();
    // eye
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(5, -3, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#1a1a1a';
    ctx.beginPath();
    ctx.arc(6, -3, 2, 0, Math.PI * 2);
    ctx.fill();
    // beak
    ctx.fillStyle = '#ff8a00';
    ctx.beginPath();
    ctx.moveTo(PLAYER_R - 2, 1);
    ctx.lineTo(PLAYER_R + 6, 4);
    ctx.lineTo(PLAYER_R - 2, 7);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    // score (canvas-rendered so screenshots include it)
    ctx.font = 'bold 56px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif';
    ctx.textAlign = 'center';
    ctx.lineWidth = 6;
    ctx.strokeStyle = '#2b2b2b';
    ctx.fillStyle = '#fff';
    ctx.strokeText(String(world.score), W / 2, 80);
    ctx.fillText(String(world.score), W / 2, 80);

    // pre-game prompt
    if (!started && !world.done) {
      drawCenterPanel('Flappy RL', 'SPACE / Click to jump', 'R to restart  ·  window.FlappyEnv');
    }

    // game-over overlay
    if (world.done) {
      drawCenterPanel('GAME OVER', `Score: ${world.score}`, 'Press R to restart');
    }
  }

  function drawCloud(x, y, scale) {
    ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
    ctx.beginPath();
    ctx.arc(x, y, 18 * scale, 0, Math.PI * 2);
    ctx.arc(x + 22 * scale, y - 6 * scale, 22 * scale, 0, Math.PI * 2);
    ctx.arc(x + 46 * scale, y, 18 * scale, 0, Math.PI * 2);
    ctx.arc(x + 24 * scale, y + 8 * scale, 18 * scale, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawCenterPanel(title, msg, sub) {
    const panelW = 320;
    const panelH = 150;
    const x = (W - panelW) / 2;
    const y = (H - panelH) / 2 - 20;
    ctx.fillStyle = 'rgba(10, 15, 25, 0.55)';
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(20, 25, 35, 0.95)';
    roundRect(x, y, panelW, panelH, 14);
    ctx.fill();
    ctx.strokeStyle = '#3b4252';
    ctx.lineWidth = 2;
    roundRect(x, y, panelW, panelH, 14);
    ctx.stroke();
    ctx.textAlign = 'center';
    ctx.fillStyle = '#ffd23f';
    ctx.font = 'bold 30px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif';
    ctx.fillText(title, W / 2, y + 50);
    ctx.fillStyle = '#e8e8e8';
    ctx.font = '18px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif';
    ctx.fillText(msg, W / 2, y + 86);
    ctx.fillStyle = '#a0a8b8';
    ctx.font = '13px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif';
    ctx.fillText(sub, W / 2, y + 116);
  }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  // -------------------------------------------------------------------
  // Human play loop. Skipped while an external driver (RL agent) is in control.
  // -------------------------------------------------------------------
  let externalDriverActive = false;
  let started = false;
  let jumpRequested = false;

  function loop() {
    if (!externalDriverActive) {
      if (started && !world.done) {
        const action = jumpRequested ? 1 : 0;
        jumpRequested = false;
        step(action);
      } else {
        jumpRequested = false;
      }
    }
    render();
    requestAnimationFrame(loop);
  }

  function tryStart() {
    if (!started && !world.done) {
      started = true;
      jumpRequested = true;
    }
  }

  function tryRestart() {
    resetGame();
    started = true;
    jumpRequested = true;
    externalDriverActive = false; // R returns control to human
  }

  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      if (externalDriverActive) return;
      if (!started) tryStart();
      else if (!world.done) jumpRequested = true;
    } else if (e.code === 'KeyR') {
      e.preventDefault();
      tryRestart();
    }
  });
  const onPress = (e) => {
    e.preventDefault();
    if (externalDriverActive) return;
    if (world.done) tryRestart();
    else if (!started) tryStart();
    else jumpRequested = true;
  };
  canvas.addEventListener('mousedown', onPress);
  canvas.addEventListener('touchstart', onPress, { passive: false });

  // -------------------------------------------------------------------
  // Init + RL interface
  // -------------------------------------------------------------------
  resetGame();
  requestAnimationFrame(loop);

  window.FlappyEnv = {
    resetGame: (seed) => {
      externalDriverActive = true;
      started = true;
      return resetGame(seed);
    },
    step: (action) => {
      externalDriverActive = true;
      started = true;
      return step(action);
    },
    getState: () => getState(),
    isDone: () => isDone(),
    releaseControl: () => {
      externalDriverActive = false;
    },
    config: {
      width: W,
      height: H,
      groundH: GROUND_H,
      playerX: PLAYER_X,
      playerR: PLAYER_R,
      gravity: GRAVITY,
      jumpVy: JUMP_VY,
      pipeWidth: PIPE_W,
      pipeGap: PIPE_GAP,
      pipeSpeed: PIPE_SPEED,
      pipeSpawnInterval: PIPE_SPAWN_INTERVAL,
    },
  };
})();
