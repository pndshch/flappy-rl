/*
 * train.js — RL training visualizer for Flappy RL.
 *
 * One shared multi-bird env: every generation, 24 birds with sampled
 * weight vectors play the same sequence of pipes in lockstep. Each bird's
 * final score is the fitness of its policy. Cross-Entropy Method keeps
 * the top 6 and resamples around their mean.
 *
 * Self-contained — does not depend on game.js.
 */
(() => {
  'use strict';

  // ---- env constants (mirror game.js) ----
  const W = 480, H = 640;
  const GROUND_H = 60;
  const PLAYER_X = 120;
  const PLAYER_R = 14;
  const GRAVITY = 0.5;
  const JUMP_VY = -8;
  const PIPE_W = 60;
  const PIPE_GAP = 150;
  const PIPE_SPEED = 2.5;
  const PIPE_SPAWN_INTERVAL = 90;
  const GAP_MIN_Y = 80 + PIPE_GAP / 2;
  const GAP_MAX_Y = (H - GROUND_H) - 80 - PIPE_GAP / 2;

  // ---- CEM hyperparams ----
  const POP = 24;
  const ELITE = 6;
  const FEAT_DIM = 5;
  const STEPS_PER_FRAME = 5;        // playback speed-up (5x realtime)
  const MAX_GEN_FRAMES = 1200;      // cap so a strong candidate doesn't lock the gen (~13 pipes mastery)
  const GEN_PAUSE_MS = 240;         // brief pause between gens for readability
  const INIT_STD = 3.0;             // wide initial exploration
  const MIN_STD = 0.20;             // entropy floor — keep exploring
  const HISTORY_LEN = 60;

  // ---- linear policy ----
  function features(playerY, playerVel, nextObsX, gapCenterY) {
    return [
      (gapCenterY - playerY) / H,        // signed vertical distance to gap center
      playerVel / 10,                     // normalized velocity
      (nextObsX - PLAYER_X) / W,          // normalized horizontal distance
      playerY / H,                        // raw vertical position
      1,                                  // bias
    ];
  }
  function policyAct(theta, playerY, playerVel, nextObsX, gapCenterY) {
    const f = features(playerY, playerVel, nextObsX, gapCenterY);
    let z = 0;
    for (let i = 0; i < FEAT_DIM; i++) z += theta[i] * f[i];
    return z > 0 ? 1 : 0;
  }

  // ---- multi-bird shared-pipe env ----
  class MultiFlappy {
    constructor() {
      this.rngState = 1;
      this.pipes = [];
      this.spawnTimer = PIPE_SPAWN_INTERVAL;
      this.birds = [];
      this.frame = 0;
    }
    seed(s) { this.rngState = (s >>> 0) || 1; }
    rand() {
      this.rngState = (this.rngState + 0x6D2B79F5) | 0;
      let t = this.rngState;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    reset(seed, thetas) {
      this.seed(seed);
      this.pipes = [];
      this.spawnTimer = PIPE_SPAWN_INTERVAL;
      this.frame = 0;
      this.birds = thetas.map((theta, i) => ({
        y: H / 2, vy: 0,
        done: false, score: 0,
        framesAlive: 0,
        scored: new Set(),
        theta,
        // hue spread, skipping pure red (0) so dead-bird wash doesn't look gory
        hue: 36 + (i / Math.max(1, thetas.length - 1)) * 300,
      }));
    }
    step() {
      const obs = this._currentObservable();
      // each living bird picks an action (shared pipes)
      for (const b of this.birds) {
        if (b.done) continue;
        const a = policyAct(b.theta, b.y, b.vy, obs.nextObsX, obs.gapCenterY);
        if (a === 1) b.vy = JUMP_VY;
        b.vy += GRAVITY;
        b.y += b.vy;
        b.framesAlive++;
      }
      // pipes
      this.spawnTimer++;
      if (this.spawnTimer >= PIPE_SPAWN_INTERVAL) {
        const gapY = GAP_MIN_Y + this.rand() * (GAP_MAX_Y - GAP_MIN_Y);
        this.pipes.push({ x: W, gapCenterY: gapY });
        this.spawnTimer = 0;
      }
      for (const p of this.pipes) p.x -= PIPE_SPEED;
      this.pipes = this.pipes.filter(p => p.x + PIPE_W > -10);
      // per-bird collision + score
      for (const b of this.birds) {
        if (b.done) continue;
        // walls
        if (b.y - PLAYER_R <= 0) { b.y = PLAYER_R; b.done = true; continue; }
        if (b.y + PLAYER_R >= H - GROUND_H) { b.y = H - GROUND_H - PLAYER_R; b.done = true; continue; }
        // pipes
        for (const p of this.pipes) {
          if (p.x < PLAYER_X + PLAYER_R && p.x + PIPE_W > PLAYER_X - PLAYER_R) {
            const topY2 = p.gapCenterY - PIPE_GAP / 2;
            const botY1 = p.gapCenterY + PIPE_GAP / 2;
            if (b.y - PLAYER_R < topY2 || b.y + PLAYER_R > botY1) {
              const closestX = Math.max(p.x, Math.min(PLAYER_X, p.x + PIPE_W));
              const dx = PLAYER_X - closestX;
              let dy = 0;
              if (b.y < topY2) dy = b.y - topY2;
              else if (b.y > botY1) dy = b.y - botY1;
              if (dx * dx + dy * dy < PLAYER_R * PLAYER_R) {
                b.done = true;
                break;
              }
            }
          }
        }
        if (b.done) continue;
        // score (each pipe scored once per bird)
        for (const p of this.pipes) {
          if (!b.scored.has(p) && (p.x + PIPE_W) < PLAYER_X) {
            b.score += 1;
            b.scored.add(p);
          }
        }
      }
      this.frame++;
    }
    _currentObservable() {
      let nextObsX = W, gapCenterY = H / 2;
      let bestD = Infinity;
      for (const p of this.pipes) {
        if (p.x + PIPE_W >= PLAYER_X) {
          const d = p.x - PLAYER_X;
          if (d < bestD) { bestD = d; nextObsX = p.x; gapCenterY = p.gapCenterY; }
        }
      }
      return { nextObsX, gapCenterY };
    }
    allDead() { return this.birds.every(b => b.done); }
  }

  // ---- CEM state ----
  let mean = new Array(FEAT_DIM).fill(0);
  let std = new Array(FEAT_DIM).fill(INIT_STD);
  let generation = 0;
  let bestEver = 0;
  let bestThetaEver = mean.slice();
  let avgsHistory = [];
  let bestHistory = [];
  let totalEpisodes = 0;
  let startTime = performance.now();

  function gauss() {
    const u = Math.random(), v = Math.random();
    return Math.sqrt(-2 * Math.log(u + 1e-9)) * Math.cos(2 * Math.PI * v);
  }

  function sampleThetas() {
    const out = [];
    for (let i = 0; i < POP; i++) {
      out.push(mean.map((m, j) => m + std[j] * gauss()));
    }
    return out;
  }

  function updateFromElite(thetas, fitnesses) {
    const idx = fitnesses.map((_, i) => i).sort((a, b) => fitnesses[b] - fitnesses[a]);
    const elite = idx.slice(0, ELITE);
    for (let j = 0; j < FEAT_DIM; j++) {
      let m = 0;
      for (const i of elite) m += thetas[i][j];
      mean[j] = m / ELITE;
    }
    for (let j = 0; j < FEAT_DIM; j++) {
      let v = 0;
      for (const i of elite) v += (thetas[i][j] - mean[j]) ** 2;
      std[j] = Math.sqrt(v / ELITE) + MIN_STD;
    }
  }

  // ---- renderer ----
  const canvas = document.getElementById('train-canvas');
  const ctx = canvas.getContext('2d');

  function drawWorld(env) {
    // sky
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, '#7ec8ff');
    grad.addColorStop(1, '#bde7ff');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);

    // a faint cloud or two
    ctx.fillStyle = 'rgba(255, 255, 255, 0.55)';
    ctx.beginPath();
    ctx.arc(80, 90, 18, 0, Math.PI * 2);
    ctx.arc(108, 84, 22, 0, Math.PI * 2);
    ctx.arc(134, 90, 18, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(360, 180, 14, 0, Math.PI * 2);
    ctx.arc(382, 175, 18, 0, Math.PI * 2);
    ctx.arc(404, 180, 14, 0, Math.PI * 2);
    ctx.fill();

    // pipes
    for (const p of env.pipes) {
      const topH = p.gapCenterY - PIPE_GAP / 2;
      const botY = p.gapCenterY + PIPE_GAP / 2;
      const botH = (H - GROUND_H) - botY;
      drawPipeColumn(p.x, 0, PIPE_W, topH, true);
      drawPipeColumn(p.x, botY, PIPE_W, botH, false);
    }

    // ground
    ctx.fillStyle = '#ded895';
    ctx.fillRect(0, H - GROUND_H, W, GROUND_H);
    ctx.fillStyle = '#c0b46c';
    ctx.fillRect(0, H - GROUND_H, W, 6);
    ctx.fillStyle = '#9c9054';
    const stripeOffset = (env.frame * PIPE_SPEED) % 24;
    for (let x = -stripeOffset; x < W; x += 24) {
      ctx.fillRect(x, H - GROUND_H + 12, 12, 4);
    }

    // birds — dead first (faded), alive on top (vibrant)
    for (const b of env.birds) {
      if (!b.done) continue;
      drawBird(b.y, b.vy, b.hue, 0.16);
    }
    for (const b of env.birds) {
      if (b.done) continue;
      drawBird(b.y, b.vy, b.hue, 0.78);
    }
  }

  function drawPipeColumn(x, y, w, h, isTop) {
    ctx.fillStyle = '#5fb84a';
    ctx.fillRect(x, y, w, h);
    ctx.fillStyle = '#82d36a';
    ctx.fillRect(x + 4, y, 8, h);
    ctx.fillStyle = '#3a8a2e';
    ctx.fillRect(x + w - 4, y, 4, h);
    if (isTop) {
      ctx.fillStyle = '#5fb84a';
      ctx.fillRect(x - 4, y + h - 18, w + 8, 18);
      ctx.fillStyle = '#3a8a2e';
      ctx.fillRect(x - 4, y + h - 4, w + 8, 4);
    } else {
      ctx.fillStyle = '#5fb84a';
      ctx.fillRect(x - 4, y, w + 8, 18);
      ctx.fillStyle = '#3a8a2e';
      ctx.fillRect(x - 4, y, w + 8, 4);
    }
  }

  function drawBird(y, vy, hue, alpha) {
    const tilt = Math.max(-0.5, Math.min(1.2, vy / 10));
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.translate(PLAYER_X, y);
    ctx.rotate(tilt);
    ctx.fillStyle = `hsl(${hue}, 90%, 55%)`;
    ctx.beginPath();
    ctx.arc(0, 0, PLAYER_R, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = `hsl(${hue}, 70%, 25%)`;
    ctx.stroke();
    // eye
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(5, -3, 3.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#1a1a1a';
    ctx.beginPath();
    ctx.arc(6, -3, 1.6, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  // ---- DOM hooks ----
  const $gen = document.getElementById('s-gen');
  const $best = document.getElementById('s-best');
  const $bestEver = document.getElementById('s-bestever');
  const $avg = document.getElementById('s-avg');
  const $alive = document.getElementById('s-alive');
  const $eps = document.getElementById('s-eps');
  const $time = document.getElementById('s-time');
  const $frame = document.getElementById('s-frame');
  const $banner = document.getElementById('gen-banner');
  const chartCtx = document.getElementById('chart').getContext('2d');

  function drawChart() {
    const cw = chartCtx.canvas.width;
    const ch = chartCtx.canvas.height;
    chartCtx.clearRect(0, 0, cw, ch);
    chartCtx.fillStyle = '#11141b';
    chartCtx.fillRect(0, 0, cw, ch);
    if (avgsHistory.length === 0) return;
    const allMax = Math.max(...avgsHistory, ...bestHistory, 1);
    const slots = Math.max(40, avgsHistory.length);
    const stepX = cw / slots;
    // bars: avg
    chartCtx.fillStyle = 'rgba(95, 184, 74, 0.75)';
    for (let i = 0; i < avgsHistory.length; i++) {
      const h = (avgsHistory[i] / allMax) * (ch - 8);
      chartCtx.fillRect(i * stepX, ch - h - 4, Math.max(1, stepX - 1), h);
    }
    // line: best
    chartCtx.strokeStyle = '#ffd23f';
    chartCtx.lineWidth = 2;
    chartCtx.beginPath();
    for (let i = 0; i < bestHistory.length; i++) {
      const x = i * stepX + stepX / 2;
      const y = ch - 4 - (bestHistory[i] / allMax) * (ch - 8);
      if (i === 0) chartCtx.moveTo(x, y);
      else chartCtx.lineTo(x, y);
    }
    chartCtx.stroke();
    // y-max label
    chartCtx.fillStyle = '#a0a8b8';
    chartCtx.font = '10px ui-monospace, monospace';
    chartCtx.fillText(`max ${allMax.toFixed(0)}`, 4, 12);
  }

  function updateStats(env, phaseLabel) {
    $gen.textContent = generation;
    $bestEver.textContent = bestEver;
    $eps.textContent = totalEpisodes;
    const elapsed = (performance.now() - startTime) / 1000;
    const m = Math.floor(elapsed / 60);
    const s = Math.floor(elapsed % 60);
    $time.textContent = `${m}:${String(s).padStart(2, '0')}`;
    if (env) {
      const alive = env.birds.filter(b => !b.done).length;
      const best = env.birds.length ? Math.max(0, ...env.birds.map(b => b.score)) : 0;
      $alive.textContent = `${alive} / ${env.birds.length}`;
      $best.textContent = best;
      $frame.textContent = env.frame;
    }
    if (phaseLabel) $banner.textContent = phaseLabel;
    drawChart();
  }

  // ---- training driver ----
  let env = new MultiFlappy();
  let running = false;
  let phase = 'idle'; // 'idle' | 'playing' | 'paused-between'
  let phaseTimer = 0;

  function startNewGeneration() {
    generation++;
    const thetas = sampleThetas();
    env.reset(generation * 31337 + 1, thetas);
    phase = 'playing';
    updateStats(env, `GEN ${generation} · LIVE`);
  }

  function finalizeGeneration() {
    const thetas = env.birds.map(b => b.theta);
    const scores = env.birds.map(b => b.score);
    // Fitness shaping: pipes score is the headline metric, but credit
    // surviving longer too — gives CEM a continuous gradient to climb
    // before the first pipe is ever cleared. Score multiplier is large so
    // an actual scorer always beats a "just survives without scoring" bird.
    const fitnesses = env.birds.map(b => b.score * 1000 + b.framesAlive);
    updateFromElite(thetas, fitnesses);
    const genBest = Math.max(...scores);
    if (genBest > bestEver) {
      bestEver = genBest;
      bestThetaEver = thetas[scores.indexOf(genBest)].slice();
    }
    avgsHistory.push(scores.reduce((s, v) => s + v, 0) / scores.length);
    bestHistory.push(genBest);
    if (avgsHistory.length > HISTORY_LEN) {
      avgsHistory.shift();
      bestHistory.shift();
    }
    totalEpisodes += POP;
    $avg.textContent = (scores.reduce((s, v) => s + v, 0) / scores.length).toFixed(1);
    updateStats(env, `GEN ${generation} · BEST ${genBest}`);
  }

  function tick() {
    if (running) {
      if (phase === 'idle') {
        startNewGeneration();
      } else if (phase === 'playing') {
        for (let s = 0; s < STEPS_PER_FRAME; s++) {
          env.step();
          if (env.allDead() || env.frame >= MAX_GEN_FRAMES) break;
        }
        if (env.allDead() || env.frame >= MAX_GEN_FRAMES) {
          finalizeGeneration();
          phase = 'paused-between';
          phaseTimer = performance.now() + GEN_PAUSE_MS;
        } else {
          updateStats(env, `GEN ${generation} · LIVE`);
        }
      } else if (phase === 'paused-between') {
        if (performance.now() >= phaseTimer) {
          startNewGeneration();
        }
      }
    }
    drawWorld(env);
    requestAnimationFrame(tick);
  }

  // ---- buttons ----
  const $btnStart = document.getElementById('btn-start');
  const $btnReset = document.getElementById('btn-reset');
  $btnStart.addEventListener('click', () => {
    running = !running;
    $btnStart.textContent = running ? '⏸ Pause' : '▶ Resume';
    if (running) {
      if (startTime === 0) startTime = performance.now();
      if (phase === 'idle') startNewGeneration();
    }
  });
  $btnReset.addEventListener('click', () => {
    mean = new Array(FEAT_DIM).fill(0);
    std = new Array(FEAT_DIM).fill(INIT_STD);
    generation = 0;
    bestEver = 0;
    bestThetaEver = mean.slice();
    avgsHistory = [];
    bestHistory = [];
    totalEpisodes = 0;
    startTime = performance.now();
    phase = 'idle';
    env = new MultiFlappy();
    env.reset(0, [mean.slice()]); // dummy single bird for the idle render
    updateStats(env, 'GEN 0 · WAITING');
  });

  // ---- expose for headless / programmatic recording ----
  window.FlappyTrain = {
    start: () => { if (!running) $btnStart.click(); },
    pause: () => { if (running) $btnStart.click(); },
    reset: () => $btnReset.click(),
    getStats: () => ({
      generation, bestEver, totalEpisodes,
      avgsHistory: avgsHistory.slice(),
      bestHistory: bestHistory.slice(),
      mean: mean.slice(), std: std.slice(),
    }),
  };

  // initial idle render
  env.reset(0, [mean.slice()]);
  updateStats(env, 'GEN 0 · WAITING');
  requestAnimationFrame(tick);
})();
