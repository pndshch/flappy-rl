# Flappy RL

A tiny browser Flappy Bird clone whose game logic is decoupled from rendering,
so a reinforcement-learning agent can drive it from JavaScript.

**Play:** <https://pndshch.github.io/flappy-rl/>

- `Space` / click / tap → jump
- `R` → restart

## RL interface

Once the page is loaded, the env is available as `window.FlappyEnv`:

```js
FlappyEnv.resetGame(seed?)   // returns initial state; seed is optional integer
FlappyEnv.step(action)       // action: 0 = noop, 1 = jump; returns next state
FlappyEnv.getState()         // returns current state
FlappyEnv.isDone()           // returns boolean
FlappyEnv.releaseControl()   // hand control back to the human play loop
FlappyEnv.config             // { width, height, gravity, pipeGap, ... }
```

State object:

```js
{
  playerY:        number,  // y of player center, in canvas pixels
  playerVelocity: number,  // vertical velocity, px / frame
  nextObstacleX:  number,  // x of next pipe pair (left edge), or canvas width if none
  gapCenterY:     number,  // y of that pipe pair's gap center
  gapSize:        number,  // height of the gap
  score:          number,
  done:           boolean,
}
```

Calling `FlappyEnv.step` or `FlappyEnv.resetGame` pauses the human play loop so
the agent has full control. Press `R` (or call `releaseControl()`) to resume
human input.

The RNG is seedable (mulberry32), so `resetGame(7)` then a deterministic action
sequence reproduces the same trajectory.

## Minimal agent example

```js
// Greedy hand-tuned policy: jump if you're below the gap center.
FlappyEnv.resetGame(0);
const id = setInterval(() => {
  const s = FlappyEnv.getState();
  if (s.done) return clearInterval(id);
  FlappyEnv.step(s.playerY > s.gapCenterY ? 1 : 0);
}, 1000 / 60);
```

## Files

- `index.html` — single canvas + small RL badge
- `style.css`  — responsive 480×640 stage
- `game.js`   — `world`, `step`, `render`, and `window.FlappyEnv`

No build step, no dependencies, no framework.
