/* ============================================================================
 * Chrome Dino — Reinforcement Learning agent
 * ----------------------------------------------------------------------------
 * Tabular Q-learning. The agent hooks straight into the game's own state via
 * `Runner.instance_` (the singleton created in dino.html) — it does not read
 * pixels. Every animation frame it:
 *    1. reads the game state (gap to next obstacle, speed, jump status),
 *    2. discretises it into a Q-table key,
 *    3. picks an action (do nothing / jump) with epsilon-greedy exploration,
 *    4. applies the action and, on the next frame, learns from the reward.
 *
 * The Q-table is persisted to localStorage so training survives page reloads.
 * ==========================================================================*/
(function () {
  'use strict';

  // ---- Hyperparameters ------------------------------------------------------
  const ALPHA          = 0.15;   // learning rate
  const GAMMA          = 0.99;   // discount factor
  const EPS_START      = 1.00;   // initial exploration rate
  const EPS_MIN        = 0.01;   // floor exploration rate
  const EPS_DECAY      = 0.96;   // multiplied per episode
  const REWARD_ALIVE   =  1;     // per surviving frame
  const REWARD_JUMP    = -1;     // small cost so the agent jumps only when useful
  const REWARD_CRASH   = -100;   // terminal penalty

  // ---- State discretisation -------------------------------------------------
  // distance to the next obstacle, bucketed; speed, bucketed; jumping flag.
  const DIST_BUCKET = 12;        // px per distance bucket
  const DIST_MAX    = 20;        // distance buckets 0..20 (20 == "far / none")
  const SPD_MIN     = 6;         // game starts at speed 6
  const SPD_BUCKETS = 8;         // speed buckets 0..7

  const ACTIONS = 2;             // 0 = do nothing, 1 = jump

  // ---- Q-table --------------------------------------------------------------
  const STORE_KEY = 'dino_rl_qtable_v1';
  let Q = {};                    // stateKey -> [q_noop, q_jump]
  let episode = 0;
  let bestScore = 0;

  function load() {
    try {
      const raw = JSON.parse(localStorage.getItem(STORE_KEY) || '{}');
      Q = raw.Q || {};
      episode = raw.episode || 0;
      bestScore = raw.bestScore || 0;
    } catch (e) { Q = {}; }
  }
  function save() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({ Q, episode, bestScore }));
    } catch (e) {/* storage full / disabled — keep training in memory */}
  }
  function qRow(key) {
    if (!Q[key]) Q[key] = new Array(ACTIONS).fill(0);
    return Q[key];
  }

  // ---- Reading the live game state -----------------------------------------
  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

  // Returns a discrete string key describing the current situation.
  function encodeState(R) {
    const trex = R.tRex;
    const obstacles = R.horizon.obstacles;

    // first obstacle that is still in front of (or overlapping) the dino
    let next = null;
    for (let i = 0; i < obstacles.length; i++) {
      const o = obstacles[i];
      if (o.xPos + o.width > trex.xPos) { next = o; break; }
    }

    let distBin;
    if (!next) {
      distBin = DIST_MAX;                                  // nothing ahead
    } else {
      const gap = next.xPos - trex.xPos;
      distBin = clamp(Math.floor(gap / DIST_BUCKET), 0, DIST_MAX);
    }

    const spdBin  = clamp(Math.floor(R.currentSpeed) - SPD_MIN, 0, SPD_BUCKETS - 1);
    const jumping = trex.jumping ? 1 : 0;

    return distBin + '|' + spdBin + '|' + jumping;
  }

  // ---- Policy ---------------------------------------------------------------
  let epsilon = EPS_START;

  function chooseAction(key) {
    if (Math.random() < epsilon) return Math.floor(Math.random() * ACTIONS);
    const row = qRow(key);
    return row[1] > row[0] ? 1 : 0;            // ties -> do nothing
  }

  function learn(key, action, reward, nextKey) {
    const row = qRow(key);
    let target = reward;
    if (nextKey !== null) {
      const nr = qRow(nextKey);
      target += GAMMA * Math.max(nr[0], nr[1]);   // terminal state has no bootstrap
    }
    row[action] += ALPHA * (target - row[action]);
  }

  // ---- Training loop --------------------------------------------------------
  let prevKey = null, prevAction = null;
  let started = false, paused = false;
  let episodeScore = 0;
  let pendingStart = false;

  function getRunner() {
    return window.Runner && window.Runner.instance_;
  }

  // Kick the game off the intro screen by simulating a spacebar press.
  function pressSpace() {
    const down = new KeyboardEvent('keydown', { keyCode: 32, which: 32, bubbles: true });
    const up   = new KeyboardEvent('keyup',   { keyCode: 32, which: 32, bubbles: true });
    document.dispatchEvent(down);
    setTimeout(() => document.dispatchEvent(up), 30);
  }

  function endEpisode(R) {
    episode++;
    if (episodeScore > bestScore) bestScore = episodeScore;
    epsilon = Math.max(EPS_MIN, EPS_START * Math.pow(EPS_DECAY, episode));
    save();
  }

  function tick() {
    requestAnimationFrame(tick);
    const R = getRunner();
    if (!R || !R.tRex || !R.horizon) return;

    // Start the game once, on the very first frame the engine is ready.
    if (!started) {
      if (!pendingStart) { pressSpace(); pendingStart = true; }
      if (R.activated) { started = true; }
      return;
    }
    if (paused) return;

    // -------- Episode finished (the dino crashed) --------
    if (R.crashed) {
      if (prevKey !== null) {
        learn(prevKey, prevAction, REWARD_CRASH, null);   // terminal update
      }
      endEpisode(R);
      prevKey = null; prevAction = null;
      episodeScore = 0;
      R.restart();                                        // begin next episode
      return;
    }

    // Wait out the opening intro animation (obstacles don't move yet).
    if (R.playingIntro || !R.activated) return;

    // -------- Normal step --------
    const key = encodeState(R);

    // Learn from the previous step now that we can see its outcome.
    if (prevKey !== null) {
      const r = REWARD_ALIVE + (prevAction === 1 ? REWARD_JUMP : 0);
      learn(prevKey, prevAction, r, key);
    }

    const action = chooseAction(key);
    if (action === 1) R.tRex.startJump();   // no-op if already mid-jump

    prevKey = key;
    prevAction = action;
    episodeScore = Math.round(R.distanceRan);
  }

  // ---- On-screen dashboard --------------------------------------------------
  let ui;
  function buildUI() {
    ui = document.createElement('div');
    ui.style.cssText = [
      'position:fixed', 'top:10px', 'right:10px', 'z-index:99999',
      'font:12px/1.5 monospace', 'background:rgba(20,20,20,.9)', 'color:#eee',
      'padding:10px 12px', 'border-radius:8px', 'width:190px',
      'box-shadow:0 2px 12px rgba(0,0,0,.4)'
    ].join(';');
    ui.innerHTML =
      '<b style="color:#7fd">Dino RL agent</b><br>' +
      '<div id="rl-stats" style="margin:6px 0"></div>' +
      '<button id="rl-pause" style="width:100%;margin-bottom:4px">Pause</button>' +
      '<button id="rl-reset" style="width:100%">Reset training</button>';
    document.body.appendChild(ui);

    ui.querySelector('#rl-pause').onclick = function () {
      paused = !paused;
      this.textContent = paused ? 'Resume' : 'Pause';
    };
    ui.querySelector('#rl-reset').onclick = function () {
      if (!confirm('Wipe the Q-table and restart training from scratch?')) return;
      Q = {}; episode = 0; bestScore = 0; epsilon = EPS_START;
      save();
    };
  }

  function refreshUI() {
    requestAnimationFrame(refreshUI);
    const stats = ui && ui.querySelector('#rl-stats');
    if (!stats) return;
    stats.innerHTML =
      'Episode: '   + episode + '<br>' +
      'Score: '     + episodeScore + '<br>' +
      'Best: '      + bestScore + '<br>' +
      'Epsilon: '   + epsilon.toFixed(3) + '<br>' +
      'States: '    + Object.keys(Q).length;
  }

  // ---- Boot -----------------------------------------------------------------
  function boot() {
    load();
    epsilon = Math.max(EPS_MIN, EPS_START * Math.pow(EPS_DECAY, episode));
    buildUI();
    requestAnimationFrame(tick);
    requestAnimationFrame(refreshUI);
    console.log('[Dino RL] agent started — episode', episode, 'epsilon', epsilon.toFixed(3));
  }

  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    boot();
  } else {
    window.addEventListener('DOMContentLoaded', boot);
  }
})();
