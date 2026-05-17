/* ============================================================================
 * Chrome Dino — Reinforcement Learning agent
 * ----------------------------------------------------------------------------
 * Tabular Q-learning. The agent hooks straight into the game's own state via
 * `Runner.instance_` (the singleton from dino.html) — it does not read pixels.
 *
 * SPEED CONTROL
 *   The game runs on a *virtual clock* (`performance.now()` is overridden), so
 *   the driver runs N fixed 1/60s game-steps per real frame, N tracking the
 *   speed multiplier (0.5x..30x). Every step uses a normal small delta, so
 *   physics & collisions stay accurate even at 30x.
 *
 * CHECKPOINTS
 *   The live Q-table auto-saves to localStorage. On top of that the user can
 *   snapshot named checkpoints, browse them, load them back, or delete them.
 *
 * REWARDS
 *   The three reward terms (alive / jump / crash) are editable at runtime.
 * ==========================================================================*/
(function () {
  'use strict';

  // ---- Q-learning hyperparameters ------------------------------------------
  const ALPHA     = 0.15;        // learning rate
  const GAMMA     = 0.99;        // discount factor
  const EPS_START = 1.00;        // initial exploration rate
  const EPS_MIN   = 0.01;        // floor exploration rate
  const EPS_DECAY = 0.96;        // multiplied per episode

  // Editable reward terms (defaults match the original design).
  let rewards = { alive: 1, jump: -1, crash: -100 };

  // ---- State discretisation -------------------------------------------------
  const DIST_BUCKET = 12;        // px per distance bucket
  const DIST_MAX    = 20;        // distance buckets 0..20 (20 == "far / none")
  const SPD_MIN     = 6;         // game starts at speed 6
  const SPD_BUCKETS = 8;         // speed buckets 0..7
  const ACTIONS     = 2;         // 0 = do nothing, 1 = jump

  // ---- Speed control --------------------------------------------------------
  const FRAME_MS = 1000 / 60;
  const SPEED_MIN = 0.5, SPEED_MAX = 30;
  const MAX_STEPS_PER_FRAME = 120;

  let speed = 1.0;
  let stepAccum = 0;
  let virtualClock = performance.now();
  performance.now = function () { return virtualClock; };   // game's clock

  // ---- Persistence keys -----------------------------------------------------
  const STORE_KEY = 'dino_rl_qtable_v1';        // live auto-save
  const CKPT_KEY  = 'dino_rl_checkpoints_v1';   // named checkpoints

  let Q = {};                    // stateKey -> [q_noop, q_jump]
  let episode = 0;
  let bestScore = 0;
  let checkpoints = [];          // [{id,name,ts,Q,episode,bestScore,rewards}]

  function load() {
    try {
      const raw = JSON.parse(localStorage.getItem(STORE_KEY) || '{}');
      Q = raw.Q || {};
      episode = raw.episode || 0;
      bestScore = raw.bestScore || 0;
      if (raw.speed) speed = raw.speed;
      if (raw.rewards) rewards = raw.rewards;
    } catch (e) { Q = {}; }
    try {
      checkpoints = JSON.parse(localStorage.getItem(CKPT_KEY) || '[]');
    } catch (e) { checkpoints = []; }
  }
  function save() {
    try {
      localStorage.setItem(STORE_KEY,
        JSON.stringify({ Q, episode, bestScore, speed, rewards }));
    } catch (e) {/* storage disabled */}
  }
  function saveCheckpoints() {
    try { localStorage.setItem(CKPT_KEY, JSON.stringify(checkpoints)); }
    catch (e) { alert('Could not save checkpoint — localStorage may be full.'); }
  }
  function qRow(key) {
    if (!Q[key]) Q[key] = new Array(ACTIONS).fill(0);
    return Q[key];
  }

  // ---- Checkpoint operations ------------------------------------------------
  function snapshotCheckpoint() {
    const dflt = 'ep' + episode + ' · best ' + bestScore;
    const name = prompt('Checkpoint name:', dflt);
    if (name === null) return;                         // cancelled
    checkpoints.push({
      id: Date.now() + '' + Math.floor(Math.random() * 1000),
      name: name.trim() || dflt,
      ts: Date.now(),
      Q: JSON.parse(JSON.stringify(Q)),                // deep copy
      episode: episode,
      bestScore: bestScore,
      rewards: Object.assign({}, rewards)
    });
    saveCheckpoints();
    renderCheckpoints();
  }

  function loadCheckpoint(id) {
    const c = checkpoints.find((x) => x.id === id);
    if (!c) return;
    if (!confirm('Load checkpoint "' + c.name + '"? This replaces the current ' +
                 'Q-table and training progress.')) return;
    Q = JSON.parse(JSON.stringify(c.Q));
    episode = c.episode || 0;
    bestScore = c.bestScore || 0;
    if (c.rewards) rewards = Object.assign({}, c.rewards);
    epsilon = Math.max(EPS_MIN, EPS_START * Math.pow(EPS_DECAY, episode));
    prevKey = null; prevAction = null; episodeScore = 0;   // drop stale transition
    save();
    syncRewardInputs();
    renderCheckpoints();
  }

  function removeCheckpoint(id) {
    const c = checkpoints.find((x) => x.id === id);
    if (c && !confirm('Delete checkpoint "' + c.name + '"?')) return;
    checkpoints = checkpoints.filter((x) => x.id !== id);
    saveCheckpoints();
    renderCheckpoints();
  }

  // ---- State / policy / learning -------------------------------------------
  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

  function encodeState(R) {
    const trex = R.tRex;
    const obstacles = R.horizon.obstacles;
    let next = null;
    for (let i = 0; i < obstacles.length; i++) {
      const o = obstacles[i];
      if (o.xPos + o.width > trex.xPos) { next = o; break; }
    }
    let distBin;
    if (!next) distBin = DIST_MAX;
    else distBin = clamp(Math.floor((next.xPos - trex.xPos) / DIST_BUCKET), 0, DIST_MAX);
    const spdBin  = clamp(Math.floor(R.currentSpeed) - SPD_MIN, 0, SPD_BUCKETS - 1);
    return distBin + '|' + spdBin + '|' + (trex.jumping ? 1 : 0);
  }

  let epsilon = EPS_START;

  function chooseAction(key) {
    if (Math.random() < epsilon) return Math.floor(Math.random() * ACTIONS);
    const row = qRow(key);
    return row[1] > row[0] ? 1 : 0;
  }

  function learn(key, action, reward, nextKey) {
    const row = qRow(key);
    let target = reward;
    if (nextKey !== null) {
      const nr = qRow(nextKey);
      target += GAMMA * Math.max(nr[0], nr[1]);
    }
    row[action] += ALPHA * (target - row[action]);
  }

  // ---- Game-step driver -----------------------------------------------------
  let prevKey = null, prevAction = null;
  let started = false, paused = false, ownsLoop = false;
  let pendingStart = false;
  let episodeScore = 0;

  function getRunner() { return window.Runner && window.Runner.instance_; }
  function noopRaq() {}

  function pressSpace() {
    document.dispatchEvent(new KeyboardEvent('keydown',
      { keyCode: 32, which: 32, bubbles: true }));
    setTimeout(() => document.dispatchEvent(new KeyboardEvent('keyup',
      { keyCode: 32, which: 32, bubbles: true })), 30);
  }

  function endEpisode() {
    episode++;
    if (episodeScore > bestScore) bestScore = episodeScore;
    epsilon = Math.max(EPS_MIN, EPS_START * Math.pow(EPS_DECAY, episode));
    save();
  }

  function gameStep(R) {
    if (R.crashed) {
      if (prevKey !== null) learn(prevKey, prevAction, rewards.crash, null);
      endEpisode();
      prevKey = null; prevAction = null;
      episodeScore = 0;
      R.restart();
      return;
    }
    if (R.playingIntro || !R.activated) {
      virtualClock += FRAME_MS;
      R.update();
      return;
    }
    const key = encodeState(R);
    if (prevKey !== null) {
      const r = rewards.alive + (prevAction === 1 ? rewards.jump : 0);
      learn(prevKey, prevAction, r, key);
    }
    const action = chooseAction(key);
    if (action === 1) R.tRex.startJump();
    prevKey = key;
    prevAction = action;

    virtualClock += FRAME_MS;
    R.update();
    episodeScore = Math.round(R.distanceRan);
  }

  function driver() {
    requestAnimationFrame(driver);
    const R = getRunner();
    if (!R || !R.tRex || !R.horizon) return;

    if (!started) {
      if (!pendingStart) { pressSpace(); pendingStart = true; }
      if (R.activated) started = true;
      return;
    }
    if (!ownsLoop) {
      R.raq = noopRaq;
      R.playSound = function () {};
      ownsLoop = true;
    }
    if (paused) return;

    stepAccum += speed;
    let budget = 0;
    while (stepAccum >= 1 && budget < MAX_STEPS_PER_FRAME) {
      stepAccum -= 1; budget++;
      gameStep(R);
    }
    if (stepAccum > MAX_STEPS_PER_FRAME) stepAccum = 0;
  }

  // ---- UI -------------------------------------------------------------------
  let ui;

  function setSpeed(v) {
    speed = clamp(Number(v) || 1, SPEED_MIN, SPEED_MAX);
    speed = Math.round(speed * 100) / 100;
    const slider = ui && ui.querySelector('#rl-speed');
    const box    = ui && ui.querySelector('#rl-speed-num');
    if (slider) slider.value = clamp(speed, SPEED_MIN, SPEED_MAX);
    if (box)    box.value = speed;
    save();
  }

  function syncRewardInputs() {
    if (!ui) return;
    ui.querySelector('#rw-alive').value = rewards.alive;
    ui.querySelector('#rw-jump').value  = rewards.jump;
    ui.querySelector('#rw-crash').value = rewards.crash;
  }

  function fmtTime(ts) {
    const d = new Date(ts);
    const p = (n) => (n < 10 ? '0' + n : '' + n);
    return p(d.getMonth() + 1) + '/' + p(d.getDate()) + ' ' +
           p(d.getHours()) + ':' + p(d.getMinutes());
  }

  function renderCheckpoints() {
    const list = ui && ui.querySelector('#rl-ckpt-list');
    if (!list) return;
    list.innerHTML = '';
    if (checkpoints.length === 0) {
      list.innerHTML = '<div style="color:#888;padding:4px 0">No checkpoints yet</div>';
      return;
    }
    checkpoints.slice().reverse().forEach((c) => {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;align-items:center;gap:4px;' +
        'padding:3px 4px;margin:2px 0;background:#2a2a2a;border-radius:4px';

      const info = document.createElement('div');
      info.style.cssText = 'flex:1;cursor:pointer;overflow:hidden';
      info.title = 'Load this checkpoint';
      info.innerHTML = '<div style="color:#7fd;white-space:nowrap;' +
        'overflow:hidden;text-overflow:ellipsis">' + escapeHtml(c.name) + '</div>' +
        '<div style="color:#999;font-size:10px">ep ' + (c.episode || 0) +
        ' · best ' + (c.bestScore || 0) + ' · ' + fmtTime(c.ts) + '</div>';
      info.onclick = () => loadCheckpoint(c.id);

      const del = document.createElement('button');
      del.textContent = '×';
      del.title = 'Delete checkpoint';
      del.style.cssText = 'flex:none;width:20px;height:20px;line-height:1;' +
        'background:#552; color:#fbb;border:1px solid #744;border-radius:4px;cursor:pointer';
      del.onclick = (e) => { e.stopPropagation(); removeCheckpoint(c.id); };

      row.appendChild(info);
      row.appendChild(del);
      list.appendChild(row);
    });
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  }

  function rewardInput(id, val) {
    return '<input id="' + id + '" type="number" step="1" value="' + val + '" ' +
      'style="width:56px;background:#333;color:#eee;border:1px solid #555">';
  }

  function buildUI() {
    ui = document.createElement('div');
    ui.style.cssText = [
      'position:fixed', 'top:10px', 'right:10px', 'z-index:99999',
      'font:12px/1.5 monospace', 'background:rgba(20,20,20,.93)', 'color:#eee',
      'padding:10px 12px', 'border-radius:8px', 'width:236px',
      'max-height:94vh', 'overflow:auto',
      'box-shadow:0 2px 12px rgba(0,0,0,.4)'
    ].join(';');

    const hr = '<hr style="border:0;border-top:1px solid #444;margin:8px 0">';
    const row = (a, b) =>
      '<div style="display:flex;justify-content:space-between;align-items:center">' +
      a + b + '</div>';

    ui.innerHTML =
      '<b style="color:#7fd">Dino RL agent</b>' +
      '<div id="rl-stats" style="margin:6px 0"></div>' +
      hr +
      '<div style="margin-bottom:4px">' +
        row('<span>Game speed</span>',
            '<input id="rl-speed-num" type="number" min="' + SPEED_MIN + '" max="' +
            SPEED_MAX + '" step="0.5" style="width:56px;background:#333;color:#eee;' +
            'border:1px solid #555">') +
        '<input id="rl-speed" type="range" min="' + SPEED_MIN + '" max="' + SPEED_MAX +
        '" step="0.5" style="width:100%">' +
      '</div>' +
      hr +
      '<div style="color:#7fd;margin-bottom:3px">Rewards</div>' +
        row('<span>Alive / step</span>', rewardInput('rw-alive', rewards.alive)) +
        row('<span>Jump</span>',         rewardInput('rw-jump',  rewards.jump)) +
        row('<span>Crash</span>',        rewardInput('rw-crash', rewards.crash)) +
      hr +
      '<div style="color:#7fd;margin-bottom:3px">Checkpoints</div>' +
      '<button id="rl-ckpt-save" style="width:100%;margin-bottom:4px">' +
        'Save checkpoint</button>' +
      '<div id="rl-ckpt-list" style="max-height:160px;overflow:auto"></div>' +
      hr +
      '<button id="rl-pause" style="width:100%;margin-bottom:4px">Pause</button>' +
      '<button id="rl-reset" style="width:100%">Reset training</button>';
    document.body.appendChild(ui);

    // speed
    const slider = ui.querySelector('#rl-speed');
    const box    = ui.querySelector('#rl-speed-num');
    slider.addEventListener('input', () => setSpeed(slider.value));
    box.addEventListener('change', () => setSpeed(box.value));
    box.addEventListener('keydown', (e) => { if (e.key === 'Enter') setSpeed(box.value); });

    // rewards
    const bindReward = (id, key) => {
      const el = ui.querySelector(id);
      const apply = () => {
        const n = Number(el.value);
        if (!isNaN(n)) { rewards[key] = n; save(); }
        el.value = rewards[key];
      };
      el.addEventListener('change', apply);
      el.addEventListener('keydown', (e) => { if (e.key === 'Enter') apply(); });
    };
    bindReward('#rw-alive', 'alive');
    bindReward('#rw-jump',  'jump');
    bindReward('#rw-crash', 'crash');

    // checkpoints
    ui.querySelector('#rl-ckpt-save').onclick = snapshotCheckpoint;

    // controls
    ui.querySelector('#rl-pause').onclick = function () {
      paused = !paused;
      this.textContent = paused ? 'Resume' : 'Pause';
    };
    ui.querySelector('#rl-reset').onclick = function () {
      if (!confirm('Wipe the live Q-table and restart training from scratch?\n' +
                   '(Saved checkpoints are kept.)')) return;
      Q = {}; episode = 0; bestScore = 0; epsilon = EPS_START;
      prevKey = null; prevAction = null; episodeScore = 0;
      save();
    };

    setSpeed(speed);
    renderCheckpoints();
  }

  function refreshUI() {
    requestAnimationFrame(refreshUI);
    const stats = ui && ui.querySelector('#rl-stats');
    if (!stats) return;
    stats.innerHTML =
      'Episode: ' + episode + '<br>' +
      'Score: '   + episodeScore + '<br>' +
      'Best: '    + bestScore + '<br>' +
      'Epsilon: ' + epsilon.toFixed(3) + '<br>' +
      'States: '  + Object.keys(Q).length;
  }

  // ---- Boot -----------------------------------------------------------------
  function boot() {
    load();
    epsilon = Math.max(EPS_MIN, EPS_START * Math.pow(EPS_DECAY, episode));
    buildUI();
    requestAnimationFrame(driver);
    requestAnimationFrame(refreshUI);
    console.log('[Dino RL] agent started — episode', episode,
                'epsilon', epsilon.toFixed(3));
  }

  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    boot();
  } else {
    window.addEventListener('DOMContentLoaded', boot);
  }
})();
