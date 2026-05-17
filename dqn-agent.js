/* ============================================================================
 * Chrome Dino — Deep Q-Network (DQN) agent
 * ----------------------------------------------------------------------------
 * A proper value-based deep-RL agent, replacing the primitive tabular table.
 * Same hooks as before: it reads the game's own state via `Runner.instance_`.
 *
 * ALGORITHM — DQN (Mnih et al. 2015), implemented from scratch in plain JS:
 *   - Q(s,a) approximated by a small MLP  (8 -> 32 -> 32 -> 2), ReLU hidden.
 *   - Experience replay buffer: decorrelates samples, reuses transitions.
 *   - Separate target network: stabilises the bootstrap target, synced
 *     periodically from the online network.
 *   - Huber / gradient-clipped TD error: robust to large reward magnitudes.
 *   - Adam optimiser, epsilon-greedy exploration with step-based decay.
 *
 * Unlike the table version the state is a *continuous* normalised feature
 * vector, so the network generalises across situations instead of memorising
 * discrete buckets.
 *
 * Keeps the existing UI: 0.5x-30x speed, named checkpoints, editable rewards.
 * ==========================================================================*/
(function () {
  'use strict';

  // ---- RL hyperparameters ---------------------------------------------------
  const GAMMA          = 0.99;
  const LR             = 5e-4;        // Adam learning rate
  const BATCH          = 32;
  const REPLAY_CAP     = 50000;
  const REPLAY_WARMUP  = 1000;        // steps of random play before training
  const TRAIN_EVERY    = 2;           // gradient step every N game-steps
  const TARGET_SYNC    = 1000;        // copy online -> target every N train steps
  const EPS_START      = 1.0;
  const EPS_MIN        = 0.05;
  const EPS_DECAY_STEPS = 30000;      // linear epsilon decay horizon

  // Editable reward terms.
  let rewards = { alive: 1, jump: -1, crash: -100 };

  // ---- Feature extraction ---------------------------------------------------
  const N_FEATURES = 8;
  const ACTIONS    = 2;               // 0 = do nothing, 1 = jump
  const LAYOUT     = [N_FEATURES, 32, 32, ACTIONS];

  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

  // Build a normalised continuous feature vector from the live game state.
  function features(R) {
    const trex = R.tRex;
    const obs = R.horizon.obstacles;
    const o1 = obs[0] && (obs[0].xPos + obs[0].width > trex.xPos) ? obs[0] : obs[1];
    const o2 = (o1 === obs[0]) ? obs[1] : obs[2];

    let gap1 = 1, w1 = 0, h1 = 0, gap2 = 1;
    if (o1) {
      gap1 = clamp((o1.xPos - trex.xPos) / 600, -0.1, 1);
      w1   = o1.width / 75;
      h1   = o1.typeConfig.height / 50;
    }
    if (o2) gap2 = clamp((o2.xPos - trex.xPos) / 600, -0.1, 1);

    const groundY = trex.groundYPos || 93;
    return [
      gap1,
      w1,
      h1,
      gap2,
      clamp((R.currentSpeed - 6) / 7, 0, 1),          // game speed
      clamp((groundY - trex.yPos) / 60, 0, 1.2),      // current jump height
      clamp((trex.jumpVelocity || 0) / 12, -1.2, 1.2),// vertical velocity
      trex.jumping ? 1 : 0
    ];
  }

  // ===========================================================================
  // Minimal neural network (MLP) with Adam — no external libraries.
  // ===========================================================================
  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function makeNet(layout) {
    const layers = [];
    for (let l = 1; l < layout.length; l++) {
      const nIn = layout[l - 1], nOut = layout[l];
      const std = Math.sqrt(2 / nIn);                 // He initialisation
      const W = [], b = [], mW = [], vW = [], mb = [], vb = [];
      for (let o = 0; o < nOut; o++) {
        const row = [], mRow = [], vRow = [];
        for (let i = 0; i < nIn; i++) {
          row.push(randn() * std); mRow.push(0); vRow.push(0);
        }
        W.push(row); mW.push(mRow); vW.push(vRow);
        b.push(0); mb.push(0); vb.push(0);
      }
      layers.push({ nIn, nOut, W, b, mW, vW, mb, vb });
    }
    return { layers, adamT: 0 };
  }

  // Forward pass; caches pre-activations for backprop. Hidden layers use ReLU,
  // output layer is linear (Q-values).
  function forward(net, x) {
    const cache = [{ a: x }];
    let a = x;
    for (let l = 0; l < net.layers.length; l++) {
      const L = net.layers[l];
      const last = l === net.layers.length - 1;
      const z = new Array(L.nOut);
      const out = new Array(L.nOut);
      for (let o = 0; o < L.nOut; o++) {
        let s = L.b[o];
        const Wo = L.W[o];
        for (let i = 0; i < L.nIn; i++) s += Wo[i] * a[i];
        z[o] = s;
        out[o] = last ? s : (s > 0 ? s : 0);          // linear / ReLU
      }
      cache.push({ z: z, a: out, in: a });
      a = out;
    }
    return { out: a, cache: cache };
  }

  function zeroGrads(net) {
    return net.layers.map((L) => ({
      W: L.W.map((row) => row.map(() => 0)),
      b: L.b.map(() => 0)
    }));
  }

  // Backprop a single sample's output-gradient into accumulated `grads`.
  function backward(net, cache, dOut, grads) {
    let delta = dOut;
    for (let l = net.layers.length - 1; l >= 0; l--) {
      const L = net.layers[l];
      const inp = cache[l].a;                          // input to layer l
      const g = grads[l];
      for (let o = 0; o < L.nOut; o++) {
        const d = delta[o];
        g.b[o] += d;
        const gWo = g.W[o];
        for (let i = 0; i < L.nIn; i++) gWo[i] += d * inp[i];
      }
      if (l > 0) {
        const newDelta = new Array(L.nIn).fill(0);
        for (let o = 0; o < L.nOut; o++) {
          const d = delta[o], Wo = L.W[o];
          for (let i = 0; i < L.nIn; i++) newDelta[i] += d * Wo[i];
        }
        const zPrev = cache[l].z;                      // ReLU derivative
        for (let i = 0; i < L.nIn; i++) if (zPrev[i] <= 0) newDelta[i] = 0;
        delta = newDelta;
      }
    }
  }

  // Adam parameter update from gradients accumulated over `batch` samples.
  function adamStep(net, grads, batch) {
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    net.adamT++;
    const t = net.adamT;
    const c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);
    for (let l = 0; l < net.layers.length; l++) {
      const L = net.layers[l], g = grads[l];
      for (let o = 0; o < L.nOut; o++) {
        // bias
        let gb = g.b[o] / batch;
        L.mb[o] = b1 * L.mb[o] + (1 - b1) * gb;
        L.vb[o] = b2 * L.vb[o] + (1 - b2) * gb * gb;
        L.b[o] -= LR * (L.mb[o] / c1) / (Math.sqrt(L.vb[o] / c2) + eps);
        // weights
        const Wo = L.W[o], mWo = L.mW[o], vWo = L.vW[o], gWo = g.W[o];
        for (let i = 0; i < L.nIn; i++) {
          const gw = gWo[i] / batch;
          mWo[i] = b1 * mWo[i] + (1 - b1) * gw;
          vWo[i] = b2 * vWo[i] + (1 - b2) * gw * gw;
          Wo[i] -= LR * (mWo[i] / c1) / (Math.sqrt(vWo[i] / c2) + eps);
        }
      }
    }
  }

  // Serialise / restore just the parameters (for checkpoints).
  function netToJSON(net) {
    return net.layers.map((L) => ({ W: L.W, b: L.b }));
  }
  function netFromJSON(json) {
    const net = makeNet(LAYOUT);
    for (let l = 0; l < net.layers.length; l++) {
      net.layers[l].W = json[l].W.map((r) => r.slice());
      net.layers[l].b = json[l].b.slice();
    }
    return net;
  }
  function copyNet(src, dst) {
    for (let l = 0; l < src.layers.length; l++) {
      const S = src.layers[l], D = dst.layers[l];
      for (let o = 0; o < S.nOut; o++) {
        D.b[o] = S.b[o];
        for (let i = 0; i < S.nIn; i++) D.W[o][i] = S.W[o][i];
      }
    }
  }

  // ===========================================================================
  // DQN agent
  // ===========================================================================
  let online = makeNet(LAYOUT);
  let target = makeNet(LAYOUT);
  copyNet(online, target);

  let replay = [];
  let replayHead = 0;
  let totalSteps = 0;            // game-steps taken
  let trainSteps = 0;            // gradient updates done
  let episode = 0;
  let bestScore = 0;
  let lossAvg = 0;               // running average Huber loss
  let epsilon = EPS_START;
  let scores = [];               // per-episode score history (for the graph)
  const SCORE_HISTORY_CAP = 4000;

  function pushReplay(tr) {
    if (replay.length < REPLAY_CAP) replay.push(tr);
    else { replay[replayHead] = tr; replayHead = (replayHead + 1) % REPLAY_CAP; }
  }

  function argmaxQ(net, x) {
    const q = forward(net, x).out;
    return q[1] > q[0] ? 1 : 0;
  }

  function selectAction(x) {
    if (Math.random() < epsilon) return Math.floor(Math.random() * ACTIONS);
    return argmaxQ(online, x);
  }

  // One DQN gradient update on a random minibatch.
  function trainBatch() {
    if (replay.length < REPLAY_WARMUP) return;
    const grads = zeroGrads(online);
    let batchLoss = 0;
    for (let k = 0; k < BATCH; k++) {
      const tr = replay[Math.floor(Math.random() * replay.length)];
      const f = forward(online, tr.s);

      let td = tr.r;
      if (tr.s2) {
        // Double-DQN: online picks the action, target evaluates it.
        const aStar = argmaxQ(online, tr.s2);
        td += GAMMA * forward(target, tr.s2).out[aStar];
      }
      let err = f.out[tr.a] - td;
      batchLoss += Math.min(0.5 * err * err, Math.abs(err) - 0.5); // Huber
      err = clamp(err, -1, 1);                                     // clipped grad
      const dOut = new Array(ACTIONS).fill(0);
      dOut[tr.a] = err;
      backward(online, f.cache, dOut, grads);
    }
    adamStep(online, grads, BATCH);
    trainSteps++;
    lossAvg = 0.99 * lossAvg + 0.01 * (batchLoss / BATCH);
    if (trainSteps % TARGET_SYNC === 0) copyNet(online, target);
  }

  // ===========================================================================
  // Speed control — virtual clock + fixed-step driver (unchanged design).
  // ===========================================================================
  const FRAME_MS = 1000 / 60;
  const SPEED_MIN = 0.5, SPEED_MAX = 30;
  const MAX_STEPS_PER_FRAME = 120;
  let speed = 1.0, stepAccum = 0;
  let virtualClock = performance.now();
  performance.now = function () { return virtualClock; };

  // ---- Persistence ----------------------------------------------------------
  const STORE_KEY = 'dino_dqn_v1';
  const CKPT_KEY  = 'dino_dqn_checkpoints_v1';
  let checkpoints = [];

  function load() {
    try {
      const raw = JSON.parse(localStorage.getItem(STORE_KEY) || '{}');
      if (raw.net) { online = netFromJSON(raw.net); copyNet(online, target); }
      episode    = raw.episode    || 0;
      bestScore  = raw.bestScore  || 0;
      totalSteps = raw.totalSteps || 0;
      scores     = raw.scores     || [];
      if (raw.speed)   speed   = raw.speed;
      if (raw.rewards) rewards = raw.rewards;
    } catch (e) {/* fresh start */}
    try { checkpoints = JSON.parse(localStorage.getItem(CKPT_KEY) || '[]'); }
    catch (e) { checkpoints = []; }
    epsilon = curEpsilon();
  }
  function save() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({
        net: netToJSON(online), episode, bestScore, totalSteps, speed, rewards, scores
      }));
    } catch (e) {/* storage disabled */}
  }
  function saveCheckpoints() {
    try { localStorage.setItem(CKPT_KEY, JSON.stringify(checkpoints)); }
    catch (e) { alert('Could not save checkpoint — localStorage may be full.'); }
  }

  function curEpsilon() {
    return Math.max(EPS_MIN, EPS_START - totalSteps / EPS_DECAY_STEPS);
  }

  // ---- Checkpoints ----------------------------------------------------------
  function snapshotCheckpoint() {
    const dflt = 'ep' + episode + ' · best ' + bestScore;
    const name = prompt('Checkpoint name:', dflt);
    if (name === null) return;
    checkpoints.push({
      id: Date.now() + '' + Math.floor(Math.random() * 1000),
      name: name.trim() || dflt,
      ts: Date.now(),
      net: netToJSON(online),
      episode: episode,
      bestScore: bestScore,
      totalSteps: totalSteps,
      rewards: Object.assign({}, rewards),
      scores: scores.slice()
    });
    saveCheckpoints();
    renderCheckpoints();
  }
  function loadCheckpoint(id) {
    const c = checkpoints.find((x) => x.id === id);
    if (!c) return;
    if (!confirm('Load checkpoint "' + c.name + '"? This replaces the current ' +
                 'network and training progress.')) return;
    online = netFromJSON(c.net);
    target = makeNet(LAYOUT); copyNet(online, target);
    replay = []; replayHead = 0; trainSteps = 0; lossAvg = 0;
    episode    = c.episode    || 0;
    bestScore  = c.bestScore  || 0;
    totalSteps = c.totalSteps || 0;
    scores     = (c.scores || []).slice();
    if (c.rewards) rewards = Object.assign({}, c.rewards);
    epsilon = curEpsilon();
    prevX = null; prevA = null; episodeScore = 0;
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

  // ===========================================================================
  // Game-step driver
  // ===========================================================================
  let prevX = null, prevA = null;
  let started = false, paused = false, ownsLoop = false, pendingStart = false;
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
    scores.push(episodeScore);
    if (scores.length > SCORE_HISTORY_CAP) scores.shift();
    save();
  }

  function gameStep(R) {
    // -------- Crash: store terminal transition, restart --------
    if (R.crashed) {
      if (prevX !== null) {
        pushReplay({ s: prevX, a: prevA, r: rewards.crash, s2: null });
      }
      endEpisode();
      prevX = null; prevA = null;
      episodeScore = 0;
      R.restart();
      return;
    }
    // -------- Intro: world frozen, just tick time --------
    if (R.playingIntro || !R.activated) {
      virtualClock += FRAME_MS;
      R.update();
      return;
    }
    // -------- Normal step --------
    const x = features(R);
    if (prevX !== null) {
      const r = rewards.alive + (prevA === 1 ? rewards.jump : 0);
      pushReplay({ s: prevX, a: prevA, r: r, s2: x });
    }
    const a = selectAction(x);
    if (a === 1) R.tRex.startJump();
    prevX = x; prevA = a;

    totalSteps++;
    epsilon = curEpsilon();
    if (totalSteps % TRAIN_EVERY === 0) trainBatch();

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

  // ===========================================================================
  // UI
  // ===========================================================================
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
    const d = new Date(ts), p = (n) => (n < 10 ? '0' + n : '' + n);
    return p(d.getMonth() + 1) + '/' + p(d.getDate()) + ' ' +
           p(d.getHours()) + ':' + p(d.getMinutes());
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
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
        'background:#552;color:#fbb;border:1px solid #744;border-radius:4px;cursor:pointer';
      del.onclick = (e) => { e.stopPropagation(); removeCheckpoint(c.id); };
      row.appendChild(info); row.appendChild(del);
      list.appendChild(row);
    });
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
      'padding:10px 12px', 'border-radius:8px', 'width:240px',
      'max-height:94vh', 'overflow:auto', 'box-shadow:0 2px 12px rgba(0,0,0,.4)'
    ].join(';');
    const hr = '<hr style="border:0;border-top:1px solid #444;margin:8px 0">';
    const row = (a, b) =>
      '<div style="display:flex;justify-content:space-between;align-items:center">' +
      a + b + '</div>';

    ui.innerHTML =
      '<b style="color:#7fd">Dino DQN agent</b>' +
      '<div style="color:#888;font-size:10px">Double-DQN · replay · target net</div>' +
      '<div id="rl-stats" style="margin:6px 0"></div>' +
      '<div style="color:#7fd;margin-bottom:3px">Score / episode</div>' +
      '<canvas id="rl-chart" width="216" height="84" ' +
        'style="width:100%;display:block;background:#1a1a1a;border-radius:4px"></canvas>' +
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

    const slider = ui.querySelector('#rl-speed');
    const box    = ui.querySelector('#rl-speed-num');
    slider.addEventListener('input', () => setSpeed(slider.value));
    box.addEventListener('change', () => setSpeed(box.value));
    box.addEventListener('keydown', (e) => { if (e.key === 'Enter') setSpeed(box.value); });

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

    ui.querySelector('#rl-ckpt-save').onclick = snapshotCheckpoint;
    ui.querySelector('#rl-pause').onclick = function () {
      paused = !paused;
      this.textContent = paused ? 'Resume' : 'Pause';
    };
    ui.querySelector('#rl-reset').onclick = function () {
      if (!confirm('Wipe the network and restart training from scratch?\n' +
                   '(Saved checkpoints are kept.)')) return;
      online = makeNet(LAYOUT); target = makeNet(LAYOUT); copyNet(online, target);
      replay = []; replayHead = 0;
      episode = 0; bestScore = 0; totalSteps = 0; trainSteps = 0; lossAvg = 0;
      scores = [];
      epsilon = EPS_START;
      prevX = null; prevA = null; episodeScore = 0;
      save();
    };

    setSpeed(speed);
    renderCheckpoints();
  }

  // Line chart of score per episode — sliding window of the last CHART_WINDOW
  // episodes: faint raw trace + bright moving average.
  const CHART_WINDOW = 50;
  function drawChart() {
    const cv = ui && ui.querySelector('#rl-chart');
    if (!cv) return;
    const ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, W, H);

    if (scores.length < 2) {
      ctx.fillStyle = '#888';
      ctx.font = '10px monospace';
      ctx.fillText('no episodes yet', 8, H / 2);
      return;
    }
    const view = scores.slice(-CHART_WINDOW);     // only the last N episodes
    const n = view.length;
    const firstEp = episode - n + 1;              // episode number of view[0]
    let maxV = 1;
    for (let i = 0; i < n; i++) if (view[i] > maxV) maxV = view[i];
    const xAt = (i) => 2 + (n > 1 ? i / (n - 1) : 0) * (W - 4);
    const yAt = (v) => H - 3 - (v / maxV) * (H - 14);

    // raw scores
    ctx.strokeStyle = '#3a6';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = xAt(i), y = yAt(view[i]);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();

    // moving average over the visible window
    const win = Math.max(1, Math.floor(n / 8));
    ctx.strokeStyle = '#7fd';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += view[i];
      if (i >= win) sum -= view[i - win];
      const avg = sum / Math.min(i + 1, win);
      const x = xAt(i), y = yAt(avg);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();

    // labels
    ctx.fillStyle = '#888';
    ctx.font = '9px monospace';
    ctx.fillText('max ' + maxV, 3, 10);
    const epLbl = 'ep ' + firstEp + '-' + episode;
    ctx.fillText(epLbl, W - ctx.measureText(epLbl).width - 3, H - 3);
  }

  let lastCharted = -1;
  function refreshUI() {
    requestAnimationFrame(refreshUI);
    const stats = ui && ui.querySelector('#rl-stats');
    if (!stats) return;
    if (scores.length !== lastCharted) {     // redraw only when a new episode ends
      lastCharted = scores.length;
      drawChart();
    }
    const phase = replay.length < REPLAY_WARMUP
      ? 'warmup ' + replay.length + '/' + REPLAY_WARMUP
      : 'training';
    stats.innerHTML =
      'Episode: '  + episode + '<br>' +
      'Score: '    + episodeScore + '<br>' +
      'Best: '     + bestScore + '<br>' +
      'Epsilon: '  + epsilon.toFixed(3) + '<br>' +
      'Steps: '    + totalSteps + '<br>' +
      'Updates: '  + trainSteps + '<br>' +
      'Loss: '     + lossAvg.toFixed(4) + '<br>' +
      '<span style="color:#888">' + phase + '</span>';
  }

  // ---- Boot -----------------------------------------------------------------
  function boot() {
    load();
    buildUI();
    requestAnimationFrame(driver);
    requestAnimationFrame(refreshUI);
    console.log('[Dino DQN] agent started — episode', episode,
                'steps', totalSteps, 'epsilon', epsilon.toFixed(3));
  }

  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    boot();
  } else {
    window.addEventListener('DOMContentLoaded', boot);
  }
})();
