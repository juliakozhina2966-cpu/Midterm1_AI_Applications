// CreditPulse front end: draws the ranked portfolio once, then places the client in it
// every time the statement history changes.

// Three real clients from the UCI dataset, one per risk band.
const PRESETS = {
  low: {
    LIMIT_BAL: 250000, EDUCATION: 1,
    PAY_0: 0, PAY_2: 0, PAY_3: 0, PAY_4: 0, PAY_5: 0, PAY_6: 0,
    BILL_AMT1: 70887, BILL_AMT2: 67060, BILL_AMT3: 63561, BILL_AMT4: 59696, BILL_AMT5: 56875, BILL_AMT6: 55512,
    PAY_AMT1: 3000, PAY_AMT2: 3000, PAY_AMT3: 3000, PAY_AMT4: 3000, PAY_AMT5: 3000, PAY_AMT6: 3000,
  },
  medium: {
    LIMIT_BAL: 50000, EDUCATION: 3,
    PAY_0: 1, PAY_2: 2, PAY_3: 0, PAY_4: 0, PAY_5: 0, PAY_6: 0,
    BILL_AMT1: 50614, BILL_AMT2: 29173, BILL_AMT3: 28116, BILL_AMT4: 28771, BILL_AMT5: 29531, BILL_AMT6: 30211,
    PAY_AMT1: 0, PAY_AMT2: 1500, PAY_AMT3: 1100, PAY_AMT4: 1200, PAY_AMT5: 1300, PAY_AMT6: 1100,
  },
  high: {
    LIMIT_BAL: 30000, EDUCATION: 3,
    PAY_0: 2, PAY_2: 2, PAY_3: 7, PAY_4: 7, PAY_5: 7, PAY_6: 7,
    BILL_AMT1: 2450, BILL_AMT2: 2450, BILL_AMT3: 2450, BILL_AMT4: 2450, BILL_AMT5: 2450, BILL_AMT6: 2450,
    PAY_AMT1: 0, PAY_AMT2: 0, PAY_AMT3: 0, PAY_AMT4: 0, PAY_AMT5: 0, PAY_AMT6: 0,
  },
};

// Field suffix 1 = September ... 6 = April (PAY_0 is September's status).
const MONTHS = ['September', 'August', 'July', 'June', 'May', 'April'];
const DECISIONS = {
  high: 'Call this client this month',
  medium: 'Send a reminder, no call',
  low: 'No action this month',
};
const W = 1000;   // svg viewBox width
const H = 300;    // svg viewBox height

const form = document.getElementById('client-form');
const errorBox = document.getElementById('form-error');
const queue = document.getElementById('queue');
const marker = document.getElementById('marker');
const $ = (id) => document.getElementById(id);
const fmt = (n) => n.toLocaleString('en-US');
const ordinal = (n) => {
  const s = ['th', 'st', 'nd', 'rd'];
  const v = n % 100;
  return fmt(n) + (s[(v - 20) % 10] || s[v] || s[0]);
};

// ---------- the ranked portfolio ----------
async function drawQueue() {
  const { scores, capacity } = await (await fetch('/api/portfolio')).json();
  const n = scores.length;
  const x = (i) => (i / n) * W;
  const y = (s) => (1 - s) * H;

  const step = Math.max(1, Math.floor(n / 900));
  const points = [];
  for (let i = 0; i < n; i += step) points.push([x(i), y(scores[i])]);
  points.push([W, y(scores[n - 1])]);

  const line = points.map(([px, py], k) => `${k ? 'L' : 'M'}${px.toFixed(1)},${py.toFixed(1)}`).join('');
  const cutX = x(capacity);
  const callPts = points.filter(([px]) => px <= cutX);
  callPts.push([cutX, y(scores[capacity - 1])]);
  const callLine = callPts.map(([px, py], k) => `${k ? 'L' : 'M'}${px.toFixed(1)},${py.toFixed(1)}`).join('');

  $('curve').setAttribute('d', line);
  $('area-rest').setAttribute('d', `${line}L${W},${H}L0,${H}Z`);
  $('area-call').setAttribute('d', `${callLine}L${cutX.toFixed(1)},${H}L0,${H}Z`);

  const share = (capacity / n) * 100;
  $('call-bracket').style.width = `${share}%`;
  $('call-bracket').querySelector('span').textContent = `Called this month, ${fmt(capacity)} clients`;
  $('cut-label').style.left = `${share}%`;
  $('cut-label').textContent = `${ordinal(capacity)} client: ${Math.round(scores[capacity - 1] * 100)} in 100`;
}

function placeClient(answer) {
  const { rank, portfolio_size: size, capacity, probability } = answer;
  const left = ((rank - 0.5) / size) * 100;
  marker.hidden = false;
  marker.style.left = `${left}%`;
  marker.classList.toggle('flip', left > 62);
  $('marker-dot').style.top = `${(1 - probability) * 100}%`;
  queue.dataset.band = answer.band;

  $('rank-value').textContent = ordinal(rank);
  $('rank-gap').textContent = rank <= capacity
    ? `On the call list, ${fmt(capacity - rank)} places inside the cut`
    : `${fmt(rank - capacity)} places after the last call`;
  $('rank-odds').textContent = `${Math.round(probability * 100)} in 100 clients with this history missed the next payment`;
}

// ---------- the form ----------
function describe(name) {
  if (name === 'LIMIT_BAL') return 'credit limit';
  const k = Number(name.match(/\d+$/)[0]);
  if (name.startsWith('BILL_AMT')) return `${MONTHS[k - 1]} bill`;
  if (name.startsWith('PAY_AMT')) return `${MONTHS[k - 1]} payment`;
  return name;
}

function paintStatus() {
  for (const select of form.querySelectorAll('.status-row select')) select.dataset.s = select.value;
}

function fill(values) {
  for (const [name, value] of Object.entries(values)) {
    if (form.elements[name]) form.elements[name].value = value;
  }
  paintStatus();
}

function readForm() {
  const data = {};
  const missing = [];
  for (const field of form.elements) {
    if (!field.name) continue;
    if (field.value === '') missing.push(describe(field.name));
    data[field.name] = field.value === '' ? null : Number(field.value);
  }
  return { data, missing };
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = !message;
}

function waiting(text) {
  marker.hidden = true;
  queue.dataset.band = 'none';
  $('decision-title').textContent = 'Waiting for a complete history';
  $('decision-text').textContent = text;
  $('reasons').replaceChildren();
}

function render(answer) {
  placeClient(answer);
  $('decision-title').textContent = DECISIONS[answer.band];
  $('decision-text').textContent = answer.action;

  const list = $('reasons');
  list.replaceChildren();
  if (!answer.reasons.length) {
    const li = document.createElement('li');
    li.className = 'empty';
    li.textContent = 'No single sign stands out. The estimate comes from the whole history.';
    list.append(li);
  }
  for (const reason of answer.reasons) {
    const li = document.createElement('li');
    li.className = reason.direction;
    li.textContent = reason.text;
    list.append(li);
  }
}

let pending = 0;
async function score() {
  const { data, missing } = readForm();
  if (missing.length) {
    showError(`Missing: ${missing.join(', ')}.`);
    waiting('Fill in the missing amounts to place the client in the queue.');
    return;
  }
  const ticket = ++pending;
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    const answer = await response.json();
    if (ticket !== pending) return;          // a newer edit is already on its way
    if (!response.ok) {
      showError(answer.error || 'The server could not score this client.');
      waiting('Correct the statements to place the client in the queue.');
      return;
    }
    showError('');
    render(answer);
  } catch {
    if (ticket === pending) showError('The scoring server is not reachable. Check that app.py is running.');
  }
}

let timer;
form.addEventListener('input', () => {
  paintStatus();
  clearTimeout(timer);
  timer = setTimeout(score, 250);
});
form.addEventListener('submit', (event) => { event.preventDefault(); score(); });
for (const button of document.querySelectorAll('[data-preset]')) {
  button.addEventListener('click', () => { fill(PRESETS[button.dataset.preset]); score(); });
}

// #low, #medium or #high in the URL opens that example (used for the slide screenshots)
fill(PRESETS[location.hash.slice(1)] || PRESETS.medium);
drawQueue().then(score);
