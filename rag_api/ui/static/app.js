const els = {
  statusText: document.getElementById('statusText'),
  btnHealth: document.getElementById('btnHealth'),
  latency: document.getElementById('latency'),
  queryInput: document.getElementById('queryInput'),
  modeSelect: document.getElementById('modeSelect'),
  nResults: document.getElementById('nResults'),
  toggleRerank: document.getElementById('toggleRerank'),
  toggleScores: document.getElementById('toggleScores'),
  btnQuery: document.getElementById('btnQuery'),
  btnCopy: document.getElementById('btnCopy'),
  btnRaw: document.getElementById('btnRaw'),
  answerBox: document.getElementById('answerBox'),
  rawBox: document.getElementById('rawBox'),
  sources: document.getElementById('sources'),
  addText: document.getElementById('addText'),
  addChunk: document.getElementById('addChunk'),
  addStrategy: document.getElementById('addStrategy'),
  btnAdd: document.getElementById('btnAdd'),
  addResult: document.getElementById('addResult'),
  fileInput: document.getElementById('fileInput'),
  upStrategy: document.getElementById('upStrategy'),
  upChunkSize: document.getElementById('upChunkSize'),
  upOverlap: document.getElementById('upOverlap'),
  btnUpload: document.getElementById('btnUpload'),
  uploadResult: document.getElementById('uploadResult'),
  toast: document.getElementById('toast'),
};

let state = {
  rerank: true,
  includeScores: true,
  lastJson: null,
};

function toast(msg, kind = 'info') {
  const bg = kind === 'error' ? 'rgba(239, 68, 68, 0.20)' : kind === 'success' ? 'rgba(16, 185, 129, 0.20)' : 'rgba(255,255,255,0.10)';
  els.toast.style.background = bg;
  els.toast.textContent = msg;
  els.toast.classList.remove('hidden');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => els.toast.classList.add('hidden'), 2200);
}

function setBusy(btn, busy, labelBusy = 'Working...') {
  if (!btn) return;
  btn.disabled = busy;
  btn.dataset._label = btn.dataset._label || btn.textContent;
  btn.textContent = busy ? labelBusy : btn.dataset._label;
  btn.classList.toggle('opacity-70', busy);
  btn.classList.toggle('cursor-not-allowed', busy);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function renderSources(sources) {
  els.sources.innerHTML = '';
  if (!sources || sources.length === 0) {
    els.sources.innerHTML = '<div class="text-sm text-slate-300">No sources returned.</div>';
    return;
  }

  sources.forEach((src, idx) => {
    const preview = src.text_preview || (src.text ? (src.text.slice(0, 150) + (src.text.length > 150 ? '...' : '')) : '');
    const fullText = src.text || '';
    const meta = src.metadata || {};
    const scores = src.scores || {};

    const scoreLine = Object.keys(scores).length
      ? `<div class="mt-2 text-[11px] text-slate-300">${Object.entries(scores).map(([k,v]) => `<span class="mr-2 rounded bg-white/5 px-1.5 py-0.5">${escapeHtml(k)}: ${Number(v).toFixed(4)}</span>`).join('')}</div>`
      : '';

    const metaLine = Object.keys(meta).length
      ? `<div class="mt-2 text-[11px] text-slate-300">${Object.entries(meta).slice(0, 6).map(([k,v]) => `<span class="mr-2 rounded bg-white/5 px-1.5 py-0.5">${escapeHtml(k)}: ${escapeHtml(v)}</span>`).join('')}</div>`
      : '';

    const el = document.createElement('div');
    el.className = 'rounded-xl border border-white/10 bg-black/30 p-3';
    el.innerHTML = `
      <button class="w-full text-left" data-idx="${idx}">
        <div class="flex items-center justify-between gap-3">
          <div>
            <div class="text-xs text-slate-400">Source #${idx + 1} • Rank ${escapeHtml(src.rank ?? '')}</div>
            <div class="mt-1 text-sm text-slate-100">${escapeHtml(preview)}</div>
          </div>
          <div class="text-xs text-slate-400">Expand</div>
        </div>
      </button>
      <div class="mt-3 hidden" data-body="${idx}">
        ${scoreLine}
        ${metaLine}
        <div class="mt-3 whitespace-pre-wrap rounded-lg border border-white/10 bg-black/20 p-3 text-xs text-slate-200">${escapeHtml(fullText)}</div>
      </div>
    `;

    el.querySelector('button').addEventListener('click', () => {
      const body = el.querySelector(`[data-body="${idx}"]`);
      body.classList.toggle('hidden');
    });

    els.sources.appendChild(el);
  });
}

async function apiJson(path, method, body) {
  const res = await fetch(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data?.error || data?.detail || `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return data;
}

async function checkHealth() {
  try {
    const t0 = performance.now();
    const res = await fetch('/health');
    const ms = Math.round(performance.now() - t0);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    els.statusText.textContent = `Healthy • v${data.version}`;
    els.latency.textContent = `Latency: ${ms}ms`;
    toast('Health OK', 'success');
  } catch (e) {
    els.statusText.textContent = 'API unreachable';
    toast(`Health failed: ${e.message}`, 'error');
  }
}

async function runQuery() {
  const q = els.queryInput.value.trim();
  if (!q) return toast('Type a question first', 'error');

  const mode = els.modeSelect.value;
  const n = Number(els.nResults.value || 5);

  setBusy(els.btnQuery, true, 'Searching...');
  els.answerBox.textContent = '';
  els.rawBox.classList.add('hidden');
  els.rawBox.textContent = '';
  renderSources([]);

  try {
    const t0 = performance.now();
    const data = await apiJson('/v1/query', 'POST', {
      q,
      mode,
      n_results: n,
      rerank: state.rerank,
      include_scores: state.includeScores,
    });
    const ms = Math.round(performance.now() - t0);

    state.lastJson = data;
    els.answerBox.textContent = data.answer || '';
    renderSources(data.sources || []);
    els.latency.textContent = `Query: ${ms}ms • Results: ${data.total_results ?? 0}`;
    toast('Query complete', 'success');
  } catch (e) {
    toast(e.message, 'error');
    els.answerBox.textContent = `Error: ${e.message}`;
  } finally {
    setBusy(els.btnQuery, false);
  }
}

async function addKnowledge() {
  const text = els.addText.value.trim();
  if (!text) return toast('Paste some text first', 'error');

  setBusy(els.btnAdd, true, 'Adding...');
  els.addResult.textContent = '';

  try {
    const data = await apiJson('/v1/add', 'POST', {
      text,
      chunk: !!els.addChunk.checked,
      strategy: els.addStrategy.value,
    });

    els.addResult.textContent = data.message || 'Added.';
    toast('Knowledge added', 'success');
    els.addText.value = '';
  } catch (e) {
    toast(e.message, 'error');
    els.addResult.textContent = `Error: ${e.message}`;
  } finally {
    setBusy(els.btnAdd, false);
  }
}

async function uploadDoc() {
  const file = els.fileInput.files?.[0];
  if (!file) return toast('Choose a file to upload', 'error');

  setBusy(els.btnUpload, true, 'Uploading...');
  els.uploadResult.textContent = '';

  try {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('strategy', els.upStrategy.value);
    fd.append('chunk_size', String(Number(els.upChunkSize.value || 1000)));
    fd.append('chunk_overlap', String(Number(els.upOverlap.value || 200)));

    const res = await fetch('/v1/upload', { method: 'POST', body: fd });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const msg = data?.error || data?.detail || `Upload failed (${res.status})`;
      throw new Error(msg);
    }

    els.uploadResult.textContent = `Uploaded ${data.filename} • ${data.chunks} chunks`;
    toast('Upload complete', 'success');
    els.fileInput.value = '';
  } catch (e) {
    toast(e.message, 'error');
    els.uploadResult.textContent = `Error: ${e.message}`;
  } finally {
    setBusy(els.btnUpload, false);
  }
}

function toggle(btn, value, onText = 'On', offText = 'Off') {
  btn.textContent = value ? onText : offText;
  btn.classList.toggle('bg-emerald-400/10', value);
  btn.classList.toggle('bg-white/5', !value);
}

els.toggleRerank.addEventListener('click', () => {
  state.rerank = !state.rerank;
  toggle(els.toggleRerank, state.rerank);
});

toggle(els.toggleRerank, state.rerank);

els.toggleScores.addEventListener('click', () => {
  state.includeScores = !state.includeScores;
  toggle(els.toggleScores, state.includeScores);
});

toggle(els.toggleScores, state.includeScores);

els.btnHealth.addEventListener('click', checkHealth);
els.btnQuery.addEventListener('click', runQuery);
els.btnAdd.addEventListener('click', addKnowledge);
els.btnUpload.addEventListener('click', uploadDoc);

els.btnCopy.addEventListener('click', async () => {
  const text = els.answerBox.textContent || '';
  if (!text.trim()) return toast('Nothing to copy', 'error');
  await navigator.clipboard.writeText(text);
  toast('Copied', 'success');
});

els.btnRaw.addEventListener('click', () => {
  if (!state.lastJson) return toast('Run a query first', 'error');
  const open = !els.rawBox.classList.contains('hidden');
  if (open) {
    els.rawBox.classList.add('hidden');
    return;
  }
  els.rawBox.textContent = JSON.stringify(state.lastJson, null, 2);
  els.rawBox.classList.remove('hidden');
});

els.queryInput.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') runQuery();
});

checkHealth();
