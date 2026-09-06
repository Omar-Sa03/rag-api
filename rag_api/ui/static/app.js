/**
 * Company Assistant - Minimalist Dark Chat Bot Client
 */

// DOM Elements
const els = {
  chatMessages: document.getElementById('chatMessages'),
  welcomeState: document.getElementById('welcomeState'),
  messagesList: document.getElementById('messagesList'),
  typingIndicator: document.getElementById('typingIndicator'),
  chatForm: document.getElementById('chatForm'),
  chatInput: document.getElementById('chatInput'),
  btnSend: document.getElementById('btnSend'),
  sendIcon: document.getElementById('sendIcon'),
  sendSpinner: document.getElementById('sendSpinner'),
  deptScopeSelect: document.getElementById('deptScopeSelect'),
  btnNewChat: document.getElementById('btnNewChat'),
  systemStatus: document.getElementById('systemStatus'),
  toast: document.getElementById('toast'),

  // Upload modal elements
  btnAttach: document.getElementById('btnAttach'),
  btnOpenUpload: document.getElementById('btnOpenUpload'),
  uploadModal: document.getElementById('uploadModal'),
  btnCloseModal: document.getElementById('btnCloseModal'),
  btnCancelModal: document.getElementById('btnCancelModal'),
  uploadForm: document.getElementById('uploadForm'),
  modalFileInput: document.getElementById('modalFileInput'),
  fileChosenLabel: document.getElementById('fileChosenLabel'),
  modalDept: document.getElementById('modalDept'),
  modalCat: document.getElementById('modalCat'),
  modalUploadResult: document.getElementById('modalUploadResult'),
  btnModalSubmit: document.getElementById('btnModalSubmit'),
};

// Department and Category Display Labels
const DEPT_LABELS = {
  hr: 'HR',
  engineering: 'Engineering',
  finance: 'Finance',
  legal: 'Legal',
  operations: 'Operations',
  marketing: 'Marketing',
};

const CAT_LABELS = {
  policy: 'Policy',
  onboarding: 'Onboarding',
  benefits: 'Benefits',
  'org-chart': 'Org Chart',
  handbook: 'Handbook',
  procedure: 'Procedure',
  faq: 'FAQ',
  announcement: 'Announcement',
};

let isGenerating = false;

/**
 * Toast Notification (Minimalist Dark)
 */
function showToast(msg, kind = 'info') {
  const borderColors = {
    error: 'border-rose-500/40 text-rose-300',
    success: 'border-emerald-500/40 text-emerald-300',
    info: 'border-white/[0.12] text-zinc-200',
  };
  els.toast.className = `fixed bottom-20 left-1/2 -translate-x-1/2 z-50 px-4 py-2 rounded-full text-xs font-medium bg-[#1A1A24] border backdrop-blur-md shadow-xl transition duration-200 ${borderColors[kind] || borderColors.info}`;
  els.toast.textContent = msg;
  els.toast.classList.remove('hidden');

  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => {
    els.toast.classList.add('hidden');
  }, 2400);
}

/**
 * Safe HTML escaping helper
 */
function escapeHtml(str) {
  if (!str) return '';
  return String(str).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[c]));
}

/**
 * Markdown parser for assistant responses
 */
function renderMarkdown(text) {
  if (!text) return '';

  let html = escapeHtml(text);

  // Bold
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  const lines = html.split('\n');
  let result = [];
  let inList = false;
  let listType = 'ul';

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    if (!line) {
      if (inList) {
        result.push(`</${listType}>`);
        inList = false;
      }
      continue;
    }

    // Bullet List
    const bulletMatch = line.match(/^[-*]\s+(.*)$/);
    if (bulletMatch) {
      if (!inList || listType !== 'ul') {
        if (inList) result.push(`</${listType}>`);
        result.push('<ul>');
        inList = true;
        listType = 'ul';
      }
      result.push(`<li>${bulletMatch[1]}</li>`);
      continue;
    }

    // Numbered List
    const numMatch = line.match(/^\d+\.\s+(.*)$/);
    if (numMatch) {
      if (!inList || listType !== 'ol') {
        if (inList) result.push(`</${listType}>`);
        result.push('<ol>');
        inList = true;
        listType = 'ol';
      }
      result.push(`<li>${numMatch[1]}</li>`);
      continue;
    }

    if (inList) {
      result.push(`</${listType}>`);
      inList = false;
    }

    // Headings
    if (line.startsWith('### ')) {
      result.push(`<h4 class="font-display font-semibold text-sm text-zinc-100 mt-3 mb-1.5">${line.slice(4)}</h4>`);
      continue;
    }
    if (line.startsWith('## ')) {
      result.push(`<h3 class="font-display font-semibold text-base text-white mt-4 mb-2">${line.slice(3)}</h3>`);
      continue;
    }

    // Standard paragraph
    result.push(`<p>${line}</p>`);
  }

  if (inList) {
    result.push(`</${listType}>`);
  }

  return result.join('');
}

/**
 * Scroll chat smoothly to bottom
 */
function scrollToBottom() {
  requestAnimationFrame(() => {
    els.chatMessages.scrollTo({
      top: els.chatMessages.scrollHeight,
      behavior: 'smooth',
    });
  });
}

/**
 * Auto-resize textarea input
 */
function autoResizeInput() {
  els.chatInput.style.height = 'auto';
  const nextHeight = Math.min(els.chatInput.scrollHeight, 140);
  els.chatInput.style.height = `${nextHeight}px`;

  const hasContent = els.chatInput.value.trim().length > 0;
  els.btnSend.disabled = !hasContent || isGenerating;
}

/**
 * Append User Message Bubble (Normal, Clean Slate Bubble on Right)
 */
function appendUserMessage(text) {
  els.welcomeState.classList.add('hidden');

  const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const msgEl = document.createElement('div');
  msgEl.className = 'flex justify-end';
  msgEl.innerHTML = `
    <div class="max-w-[85%] sm:max-w-[75%] space-y-1 text-right">
      <div class="inline-block text-left bg-[#1A1A24] border border-white/[0.08] text-[#FAFAFA] px-4 py-3 rounded-2xl rounded-tr-sm text-sm leading-relaxed whitespace-pre-wrap select-text shadow-sm">
        ${escapeHtml(text)}
      </div>
      <div class="text-[10px] text-zinc-500 pr-1">${timeStr}</div>
    </div>
  `;
  els.messagesList.appendChild(msgEl);
  scrollToBottom();
}

/**
 * Append Assistant Message Bubble (Glass Card with Avatar on Left)
 */
function appendAssistantMessage(data) {
  const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const msgId = `msg-${Date.now()}`;
  const answer = data.answer || 'I could not find an answer in the company documents.';
  const sources = data.sources || [];
  const disclaimer = data.disclaimer || 'Company Assistant • Grounded in internal documentation.';

  const container = document.createElement('div');
  container.className = 'flex items-start gap-3';
  container.id = msgId;

  // Bot Avatar
  const avatarHtml = `
    <div class="w-8 h-8 rounded-full bg-[#1A1A24] border border-white/[0.08] flex items-center justify-center text-amber-500 flex-shrink-0 mt-1">
      <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
      </svg>
    </div>
  `;

  // Sources Accordion HTML
  let sourcesHtml = '';
  if (sources.length > 0) {
    const sourceCards = sources.map((src, idx) => {
      const meta = src.metadata || {};
      const deptName = DEPT_LABELS[meta.department] || meta.department || 'General';
      const catName = CAT_LABELS[meta.category] || meta.category || 'Policy';
      const filename = meta.filename || meta.source || `Document ${idx + 1}`;
      const textPreview = src.text_preview || (src.text ? src.text.slice(0, 180) + '...' : '');

      return `
        <div class="p-3 rounded-lg bg-[#12121A] border border-white/[0.06] text-xs space-y-1">
          <div class="flex items-center justify-between gap-2 text-zinc-300">
            <span class="font-medium truncate">${escapeHtml(filename)}</span>
            <div class="flex items-center gap-1.5 flex-shrink-0 text-[10px]">
              <span class="px-1.5 py-0.5 rounded bg-white/[0.05] text-zinc-400">${escapeHtml(deptName)}</span>
              <span class="px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 font-medium">${escapeHtml(catName)}</span>
            </div>
          </div>
          <p class="text-[11px] text-zinc-400 leading-relaxed">${escapeHtml(textPreview)}</p>
        </div>
      `;
    }).join('');

    sourcesHtml = `
      <div class="mt-3 pt-2.5 border-t border-white/[0.06]">
        <button type="button" class="btn-toggle-sources flex items-center gap-1.5 text-xs text-amber-400/90 hover:text-amber-300 transition-colors font-medium focus:outline-none">
          <svg class="w-3.5 h-3.5 chevron-icon transition-transform duration-200" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
          </svg>
          <span>${sources.length} referenced ${sources.length === 1 ? 'document' : 'documents'}</span>
        </button>
        <div class="sources-content hidden mt-2.5 space-y-2">
          ${sourceCards}
        </div>
      </div>
    `;
  }

  container.innerHTML = `
    ${avatarHtml}
    <div class="max-w-[88%] sm:max-w-[82%] space-y-1">
      <div class="glass-card px-4 py-3.5 rounded-2xl rounded-tl-sm text-sm">
        
        <!-- Answer Content -->
        <div class="chat-markdown">
          ${renderMarkdown(answer)}
        </div>

        <!-- Sources Accordion -->
        ${sourcesHtml}

        <!-- Disclaimer -->
        <div class="mt-3 pt-2 border-t border-white/[0.06] text-[11px] text-zinc-500 leading-tight">
          ${escapeHtml(disclaimer)}
        </div>

      </div>

      <!-- Action Footer -->
      <div class="flex items-center justify-between text-[10px] text-zinc-500 px-1 pt-0.5">
        <button class="btn-copy hover:text-zinc-300 transition-colors flex items-center gap-1" title="Copy answer">
          <svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          <span>Copy</span>
        </button>
        <span>${timeStr}</span>
      </div>
    </div>
  `;

  // Wire Events
  const toggleBtn = container.querySelector('.btn-toggle-sources');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const content = container.querySelector('.sources-content');
      const chevron = toggleBtn.querySelector('.chevron-icon');
      const isOpen = !content.classList.contains('hidden');
      if (isOpen) {
        content.classList.add('hidden');
        chevron.classList.remove('rotate-180');
      } else {
        content.classList.remove('hidden');
        chevron.classList.add('rotate-180');
        scrollToBottom();
      }
    });
  }

  const copyBtn = container.querySelector('.btn-copy');
  if (copyBtn) {
    copyBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(answer);
        showToast('Answer copied', 'success');
        copyBtn.innerHTML = `
          <svg class="w-3 h-3 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
          </svg>
          <span class="text-emerald-400">Copied</span>
        `;
        setTimeout(() => {
          copyBtn.innerHTML = `
            <svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            <span>Copy</span>
          `;
        }, 2000);
      } catch {
        showToast('Unable to copy', 'error');
      }
    });
  }

  els.messagesList.appendChild(container);
  scrollToBottom();
}

/**
 * Send user query to /v1/query
 */
async function sendQuery(text) {
  const query = (text || els.chatInput.value).trim();
  if (!query || isGenerating) return;

  isGenerating = true;
  els.chatInput.value = '';
  autoResizeInput();
  els.btnSend.disabled = true;
  els.sendIcon.classList.add('hidden');
  els.sendSpinner.classList.remove('hidden');

  appendUserMessage(query);

  els.typingIndicator.classList.remove('hidden');
  scrollToBottom();

  const dept = els.deptScopeSelect.value || null;

  try {
    const payload = {
      q: query,
      mode: 'hybrid',
      n_results: 5,
      rerank: true,
      include_scores: true,
      generate_answer: true,
    };
    if (dept) {
      payload.department = dept;
    }

    const res = await fetch('/v1/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      const errDetail = data.detail || data.error || `Server returned ${res.status}`;
      throw new Error(errDetail);
    }

    appendAssistantMessage(data);

  } catch (err) {
    console.error('Query error:', err);
    appendAssistantMessage({
      answer: 'Sorry, I encountered an issue while searching the company documents.\n\n' + err.message + '\n\n*Please ensure Ollama and the backend service are running.*',
      sources: [],
      disclaimer: 'System connection error.',
    });
    showToast(err.message, 'error');
  } finally {
    isGenerating = false;
    els.typingIndicator.classList.add('hidden');
    els.sendIcon.classList.remove('hidden');
    els.sendSpinner.classList.add('hidden');
    autoResizeInput();
    els.chatInput.focus();
  }
}

/**
 * Reset conversation
 */
function resetChat() {
  els.messagesList.innerHTML = '';
  els.welcomeState.classList.remove('hidden');
  els.chatInput.value = '';
  autoResizeInput();
  els.chatInput.focus();
  showToast('Chat cleared', 'info');
}

/**
 * Document Upload Handling
 */
function openUploadModal() {
  els.uploadModal.classList.remove('hidden');
  els.modalUploadResult.classList.add('hidden');
  els.modalUploadResult.textContent = '';
  els.fileChosenLabel.innerHTML = '<span class="font-medium text-amber-400">Click to browse</span> or drag file here';
  els.modalFileInput.value = '';
}

function closeUploadModal() {
  els.uploadModal.classList.add('hidden');
}

async function handleUpload(e) {
  e.preventDefault();
  const file = els.modalFileInput.files?.[0];
  if (!file) {
    showToast('Select a file to upload', 'error');
    return;
  }

  const dept = els.modalDept.value;
  const cat = els.modalCat.value;

  els.btnModalSubmit.disabled = true;
  els.btnModalSubmit.textContent = 'Indexing...';

  try {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('strategy', 'recursive');
    fd.append('chunk_size', '1000');
    fd.append('chunk_overlap', '200');
    fd.append('department', dept);
    if (cat) fd.append('category', cat);

    const res = await fetch('/v1/upload', {
      method: 'POST',
      body: fd,
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      throw new Error(data.error || data.detail || `Upload failed (${res.status})`);
    }

    els.modalUploadResult.className = 'block text-xs p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-300';
    els.modalUploadResult.textContent = `"${data.filename}" uploaded successfully (${data.chunks} chunks indexed).`;

    showToast('Document uploaded', 'success');

    setTimeout(() => {
      closeUploadModal();
      els.modalUploadResult.classList.add('hidden');
    }, 2000);

  } catch (err) {
    els.modalUploadResult.className = 'block text-xs p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300';
    els.modalUploadResult.textContent = `Upload error: ${err.message}`;
    showToast(err.message, 'error');
  } finally {
    els.btnModalSubmit.disabled = false;
    els.btnModalSubmit.textContent = 'Upload & Index';
  }
}

/**
 * Health check on startup
 */
async function checkHealth() {
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    els.systemStatus.innerHTML = `
      <span class="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
      <span>Online • Ready to help with company questions</span>
    `;
  } catch (e) {
    els.systemStatus.innerHTML = `
      <span class="inline-block w-1.5 h-1.5 rounded-full bg-amber-400"></span>
      <span>Offline or Starting up</span>
    `;
  }
}

/**
 * Initialize
 */
function init() {
  els.chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    sendQuery();
  });

  els.chatInput.addEventListener('input', autoResizeInput);
  els.chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendQuery();
    }
  });

  // Prompt Cards
  document.querySelectorAll('.prompt-card').forEach((btn) => {
    btn.addEventListener('click', () => {
      const q = btn.querySelector('p')?.textContent.trim();
      if (q) {
        sendQuery(q);
      }
    });
  });

  els.btnNewChat.addEventListener('click', resetChat);

  els.btnAttach.addEventListener('click', openUploadModal);
  els.btnOpenUpload.addEventListener('click', openUploadModal);
  els.btnCloseModal.addEventListener('click', closeUploadModal);
  els.btnCancelModal.addEventListener('click', closeUploadModal);
  els.uploadModal.addEventListener('click', (e) => {
    if (e.target === els.uploadModal) closeUploadModal();
  });

  els.modalFileInput.addEventListener('change', () => {
    const file = els.modalFileInput.files?.[0];
    if (file) {
      els.fileChosenLabel.innerHTML = `<span class="font-medium text-white">${escapeHtml(file.name)}</span> (${Math.round(file.size / 1024)} KB)`;
    }
  });

  els.uploadForm.addEventListener('submit', handleUpload);

  els.chatInput.focus();
  checkHealth();
}

document.addEventListener('DOMContentLoaded', init);
