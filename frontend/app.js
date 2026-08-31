'use strict';

const API_BASE = 'http://127.0.0.1:8000';
const CHAT_URL = `${API_BASE}/api/chat`;
const ANALYZE_URL = `${API_BASE}/api/diabetes/analyze`;

// Global State
let activeChatId = null;
let currentFeatures = {};
let analysisCompleted = false;
let messageHistory = [];

const chatHistory = document.getElementById('chat-history');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const typingIndicator = document.getElementById('typing-indicator');

const sidebar = document.getElementById('sidebar');
const sidebarCloseBtn = document.getElementById('sidebar-close-btn');
const sidebarOpenBtn = document.getElementById('sidebar-open-btn');
const newChatBtn = document.getElementById('new-chat-btn');
const historyList = document.getElementById('history-list');

// SVGs for Chat Avatars
const assistantSvg = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>`;
const userSvg = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>`;

const welcomeMessage = `
  <p>Hello! I am Clinicagen, your medical AI assistant. To assess your diabetes risk using our machine learning model, I'll need some medical details from you.</p>
  <p>Could you please provide information such as your Age, BMI, Glucose levels, Blood Pressure, Insulin, Skin Thickness, number of Pregnancies, and Diabetes Pedigree Function?</p>
  <p>You can describe this naturally (e.g., <em>"I am 30 years old with a BMI of 25..."</em>).</p>
`;

// Render Welcome Suggestion Grid (ChatGPT style dashboard)
function renderSuggestionsGrid() {
  chatHistory.innerHTML = `
    <div class="welcome-container">
      <h2 class="welcome-title">How can I help you today?</h2>
      <div class="suggestions-grid">
        <button class="suggestion-card" onclick="selectSuggestion('Check risk for: Age 45, BMI 32.5, Glucose 140, BP 80, Insulin 120, Skin 30, Preg 2, DPF 0.6')">
          <strong>Assess sample patient</strong>
          <span>Check risk for BMI 32.5, Age 45</span>
        </button>
        <button class="suggestion-card" onclick="selectSuggestion('Explain insulin resistance and its relation to diabetes risk')">
          <strong>Explain insulin resistance</strong>
          <span>What is its relation to diabetes risk?</span>
        </button>
        <button class="suggestion-card" onclick="selectSuggestion('What are the clinical guidelines for fasting glucose levels?')">
          <strong>Fasting glucose ranges</strong>
          <span>What are the clinical guidelines?</span>
        </button>
        <button class="suggestion-card" onclick="selectSuggestion('Tell me about modifiable risk factors for diabetes')">
          <strong>Modifiable risk factors</strong>
          <span>How weight and activity impact risk</span>
        </button>
      </div>
    </div>
  `;
}

window.selectSuggestion = function(text) {
  chatInput.value = text;
  chatForm.dispatchEvent(new Event('submit'));
};

// ---------------------------------------------------------
// LocalStorage Persistence Layer
// ---------------------------------------------------------
function getStoredChats() {
  const data = localStorage.getItem('clinicagen_saved_chats');
  return data ? JSON.parse(data) : {};
}

function saveStoredChats(chats) {
  localStorage.setItem('clinicagen_saved_chats', JSON.stringify(chats));
}

function saveCurrentChatState() {
  if (!activeChatId) return;
  const chats = getStoredChats();
  chats[activeChatId] = {
    id: activeChatId,
    title: chats[activeChatId]?.title || 'New chat',
    currentFeatures,
    analysisCompleted,
    history: messageHistory
  };
  saveStoredChats(chats);
}

function loadChatState(chatId) {
  const chats = getStoredChats();
  const chat = chats[chatId];
  if (!chat) return;

  activeChatId = chatId;
  currentFeatures = chat.currentFeatures || {};
  analysisCompleted = chat.analysisCompleted || false;
  messageHistory = chat.history || [];

  // Re-render Chat Area DOM
  chatHistory.innerHTML = '';
  if (messageHistory.length <= 1) {
    renderSuggestionsGrid();
  } else {
    messageHistory.forEach(msg => {
      appendMessageDOM(msg.role, msg.content);
    });
  }
  
  reconstructSourcesFromHistory();
  scrollToBottom();
}

function createNewChat() {
  saveCurrentChatState();

  const chats = getStoredChats();
  const newId = 'chat_' + Date.now();
  const newChat = {
    id: newId,
    title: 'New Diabetes Chat',
    currentFeatures: {},
    analysisCompleted: false,
    history: [{ role: 'assistant', content: welcomeMessage }]
  };

  chats[newId] = newChat;
  saveStoredChats(chats);

  activeChatId = newId;
  currentFeatures = {};
  analysisCompleted = false;
  messageHistory = [...newChat.history];

  renderSuggestionsGrid();
  renderSidebarChats();
  scrollToBottom();
}

function deleteChat(chatId, event) {
  if (event) event.stopPropagation();
  const chats = getStoredChats();
  delete chats[chatId];
  saveStoredChats(chats);

  const keys = Object.keys(chats);
  if (activeChatId === chatId) {
    if (keys.length > 0) {
      loadChatState(keys[keys.length - 1]);
    } else {
      createNewChat();
    }
  } else {
    renderSidebarChats();
  }
}

function renameChatTitleIfDefault(chatId, text) {
  const chats = getStoredChats();
  if (chats[chatId] && (chats[chatId].title === 'New Diabetes Chat' || chats[chatId].title === 'New chat')) {
    let cleanText = text.replace(/<[^>]+>/g, '').trim();
    if (cleanText.length > 25) {
      cleanText = cleanText.substring(0, 25) + '...';
    }
    chats[chatId].title = cleanText || 'Diabetes Chat';
    saveStoredChats(chats);
    renderSidebarChats();
  }
}

function renderSidebarChats() {
  if (!historyList) return;
  const chats = getStoredChats();
  historyList.innerHTML = '';
  
  const sortedKeys = Object.keys(chats).sort((a, b) => {
    const ta = parseInt(a.replace('chat_', '')) || 0;
    const tb = parseInt(b.replace('chat_', '')) || 0;
    return tb - ta;
  });

  if (sortedKeys.length === 0) {
    setTimeout(() => createNewChat(), 10);
    return;
  }

  sortedKeys.forEach(chatId => {
    const chat = chats[chatId];
    const item = document.createElement('button');
    item.className = `history-item ${chatId === activeChatId ? 'active' : ''}`;
    item.onclick = () => {
      saveCurrentChatState();
      loadChatState(chatId);
      renderSidebarChats();
    };

    item.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
      <span>${escapeHtml(chat.title)}</span>
      <span class="delete-chat-icon" onclick="deleteChat('${chatId}', event)" style="margin-left: auto; padding: 2px; color: var(--text-muted); opacity: 0.7; transition: opacity 0.15s; font-weight: bold;" title="Delete chat">×</span>
    `;

    historyList.appendChild(item);
  });
}

function reconstructSourcesFromHistory() {
  window.currentSources = {};
  messageHistory.forEach(msg => {
    if (msg.role === 'assistant' && msg.content.includes('class="result-card"')) {
      // Rebuild window.currentSources from parsed chunks if present in HTML
    }
  });
}

// ---------------------------------------------------------
// DOM Rendering / Scroller Helpers
// ---------------------------------------------------------
function scrollToBottom() {
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatMarkdownToHtml(text) {
  if (!text) return '';
  let escaped = escapeHtml(text);
  
  // Replace bold syntax **bold** -> <strong>bold</strong>
  escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  
  // Replace headers: e.g. ### Header -> <h4>Header</h4>
  escaped = escaped.replace(/(?:^|\n)###\s+([^\n]+)/g, '\n<h4 style="margin: 10px 0 5px 0; color: var(--text-main); font-size: 0.95rem;">$1</h4>');
  escaped = escaped.replace(/(?:^|\n)##\s+([^\n]+)/g, '\n<h4 style="margin: 12px 0 6px 0; color: var(--text-main); font-size: 1rem;">$1</h4>');
  
  // Replace bullet lists: * item or - item -> <li>item</li>
  escaped = escaped.replace(/(?:^|\n)[*\-]\s+([^\n]+)/g, '\n<li>$1</li>');
  
  // Group consecutive <li> items into <ul> blocks
  // Since we have \n<li>, we can replace sequences of them
  escaped = escaped.replace(/((?:\n<li>.*?<\/li>)+)/g, '\n<ul style="margin: 8px 0 8px 20px; padding-left: 0;">$1\n</ul>');
  
  // Replace numbered lists: 1. item -> <li>item</li>
  escaped = escaped.replace(/(?:^|\n)(\d+)\.\s+([^\n]+)/g, '\n<li data-index="$1">$2</li>');
  escaped = escaped.replace(/((?:\n<li data-index=.*?<\/li>)+)/g, '\n<ol style="margin: 8px 0 8px 20px; padding-left: 0;">$1\n</ol>');
  
  // Split paragraphs by double newlines or single newlines that are not inside lists
  const lines = escaped.split(/\n\n+/);
  const formattedParagraphs = lines.map(p => {
    const trimmed = p.trim();
    if (!trimmed) return '';
    if (trimmed.startsWith('<ul') || trimmed.startsWith('<ol') || trimmed.startsWith('<h') || trimmed.startsWith('<li')) {
      return trimmed;
    }
    return `<p style="margin-bottom: 8px; line-height: 1.55;">${trimmed.replace(/\n/g, '<br>')}</p>`;
  });
  
  return formattedParagraphs.filter(Boolean).join('');
}

// Low-level DOM injection
function appendMessageDOM(sender, htmlContent) {
  const rowDiv = document.createElement('div');
  rowDiv.className = `message-row ${sender}`;
  
  const innerDiv = document.createElement('div');
  innerDiv.className = 'message-inner';
  
  const avatar = document.createElement('div');
  avatar.className = `avatar ${sender}-avatar`;
  avatar.innerHTML = sender === 'user' ? userSvg : assistantSvg;
  
  const content = document.createElement('div');
  content.className = 'content';
  content.innerHTML = htmlContent;
  
  innerDiv.appendChild(avatar);
  innerDiv.appendChild(content);
  rowDiv.appendChild(innerDiv);
  
  chatHistory.appendChild(rowDiv);
}

// High-level messaging (DOM + History State + Storage Sync)
function appendMessage(sender, htmlContent) {
  // If we are currently displaying suggestions grid, clear it before appending first message
  if (messageHistory.length <= 1 && chatHistory.querySelector('.welcome-container')) {
    chatHistory.innerHTML = '';
    // Append the hardcoded welcome message to match background history log
    appendMessageDOM('assistant', welcomeMessage);
  }
  
  appendMessageDOM(sender, htmlContent);
  messageHistory.push({ role: sender, content: htmlContent });
  saveCurrentChatState();
}

function showTyping() {
  typingIndicator.removeAttribute('hidden');
  scrollToBottom();
}

function hideTyping() {
  typingIndicator.setAttribute('hidden', '');
}

// ---------------------------------------------------------
// Pipeline Explanation Generator
// ---------------------------------------------------------
function generateResultHTML(data) {
  const pred = data.prediction;
  const prob = data.model_probability;
  const label = data.risk_label || '';
  const badgeClass = pred === 1 ? 'positive' : 'negative';
  const badgeText = pred === 1 ? 'Positive Class' : 'Negative Class';
  
  const chunkMap = {};
  const sourcesList = data.explanation?.sources || [];
  
  sourcesList.forEach((s, idx) => {
    chunkMap[s.chunk_id] = {
      index: idx + 1,
      source: s.source,
      publisher: s.publisher,
      page: s.page,
      url: s.url,
      text: s.text,
      score: s.score
    };
  });
  
  window.currentSources = chunkMap;
  
  function renderCitations(citationIds) {
    if (!citationIds || citationIds.length === 0) return '';
    return citationIds
      .map(id => {
        const info = chunkMap[id];
        if (!info) return '';
        return `<span class="citation-badge" data-chunk-id="${id}" onclick="showCitationTooltip(event, '${id}')" title="Grounded Medical Evidence from ${escapeHtml(info.publisher)}">[${info.index}]</span>`;
      })
      .filter(Boolean)
      .join('');
  }
  
  let html = `
    <p>I have gathered all the necessary information and completed the analysis.</p>
    <div class="result-card">
      <div class="badge-row">
        <div class="result-badge ${badgeClass}">${badgeText}</div>
      </div>
      <div class="metric-grid">
        <div class="metric-item">
          <span class="metric-label">Confidence</span>
          <span class="metric-value">${(prob * 100).toFixed(1)}%</span>
        </div>
      </div>
      <p style="font-size: 0.9rem; margin-bottom: 15px;">${escapeHtml(label)}</p>
  `;

  if (data.explanation_status === 'success' && data.explanation) {
    const expl = data.explanation;
    const summaryCitations = renderCitations(expl.medical_context_citation_chunk_ids);
    
    html += `
      <div class="analysis-explanation-section" style="margin-top: 15px; border-top: 1px solid var(--border-color); padding-top: 15px;">
        <p><strong>Analysis:</strong> ${escapeHtml(expl.summary)} ${summaryCitations}</p>
        <p style="margin-top: 8px;">${escapeHtml(expl.prediction_explanation)}</p>
      </div>
    `;

    if (expl.important_factors && expl.important_factors.length > 0) {
      html += `<div class="important-factors"><h4>Key Factors</h4>`;
      expl.important_factors.forEach(f => {
        const factorCitations = renderCitations(f.citation_chunk_ids);
        html += `
          <div class="factor-item">
            <div class="factor-title-row" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
              <span class="factor-name">${escapeHtml(f.factor)}</span>
              ${factorCitations}
            </div>
            <p class="factor-explanation">${escapeHtml(f.explanation)}</p>
          </div>
        `;
      });
      html += `</div>`;
    }

    if (expl.medical_context) {
      const contextCitations = renderCitations(expl.medical_context_citation_chunk_ids);
      html += `
        <div class="medical-context-container" style="margin-top: 15px; padding: 12px; background: rgba(139, 92, 246, 0.05); border-left: 3px solid var(--accent-secondary); border-radius: 4px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
            <h4 style="color: var(--accent-secondary); margin: 0; font-size: 0.95rem;">Grounded RAG Insights</h4>
            ${contextCitations}
          </div>
          <p style="font-size: 0.9rem; margin: 0; line-height: 1.4;">${escapeHtml(expl.medical_context)}</p>
        </div>
      `;
    }

    if (expl.recommendation) {
      html += `
        <div style="margin-top: 15px; padding: 12px; background: rgba(59, 130, 246, 0.05); border-left: 3px solid var(--accent-primary); border-radius: 4px;">
          <strong>Recommendation:</strong> ${escapeHtml(expl.recommendation)}
        </div>
      `;
    }

    if (expl.sources && expl.sources.length > 0) {
      html += `<div class="sources-section"><h4>Sources Cited</h4><ul style="padding-left: 20px; margin-top: 5px; font-size: 0.85rem;">`;
      expl.sources.forEach((s, idx) => {
        const urlPart = s.url ? ` <a href="${escapeHtml(s.url)}" target="_blank" style="color: var(--accent-primary); text-decoration: none;">[Link]</a>` : '';
        html += `<li style="margin-bottom: 4px;"><strong>[${idx + 1}] ${escapeHtml(s.source)}</strong> (${escapeHtml(s.publisher)}, pg ${s.page})${urlPart}</li>`;
      });
      html += `</ul></div>`;
    }

    if (expl.disclaimer) {
      html += `<p style="font-size: 0.75rem; color: var(--text-muted); margin-top: 15px; text-align: center; line-height: 1.3;">${escapeHtml(expl.disclaimer)}</p>`;
    }
  } else if (data.explanation_status === 'fallback') {
      html += `<p style="margin-top:10px; color: #ca8a04;">${escapeHtml(data.explanation_message)}</p>`;
  } else {
      html += `<p style="margin-top:10px; color: var(--error);">${escapeHtml(data.explanation_message)}</p>`;
  }

  if (data.rag_query || (data.retrieved_sources && data.retrieved_sources.length > 0)) {
    const queryStr = data.rag_query || "N/A";
    const sourcesCount = data.retrieved_sources ? data.retrieved_sources.length : 0;
    
    html += `
      <div class="rag-inspector" style="margin-top: 20px; border-top: 1px solid var(--border-color); padding-top: 15px;">
        <button type="button" class="rag-inspector-toggle" onclick="toggleRagInspector(this)" style="background: var(--bg-input); border: 1px solid var(--border-color); width: 100%; text-align: left; padding: 10px 15px; border-top-left-radius: 8px; border-top-right-radius: 8px; border-bottom-left-radius: 0px; border-bottom-right-radius: 0px; color: var(--text-main); font-family: inherit; font-size: 0.85rem; cursor: pointer; display: flex; justify-content: space-between; align-items: center;">
          <span>🔍 RAG Pipeline Inspector (Top 5 Retrieved Chunks)</span>
          <span class="toggle-icon">▲</span>
        </button>
        <div class="rag-inspector-content" style="display: block; padding: 12px; background: #f9fafb; border: 1px solid var(--border-color); border-top: none; border-bottom-left-radius: 8px; border-bottom-right-radius: 8px; font-size: 0.8rem;">
          <div style="margin-bottom: 10px;">
            <strong style="color: var(--accent-primary); display: block; margin-bottom: 3px;">Generated Search Query:</strong>
            <div style="background: var(--bg-main); padding: 8px; border-radius: 4px; font-family: monospace; border: 1px solid var(--border-color); line-height: 1.3;">${escapeHtml(queryStr)}</div>
          </div>
          <div>
            <strong style="color: var(--accent-secondary); display: block; margin-bottom: 5px;">FAISS Retrieved Chunks (Similarity Threshold: 0.35):</strong>
            <div class="retrieved-documents-list" style="display: flex; flex-direction: column; gap: 8px; max-height: 250px; overflow-y: auto; padding-right: 5px;">
    `;
    
    if (sourcesCount > 0) {
      data.retrieved_sources.forEach((s, idx) => {
        const urlLink = s.url ? `<a href="${escapeHtml(s.url)}" target="_blank" style="color: var(--accent-primary); text-decoration: none; font-weight: 600;">Link ↗</a>` : '';
        const matchPct = (s.score * 100).toFixed(1);
        
        let scoreColor = '#dc2626'; // low
        if (s.score >= 0.55) {
          scoreColor = '#16a34a'; // high
        } else if (s.score >= 0.4) {
          scoreColor = '#ca8a04'; // medium
        }
        
        html += `
          <div class="retrieved-doc-item" style="background: var(--bg-main); border: 1px solid var(--border-color); padding: 8px; border-radius: 6px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; border-bottom: 1px solid var(--border-color); padding-bottom: 4px; font-size: 0.75rem; color: var(--text-muted);">
              <span><strong>Chunk ${idx + 1}</strong> (${escapeHtml(s.publisher)}, pg ${s.page})</span>
              <div style="display: flex; gap: 10px; align-items: center;">
                <span style="color: ${scoreColor}; font-weight: 600;">Match: ${matchPct}%</span>
                ${urlLink}
              </div>
            </div>
            <p style="margin: 0; line-height: 1.35; color: var(--text-muted); font-style: italic;">"${escapeHtml(s.text)}"</p>
          </div>
        `;
      });
    } else {
      html += `<p style="margin: 0; color: #dc2626;">No documents met the similarity threshold.</p>`;
    }
    
    html += `
            </div>
          </div>
        </div>
      </div>
    `;
  }

  html += `</div>`;
  return html;
}

// ---------------------------------------------------------
// Tooltip & Inspector callbacks
// ---------------------------------------------------------
window.showCitationTooltip = function(event, chunkId) {
  event.stopPropagation();
  const info = window.currentSources ? window.currentSources[chunkId] : null;
  if (!info) return;

  let popover = document.getElementById('citation-popover');
  if (!popover) {
    popover = document.createElement('div');
    popover.id = 'citation-popover';
    popover.className = 'citation-popover';
    document.body.appendChild(popover);
  }

  const urlPart = info.url ? `<a href="${escapeHtml(info.url)}" target="_blank" style="color: var(--accent-primary); text-decoration: none; font-weight: 600; display: inline-flex; align-items: center; gap: 3px;">Full Source Document <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg></a>` : '';
  const scoreBadge = info.score ? `<span style="background: rgba(139, 92, 246, 0.1); color: var(--accent-secondary); padding: 2px 6px; border-radius: 4px; font-weight: 600;">Match: ${(info.score * 100).toFixed(1)}%</span>` : '';

  popover.innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; border-bottom: 1px solid var(--border-color); padding-bottom: 6px;">
      <span style="font-weight: 600; color: var(--accent-secondary); font-size: 0.9rem;">Grounded RAG Context [${info.index}]</span>
      <button onclick="closeCitationTooltip(event)" style="background: none; border: none; color: var(--text-muted); font-size: 1.2rem; cursor: pointer; padding: 0 4px; line-height: 1;">×</button>
    </div>
    <div style="font-size: 0.75rem; color: var(--text-muted); display: flex; gap: 10px; margin-bottom: 10px; align-items: center;">
      <span>Publisher: ${escapeHtml(info.publisher)}</span>
      <span>Page: ${info.page}</span>
      ${scoreBadge}
    </div>
    <div style="background: var(--bg-input); border: 1px solid var(--border-color); padding: 10px; border-radius: 6px; font-size: 0.8rem; line-height: 1.4; color: var(--text-main); font-style: italic; max-height: 140px; overflow-y: auto; margin-bottom: 10px;">
      "${escapeHtml(info.text)}"
    </div>
    <div style="text-align: right; font-size: 0.8rem;">
      ${urlPart}
    </div>
  `;

  const rect = event.target.getBoundingClientRect();
  const popoverWidth = 320;
  
  let left = rect.left + window.scrollX - popoverWidth / 2 + rect.width / 2;
  let top = rect.bottom + window.scrollY + 8;
  
  if (left < 10) left = 10;
  if (left + popoverWidth > window.innerWidth - 10) {
    left = window.innerWidth - popoverWidth - 10;
  }
  
  popover.style.left = `${left}px`;
  popover.style.top = `${top}px`;
  popover.style.display = 'block';

  setTimeout(() => {
    document.addEventListener('click', closeCitationTooltip);
  }, 10);
};

window.closeCitationTooltip = function(event) {
  if (event) {
    const popover = document.getElementById('citation-popover');
    if (popover && popover.contains(event.target) && !event.target.onclick?.toString().includes('closeCitationTooltip')) {
      return;
    }
    event.stopPropagation();
  }
  const popover = document.getElementById('citation-popover');
  if (popover) {
    popover.style.display = 'none';
  }
  document.removeEventListener('click', closeCitationTooltip);
};

window.toggleRagInspector = function(btn) {
  const content = btn.nextElementSibling;
  const icon = btn.querySelector('.toggle-icon');
  if (content.style.display === 'none') {
    content.style.display = 'block';
    icon.textContent = '▲';
    btn.style.borderBottomLeftRadius = '0px';
    btn.style.borderBottomRightRadius = '0px';
  } else {
    content.style.display = 'none';
    icon.textContent = '▼';
    btn.style.borderBottomLeftRadius = '8px';
    btn.style.borderBottomRightRadius = '8px';
  }
};

// ---------------------------------------------------------
// Submit Handler (Conversational Turn or Prediction Trigger)
// ---------------------------------------------------------
async function handleChatSubmit(e) {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;

  // Append user text
  appendMessage('user', `<p>${escapeHtml(text)}</p>`);
  chatInput.value = '';
  showTyping();

  // Dynamically rename chat item based on user's first prompt
  renameChatTitleIfDefault(activeChatId, text);

  try {
    // Clean and compress chat history to prevent 413 Payload Too Large (max 4 KB)
    const cleanedHistory = messageHistory.slice(0, -1).map(msg => {
      let text = msg.content;
      try {
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = text;
        
        // Remove massive blocks like the RAG Pipeline Inspector
        const inspector = tempDiv.querySelector('.rag-inspector');
        if (inspector) inspector.remove();
        
        // Remove citation badges
        tempDiv.querySelectorAll('.citation-badge').forEach(el => el.remove());
        
        text = tempDiv.innerText || tempDiv.textContent || '';
      } catch (e) {
        // Fallback to regex cleaning if DOM parsing fails
        text = text.replace(/<[^>]+>/g, '');
      }
      return {
        role: msg.role,
        content: text.trim()
      };
    });

    // 1. Call Chat Endpoint with full context history to extract features and provide insights
    const chatRes = await fetch(CHAT_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        message: text, 
        current_features: currentFeatures,
        history: cleanedHistory
      })
    });
    
    if (!chatRes.ok) throw new Error('Failed to process message.');
    const chatData = await chatRes.json();
    currentFeatures = chatData.extracted_features;
    
    // Save current features to chat object
    saveCurrentChatState();

    // 2. Check if we need more features
    if (chatData.missing_features && chatData.missing_features.length > 0) {
      hideTyping();
      appendMessage('assistant', formatMarkdownToHtml(chatData.follow_up_message));
      return;
    }

    // 3. If features are complete and we haven't run the analysis yet, run it!
    if (!analysisCompleted) {
      appendMessage('assistant', `<p>Thank you! All features collected. Running the ML pipeline and generating grounded explanations...</p>`);
      showTyping();

      const analysisRes = await fetch(ANALYZE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(currentFeatures)
      });

      if (!analysisRes.ok) {
        let errStr = 'Failed to run analysis.';
        try {
          const errData = await analysisRes.json();
          if(errData.detail) errStr = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail);
        } catch(e){}
        throw new Error(errStr);
      }

      const analysisData = await analysisRes.json();
      analysisCompleted = true;
      saveCurrentChatState();

      hideTyping();
      
      const resultHtml = generateResultHTML(analysisData);
      appendMessage('assistant', resultHtml);
    } else {
      // If analysis is already completed, the LLM returned answers directly in the follow_up_message
      hideTyping();
      appendMessage('assistant', formatMarkdownToHtml(chatData.follow_up_message));
    }

  } catch (err) {
    hideTyping();
    appendMessage('assistant', `<p style="color: var(--error)">Error: ${escapeHtml(err.message)}</p>`);
  }
}

// ---------------------------------------------------------
// Startup & Navigation Event Listeners
// ---------------------------------------------------------
sidebarCloseBtn.addEventListener('click', () => {
  sidebar.classList.add('collapsed');
  sidebarOpenBtn.style.display = 'flex';
});

sidebarOpenBtn.addEventListener('click', () => {
  sidebar.classList.remove('collapsed');
  sidebarOpenBtn.style.display = 'none';
});

newChatBtn.addEventListener('click', createNewChat);
chatForm.addEventListener('submit', handleChatSubmit);

// Initializer on Page Load
function init() {
  // Wipe past history as requested by the user
  localStorage.removeItem('clinicagen_saved_chats');

  const chats = getStoredChats();
  const keys = Object.keys(chats);
  if (keys.length > 0) {
    // Load last active chat
    loadChatState(keys[keys.length - 1]);
  } else {
    // Create first chat
    createNewChat();
  }
  renderSidebarChats();
}

window.onload = init;
