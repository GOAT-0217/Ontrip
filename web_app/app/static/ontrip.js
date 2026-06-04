// ontrip.js — 核心交互逻辑

const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const pendingActionModal = document.getElementById('pending-action-modal');
const pendingActionDetails = document.getElementById('pending-action-details');
const loadingIndicator = document.getElementById('loading-indicator');
const operationLogContent = document.getElementById('operation-log-content');

// ── Session tracking ──────────────────────────────────────────────

let currentSessionId = (
    new URLSearchParams(window.location.search).get('session_id') ||
    document.cookie.replace(/(?:(?:^|.*;\s*)session_id\s*\=\s*([^;]*).*$)|^.*$/, '$1') ||
    ''
);
if (currentSessionId) {
    document.cookie = 'session_id=' + currentSessionId + ';path=/;max-age=86400;SameSite=Lax';
    if (window.location.search.includes('session_id')) {
        window.history.replaceState({}, '', '/');
    }
}

function getCurrentSessionId() {
    return currentSessionId;
}

// ── Unified fetch wrapper — injects X-Session-Id header ───────────

async function apiFetch(url, options) {
    if (!options) options = {};
    if (!options.headers) options.headers = {};
    options.headers['X-Session-Id'] = currentSessionId;
    return fetch(url, options);
}

// ── Message rendering ─────────────────────────────────────────────

function renderMarkdown(text) {
    // Step 1: Convert markdown-style links [text](url) to clickable <a> tags
    //         Guaranteed to work — no dependency on external library
    text = text.replace(
        /\[([^\]]*?)\]\((https?:\/\/[^\s\)]+)\)/g,
        '<a target="_blank" rel="noopener noreferrer" href="$2">$1</a>'
    );

    // Step 2: Let marked handle bold / italic / lists / blockquotes
    if (typeof marked !== 'undefined') {
        try {
            text = marked.parse(text, { breaks: true, gfm: true });
        } catch (e) { /* fall through */ }
    }

    // Step 3: Safety net — convert any [text](url) that survived Step 1+2
    //         (e.g. if marked escaped them back to plain text)
    text = text.replace(
        /\[([^\]]*?)\]\((https?:\/\/[^\s\)]+)\)/g,
        '<a target="_blank" rel="noopener noreferrer" href="$2">$1</a>'
    );

    // Step 4: Convert leftover \n to <br>
    text = text.replace(/\n/g, '<br>');

    return text;
}

function addMessage(sender, message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message ' + (sender === 'user' ? 'user-message' : 'ai-message');
    // User messages get <br>, AI messages get full markdown rendering
    const formatted = sender === 'user' ? message.replace(/\n/g, '<br>') : renderMarkdown(message);
    messageDiv.innerHTML = '<div class="message-content"><div class="message-sender">' +
        (sender === 'user' ? '你' : 'Ontrip') + '</div>' + formatted + '</div>';
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Convert server-rendered AI messages (Jinja2) to markdown on page load
(function convertInitialMessages() {
    document.querySelectorAll('.ai-message .message-content').forEach(function(content) {
        // Skip if already processed
        if (content.dataset.rendered) return;
        content.dataset.rendered = '1';
        // Extract sender div, render the rest as markdown
        var sender = content.querySelector('.message-sender');
        if (!sender) return;
        // Get raw text (browser gives us the literal content including \n as text, not HTML)
        var raw = content.textContent || '';
        // Remove the sender label from the raw text
        var label = sender.textContent || '';
        if (raw.startsWith(label)) raw = raw.slice(label.length);
        content.innerHTML = '';
        content.appendChild(sender);
        // Insert rendered markdown after the sender label
        var rendered = document.createElement('span');
        rendered.innerHTML = renderMarkdown(raw.trim());
        content.appendChild(rendered);
    });
})();

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;
    userInput.disabled = true;
    sendBtn.disabled = true;
    loadingIndicator.style.display = 'flex';
    try {
        addMessage('user', message);
        userInput.value = '';
        const resp = await apiFetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });
        const data = await resp.json();
        if (data.error) addMessage('assistant', '错误: ' + data.error);
        else addMessage('assistant', data.response);
    } catch (e) {
        addMessage('assistant', '请求失败，请稍后重试');
    } finally {
        userInput.disabled = false;
        sendBtn.disabled = false;
        loadingIndicator.style.display = 'none';
        userInput.focus();
    }
}

async function checkPendingAction() {
    try {
        const resp = await apiFetch('/pending-action');
        const data = await resp.json();
        if (data.pending_action) {
            let h = '<h4>待批准操作:</h4><ul style="margin-top:12px">';
            data.pending_action.tool_calls.forEach(function(tc) {
                h += '<li><strong>' + tc.name + '</strong>: ' + JSON.stringify(tc.args) + '</li>';
            });
            h += '</ul>';
            pendingActionDetails.innerHTML = h;
            pendingActionModal.style.display = 'flex';
        }
    } catch (e) { console.error('checkPendingAction', e); }
}

async function submitDecision(decision) {
    try {
        const resp = await apiFetch(
            decision === 'approve' ? '/approve-action' : '/reject-action',
            { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ decision: decision }) }
        );
        const data = await resp.json();
        if (data.error) addMessage('assistant', '错误: ' + data.error);
        else addMessage('assistant', data.response);
        pendingActionModal.style.display = 'none';
    } catch (e) {
        addMessage('assistant', '操作失败，请重试');
        pendingActionModal.style.display = 'none';
    }
}

function approveAction() { submitDecision('approve'); }
function rejectAction() { submitDecision('reject'); }

async function fetchOperationLog() {
    try {
        const resp = await apiFetch('/operation-log');
        const data = await resp.json();
        if (!data.error) displayOperationLog(data.operation_log);
    } catch (e) {}
}

function displayOperationLog(logEntries) {
    if (!logEntries || logEntries.length === 0) {
        operationLogContent.innerHTML = '<div class="log-entry">暂无日志</div>';
        return;
    }
    var html = '';
    logEntries.slice().reverse().forEach(function(entry) {
        var time = new Date(entry.timestamp).toLocaleTimeString();
        html += '<div class="log-entry ' + entry.type + '"><div class="log-title">' + entry.title +
            '</div><div class="log-content">' + entry.content +
            '</div><div class="log-timestamp">' + time + '</div></div>';
    });
    operationLogContent.innerHTML = html;
}

function clearOperationLog() {
    operationLogContent.innerHTML = '<div class="log-entry">暂无日志</div>';
}

function toggleOperationLog() {
    var panel = document.getElementById('operation-log-panel');
    if (panel) { panel.classList.toggle('collapsed'); }
}

function startNewChat() { newConversation(); }

// ── Conversation management ────────────────────────────────────────

// ── Storage keys ───────────────────────────────────────────────────
var STORAGE_KEY_CONV = 'ontrip_conversations';
var STORAGE_KEY_ACTIVE = 'ontrip_active_session';

// SVG icon for conversation items
var CONV_ICON_SVG = '<svg viewBox="0 0 24 24"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>';
var switchingConversation = false;

// ── localStorage helpers ──────────────────────────────────────────

/** Load conversations from localStorage, return array or null if empty */
function loadFromStorage() {
    try {
        var raw = localStorage.getItem(STORAGE_KEY_CONV);
        return raw ? JSON.parse(raw) : null;
    } catch (e) { return null; }
}

/** Save conversations array to localStorage */
function saveToStorage(conversations) {
    try {
        localStorage.setItem(STORAGE_KEY_CONV, JSON.stringify(conversations));
    } catch (e) { /* quota exceeded, ignore */ }
}

/** Save active session id to localStorage */
function saveActiveToStorage(sessionId) {
    try { localStorage.setItem(STORAGE_KEY_ACTIVE, sessionId); } catch (e) {}
}

/** Get active session id from localStorage */
function getActiveFromStorage() {
    try { return localStorage.getItem(STORAGE_KEY_ACTIVE); } catch (e) { return null; }
}

// ── Loading / UI helpers ──────────────────────────────────────────

function showSwitchLoading() {
    chatMessages.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:rgba(255,255,255,0.3);gap:10px;flex-direction:column;">' +
        '<div class="spinner"></div><span>加载对话中...</span></div>';
    operationLogContent.innerHTML = '<div class="log-entry">暂无日志</div>';
}

function clearChatArea() {
    chatMessages.innerHTML = '';
    operationLogContent.innerHTML = '<div class="log-entry">暂无日志</div>';
}

// ── Data loading (server first, localStorage fallback) ────────────

async function loadSessionMessages(sessionId) {
    try {
        var resp = await apiFetch('/session-data/' + encodeURIComponent(sessionId));
        if (resp.ok) {
            var data = await resp.json();
            clearChatArea();
            if (data.chat_history && data.chat_history.length > 0) {
                data.chat_history.forEach(function(msg) {
                    addMessage('user', msg.user_message);
                    addMessage('assistant', msg.ai_response);
                });
            } else {
                addMessage('assistant', '已开始新对话，有什么可以帮您的？');
            }
            if (data.operation_log) displayOperationLog(data.operation_log);
            return;
        }
    } catch (e) { console.error('loadSessionMessages (server)', e); }

    // Fallback: try localStorage
    var cached = loadFromStorage();
    if (cached) {
        var found = cached.find(function(c) { return c.session_id === sessionId; });
        if (found && found.messages && found.messages.length > 0) {
            clearChatArea();
            found.messages.forEach(function(msg) {
                addMessage('user', msg.user_message);
                addMessage('assistant', msg.ai_response);
            });
        } else {
            clearChatArea();
            addMessage('assistant', '已开始新对话，有什么可以帮您的？');
        }
    }
}

async function loadConversations() {
    try {
        var resp = await apiFetch('/conversations');
        var data = await resp.json();
        if (!data.error && data.conversations && data.conversations.length > 0) {
            saveToStorage(data.conversations);
            renderConversationList(data.conversations);
            return;
        }
    } catch (e) { console.error('loadConversations (server)', e); }

    // Fallback: localStorage
    var cached = loadFromStorage();
    if (cached && cached.length > 0) {
        renderConversationList(cached);
    } else {
        // Ultimate fallback: demo data for first-time users
        renderConversationList([]);
    }
}

// ── Rendering ─────────────────────────────────────────────────────

/**
 * Group conversations into 3 date buckets and render the list.
 * Buckets: 今天 / 7天内 / 更早
 */
function renderConversationList(conversations) {
    var list = document.getElementById('conversation-list');
    if (!conversations || conversations.length === 0) {
        list.innerHTML = '<div class="conv-empty">暂无历史对话<br>点击上方按钮开始新对话</div>';
        return;
    }

    var activeId = currentSessionId;
    var now = new Date();
    var todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    var weekStart = new Date(todayStart.getTime() - 6 * 86400000);

    // 3 buckets: 今天, 7天内, 更早
    var groups = [
        { label: '今天', items: [] },
        { label: '7天内', items: [] },
        { label: '更早', items: [] }
    ];

    conversations.forEach(function(c) {
        var d = new Date(c.updated_at + 'Z');
        if (d >= todayStart) groups[0].items.push(c);
        else if (d >= weekStart) groups[1].items.push(c);
        else groups[2].items.push(c);
    });

    var html = '';
    groups.forEach(function(g) {
        if (g.items.length === 0) return;
        html += '<div class="conv-date-group">' + g.label + '</div>';
        g.items.forEach(function(c) {
            var isActive = c.session_id === activeId;
            var escapedId = c.session_id.replace(/'/g, "\\'");
            // dblclick to rename, onclick to switch
            html += '<div class="conv-item' + (isActive ? ' active' : '') + '" ' +
                'onclick="switchConversation(\'' + escapedId + '\')" ' +
                'ondblclick="startRename(event, \'' + escapedId + '\')">' +
                '<div class="conv-icon">' + CONV_ICON_SVG + '</div>' +
                '<div class="conv-title" id="conv-title-' + escapedId + '" title="' + escapeHtml(c.title) + '">' + escapeHtml(c.title) + '</div>' +
                '<button class="conv-delete" onclick="event.stopPropagation(); deleteConversation(\'' + escapedId + '\')" title="删除对话">×</button>' +
                '</div>';
        });
    });
    list.innerHTML = html;
}

function escapeHtml(str) {
    var div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ── Switch / create / delete ──────────────────────────────────────

async function switchConversation(sessionId) {
    if (switchingConversation) return;
    if (sessionId === currentSessionId) return;

    switchingConversation = true;
    currentSessionId = sessionId;
    document.cookie = 'session_id=' + sessionId + ';path=/;max-age=86400;SameSite=Lax';
    saveActiveToStorage(sessionId);
    showSwitchLoading();
    try { await loadSessionMessages(sessionId); } catch (e) { console.error(e); }
    loadConversations();
    switchingConversation = false;
}

async function newConversation() {
    try {
        var resp = await apiFetch('/new-chat', { method: 'POST' });
        var data = await resp.json();
        if (data.session_id) {
            currentSessionId = data.session_id;
            document.cookie = 'session_id=' + data.session_id + ';path=/;max-age=86400;SameSite=Lax';
            saveActiveToStorage(data.session_id);
            clearChatArea();
            addMessage('assistant', '已开始新对话，有什么可以帮您的？');
            loadConversations();
        }
    } catch (e) {
        // Offline fallback: generate local-only session
        var fallbackId = 'local-' + Date.now();
        currentSessionId = fallbackId;
        document.cookie = 'session_id=' + fallbackId + ';path=/;max-age=86400;SameSite=Lax';
        saveActiveToStorage(fallbackId);
        clearChatArea();
        addMessage('assistant', '已开始新对话，有什么可以帮您的？');
        // Add to localStorage cache
        var cached = loadFromStorage() || [];
        var count = cached.length + 1;
        cached.unshift({ session_id: fallbackId, title: '新对话 ' + count, updated_at: new Date().toISOString() });
        saveToStorage(cached);
        renderConversationList(cached);
    }
}

async function deleteConversation(sessionId) {
    if (!confirm('确定要删除这个对话吗？此操作不可撤销。')) return;
    try { await apiFetch('/conversations/' + encodeURIComponent(sessionId), { method: 'DELETE' }); } catch (e) {}

    // Also remove from localStorage cache
    var cached = loadFromStorage();
    if (cached) {
        cached = cached.filter(function(c) { return c.session_id !== sessionId; });
        saveToStorage(cached);
    }

    if (sessionId === currentSessionId) {
        // Switch to next available conversation
        var next = (cached && cached.length > 0) ? cached[0] : null;
        if (next) {
            await switchConversation(next.session_id);
        } else {
            await newConversation();
        }
    } else {
        loadConversations();
    }
}

// ── Inline rename (double-click) ──────────────────────────────────

var _renamingSessionId = null;

/** Called on dblclick of a conversation item. Replaces title with an input. */
function startRename(event, sessionId) {
    event.stopPropagation();
    if (_renamingSessionId === sessionId) return;  // already editing

    var titleEl = document.getElementById('conv-title-' + sessionId);
    if (!titleEl) return;

    _renamingSessionId = sessionId;
    var currentTitle = titleEl.textContent || titleEl.innerText || '';
    titleEl.innerHTML = '<input class="conv-rename-input" id="rename-input-' + sessionId +
        '" value="' + escapeHtml(currentTitle) + '" maxlength="50">';

    var input = document.getElementById('rename-input-' + sessionId);
    input.focus();
    input.select();

    input.addEventListener('blur', function() { finishRename(sessionId); });
    input.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') { e.preventDefault(); finishRename(sessionId); }
        if (e.key === 'Escape') { cancelRename(sessionId, currentTitle); }
    });
}

/** Save the new title to server + cache, restore normal display. */
async function finishRename(sessionId) {
    var input = document.getElementById('rename-input-' + sessionId);
    var newTitle = (input ? input.value.trim() : '') || '未命名对话';
    newTitle = newTitle.substring(0, 50);
    _renamingSessionId = null;

    // Update server
    try {
        await apiFetch('/conversations/' + encodeURIComponent(sessionId) + '/rename', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: newTitle })
        });
    } catch (e) { /* offline — just update localStorage */ }

    // Update localStorage cache
    var cached = loadFromStorage();
    if (cached) {
        cached = cached.map(function(c) {
            if (c.session_id === sessionId) { c.title = newTitle; }
            return c;
        });
        saveToStorage(cached);
    }

    // Restore display
    var titleEl = document.getElementById('conv-title-' + sessionId);
    if (titleEl) { titleEl.textContent = newTitle; }
    loadConversations();  // refresh to pick up any server-side changes
}

/** Cancel rename, restore original title. */
function cancelRename(sessionId, originalTitle) {
    _renamingSessionId = null;
    var titleEl = document.getElementById('conv-title-' + sessionId);
    if (titleEl) { titleEl.textContent = originalTitle; }
}

// ── Init ──────────────────────────────────────────────────────────

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', function(e) { if (e.key === 'Enter') sendMessage(); });
setInterval(checkPendingAction, 5000);
setInterval(fetchOperationLog, 8000);
fetchOperationLog();
loadConversations();
setInterval(loadConversations, 15000);
userInput.focus();

function logout() {
    localStorage.removeItem(STORAGE_KEY_ACTIVE);
    window.location.href = '/logout';
}
