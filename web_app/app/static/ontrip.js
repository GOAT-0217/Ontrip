// ontrip.js — 核心交互逻辑

const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const pendingActionModal = document.getElementById('pending-action-modal');
const pendingActionDetails = document.getElementById('pending-action-details');
const loadingIndicator = document.getElementById('loading-indicator');
const operationLogContent = document.getElementById('operation-log-content');

function addMessage(sender, message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender === 'user' ? 'user-message' : 'ai-message'}`;
    messageDiv.innerHTML = `<div class="message-content"><div class="message-sender">${sender === 'user' ? '你' : 'Ontrip'}</div>${message}</div>`;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;
    userInput.disabled = true;
    sendBtn.disabled = true;
    loadingIndicator.style.display = 'flex';
    try {
        addMessage('user', message);
        userInput.value = '';
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });
        const data = await response.json();
        if (data.error) addMessage('assistant', `错误: ${data.error}`);
        else addMessage('assistant', data.response);
    } catch (error) {
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
        const response = await fetch('/pending-action');
        const data = await response.json();
        if (data.pending_action) {
            let detailsHTML = '<h4>待批准操作:</h4><ul style="margin-top:12px">';
            data.pending_action.tool_calls.forEach(tc => {
                detailsHTML += `<li><strong>${tc.name}</strong>: ${JSON.stringify(tc.args)}</li>`;
            });
            detailsHTML += '</ul>';
            pendingActionDetails.innerHTML = detailsHTML;
            pendingActionModal.style.display = 'flex';
        }
    } catch (error) {
        console.error('检查待处理操作失败', error);
    }
}

async function approveAction() {
    await submitDecision('approve');
}

async function rejectAction() {
    await submitDecision('reject');
}

async function submitDecision(decision) {
    try {
        const response = await fetch(decision === 'approve' ? '/approve-action' : '/reject-action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ decision })
        });
        const data = await response.json();
        if (data.error) addMessage('assistant', `错误: ${data.error}`);
        else addMessage('assistant', data.response);
        pendingActionModal.style.display = 'none';
    } catch (error) {
        addMessage('assistant', '操作失败，请重试');
        pendingActionModal.style.display = 'none';
    }
}

async function fetchOperationLog() {
    try {
        const response = await fetch('/operation-log');
        const data = await response.json();
        if (!data.error) displayOperationLog(data.operation_log);
    } catch (error) {}
}

function displayOperationLog(logEntries) {
    if (!logEntries || logEntries.length === 0) {
        operationLogContent.innerHTML = '<div class="log-entry">暂无日志</div>';
        return;
    }
    let html = '';
    logEntries.slice().reverse().forEach(entry => {
        const time = new Date(entry.timestamp).toLocaleTimeString();
        html += `<div class="log-entry ${entry.type}"><div class="log-title">${entry.title}</div><div class="log-content">${entry.content}</div><div class="log-timestamp">${time}</div></div>`;
    });
    operationLogContent.innerHTML = html;
}

function clearOperationLog() { fetchOperationLog(); }

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });
setInterval(checkPendingAction, 5000);
setInterval(fetchOperationLog, 8000);
fetchOperationLog();
userInput.focus();