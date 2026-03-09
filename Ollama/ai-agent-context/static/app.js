/**
 * AI Agent Context Manager - Frontend Application
 * Implements 6 Types of Context for AI Agents
 */

const API_BASE = '';
let contextConfig = {
    instructions: true,
    tools: true,
    memory: true,
    knowledge: true,
    examples: true,
    session_id: 'default-' + Date.now()
};

// List storage for instruction fields
let listData = {
    'inst-steps': [],
    'inst-conventions': [],
    'inst-constraints': []
};

// ===== Initialization =====
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    loadAllData();
    autoResizeTextarea();
});

// ===== Health Check =====
async function checkHealth() {
    try {
        const resp = await fetch(`${API_BASE}/api/health`);
        const data = await resp.json();
        const dot = document.getElementById('statusDot');
        const badge = document.getElementById('modelName');
        if (data.model_available) {
            dot.classList.remove('offline');
            badge.textContent = data.model.replace(':latest', '');
        } else {
            dot.classList.add('offline');
            badge.textContent = 'Model not found';
        }
    } catch (e) {
        document.getElementById('statusDot').classList.add('offline');
        document.getElementById('modelName').textContent = 'Offline';
    }
}

// ===== Load All Data =====
async function loadAllData() {
    await Promise.all([
        loadInstructions(),
        loadTools(),
        loadMemory(),
        loadKnowledge(),
        loadExamples()
    ]);
}

// ===== Panel Switching =====
function switchPanel(panelName) {
    // Update buttons
    document.querySelectorAll('.ctx-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.panel === panelName);
    });
    // Update panels
    document.querySelectorAll('.panel').forEach(panel => {
        panel.classList.toggle('active', panel.id === `panel-${panelName}`);
    });
}

// ===== Instructions =====
async function loadInstructions() {
    try {
        const resp = await fetch(`${API_BASE}/api/instructions`);
        const data = await resp.json();

        document.getElementById('inst-role').value = data.role || '';
        document.getElementById('inst-role-desc').value = data.role_description || '';
        document.getElementById('inst-objective').value = data.objective || '';
        document.getElementById('inst-motivation').value = data.objective_motivation || '';
        document.getElementById('inst-format').value = data.requirements_response_format || 'plain text';

        listData['inst-steps'] = data.requirements_steps || [];
        listData['inst-conventions'] = data.requirements_conventions || [];
        listData['inst-constraints'] = data.requirements_constraints || [];

        renderList('inst-steps');
        renderList('inst-conventions');
        renderList('inst-constraints');
    } catch (e) {
        console.error('Failed to load instructions:', e);
    }
}

async function saveInstructions() {
    const data = {
        role: document.getElementById('inst-role').value,
        role_description: document.getElementById('inst-role-desc').value,
        objective: document.getElementById('inst-objective').value,
        objective_motivation: document.getElementById('inst-motivation').value,
        requirements_steps: listData['inst-steps'],
        requirements_conventions: listData['inst-conventions'],
        requirements_constraints: listData['inst-constraints'],
        requirements_response_format: document.getElementById('inst-format').value
    };

    try {
        await fetch(`${API_BASE}/api/instructions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        showToast('Instructions saved successfully', 'success');
    } catch (e) {
        showToast('Failed to save instructions', 'error');
    }
}

// ===== List Management =====
function addToList(listId, inputId) {
    const input = document.getElementById(inputId);
    const value = input.value.trim();
    if (!value) return;

    listData[listId].push(value);
    input.value = '';
    renderList(listId);
}

function removeFromList(listId, index) {
    listData[listId].splice(index, 1);
    renderList(listId);
}

function renderList(listId) {
    const container = document.getElementById(`${listId}-list`);
    if (!container) return;

    container.innerHTML = listData[listId].map((item, i) => `
        <span class="tag">
            ${escapeHtml(item)}
            <button class="tag-remove" onclick="removeFromList('${listId}', ${i})">×</button>
        </span>
    `).join('');
}

// ===== Tools =====
async function loadTools() {
    try {
        const resp = await fetch(`${API_BASE}/api/tools`);
        const data = await resp.json();
        renderTools(data.tools || []);
    } catch (e) {
        console.error('Failed to load tools:', e);
    }
}

function renderTools(tools) {
    const container = document.getElementById('tools-list');
    if (tools.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity="0.3"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>
                <p>No tools defined yet. Add tools to give your agent capabilities.</p>
            </div>`;
        return;
    }

    container.innerHTML = tools.map(tool => `
        <div class="item-card">
            <div class="item-content">
                <div class="item-title">${escapeHtml(tool.name)}</div>
                <div class="item-text">${escapeHtml(tool.description || '')}</div>
                ${tool.parameters && tool.parameters.length > 0 ? `
                    <div class="item-text" style="margin-top:6px; font-family: var(--font-mono); font-size: 0.72rem;">
                        Params: ${tool.parameters.map(p => p.name).join(', ')}
                    </div>` : ''}
            </div>
            <div class="item-actions">
                <button class="item-delete" onclick="deleteTool('${tool.id}')" title="Delete">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                </button>
            </div>
        </div>
    `).join('');
}

async function addTool() {
    const data = {
        name: document.getElementById('tool-name').value,
        description: document.getElementById('tool-desc').value,
        what_it_does: document.getElementById('tool-what').value,
        how_to_use: document.getElementById('tool-how').value,
        return_value: document.getElementById('tool-return').value,
        parameters: []
    };

    if (!data.name) {
        showToast('Tool name is required', 'error');
        return;
    }

    try {
        await fetch(`${API_BASE}/api/tools`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        showToast('Tool added', 'success');
        loadTools();
        toggleAddForm('add-tool-form');
        // Clear form
        ['tool-name', 'tool-desc', 'tool-what', 'tool-how', 'tool-return'].forEach(id => {
            document.getElementById(id).value = '';
        });
    } catch (e) {
        showToast('Failed to add tool', 'error');
    }
}

async function deleteTool(id) {
    try {
        await fetch(`${API_BASE}/api/tools/${id}`, { method: 'DELETE' });
        showToast('Tool deleted', 'success');
        loadTools();
    } catch (e) {
        showToast('Failed to delete tool', 'error');
    }
}

// ===== Memory =====
async function loadMemory() {
    try {
        const resp = await fetch(`${API_BASE}/api/memory`);
        const data = await resp.json();
        renderMemory(data.long_term || []);
    } catch (e) {
        console.error('Failed to load memory:', e);
    }
}

function renderMemory(entries) {
    const container = document.getElementById('memory-list');
    if (entries.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity="0.3"><path d="M12 2a4 4 0 0 1 4 4c0 1.95-1.4 3.58-3.25 3.93L12 22l-.75-12.07A4.001 4.001 0 0 1 12 2z"/></svg>
                <p>No long-term memory entries. Add facts, experiences, and instructions.</p>
            </div>`;
        return;
    }

    container.innerHTML = entries.map(entry => `
        <div class="item-card">
            <span class="item-type-badge ${entry.type}">${entry.type}</span>
            <div class="item-content">
                <div class="item-text">${escapeHtml(entry.content)}</div>
            </div>
            <div class="item-actions">
                <button class="item-delete" onclick="deleteMemory('${entry.id}')" title="Delete">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                </button>
            </div>
        </div>
    `).join('');
}

async function addMemory() {
    const data = {
        type: document.getElementById('memory-type').value,
        content: document.getElementById('memory-content').value
    };

    if (!data.content) {
        showToast('Memory content is required', 'error');
        return;
    }

    try {
        await fetch(`${API_BASE}/api/memory`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        showToast('Memory added', 'success');
        loadMemory();
        toggleAddForm('add-memory-form');
        document.getElementById('memory-content').value = '';
    } catch (e) {
        showToast('Failed to add memory', 'error');
    }
}

async function deleteMemory(id) {
    try {
        await fetch(`${API_BASE}/api/memory/${id}`, { method: 'DELETE' });
        showToast('Memory deleted', 'success');
        loadMemory();
    } catch (e) {
        showToast('Failed to delete memory', 'error');
    }
}

// ===== Knowledge =====
async function loadKnowledge() {
    try {
        const resp = await fetch(`${API_BASE}/api/knowledge`);
        const data = await resp.json();
        renderKnowledge(data.entries || []);
    } catch (e) {
        console.error('Failed to load knowledge:', e);
    }
}

function renderKnowledge(entries) {
    const container = document.getElementById('knowledge-list');
    if (entries.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity="0.3"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
                <p>No knowledge entries. Add domain expertise, workflows, and documents.</p>
            </div>`;
        return;
    }

    container.innerHTML = entries.map(entry => `
        <div class="item-card">
            <span class="item-type-badge ${entry.category}">${entry.category.replace('_', ' ')}</span>
            <div class="item-content">
                <div class="item-title">${escapeHtml(entry.title)}</div>
                <div class="item-text">${escapeHtml(entry.content)}</div>
            </div>
            <div class="item-actions">
                <button class="item-delete" onclick="deleteKnowledge('${entry.id}')" title="Delete">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                </button>
            </div>
        </div>
    `).join('');
}

async function addKnowledge() {
    const data = {
        category: document.getElementById('knowledge-category').value,
        title: document.getElementById('knowledge-title').value,
        content: document.getElementById('knowledge-content').value
    };

    if (!data.title || !data.content) {
        showToast('Title and content are required', 'error');
        return;
    }

    try {
        await fetch(`${API_BASE}/api/knowledge`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        showToast('Knowledge added', 'success');
        loadKnowledge();
        toggleAddForm('add-knowledge-form');
        document.getElementById('knowledge-title').value = '';
        document.getElementById('knowledge-content').value = '';
    } catch (e) {
        showToast('Failed to add knowledge', 'error');
    }
}

async function deleteKnowledge(id) {
    try {
        await fetch(`${API_BASE}/api/knowledge/${id}`, { method: 'DELETE' });
        showToast('Knowledge deleted', 'success');
        loadKnowledge();
    } catch (e) {
        showToast('Failed to delete knowledge', 'error');
    }
}

// ===== Examples =====
async function loadExamples() {
    try {
        const resp = await fetch(`${API_BASE}/api/examples`);
        const data = await resp.json();
        renderExamples(data.entries || []);
    } catch (e) {
        console.error('Failed to load examples:', e);
    }
}

function renderExamples(entries) {
    const container = document.getElementById('examples-list');
    if (entries.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity="0.3"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
                <p>No examples yet. Add behavior and response examples.</p>
            </div>`;
        return;
    }

    const typeLabels = {
        'behavior_positive': '✓ Behavior+',
        'behavior_negative': '✗ Behavior-',
        'response_positive': '✓ Response+',
        'response_negative': '✗ Response-'
    };

    container.innerHTML = entries.map(entry => `
        <div class="item-card">
            <span class="item-type-badge ${entry.type}">${typeLabels[entry.type] || entry.type}</span>
            <div class="item-content">
                <div class="item-text"><strong>Input:</strong> ${escapeHtml(entry.input_text)}</div>
                <div class="item-text"><strong>Output:</strong> ${escapeHtml(entry.output_text)}</div>
                ${entry.description ? `<div class="item-text" style="font-style:italic; margin-top:4px;">${escapeHtml(entry.description)}</div>` : ''}
            </div>
            <div class="item-actions">
                <button class="item-delete" onclick="deleteExample('${entry.id}')" title="Delete">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                </button>
            </div>
        </div>
    `).join('');
}

async function addExample() {
    const data = {
        type: document.getElementById('example-type').value,
        input_text: document.getElementById('example-input').value,
        output_text: document.getElementById('example-output').value,
        description: document.getElementById('example-desc').value
    };

    if (!data.input_text || !data.output_text) {
        showToast('Input and output are required', 'error');
        return;
    }

    try {
        await fetch(`${API_BASE}/api/examples`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        showToast('Example added', 'success');
        loadExamples();
        toggleAddForm('add-example-form');
        document.getElementById('example-input').value = '';
        document.getElementById('example-output').value = '';
        document.getElementById('example-desc').value = '';
    } catch (e) {
        showToast('Failed to add example', 'error');
    }
}

async function deleteExample(id) {
    try {
        await fetch(`${API_BASE}/api/examples/${id}`, { method: 'DELETE' });
        showToast('Example deleted', 'success');
        loadExamples();
    } catch (e) {
        showToast('Failed to delete example', 'error');
    }
}

// ===== Chat =====
let isGenerating = false;

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    if (!message || isGenerating) return;

    input.value = '';
    autoResizeTextarea();

    // Remove welcome message
    const welcome = document.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    // Add user message
    addChatMessage(message, 'user');

    // Add loading indicator
    const loadingId = addLoadingMessage();

    isGenerating = true;
    document.getElementById('sendBtn').disabled = true;

    try {
        const resp = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                context_config: contextConfig
            })
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Request failed');
        }

        const data = await resp.json();

        // Remove loading
        removeLoadingMessage(loadingId);

        // Add assistant message
        addChatMessage(data.response, 'assistant');

        // NEW: Load tool results if any tools were executed
        loadToolResults();

        // Update context info
        if (data.context_used) {
            document.getElementById('contextInfo').textContent =
                `Context active | History: ${data.context_used.total_messages} msgs`;
        }
    } catch (e) {
        removeLoadingMessage(loadingId);
        addChatMessage(`Error: ${e.message}`, 'assistant');
        showToast(e.message, 'error');
    } finally {
        isGenerating = false;
        document.getElementById('sendBtn').disabled = false;
    }
}

// ===== Tool Results Loading =====
async function loadToolResults() {
    try {
        const resp = await fetch(`${API_BASE}/api/tool-results`);
        const data = await resp.json();
        renderToolResults(data.results || []);
    } catch (e) {
        console.error('Failed to load tool results:', e);
    }
}

function renderToolResults(results) {
    const container = document.getElementById('tool-results-log');
    if (!container) return;

    if (results.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity="0.3"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
                <p>Tool results will appear here during chat interactions</p>
            </div>`;
        return;
    }

    container.innerHTML = results.slice().reverse().map(res => `
        <div class="item-card tool-result-card">
            <div class="item-content">
                <div class="item-title" style="color:var(--accent-primary)">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right:4px;"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
                    Executed: ${escapeHtml(res.tool)}
                </div>
                <div class="item-text" style="font-family:var(--font-mono); font-size:0.75rem; background:rgba(0,0,0,0.2); padding:8px; border-radius:6px; margin-top:8px;">
                    <div style="margin-bottom:4px; opacity:0.6;">// Parameters</div>
                    ${JSON.stringify(res.params, null, 2)}
                </div>
                <div class="item-text" style="font-family:var(--font-mono); font-size:0.75rem; background:rgba(124,92,252,0.05); padding:8px; border-radius:6px; margin-top:8px; border-left:2px solid var(--accent-primary);">
                    <div style="margin-bottom:4px; opacity:0.6;">// Result</div>
                    <pre style="white-space:pre-wrap;">${JSON.stringify(res.result, null, 2)}</pre>
                </div>
            </div>
            <div class="item-meta" style="font-size:0.65rem; opacity:0.4; margin-top:8px; text-align:right;">
                ${new Date(res.timestamp).toLocaleTimeString()}
            </div>
        </div>
    `).join('');
}

function addChatMessage(text, role) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = `msg msg-${role}`;

    const avatar = role === 'user' ? 'U' : '🤖';
    const avatarClass = role === 'user' ? '' : '';

    div.innerHTML = `
        <div class="msg-avatar">${avatar}</div>
        <div class="msg-bubble">${formatMessage(text)}</div>
    `;

    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function addLoadingMessage() {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    const id = 'loading-' + Date.now();
    div.id = id;
    div.className = 'msg msg-assistant msg-loading';
    div.innerHTML = `
        <div class="msg-avatar">🤖</div>
        <div class="msg-bubble">
            <div class="typing-dots">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeLoadingMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function formatMessage(text) {
    // Basic markdown-like formatting
    return escapeHtml(text)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code style="background:rgba(124,92,252,0.1);padding:1px 4px;border-radius:3px;font-family:var(--font-mono);font-size:0.78rem;">$1</code>')
        .replace(/\n/g, '<br>');
}

function handleChatKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

async function clearChat() {
    const container = document.getElementById('chatMessages');
    container.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">🤖</div>
            <h3>AI Agent Ready</h3>
            <p>Configure the 6 context types in the panels above, then start chatting.</p>
            <div class="welcome-tags">
                <span class="welcome-tag" style="--tag-color: #7c5cfc">Instructions</span>
                <span class="welcome-tag" style="--tag-color: #ff6b9d">Tools</span>
                <span class="welcome-tag" style="--tag-color: #ffa63e">Memory</span>
                <span class="welcome-tag" style="--tag-color: #3b82f6">Knowledge</span>
                <span class="welcome-tag" style="--tag-color: #06d6a0">Examples</span>
                <span class="welcome-tag" style="--tag-color: #00d4aa">Tool Results</span>
            </div>
        </div>`;

    try {
        await fetch(`${API_BASE}/api/chat/history/${contextConfig.session_id}`, { method: 'DELETE' });
    } catch (e) { /* ignore */ }

    // Create new session
    contextConfig.session_id = 'default-' + Date.now();
    document.getElementById('contextInfo').textContent = 'All context types active';
    showToast('Chat cleared', 'success');
}

// ===== Context Config =====
function updateContextConfig() {
    const checkboxes = document.querySelectorAll('#contextToggles input[type="checkbox"]');
    let activeCount = 0;

    checkboxes.forEach(cb => {
        contextConfig[cb.dataset.ctx] = cb.checked;
        if (cb.checked) activeCount++;
    });

    const info = document.getElementById('contextInfo');
    info.textContent = `${activeCount}/5 context types active`;
}

// ===== Context Preview =====
async function previewContext() {
    try {
        const resp = await fetch(`${API_BASE}/api/context/preview`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(contextConfig)
        });
        const data = await resp.json();

        document.getElementById('modalCharCount').textContent = data.char_count.toLocaleString();
        document.getElementById('modalTokenCount').textContent = '~' + data.token_estimate.toLocaleString();
        document.getElementById('contextPreviewText').textContent = data.system_prompt || '(empty context)';

        document.getElementById('contextModal').classList.add('active');
    } catch (e) {
        showToast('Failed to preview context', 'error');
    }
}

function closeModal() {
    document.getElementById('contextModal').classList.remove('active');
}

// Close modal on overlay click
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal-overlay')) {
        closeModal();
    }
});

// Close modal on Escape
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
});

// ===== Toggle Add Form =====
function toggleAddForm(formId) {
    const form = document.getElementById(formId);
    form.classList.toggle('collapsed');
}

// ===== Auto-resize textarea =====
function autoResizeTextarea() {
    const textarea = document.getElementById('chatInput');
    textarea.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 100) + 'px';
    });
}

// ===== Toast =====
function showToast(message, type = 'success') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icon = type === 'success'
        ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#06d6a0" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
        : '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>';

    toast.innerHTML = `${icon}<span>${escapeHtml(message)}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'toastOut 0.3s ease forwards';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ===== Utility =====
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ===== Theme Toggle (placeholder) =====
function toggleDarkMode() {
    // Currently dark mode only
    showToast('Dark mode is the default theme', 'success');
}
