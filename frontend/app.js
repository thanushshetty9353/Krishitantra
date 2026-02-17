// ============================================================
// Krishitantra SE-SLM Dashboard – app.js (v3.0)
// WhatsApp Chat UI + Full Dashboard Logic
// ============================================================

const API_BASE = window.location.origin;
const POLL_INTERVAL = 10000;

// ============================================================
// Tab Navigation
// ============================================================

document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        const target = document.getElementById('tab-' + tab.dataset.tab);
        if (target) target.classList.add('active');
    });
});

// ============================================================
// Toast Notifications
// ============================================================

function showToast(msg, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = 'toast ' + type;
    toast.textContent = msg;
    container.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 3500);
}

// ============================================================
// Time Formatting Helpers
// ============================================================

function formatTime(date) {
    if (!date) date = new Date();
    const h = date.getHours();
    const m = date.getMinutes().toString().padStart(2, '0');
    const ampm = h >= 12 ? 'PM' : 'AM';
    const h12 = h % 12 || 12;
    return h12 + ':' + m + ' ' + ampm;
}

function setWelcomeTime() {
    const el = document.getElementById('wa-welcome-time');
    if (el) el.textContent = formatTime(new Date());
}

// ============================================================
// WhatsApp Chat – Message Handling
// ============================================================

// ============================================================
// WhatsApp Chat – Message Handling
// ============================================================

function addMessage(text, type, extra) {
    const body = document.getElementById('wa-chat-body');
    const msg = document.createElement('div');
    msg.className = 'wa-message ' + (type === 'user' ? 'wa-outgoing' : 'wa-incoming');

    let metaHTML = '<span class="wa-time">' + formatTime(new Date()) + '</span>';
    if (type === 'user') {
        // Double check simulation (gray initially, blue later if we wanted)
        metaHTML += ' <span class="wa-check">\u2713\u2713</span>';
    }

    let extraInfo = '';
    if (extra && extra.latency_ms) {
        // Optional: show latency in a subtle way or just in telemetry panel
        // For "same to same" look, we might hide this from the bubble or make it very subtle
        // console.log('Latency:', extra.latency_ms);
    }

    msg.innerHTML = '<div class="wa-bubble">' +
        '<div class="wa-text">' + escapeHtml(text) + '</div>' +
        '<div class="wa-meta">' + metaHTML + '</div>' +
        '</div>';

    // Insert before the footer? No, append to body.
    body.appendChild(msg);
    scrollToBottom();

    // Update sidebar preview if it exists
    if (type === 'user') {
        updateSidebarPreview('You: ' + text);
    } else {
        updateSidebarPreview(text);
    }
}

function updateSidebarPreview(text) {
    const lastMsg = document.getElementById('wa-last-msg');
    const lastTime = document.getElementById('wa-last-time');
    if (lastMsg) lastMsg.textContent = text;
    if (lastTime) lastTime.textContent = formatTime(new Date());
}

function scrollToBottom() {
    const body = document.getElementById('wa-chat-body');
    body.scrollTop = body.scrollHeight;
}

function addTypingIndicator() {
    const body = document.getElementById('wa-chat-body');
    const msg = document.createElement('div');
    msg.className = 'wa-message wa-incoming wa-typing';
    msg.id = 'wa-typing';
    msg.innerHTML = '<div class="wa-bubble">' +
        '<div class="wa-typing-dot"></div>' +
        '<div class="wa-typing-dot"></div>' +
        '<div class="wa-typing-dot"></div>' +
        '</div>';
    body.appendChild(msg);
    scrollToBottom();
}

function removeTypingIndicator() {
    const el = document.getElementById('wa-typing');
    if (el) el.remove();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/\n/g, '<br>');
}

// ============================================================
// Send Message (Inference)
// ============================================================

async function sendMessage() {
    const input = document.getElementById('wa-input');
    const text = input.value.trim();
    if (!text) return;

    input.value = '';
    toggleSendIcon(); // Reset to mic

    addMessage(text, 'user');

    // Update status
    const statusEl = document.getElementById('wa-status');
    const originalStatus = 'TinyLlama 1.1B \u2022 Online';
    if (statusEl) statusEl.textContent = 'typing...';

    // Show typing indicator
    addTypingIndicator();

    // Disable inputs? Maybe not for "same to same" feel, but good for logic.

    try {
        const start = performance.now();
        const res = await fetch(API_BASE + '/infer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        const data = await res.json();
        const elapsed = performance.now() - start;

        removeTypingIndicator();

        if (data.response) {
            addMessage(data.response, 'ai', {
                latency_ms: data.latency_ms || elapsed,
                drift_detected: data.drift_detected
            });

            // Update telemetry panel
            updateTelemetryPanel({
                latency_ms: data.latency_ms || elapsed,
                request_id: data.request_id || '-',
                drift_detected: data.drift_detected || false,
                drift_score: data.drift_score || 0
            });
        } else if (data.detail) {
            addMessage('Error: ' + data.detail, 'ai');
        }
    } catch (err) {
        removeTypingIndicator();
        addMessage('Connection error. Please check if the server is running.', 'ai');
    }

    if (statusEl) statusEl.textContent = originalStatus;
    input.focus();
}

function updateTelemetryPanel(info) {
    // Keep this function as is, or update if we moved it
    const panel = document.getElementById('wa-telemetry-info');
    if (!panel) return;

    panel.innerHTML = '' +
        '<div class="tele-row"><span class="tele-label">Request ID</span><span class="tele-value" style="font-size:.7rem">' + (info.request_id || '-') + '</span></div>' +
        '<div class="tele-row"><span class="tele-label">Latency</span><span class="tele-value">' + (info.latency_ms ? info.latency_ms.toFixed(1) + ' ms' : '-') + '</span></div>' +
        '<div class="tele-row"><span class="tele-label">Drift</span><span class="tele-value">' + (info.drift_detected ? '<span style="color:var(--warning)">Yes (' + (info.drift_score || 0).toFixed(3) + ')</span>' : '<span style="color:var(--success)">No</span>') + '</span></div>';
}

function clearChat() {
    const body = document.getElementById('wa-chat-body');
    // Keep the encryption message
    const encMsg = '<div class="wa-encryption-msg"><span class="lock-icon">\uD83D\uDD12</span> Messages are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them. Click to learn more.</div>';

    body.innerHTML = encMsg +
        '<div class="wa-date-divider"><span>Today</span></div>' +
        '<div class="wa-message wa-incoming"><div class="wa-bubble">' +
        '<div class="wa-text">Chat cleared. How can I help you? \uD83E\uDDE0</div>' +
        '<div class="wa-meta"><span class="wa-time">' + formatTime(new Date()) + '</span></div>' +
        '</div></div>';
}

function toggleSendIcon() {
    const input = document.getElementById('wa-input');
    const micIcon = document.getElementById('wa-mic-icon');
    const sendIcon = document.getElementById('wa-send-icon');
    const btn = document.getElementById('wa-mic-btn');

    if (input.value.trim().length > 0) {
        if (micIcon) micIcon.style.display = 'none';
        if (sendIcon) sendIcon.style.display = 'block';
        if (btn) btn.title = "Send";
    } else {
        if (micIcon) micIcon.style.display = 'block';
        if (sendIcon) sendIcon.style.display = 'none';
        if (btn) btn.title = "Voice Message";
    }
}

// Handle Enter key + Input changes
document.addEventListener('DOMContentLoaded', () => {
    setWelcomeTime();
    const input = document.getElementById('wa-input');

    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        input.addEventListener('input', toggleSendIcon);
    }
});


// ============================================================
// Overview Tab
// ============================================================

async function loadOverview() {
    try {
        const res = await fetch(API_BASE + '/health');
        const data = await res.json();

        setText('ov-model', data.model_version || 'base');
        setText('header-model', 'Model: ' + (data.model_version || 'base'));

        const uptime = data.uptime_seconds || 0;
        const uptimeStr = uptime > 3600 ? (uptime / 3600).toFixed(1) + 'h' :
            uptime > 60 ? (uptime / 60).toFixed(1) + 'm' :
                uptime.toFixed(0) + 's';
        setText('header-uptime', '\u23F1 ' + uptimeStr);
        setText('ov-uptime-text', uptimeStr);
        setText('ov-requests', data.total_requests || 0);
    } catch { }

    try {
        const res = await fetch(API_BASE + '/telemetry');
        const data = await res.json();
        if (data && data.recent_requests) {
            const reqs = data.recent_requests;
            setText('ov-requests', reqs.length);
            if (reqs.length > 0) {
                const latencies = reqs.map(r => r.latency_ms || 0);
                const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;
                setText('ov-latency', avg.toFixed(0));
                setText('ov-lat-range', Math.min(...latencies).toFixed(0) + 'ms – ' + Math.max(...latencies).toFixed(0) + 'ms');

                const tokens = reqs.reduce((sum, r) => sum + (r.input_tokens || 0) + (r.output_tokens || 0), 0);
                setText('ov-tokens', tokens);

                const drifts = reqs.filter(r => r.drift_detected).length;
                setText('ov-drifts', drifts);

                // Recent activity
                const actEl = document.getElementById('ov-activity');
                if (actEl && reqs.length > 0) {
                    const recent = reqs.slice(-5).reverse();
                    actEl.innerHTML = recent.map(r => {
                        const drift = r.drift_detected ? ' <span style="color:var(--warning)">🌊</span>' : '';
                        return '<div style="padding:.4rem 0;border-bottom:1px solid var(--border);font-size:.82rem">' +
                            '<span style="color:var(--text-muted);font-size:.7rem">' + (r.request_id || '').slice(0, 8) + '</span> ' +
                            '<span style="color:var(--text-secondary)">' + (r.prompt || '').slice(0, 50) + '</span>' + drift +
                            ' <span style="float:right;color:var(--accent);font-size:.75rem">' + (r.latency_ms || 0).toFixed(0) + 'ms</span>' +
                            '</div>';
                    }).join('');
                }
            }
        }
    } catch { }

    try {
        const res = await fetch(API_BASE + '/registry');
        const data = await res.json();
        if (data && data.models) {
            setText('ov-evolutions', data.models.length);
        }
    } catch { }
}

// ============================================================
// Telemetry Tab
// ============================================================

async function loadTelemetry() {
    try {
        const res = await fetch(API_BASE + '/telemetry');
        const data = await res.json();
        if (!data || !data.recent_requests) return;

        const reqs = data.recent_requests;
        setText('tel-total', reqs.length);

        if (reqs.length > 0) {
            const latencies = reqs.map(r => r.latency_ms || 0);
            const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;
            setText('tel-avg-lat', avg.toFixed(0));
            setText('tel-min-lat', Math.min(...latencies).toFixed(0));
            setText('tel-max-lat', Math.max(...latencies).toFixed(0));

            // Latency chart
            const chart = document.getElementById('tel-latency-chart');
            if (chart) {
                const recent = latencies.slice(-30);
                const maxL = Math.max(...recent, 1);
                chart.innerHTML = recent.map((l, i) => {
                    const pct = (l / maxL) * 100;
                    return '<div style="flex:1;background:var(--accent);min-width:4px;border-radius:2px 2px 0 0;height:' + pct + '%;transition:height .3s" title="' + l.toFixed(0) + 'ms"></div>';
                }).join('');
            }
        }

        // Structural telemetry
        const structEl = document.getElementById('tel-structural');
        if (structEl && data.latest_structural) {
            structEl.textContent = JSON.stringify(data.latest_structural, null, 2);
        }
    } catch (e) {
        showToast('Failed to load telemetry', 'error');
    }
}

// ============================================================
// Analysis Tab
// ============================================================

async function runAnalysis() {
    const btn = document.getElementById('analysis-btn');
    if (btn) { btn.disabled = true; btn.textContent = 'Analyzing...'; }

    try {
        const res = await fetch(API_BASE + '/analysis');
        const data = await res.json();

        const resultEl = document.getElementById('analysis-result');
        if (resultEl) {
            // Display the nested analysis object if available
            const toShow = data.analysis || data;
            resultEl.innerHTML = '<pre class="json-view">' + JSON.stringify(toShow, null, 2) + '</pre>';
        }

        const prunableEl = document.getElementById('analysis-prunable');
        // Backend returns analysis.prunable_attention_blocks
        const blocks = data.analysis ? data.analysis.prunable_attention_blocks : data.prunable_blocks;

        if (prunableEl && blocks) {
            prunableEl.innerHTML = blocks.map(b =>
                '<div style="padding:.35rem .6rem;margin:.25rem 0;background:var(--bg-input);border-radius:6px;font-size:.82rem;display:flex;justify-content:space-between">' +
                '<span>' + b + '</span>' +
                '</div>'
            ).join('');
        }

        const recsEl = document.getElementById('analysis-recs');
        // Backend returns analysis.rewiring_recommendations
        const recs = data.analysis ? data.analysis.rewiring_recommendations : data.recommendations;

        if (recsEl && recs) {
            recsEl.innerHTML = recs.map(r =>
                '<div style="padding:.4rem 0;border-bottom:1px solid var(--border);font-size:.82rem;color:var(--text-secondary)">\u2022 ' + (r.description || r) + '</div>'
            ).join('');
        }

        showToast('Analysis complete', 'success');
    } catch (e) {
        showToast('Analysis failed', 'error');
    }
    if (btn) { btn.disabled = false; btn.textContent = 'Run Analysis'; }
}

// ============================================================
// Evolution Tab
// ============================================================

async function triggerEvolution() {
    const btn = document.getElementById('evolve-btn');
    if (btn) { btn.disabled = true; btn.textContent = '\u23F3 Evolving...'; }
    const statusEl = document.getElementById('evolution-status');
    if (statusEl) statusEl.innerHTML = '<div style="color:var(--accent);padding:1rem;text-align:center">\uD83E\uDDEC Evolution cycle running...<br><small>This may take a moment</small></div>';

    try {
        const res = await fetch(API_BASE + '/evolve', { method: 'POST' });
        const data = await res.json();

        if (statusEl) {
            statusEl.innerHTML = '<pre class="json-view">' + JSON.stringify(data, null, 2) + '</pre>';
        }

        showToast('Evolution complete: ' + (data.status || 'done'), data.status === 'success' ? 'success' : 'info');
        loadOverview();
        loadEvolutionHistory();
    } catch (e) {
        if (statusEl) statusEl.innerHTML = '<div style="color:var(--danger);padding:1rem">\u274C Evolution failed</div>';
        showToast('Evolution failed', 'error');
    }
    if (btn) { btn.disabled = false; btn.textContent = '\uD83D\uDE80 Trigger Evolution'; }
}

async function loadEvolutionHistory() {
    try {
        const res = await fetch(API_BASE + '/evolution-history');
        const data = await res.json();
        const el = document.getElementById('evolution-history');
        if (!el) return;

        if (data.entries && data.entries.length > 0) {
            el.innerHTML = data.entries.slice().reverse().map(e =>
                '<div style="padding:.5rem .75rem;margin:.25rem 0;background:var(--bg-input);border-radius:8px;border-left:3px solid ' + (e.validation_status === 'PASS' ? 'var(--success)' : 'var(--danger)') + '">' +
                '<div style="display:flex;justify-content:space-between;font-size:.82rem">' +
                '<span style="font-weight:600">' + (e.version || 'unknown') + '</span>' +
                '<span class="badge ' + (e.validation_status === 'PASS' ? 'pass' : 'fail') + '">' + (e.validation_status || '-') + '</span>' +
                '</div>' +
                '<div style="font-size:.72rem;color:var(--text-muted);margin-top:.2rem">' + (e.timestamp || '') + '</div>' +
                '</div>'
            ).join('');
        }
    } catch { }
}

async function runProfiler() {
    try {
        const res = await fetch(API_BASE + '/profiler/report');
        const data = await res.json();
        const el = document.getElementById('profiler-result');
        if (el) el.textContent = JSON.stringify(data, null, 2);
        showToast('Profiler report generated', 'success');
    } catch (e) {
        showToast('Failed to generate report', 'error');
    }
}

// ============================================================
// Registry Tab
// ============================================================

async function loadRegistry() {
    try {
        const res = await fetch(API_BASE + '/registry');
        const data = await res.json();
        const bodyEl = document.getElementById('registry-body');
        const emptyEl = document.getElementById('registry-empty');
        const summaryEl = document.getElementById('registry-summary');

        if (!data || !data.models || data.models.length === 0) {
            if (bodyEl) bodyEl.innerHTML = '';
            if (emptyEl) emptyEl.style.display = 'block';
            if (summaryEl) summaryEl.innerHTML = '';
            return;
        }

        if (emptyEl) emptyEl.style.display = 'none';

        if (summaryEl) {
            summaryEl.innerHTML = '<div style="font-size:.82rem;color:var(--text-secondary)">' +
                'Total versions: <strong>' + data.models.length + '</strong> | ' +
                'Active: <strong>' + (data.active || 'base') + '</strong></div>';
        }

        if (bodyEl) {
            bodyEl.innerHTML = data.models.map(e =>
                '<tr>' +
                '<td style="font-weight:600">' + (e.version || '-') + '</td>' +
                '<td>' + (e.parent || '-') + '</td>' +
                '<td>' + (e.optimization || '-') + '</td>' +
                '<td>' + (e.compression_ratio ? (e.compression_ratio * 100).toFixed(1) + '%' : '-') + '</td>' +
                '<td>' + (e.accuracy_drop != null ? e.accuracy_drop.toFixed(1) + '%' : '-') + '</td>' +
                '<td><span class="badge ' + (e.validation === 'PASS' ? 'pass' : 'fail') + '">' + (e.validation || '-') + '</span></td>' +
                '<td style="font-size:.72rem;color:var(--text-muted)">' + (e.timestamp || '-') + '</td>' +
                '</tr>'
            ).join('');
        }
    } catch (e) {
        showToast('Failed to load registry', 'error');
    }
}

// ============================================================
// Drift Tab
// ============================================================

let driftHistory = [];

async function loadDrift() {
    try {
        const res = await fetch(API_BASE + '/telemetry');
        const data = await res.json();
        if (!data || !data.recent_requests) return;

        const reqs = data.recent_requests.filter(r => r.drift_score !== undefined && r.drift_score !== null);
        driftHistory = reqs;

        const events = reqs.filter(r => r.drift_detected);
        setText('drift-events-count', events.length);

        if (reqs.length > 0) {
            const latest = reqs[reqs.length - 1];
            setText('drift-current-score', (latest.drift_score || 0).toFixed(3));
            setText('drift-mem-size', reqs.length);

            const circle = document.getElementById('drift-circle');
            if (circle) {
                const s = latest.drift_score || 0;
                circle.className = 'drift-circle ' + (s > 0.5 ? 'high' : s > 0.25 ? 'medium' : 'low');
            }

            // Drift chart
            const chart = document.getElementById('drift-chart');
            if (chart) {
                const recent = reqs.slice(-25);
                chart.innerHTML = recent.map(r => {
                    const s = r.drift_score || 0;
                    const pct = Math.min(s * 200, 100);
                    const color = s > 0.5 ? 'var(--danger)' : s > 0.25 ? 'var(--warning)' : 'var(--success)';
                    return '<div style="flex:1;background:' + color + ';min-width:4px;border-radius:2px 2px 0 0;height:' + pct + '%;opacity:.8" title="' + s.toFixed(3) + '"></div>';
                }).join('');
            }

            // History table
            const tbody = document.getElementById('drift-history-body');
            if (tbody) {
                tbody.innerHTML = reqs.slice(-10).reverse().map(r =>
                    '<tr>' +
                    '<td>' + (r.drift_score || 0).toFixed(3) + '</td>' +
                    '<td>' + (r.drift_detected ? '<span style="color:var(--warning)">\u26A0</span>' : '<span style="color:var(--success)">\u2713</span>') + '</td>' +
                    '<td>' + ((r.drift_components || {}).embedding_shift || 0).toFixed(3) + '</td>' +
                    '<td>' + ((r.drift_components || {}).vocab_shift || 0).toFixed(3) + '</td>' +
                    '<td style="font-size:.72rem;color:var(--text-muted)">' + (r.timestamp || '-') + '</td>' +
                    '</tr>'
                ).join('');
            }
        }
    } catch { }
}

// ============================================================
// Governance Tab
// ============================================================

async function loadGovernance() {
    try {
        const res = await fetch(API_BASE + '/governance');
        const data = await res.json();

        const summaryEl = document.getElementById('gov-summary');
        if (summaryEl) {
            summaryEl.innerHTML = '<div style="font-size:.85rem">' +
                '<div style="margin-bottom:.5rem"><strong>Active Model:</strong> ' + (data.active_model || 'base') + '</div>' +
                '<div><strong>Backup Available:</strong> ' + (data.backup_available ? '<span style="color:var(--success)">Yes</span>' : '<span style="color:var(--text-muted)">No</span>') + '</div>' +
                '</div>';
        }

        const auditEl = document.getElementById('gov-audit');
        if (auditEl && data.audit_log && data.audit_log.length > 0) {
            auditEl.innerHTML = data.audit_log.slice().reverse().map(e =>
                '<div style="padding:.5rem .75rem;margin:.25rem 0;background:var(--bg-input);border-radius:8px;font-size:.82rem;border-left:3px solid var(--accent)">' +
                '<div>' + (e.action || e.event || '-') + '</div>' +
                '<div style="font-size:.72rem;color:var(--text-muted);margin-top:.15rem">' + (e.timestamp || '-') + '</div>' +
                '</div>'
            ).join('');
        }
    } catch (e) {
        showToast('Failed to load governance', 'error');
    }
}

async function performRollback() {
    if (!confirm('Roll back to the backup model? This will replace the current optimized model.')) return;

    try {
        const res = await fetch(API_BASE + '/rollback', { method: 'POST' });
        const data = await res.json();
        showToast('Rollback: ' + (data.status || 'done'), data.status === 'OK' ? 'success' : 'error');
        loadGovernance();
        loadOverview();
    } catch (e) {
        showToast('Rollback failed', 'error');
    }
}

// ============================================================
// Utilities
// ============================================================

function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

// ============================================================
// Init + Polling
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    loadOverview();
    loadTelemetry();
    loadEvolutionHistory();
    loadRegistry();
    loadDrift();
    loadGovernance();

    setInterval(() => {
        loadOverview();
        loadTelemetry();
        loadDrift();
    }, POLL_INTERVAL);
});
