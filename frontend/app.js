// ============================================================
// Krishitantra SE-SLM Dashboard – app.js
// ============================================================

const API_BASE = window.location.origin;
const POLL_INTERVAL = 10000; // 10 seconds

// ============================================================
// Tab Navigation
// ============================================================

document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

        tab.classList.add('active');
        const target = tab.dataset.tab;
        document.getElementById(`tab-${target}`).classList.add('active');

        // Load data for the tab
        switch (target) {
            case 'overview': loadOverview(); break;
            case 'telemetry': loadTelemetry(); break;
            case 'registry': loadRegistry(); break;
            case 'drift': loadDrift(); break;
            case 'governance': loadGovernance(); break;
        }
    });
});

// Enter key triggers inference
document.getElementById('infer-input')?.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') runInference();
});

// ============================================================
// Toast Notifications
// ============================================================

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// ============================================================
// API Helpers
// ============================================================

async function apiGet(path) {
    try {
        const res = await fetch(`${API_BASE}${path}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (e) {
        console.error(`API GET ${path} failed:`, e);
        return null;
    }
}

async function apiPost(path, data = {}) {
    try {
        const res = await fetch(`${API_BASE}${path}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (e) {
        console.error(`API POST ${path} failed:`, e);
        return null;
    }
}

function formatNum(n) {
    if (n === null || n === undefined) return '0';
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return String(n);
}

function truncateId(id) {
    if (!id) return '-';
    return id.substring(0, 8) + '...';
}

function formatTime(iso) {
    if (!iso) return '-';
    try {
        const d = new Date(iso);
        return d.toLocaleTimeString() + ' ' + d.toLocaleDateString();
    } catch {
        return iso;
    }
}

function formatUptime(seconds) {
    if (!seconds) return '0s';
    if (seconds < 60) return Math.round(seconds) + 's';
    if (seconds < 3600) return Math.round(seconds / 60) + 'm';
    return (seconds / 3600).toFixed(1) + 'h';
}

// ============================================================
// Overview
// ============================================================

async function loadOverview() {
    const health = await apiGet('/health');
    const telemetry = await apiGet('/telemetry?limit=10');
    const drift = await apiGet('/drift?limit=10');
    const registry = await apiGet('/registry');

    if (health) {
        document.getElementById('header-model').textContent = `Model: ${health.model_version}`;
        document.getElementById('header-uptime').textContent = `⏱ ${formatUptime(health.uptime_seconds)}`;
        document.getElementById('ov-model').textContent = health.model_version;
        document.getElementById('ov-requests').textContent = formatNum(health.total_requests);
        document.getElementById('ov-uptime-text').textContent = formatUptime(health.uptime_seconds);
    }

    if (telemetry && telemetry.summary) {
        const s = telemetry.summary;
        document.getElementById('ov-latency').innerHTML =
            `${s.avg_latency_ms || 0}<span style="font-size:0.9rem">ms</span>`;
        document.getElementById('ov-tokens').textContent =
            formatNum((s.total_input_tokens || 0) + (s.total_output_tokens || 0));
        document.getElementById('ov-lat-range').textContent =
            `${s.min_latency_ms || 0}ms – ${s.max_latency_ms || 0}ms`;
    }

    if (drift && drift.history) {
        document.getElementById('ov-drifts').textContent = drift.history.length;
    }

    if (registry && registry.models) {
        document.getElementById('ov-evolutions').textContent = registry.models.length;
    }

    // Recent activity
    if (telemetry && telemetry.recent_requests && telemetry.recent_requests.length > 0) {
        const activityEl = document.getElementById('ov-activity');
        const timeline = document.createElement('div');
        timeline.className = 'timeline';

        telemetry.recent_requests.slice(0, 5).forEach(req => {
            timeline.innerHTML += `
                <div class="timeline-item">
                    <div class="timeline-dot"></div>
                    <div class="timeline-content">
                        <div class="timeline-title">Inference Request</div>
                        <div class="timeline-desc">${req.input_tokens} in → ${req.output_tokens} out | ${Math.round(req.latency_ms)}ms</div>
                        <div class="timeline-time">${formatTime(req.timestamp)}</div>
                    </div>
                </div>`;
        });

        activityEl.innerHTML = '';
        activityEl.appendChild(timeline);
    }
}

// ============================================================
// Inference
// ============================================================

async function runInference() {
    const input = document.getElementById('infer-input');
    const btn = document.getElementById('infer-btn');
    const text = input.value.trim();

    if (!text) {
        showToast('Please enter a prompt', 'error');
        return;
    }

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Generating...';
    document.getElementById('infer-response').textContent = 'Thinking...';

    const data = await apiPost('/infer', { text });

    btn.disabled = false;
    btn.innerHTML = 'Generate';

    if (data) {
        document.getElementById('infer-response').textContent = data.response;
        document.getElementById('infer-meta').style.display = 'flex';
        document.getElementById('infer-latency').textContent = `${data.latency_ms}ms`;
        document.getElementById('infer-id').textContent = truncateId(data.request_id);
        document.getElementById('infer-drift').textContent =
            `${data.drift_score || 0} ${data.drift_detected ? '⚠️' : '✅'}`;

        showToast('Inference completed', 'success');
        loadTelemetry();
    } else {
        document.getElementById('infer-response').textContent = 'Error: Failed to get response.';
        showToast('Inference failed', 'error');
    }
}

// ============================================================
// Telemetry
// ============================================================

async function loadTelemetry() {
    const data = await apiGet('/telemetry?limit=30');
    if (!data) return;

    const s = data.summary || {};
    document.getElementById('tel-total').textContent = formatNum(s.total_requests);
    document.getElementById('tel-avg-lat').innerHTML =
        `${s.avg_latency_ms || 0}<span style="font-size:0.9rem">ms</span>`;
    document.getElementById('tel-min-lat').innerHTML =
        `${s.min_latency_ms || 0}<span style="font-size:0.9rem">ms</span>`;
    document.getElementById('tel-max-lat').innerHTML =
        `${s.max_latency_ms || 0}<span style="font-size:0.9rem">ms</span>`;

    // Latency chart
    const chart = document.getElementById('tel-latency-chart');
    chart.innerHTML = '';

    if (data.recent_requests && data.recent_requests.length > 0) {
        const requests = [...data.recent_requests].reverse();
        const maxLat = Math.max(...requests.map(r => r.latency_ms), 1);

        requests.forEach(req => {
            const height = Math.max(4, (req.latency_ms / maxLat) * 130);
            const color = req.latency_ms > 2000 ? 'var(--danger)' :
                req.latency_ms > 1000 ? 'var(--warning)' : 'var(--accent-primary)';
            const bar = document.createElement('div');
            bar.style.cssText = `
                flex: 1; max-width: 30px; height: ${height}px;
                background: ${color}; border-radius: 3px 3px 0 0;
                transition: height 0.3s ease; cursor: pointer;
                opacity: 0.8; min-width: 6px;
            `;
            bar.title = `${Math.round(req.latency_ms)}ms`;
            bar.addEventListener('mouseenter', () => bar.style.opacity = '1');
            bar.addEventListener('mouseleave', () => bar.style.opacity = '0.8');
            chart.appendChild(bar);
        });
    }

    // History table
    const tbody = document.getElementById('infer-history-body');
    tbody.innerHTML = '';

    if (data.recent_requests) {
        data.recent_requests.slice(0, 15).forEach(req => {
            tbody.innerHTML += `<tr>
                <td><code>${truncateId(req.request_id)}</code></td>
                <td>${req.input_tokens}</td>
                <td>${req.output_tokens}</td>
                <td>${Math.round(req.latency_ms)}</td>
                <td>${formatTime(req.timestamp)}</td>
            </tr>`;
        });
    }

    // Structural telemetry
    const structural = await apiGet('/telemetry/structural?limit=5');
    if (structural && structural.structural_telemetry) {
        document.getElementById('tel-structural').textContent =
            JSON.stringify(structural.structural_telemetry.slice(0, 3), null, 2);
    }
}

// ============================================================
// Structural Analysis
// ============================================================

async function runAnalysis() {
    const btn = document.getElementById('analysis-btn');
    btn.disabled = true;
    btn.textContent = 'Analyzing...';

    const data = await apiGet('/analysis');

    btn.disabled = false;
    btn.textContent = 'Run Analysis';

    if (!data || data.status === 'NO_DATA') {
        document.getElementById('analysis-result').innerHTML = `
            <div class="empty-state">
                <div class="icon">📭</div>
                <p>${data?.message || 'No telemetry data available. Run some inferences first.'}</p>
            </div>`;
        return;
    }

    const a = data.analysis;

    // Main result
    document.getElementById('analysis-result').innerHTML = `
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Prunable Blocks</div>
                <div class="stat-value">${(a.prunable_attention_blocks || []).length}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Risk Score</div>
                <div class="stat-value">${a.pruning_risk_score || 0}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Redundant FFN</div>
                <div class="stat-value">${(a.redundant_ffn_layers || []).length}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Recommendations</div>
                <div class="stat-value">${(a.rewiring_recommendations || []).length}</div>
            </div>
        </div>`;

    // Prunable blocks
    const prunableEl = document.getElementById('analysis-prunable');
    if (a.prunable_attention_blocks && a.prunable_attention_blocks.length > 0) {
        prunableEl.innerHTML = '<ul class="rec-list">' +
            a.prunable_attention_blocks.map(b => `
                <li class="rec-item">
                    <div class="rec-type">Prunable</div>
                    <div class="rec-desc">${b}</div>
                </li>`).join('') + '</ul>';
    } else {
        prunableEl.innerHTML = '<div class="empty-state"><p>No prunable blocks found</p></div>';
    }

    // Recommendations
    const recsEl = document.getElementById('analysis-recs');
    if (a.rewiring_recommendations && a.rewiring_recommendations.length > 0) {
        recsEl.innerHTML = '<ul class="rec-list">' +
            a.rewiring_recommendations.map(r => `
                <li class="rec-item">
                    <div class="rec-type">${r.type}</div>
                    <div class="rec-desc">${r.description}</div>
                    <div style="font-size:0.75rem;color:var(--text-muted);margin-top:4px">
                        ${r.estimated_speedup ? '⚡ ' + r.estimated_speedup : ''}
                        ${r.estimated_memory_saving ? ' 💾 ' + r.estimated_memory_saving : ''}
                    </div>
                </li>`).join('') + '</ul>';
    }

    showToast('Analysis complete', 'success');
}

// ============================================================
// Evolution
// ============================================================

async function triggerEvolution() {
    const btn = document.getElementById('evolve-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Evolving...';

    const statusEl = document.getElementById('evolution-status');
    statusEl.innerHTML = '<div style="padding:1rem;text-align:center;color:var(--text-muted)">' +
        '<span class="spinner"></span> Running evolution cycle... This may take a minute.</div>';

    const data = await apiPost('/evolve', { triggered_by: 'dashboard' });

    btn.disabled = false;
    btn.innerHTML = '🚀 Trigger Evolution';

    if (data && data.result) {
        const r = data.result;
        const status = r.evolution_status || r.status || 'UNKNOWN';
        const badge = status === 'APPROVED' ? 'badge-success' :
            status === 'REJECTED' ? 'badge-danger' :
                status === 'ERROR' ? 'badge-danger' : 'badge-warning';

        // Build detail sections
        let details = '';
        if (r.architecture_diff) {
            const diff = r.architecture_diff;
            details += `
                <div style="margin-top:0.75rem;padding:0.5rem;background:var(--bg-secondary);border-radius:6px">
                    <div style="font-size:0.8rem;font-weight:600;margin-bottom:0.25rem">Architecture Changes</div>
                    <div style="font-size:0.75rem;color:var(--text-muted)">
                        Version: <strong>${diff.version || '-'}</strong> |
                        Params: ${(diff.base_parameters || 0).toLocaleString()} → ${(diff.optimized_parameters || 0).toLocaleString()} |
                        Reduction: ${diff.reduction_percent || 0}%
                    </div>
                </div>`;
        }
        if (r.validation) {
            const v = r.validation;
            details += `
                <div style="margin-top:0.5rem;padding:0.5rem;background:var(--bg-secondary);border-radius:6px">
                    <div style="font-size:0.8rem;font-weight:600;margin-bottom:0.25rem">Validation</div>
                    <div style="font-size:0.75rem;color:var(--text-muted)">
                        Similarity: ${v.similarity_score || '-'} |
                        Accuracy Drop: ${v.accuracy_drop_percent || 0}% |
                        Hallucination: ${v.hallucination_rate || 0} |
                        Status: <strong>${v.status || '-'}</strong>
                    </div>
                </div>`;
        }
        if (r.error) {
            details += `
                <div style="margin-top:0.5rem;padding:0.5rem;background:rgba(255,80,80,0.1);border-radius:6px">
                    <div style="font-size:0.8rem;font-weight:600;color:var(--danger);margin-bottom:0.25rem">Error</div>
                    <div style="font-size:0.75rem;color:var(--text-muted)">${r.error}</div>
                </div>`;
        }

        statusEl.innerHTML = `
            <div style="padding:1rem;border:1px solid var(--border-subtle);border-radius:var(--radius-md)">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem">
                    <strong>Evolution Result</strong>
                    <span class="badge ${badge}">${status}</span>
                </div>
                ${details}
                <details style="margin-top:0.75rem">
                    <summary style="font-size:0.75rem;color:var(--text-muted);cursor:pointer">Raw JSON</summary>
                    <div class="json-view" style="margin-top:0.5rem">${JSON.stringify(r, null, 2)}</div>
                </details>
            </div>`;

        const toastType = status === 'APPROVED' ? 'success' :
            status === 'ERROR' ? 'error' : 'info';
        showToast(`Evolution: ${status}`, toastType);

        // Refresh dashboard data
        loadOverview();
        loadRegistry();
    } else {
        statusEl.innerHTML = `
            <div style="padding:1rem;color:var(--danger)">
                <strong>Evolution failed</strong><br>
                <span style="font-size:0.85rem">Could not reach the server. Check that the backend is running.</span>
            </div>`;
        showToast('Evolution failed — server unreachable', 'error');
    }
}

async function runProfiler() {
    showToast('Running profiler...', 'info');
    const data = await apiPost('/profiler/run');

    if (data) {
        document.getElementById('profiler-result').textContent =
            JSON.stringify(data, null, 2);
        showToast('Profiling complete', 'success');
    } else {
        showToast('Profiling failed', 'error');
    }
}

// ============================================================
// Model Registry
// ============================================================

async function loadRegistry() {
    const data = await apiGet('/registry');
    if (!data) return;

    const tbody = document.getElementById('registry-body');
    const emptyEl = document.getElementById('registry-empty');
    const summaryEl = document.getElementById('registry-summary');

    if (data.summary) {
        const s = data.summary;
        summaryEl.innerHTML = `
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Total Versions</div>
                    <div class="stat-value">${s.total_versions || 0}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Latest Version</div>
                    <div class="stat-value" style="font-size:1.2rem">${s.latest_version || 'base'}</div>
                </div>
            </div>`;
    }

    tbody.innerHTML = '';

    if (data.models && data.models.length > 0) {
        emptyEl.style.display = 'none';

        data.models.forEach(m => {
            const validBadge = m.validation_status === 'PASS' ? 'badge-success' : 'badge-danger';
            const optText = Array.isArray(m.optimization) ? m.optimization.join(', ') :
                (m.optimization || 'unknown');

            tbody.innerHTML += `<tr>
                <td><strong>${m.version}</strong></td>
                <td>${m.parent_version || '-'}</td>
                <td>${optText}</td>
                <td>${m.compression_percent || 0}%</td>
                <td>${m.accuracy_drop_percent || 0}%</td>
                <td><span class="badge ${validBadge}">${m.validation_status || '-'}</span></td>
                <td>${formatTime(m.timestamp)}</td>
            </tr>`;
        });
    } else {
        emptyEl.style.display = 'block';
    }
}

// ============================================================
// Drift Detection
// ============================================================

async function loadDrift() {
    const data = await apiGet('/drift?limit=30');
    if (!data) return;

    // Detector status
    if (data.detector_status) {
        document.getElementById('drift-mem-size').textContent = data.detector_status.memory_size || 0;
    }

    // History chart & table
    const chart = document.getElementById('drift-chart');
    const tbody = document.getElementById('drift-history-body');
    chart.innerHTML = '';
    tbody.innerHTML = '';

    if (data.history && data.history.length > 0) {
        document.getElementById('drift-events-count').textContent = data.history.length;

        // Latest score
        const latest = data.history[0];
        const score = latest.drift_score || 0;
        document.getElementById('drift-current-score').textContent = score.toFixed(4);

        const circle = document.getElementById('drift-circle');
        circle.className = 'drift-circle ' +
            (score > 0.35 ? 'high' : score > 0.15 ? 'medium' : 'low');

        // Chart
        const entries = [...data.history].reverse().slice(-30);
        const maxScore = Math.max(...entries.map(e => e.drift_score), 0.1);

        entries.forEach(entry => {
            const height = Math.max(4, (entry.drift_score / maxScore) * 100);
            const color = entry.drift_flag ? 'var(--danger)' : 'var(--success)';
            const bar = document.createElement('div');
            bar.style.cssText = `
                flex: 1; max-width: 20px; height: ${height}px;
                background: ${color}; border-radius: 2px 2px 0 0;
                min-width: 4px; opacity: 0.8; cursor: pointer;
            `;
            bar.title = `Score: ${entry.drift_score}`;
            chart.appendChild(bar);
        });

        // Table
        data.history.slice(0, 10).forEach(d => {
            tbody.innerHTML += `<tr>
                <td>${(d.drift_score || 0).toFixed(4)}</td>
                <td><span class="badge ${d.drift_flag ? 'badge-danger' : 'badge-success'}">${d.drift_flag ? 'DRIFT' : 'OK'}</span></td>
                <td>${(d.embedding_shift || 0).toFixed(4)}</td>
                <td>${(d.vocab_shift || 0).toFixed(4)}</td>
                <td>${formatTime(d.timestamp)}</td>
            </tr>`;
        });
    }
}

// ============================================================
// Governance
// ============================================================

async function loadGovernance() {
    const data = await apiGet('/governance/audit?limit=20');
    if (!data) return;

    // Summary
    if (data.summary) {
        const s = data.summary;
        document.getElementById('gov-summary').innerHTML = `
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;font-size:0.85rem">
                <div>
                    <div style="color:var(--text-muted);font-size:0.7rem">CURRENT MODEL</div>
                    <div style="font-weight:600">${s.current_model || 'base'}</div>
                </div>
                <div>
                    <div style="color:var(--text-muted);font-size:0.7rem">TOTAL EVOLUTIONS</div>
                    <div style="font-weight:600">${s.total_evolutions || 0}</div>
                </div>
                <div>
                    <div style="color:var(--text-muted);font-size:0.7rem">AUDIT EVENTS</div>
                    <div style="font-weight:600">${s.recent_audit_events || 0}</div>
                </div>
                <div>
                    <div style="color:var(--text-muted);font-size:0.7rem">LAST ACTION</div>
                    <div style="font-weight:600">${s.last_audit_action || 'none'}</div>
                </div>
            </div>`;
    }

    // Audit log
    const auditEl = document.getElementById('gov-audit');
    if (data.audit_log && data.audit_log.length > 0) {
        const timeline = document.createElement('div');
        timeline.className = 'timeline';

        data.audit_log.forEach(evt => {
            const badge = evt.status === 'APPROVED' ? 'badge-success' :
                evt.status === 'REJECTED' ? 'badge-danger' :
                    evt.status === 'OK' ? 'badge-success' : 'badge-neutral';

            timeline.innerHTML += `
                <div class="timeline-item">
                    <div class="timeline-dot"></div>
                    <div class="timeline-content">
                        <div class="timeline-title">
                            ${evt.action} <span class="badge ${badge}">${evt.status}</span>
                        </div>
                        <div class="timeline-desc">Version: ${evt.version} | By: ${evt.triggered_by}</div>
                        <div class="timeline-time">${formatTime(evt.timestamp)}</div>
                    </div>
                </div>`;
        });

        auditEl.innerHTML = '';
        auditEl.appendChild(timeline);
    }
}

async function performRollback() {
    if (!confirm('Are you sure you want to rollback to the previous model version?')) return;

    showToast('Rolling back...', 'info');
    const data = await apiPost('/governance/rollback', { reason: 'Manual rollback from dashboard' });

    if (data) {
        showToast(`Rollback: ${data.status}`, data.status === 'OK' ? 'success' : 'error');
        loadOverview();
        loadGovernance();
    } else {
        showToast('Rollback failed', 'error');
    }
}

// ============================================================
// Auto-Refresh & Init
// ============================================================

function init() {
    loadOverview();

    // Auto-refresh overview every 10 seconds
    setInterval(() => {
        const activeTab = document.querySelector('.nav-tab.active');
        if (activeTab && activeTab.dataset.tab === 'overview') {
            loadOverview();
        }
    }, POLL_INTERVAL);
}

init();
