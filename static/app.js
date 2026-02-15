const API = '';

// ===== NAVIGATION =====
function navigate(page) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav a').forEach(a => a.classList.remove('active'));
    document.getElementById('page-' + page).classList.add('active');
    document.querySelector(`.nav a[onclick*="${page}"]`).classList.add('active');
    if (page === 'dashboard') loadDashboard();
    if (page === 'monitoring') loadMonitoring();
}

// ===== TOAST =====
function toast(msg, type = 'error') {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.className = `toast show toast-${type}`;
    setTimeout(() => t.classList.remove('show'), 4000);
}

// ===== API HELPERS =====
async function api(path, opts = {}) {
    try {
        const r = await fetch(API + path, opts);
        if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || `Erro ${r.status}`); }
        return await r.json();
    } catch (e) { toast(e.message); throw e; }
}

// ===== DASHBOARD =====
async function loadDashboard() {
    try {
        const [health, info] = await Promise.all([api('/health'), api('/model-info')]);
        const m = info.metrics || {};
        document.getElementById('m-accuracy').textContent = ((m.accuracy || 0) * 100).toFixed(1) + '%';
        document.getElementById('m-f1').textContent = ((m.f1_score || 0) * 100).toFixed(1) + '%';
        document.getElementById('m-recall').textContent = ((m.recall || 0) * 100).toFixed(1) + '%';
        document.getElementById('m-precision').textContent = ((m.precision || 0) * 100).toFixed(1) + '%';
        document.getElementById('m-auc').textContent = ((m.auc_roc || 0) * 100).toFixed(1) + '%';

        document.getElementById('m-model').textContent = `${info.model_name || '—'} v${info.model_version || '?'}`;
        document.getElementById('m-features').textContent = `${(info.feature_names || []).length} features`;
        document.getElementById('m-samples').textContent = `${info.n_training_samples || '?'} amostras`;
        document.getElementById('m-status').textContent = health.model_loaded ? '🟢 Online' : '🔴 Offline';

        // Feature importance bars
        const feats = (info.feature_importance || []).slice(0, 12);
        const maxImp = feats.length ? feats[0].importance : 1;
        const container = document.getElementById('feat-bars');
        container.innerHTML = feats.map(f => `
      <div class="feat-row">
        <span class="feat-name" title="${f.feature}">${f.feature}</span>
        <div class="feat-bar-bg"><div class="feat-bar" style="width:${(f.importance / maxImp * 100).toFixed(1)}%"></div></div>
        <span class="feat-val">${(f.importance * 100).toFixed(1)}%</span>
      </div>
    `).join('');

        // Confusion matrix
        const cm = info.confusion_matrix || {};
        document.getElementById('cm-tp').textContent = cm.true_positives ?? '—';
        document.getElementById('cm-tn').textContent = cm.true_negatives ?? '—';
        document.getElementById('cm-fp').textContent = cm.false_positives ?? '—';
        document.getElementById('cm-fn').textContent = cm.false_negatives ?? '—';
    } catch (e) { console.error('Dashboard error:', e); }
}

// ===== PREDICTION =====
async function submitPrediction() {
    const btn = document.getElementById('btn-predict');
    btn.classList.add('loading');
    try {
        const fields = ['IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN', 'INDE_22', 'Matem', 'Portug', 'Ingles',
            'Idade_22', 'Genero', 'Instituicao', 'Ano_ingresso', 'Pedra_22', 'Rec_psico',
            'Atingiu_PV', 'Indicado', 'Cg', 'Cf', 'Ct', 'Num_Av',
            'Destaque_IEG', 'Destaque_IDA', 'Destaque_IPV'];
        const data = {};
        fields.forEach(f => {
            const el = document.getElementById('f-' + f);
            if (!el) return;
            const v = el.value;
            if (v === '') return;
            const numFields = ['IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN', 'INDE_22', 'Matem', 'Portug', 'Ingles', 'Idade_22', 'Ano_ingresso', 'Cg', 'Cf', 'Ct', 'Num_Av'];
            data[el.dataset.key || f] = numFields.includes(f) ? parseFloat(v) : v;
        });
        const result = await api('/predict', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data)
        });
        showResult(result);
    } catch (e) { console.error(e); }
    btn.classList.remove('loading');
}

function showResult(r) {
    const card = document.getElementById('result-card');
    card.classList.add('show');
    const isRisk = r.prediction === 1;
    document.getElementById('r-badge').className = `result-badge ${isRisk ? 'badge-risk' : 'badge-safe'}`;
    document.getElementById('r-badge').textContent = r.label || (isRisk ? 'Em Risco' : 'Sem Risco');
    document.getElementById('r-prob').textContent = (r.probability * 100).toFixed(1) + '%';
    document.getElementById('r-prob').style.color = isRisk ? 'var(--red)' : 'var(--green)';
    document.getElementById('r-level').textContent = `Nível de Risco: ${r.risk_level}`;

    const factors = document.getElementById('r-factors');
    factors.innerHTML = (r.top_factors || []).slice(0, 5).map(f => `
    <div class="feat-row">
      <span class="feat-name">${f.feature}</span>
      <div class="feat-bar-bg"><div class="feat-bar" style="width:${(f.importance * 100 / (r.top_factors[0]?.importance || 1)).toFixed(0)}%"></div></div>
      <span class="feat-val">${(f.importance * 100).toFixed(1)}%</span>
    </div>
  `).join('');
}

// ===== BATCH PREDICTION =====
function loadBatchExample() {
    document.getElementById('batch-json').value = JSON.stringify([
        { "IAA": 7.5, "IEG": 8, "IPS": 6.5, "IDA": 7, "IPV": 5.5, "IAN": 5, "INDE 22": 7.2, "Matem": 7.5, "Portug": 6.8, "Idade 22": 14, "Gênero": "Menina", "Instituição de ensino": "Escola Pública", "Ano ingresso": 2018, "Pedra 22": "Ametista", "Rec Psicologia": "Sem limitações", "Atingiu PV": "Não", "Indicado": "Não", "Cg": 300, "Cf": 50, "Ct": 5, "Nº Av": 3, "Destaque IEG": "Não", "Destaque IDA": "Não", "Destaque IPV": "Não" },
        { "IAA": 3.2, "IEG": 4.1, "IPS": 3.5, "IDA": 2.8, "IPV": 2.0, "IAN": 0, "INDE 22": 3.1, "Matem": 3.0, "Portug": 2.5, "Idade 22": 17, "Gênero": "Menino", "Instituição de ensino": "Escola Pública", "Ano ingresso": 2020, "Pedra 22": "Quartzo", "Rec Psicologia": "Requer avaliação", "Atingiu PV": "Não", "Indicado": "Não", "Cg": 700, "Cf": 150, "Ct": 15, "Nº Av": 4, "Destaque IEG": "Não", "Destaque IDA": "Não", "Destaque IPV": "Não" }
    ], null, 2);
}

async function submitBatch() {
    const btn = document.getElementById('btn-batch');
    btn.classList.add('loading');
    try {
        const raw = document.getElementById('batch-json').value.trim();
        const students = JSON.parse(raw);
        if (!Array.isArray(students)) throw new Error('JSON deve ser um array de alunos');
        const result = await api('/predict/batch', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ students })
        });
        showBatchResults(result);
        toast(`${result.total} predições realizadas com sucesso`, 'success');
    } catch (e) { if (e instanceof SyntaxError) toast('JSON inválido'); }
    btn.classList.remove('loading');
}

function showBatchResults(data) {
    const container = document.getElementById('batch-results');
    const preds = data.predictions || [];
    container.innerHTML = `
    <div class="card" style="margin-top:20px">
      <div class="card-title"><span class="card-icon">📊</span> Resultados — ${preds.length} alunos</div>
      <div class="metrics-grid" style="margin-bottom:16px">
        <div class="metric-card green"><div class="metric-label">Sem Risco</div><div class="metric-value" style="color:var(--green)">${preds.filter(p => p.prediction === 0).length}</div></div>
        <div class="metric-card red"><div class="metric-label">Em Risco</div><div class="metric-value" style="color:var(--red)">${preds.filter(p => p.prediction === 1).length}</div></div>
      </div>
      <div class="table-wrap"><table>
        <thead><tr><th>#</th><th>Resultado</th><th>Probabilidade</th><th>Nível</th></tr></thead>
        <tbody>${preds.map((p, i) => `<tr>
          <td>${i + 1}</td>
          <td><span class="drift-status ${p.prediction === 1 ? 'drift-alert' : 'drift-ok'}">${p.label || (p.prediction === 1 ? 'Em Risco' : 'Sem Risco')}</span></td>
          <td>${(p.probability * 100).toFixed(1)}%</td>
          <td>${p.risk_level}</td>
        </tr>`).join('')}</tbody>
      </table></div>
    </div>`;
}

// ===== MONITORING =====
async function loadMonitoring() {
    try {
        const [drift, stats] = await Promise.all([api('/monitoring/drift'), api('/monitoring/stats')]);

        document.getElementById('mon-total').textContent = stats.total_predictions || 0;
        document.getElementById('mon-risk-rate').textContent = ((stats.risk_rate || 0) * 100).toFixed(1) + '%';
        document.getElementById('mon-avg-prob').textContent = ((stats.avg_probability || 0) * 100).toFixed(1) + '%';

        const statusMap = { 'OK': 'drift-ok', 'WARNING': 'drift-warn', 'DRIFT_DETECTED': 'drift-alert', 'NO_DATA': 'drift-warn', 'NO_REFERENCE': 'drift-warn' };
        const statusEl = document.getElementById('mon-drift-status');
        statusEl.className = `drift-status ${statusMap[drift.status] || 'drift-warn'}`;
        statusEl.textContent = drift.status || 'SEM DADOS';

        const container = document.getElementById('drift-details');
        const features = drift.features || {};
        const keys = Object.keys(features);
        if (keys.length === 0) {
            container.innerHTML = '<p style="color:var(--text2);text-align:center;padding:20px">Faça predições para gerar dados de monitoramento</p>';
        } else {
            container.innerHTML = `<div class="table-wrap"><table>
        <thead><tr><th>Feature</th><th>Status</th><th>P-Value</th><th>Mean Shift</th></tr></thead>
        <tbody>${keys.map(k => {
                const f = features[k];
                const sc = f.drift_status === 'OK' ? 'drift-ok' : f.drift_status === 'WARNING' ? 'drift-warn' : 'drift-alert';
                return `<tr><td>${k}</td><td><span class="drift-status ${sc}">${f.drift_status}</span></td><td>${f.p_value?.toFixed(4) ?? '—'}</td><td>${f.mean_shift?.toFixed(4) ?? '—'}</td></tr>`;
            }).join('')}</tbody>
      </table></div>`;
        }
    } catch (e) { console.error('Monitoring error:', e); }
}

// ===== FILL EXAMPLE =====
function fillExample() {
    const ex = { IAA: 7.5, IEG: 8.0, IPS: 6.5, IDA: 7.0, IPV: 5.5, IAN: 5.0, INDE_22: 7.2, Matem: 7.5, Portug: 6.8, Ingles: 7.0, Idade_22: 14, Genero: 'Menina', Instituicao: 'Escola Pública', Ano_ingresso: 2018, Pedra_22: 'Ametista', Rec_psico: 'Sem limitações', Atingiu_PV: 'Não', Indicado: 'Não', Cg: 300, Cf: 50, Ct: 5, Num_Av: 3, Destaque_IEG: 'Não', Destaque_IDA: 'Não', Destaque_IPV: 'Não' };
    Object.entries(ex).forEach(([k, v]) => { const el = document.getElementById('f-' + k); if (el) el.value = v; });
    toast('Exemplo preenchido', 'success');
}

function clearForm() {
    document.querySelectorAll('#page-predict input, #page-predict select').forEach(el => el.value = '');
    document.getElementById('result-card').classList.remove('show');
}

// ===== INIT =====
document.addEventListener('DOMContentLoaded', () => { navigate('dashboard'); });
