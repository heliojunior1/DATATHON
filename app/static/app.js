const API = '';

// ===== STATE =====
let _availableFeatures = [];
let _trainedModels = [];

// ===== NAVIGATION =====
function navigate(page) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav a').forEach(a => a.classList.remove('active'));
    document.getElementById('page-' + page).classList.add('active');
    document.querySelector(`.nav a[onclick*="${page}"]`).classList.add('active');
    if (page === 'dashboard') loadDashboard();
    if (page === 'training') loadTrainingPage();
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

// ===== LOAD TRAINED MODELS (shared) =====
async function loadTrainedModels() {
    try {
        const data = await api('/models');
        _trainedModels = data.models || [];
        populateModelDropdowns();
    } catch (e) { _trainedModels = []; }
}

function populateModelDropdowns() {
    const selectors = ['dash-model-select', 'predict-model-select'];
    selectors.forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        const prev = el.value;
        el.innerHTML = '<option value="">Mais recente</option>';
        _trainedModels.forEach(m => {
            const label = `${m.model_type || '?'} — ${m.model_id} (${((m.metrics?.f1_score || 0) * 100).toFixed(1)}% F1)`;
            el.innerHTML += `<option value="${m.model_id}">${label}</option>`;
        });
        if (prev) el.value = prev;
    });
}

// ===== DASHBOARD =====
async function loadDashboard() {
    try {
        const modelId = document.getElementById('dash-model-select')?.value || '';
        const qs = modelId ? `?model_id=${modelId}` : '';
        const [health, info] = await Promise.all([api('/health'), api('/model-info' + qs)]);
        const m = info.metrics || {};
        document.getElementById('m-accuracy').textContent = ((m.accuracy || 0) * 100).toFixed(1) + '%';
        document.getElementById('m-f1').textContent = ((m.f1_score || 0) * 100).toFixed(1) + '%';
        document.getElementById('m-recall').textContent = ((m.recall || 0) * 100).toFixed(1) + '%';
        document.getElementById('m-precision').textContent = ((m.precision || 0) * 100).toFixed(1) + '%';
        document.getElementById('m-auc').textContent = ((m.auc_roc || 0) * 100).toFixed(1) + '%';

        document.getElementById('m-model').textContent = `${info.model_type || info.model_name || '—'} • ${info.model_id || '?'}`;
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

        // Cross-Validation results
        const cvSection = document.getElementById('cv-section');
        if (info.cv_results) {
            cvSection.style.display = 'block';
            const cvGrid = document.getElementById('cv-metrics');
            const labels = { accuracy: 'Accuracy', precision: 'Precision', recall: 'Recall', f1_score: 'F1-Score', auc_roc: 'AUC-ROC' };
            const colors = { accuracy: 'green', precision: 'purple', recall: 'green', f1_score: 'blue', auc_roc: 'amber' };
            cvGrid.innerHTML = Object.entries(labels).map(([key, label]) => {
                const cv = info.cv_results[key];
                if (!cv) return '';
                return `<div class="metric-card ${colors[key]}">
                    <div class="metric-label">${label}</div>
                    <div class="metric-value" style="font-size:22px">${(cv.mean * 100).toFixed(1)}%</div>
                    <div class="metric-sub">± ${(cv.std * 100).toFixed(2)}%</div>
                </div>`;
            }).join('');
        } else {
            cvSection.style.display = 'none';
        }

        // Learning Curves
        const lcSection = document.getElementById('lc-section');
        const lcImg = document.getElementById('lc-img');
        try {
            const lcResp = await fetch('/learning-curve' + qs);
            if (lcResp.ok) {
                const blob = await lcResp.blob();
                lcImg.src = URL.createObjectURL(blob);
                lcSection.style.display = 'block';
            } else {
                lcSection.style.display = 'none';
            }
        } catch (e) {
            lcSection.style.display = 'none';
        }
    } catch (e) { console.error('Dashboard error:', e); }
}

// ===== TRAINING PAGE =====
async function loadTrainingPage() {
    // Load available features
    try {
        const data = await api('/features/available');
        _availableFeatures = data.features || [];
        renderFeatureSelection();
    } catch (e) { console.error('Error loading features:', e); }

    // Load trained models list
    await loadTrainedModelsList();
}

function renderFeatureSelection() {
    const container = document.getElementById('feature-selection');
    // Group by category
    const groups = {};
    _availableFeatures.forEach(f => {
        if (!groups[f.category]) groups[f.category] = [];
        groups[f.category].push(f);
    });

    container.innerHTML = Object.entries(groups).map(([cat, feats]) => `
    <div class="feature-category">
      <div class="feature-cat-header">${cat}</div>
      ${feats.map(f => `
        <label class="feature-item" title="${f.description}">
          <input type="checkbox" value="${f.name}" ${f.default_selected ? 'checked' : ''} class="feat-cb">
          <span>${f.name}</span>
          <small style="color:var(--text2);display:block;font-size:11px">${f.description}</small>
        </label>
      `).join('')}
    </div>
  `).join('');
}

function toggleAllFeatures(state) {
    document.querySelectorAll('.feat-cb').forEach(cb => cb.checked = state);
}

function resetDefaultFeatures() {
    document.querySelectorAll('.feat-cb').forEach(cb => {
        const feat = _availableFeatures.find(f => f.name === cb.value);
        cb.checked = feat ? feat.default_selected : false;
    });
}

function getSelectedFeatures() {
    return Array.from(document.querySelectorAll('.feat-cb:checked')).map(cb => cb.value);
}

async function submitTraining() {
    const btn = document.getElementById('btn-train');
    btn.classList.add('loading');
    const resultsDiv = document.getElementById('train-results');
    resultsDiv.innerHTML = '<div class="card" style="margin-top:20px"><div class="card-title"><span class="card-icon">⏳</span>Treinando modelo... isso pode levar de 10s a 60s</div><div class="train-progress"><div class="train-progress-bar"></div></div></div>';

    try {
        const features = getSelectedFeatures();
        if (features.length < 3) { toast('Selecione ao menos 3 features'); btn.classList.remove('loading'); return; }

        const body = {
            model_type: document.getElementById('train-model-type').value,
            features,
            optimize: document.getElementById('train-optimize').value === 'true',
            n_iter: parseInt(document.getElementById('train-n-iter').value) || 50,
            include_ian: document.getElementById('train-ian').checked,
            run_cv: document.getElementById('train-cv').checked,
            run_learning_curves: document.getElementById('train-lc').checked,
        };

        const result = await api('/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        showTrainResults(result);
        toast(result.message, 'success');

        // Refresh models list and dropdowns
        await loadTrainedModels();
        await loadTrainedModelsList();
    } catch (e) {
        resultsDiv.innerHTML = `<div class="card" style="margin-top:20px"><div class="card-title"><span class="card-icon">❌</span>Erro no treinamento</div><p style="color:var(--red)">${e.message}</p></div>`;
    }
    btn.classList.remove('loading');
}

function showTrainResults(r) {
    const m = r.metrics || {};
    const resultsDiv = document.getElementById('train-results');
    resultsDiv.innerHTML = `
    <div class="card" style="margin-top:20px">
      <div class="card-title"><span class="card-icon">✅</span>Treinamento Concluído — ${r.model_id}</div>
      <div class="metrics-grid">
        <div class="metric-card green"><div class="metric-label">Accuracy</div><div class="metric-value">${((m.accuracy || 0) * 100).toFixed(1)}%</div></div>
        <div class="metric-card blue"><div class="metric-label">F1-Score</div><div class="metric-value">${((m.f1_score || 0) * 100).toFixed(1)}%</div></div>
        <div class="metric-card green"><div class="metric-label">Recall</div><div class="metric-value">${((m.recall || 0) * 100).toFixed(1)}%</div></div>
        <div class="metric-card purple"><div class="metric-label">Precision</div><div class="metric-value">${((m.precision || 0) * 100).toFixed(1)}%</div></div>
        <div class="metric-card amber"><div class="metric-label">AUC-ROC</div><div class="metric-value">${((m.auc_roc || 0) * 100).toFixed(1)}%</div></div>
      </div>
      <div style="margin-top:12px;color:var(--text2)">
        <strong>Modelo:</strong> ${r.model_type} • <strong>Features:</strong> ${r.feature_names.length} •
        <strong>Treino:</strong> ${r.n_train} amostras • <strong>Teste:</strong> ${r.n_test} amostras
      </div>
      ${r.cv_results ? `<div style="margin-top:8px;color:var(--text2)"><strong>CV F1:</strong> ${((r.cv_results.f1_score?.mean || 0) * 100).toFixed(1)}% ± ${((r.cv_results.f1_score?.std || 0) * 100).toFixed(2)}%</div>` : ''}
    </div>`;
}

async function loadTrainedModelsList() {
    try {
        const data = await api('/models');
        const models = data.models || [];
        const container = document.getElementById('trained-models-list');
        if (models.length === 0) {
            container.innerHTML = '<p style="color:var(--text2);text-align:center;padding:20px">Nenhum modelo treinado ainda</p>';
            return;
        }
        container.innerHTML = `<div class="table-wrap"><table>
            <thead><tr><th>ID</th><th>Tipo</th><th>F1-Score</th><th>Accuracy</th><th>Features</th><th>Amostras</th><th>Treinado em</th><th></th></tr></thead>
            <tbody>${models.map(m => `<tr>
                <td><code>${m.model_id}</code></td>
                <td>${m.model_type || '—'}</td>
                <td>${((m.metrics?.f1_score || 0) * 100).toFixed(1)}%</td>
                <td>${((m.metrics?.accuracy || 0) * 100).toFixed(1)}%</td>
                <td>${m.feature_count || '—'}</td>
                <td>${m.n_training_samples || '—'}</td>
                <td>${m.trained_at ? new Date(m.trained_at).toLocaleString('pt-BR') : '—'}</td>
                <td><button class="btn btn-secondary" style="padding:4px 12px;font-size:12px" onclick="deleteModel('${m.model_id}')">🗑️</button></td>
            </tr>`).join('')}</tbody>
        </table></div>`;
    } catch (e) { console.error('Error loading models list:', e); }
}

async function deleteModel(modelId) {
    if (!confirm(`Deletar modelo ${modelId}?`)) return;
    try {
        await api(`/models/${modelId}`, { method: 'DELETE' });
        toast(`Modelo ${modelId} deletado`, 'success');
        await loadTrainedModels();
        await loadTrainedModelsList();
    } catch (e) { /* toast shown by api() */ }
}

// ===== PREDICTION =====
async function submitPrediction() {
    const btn = document.getElementById('btn-predict');
    btn.classList.add('loading');
    try {
        const fields = ['IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN', 'INDE_22', 'Matem', 'Portug', 'Tem_nota_ingles',
            'Fase', 'Idade_22', 'Genero', 'Instituicao', 'Ano_ingresso', 'Pedra_22',
            'Atingiu_PV', 'Indicado', 'Cf', 'Ct', 'Num_Av',
            'Destaque_IEG', 'Destaque_IDA', 'Destaque_IPV'];
        const data = {};
        fields.forEach(f => {
            const el = document.getElementById('f-' + f);
            if (!el) return;
            const v = el.value;
            if (v === '') return;
            const numFields = ['IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN', 'INDE_22', 'Matem', 'Portug', 'Tem_nota_ingles', 'Idade_22', 'Ano_ingresso', 'Cf', 'Ct', 'Num_Av'];
            data[el.dataset.key || f] = numFields.includes(f) ? parseFloat(v) : v;
        });

        const modelId = document.getElementById('predict-model-select')?.value || '';
        const qs = modelId ? `?model_id=${modelId}` : '';
        const result = await api('/predict' + qs, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data)
        });
        showResult(result);
    } catch (e) {
        console.error(e);
        toast(e.message);
    }
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
        { "IAA": 7.5, "IEG": 8, "IPS": 6.5, "IDA": 7, "IPV": 5.5, "IAN": 5, "INDE 22": 7.2, "Matem": 7.5, "Portug": 6.8, "Tem_nota_ingles": 1, "Fase": "Fase 3", "Idade 22": 14, "Gênero": "Menina", "Instituição de ensino": "Escola Pública", "Ano ingresso": 2018, "Pedra 22": "Ametista", "Atingiu PV": "Não", "Indicado": "Não", "Cf": 50, "Ct": 5, "Nº Av": 3, "Destaque IEG": "Não", "Destaque IDA": "Não", "Destaque IPV": "Não" },
        { "IAA": 3.2, "IEG": 4.1, "IPS": 3.5, "IDA": 2.8, "IPV": 2.0, "IAN": 0, "INDE 22": 3.1, "Matem": 3.0, "Portug": 2.5, "Tem_nota_ingles": 0, "Fase": "Fase 3", "Idade 22": 17, "Gênero": "Menino", "Instituição de ensino": "Escola Pública", "Ano ingresso": 2020, "Pedra 22": "Quartzo", "Atingiu PV": "Não", "Indicado": "Não", "Cf": 150, "Ct": 15, "Nº Av": 4, "Destaque IEG": "Não", "Destaque IDA": "Não", "Destaque IPV": "Não" }
    ], null, 2);
}

async function submitBatch() {
    const btn = document.getElementById('btn-batch');
    btn.classList.add('loading');
    try {
        const raw = document.getElementById('batch-json').value.trim();
        const students = JSON.parse(raw);
        if (!Array.isArray(students)) throw new Error('JSON deve ser um array de alunos');

        const modelId = document.getElementById('predict-model-select')?.value || '';
        const body = { students };
        if (modelId) body.model_id = modelId;

        const result = await api('/predict/batch', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
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
        const features = drift.details || {};
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
    const ex = { IAA: 7.5, IEG: 8.0, IPS: 6.5, IDA: 7.0, IPV: 5.5, IAN: 5.0, INDE_22: 7.2, Matem: 7.5, Portug: 6.8, Tem_nota_ingles: 1, Fase: 'Fase 3', Idade_22: 14, Genero: 'Menina', Instituicao: 'Escola Pública', Ano_ingresso: 2018, Pedra_22: 'Ametista', Atingiu_PV: 'Não', Indicado: 'Não', Cf: 50, Ct: 5, Num_Av: 3, Destaque_IEG: 'Não', Destaque_IDA: 'Não', Destaque_IPV: 'Não' };
    Object.entries(ex).forEach(([k, v]) => { const el = document.getElementById('f-' + k); if (el) el.value = v; });
    toast('Exemplo preenchido', 'success');
}

function clearForm() {
    document.querySelectorAll('#page-predict input, #page-predict select').forEach(el => {
        if (el.id === 'predict-model-select') return; // Preserve model selection
        el.value = '';
    });
    document.getElementById('result-card').classList.remove('show');
}

// ===== MODEL TYPE CHANGE HANDLER =====
function onModelTypeChange() {
    const modelType = document.getElementById('train-model-type').value;
    const optimizeSelect = document.getElementById('train-optimize');
    const nIterInput = document.getElementById('train-n-iter');

    // TabPFN não suporta otimização de hiperparâmetros
    const noOptimize = (modelType === 'tabpfn');
    optimizeSelect.disabled = noOptimize;
    nIterInput.disabled = noOptimize;
    if (noOptimize) {
        optimizeSelect.value = 'false';
    }
}

// ===== INIT =====
document.addEventListener('DOMContentLoaded', async () => {
    await loadTrainedModels();

    // Attach model type change handler
    const modelSelect = document.getElementById('train-model-type');
    if (modelSelect) {
        modelSelect.onchange = onModelTypeChange;
    }

    navigate('dashboard');
});
