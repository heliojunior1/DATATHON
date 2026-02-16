# Comparação de Modelos — Datathon Passos Mágicos

> **Data:** 16/02/2026 &nbsp;|&nbsp; **Dataset:** 860 alunos (688 treino / 172 teste) &nbsp;|&nbsp; **Features:** 35 &nbsp;|&nbsp; **Target:** Risco de evasão (69.9% positivo)

---

## 1. Métricas no Test Set

| Modelo | Accuracy | F1 Score | Precision | Recall | AUC-ROC |
|--------|:--------:|:--------:|:---------:|:------:|:-------:|
| **CatBoost** 🥇 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Reg. Logística** 🥇 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **LightGBM** 🥉 | 0.9942 | 0.9959 | 0.9917 | **1.0000** | **1.0000** |
| **TabPFN** | 0.9826 | 0.9876 | 0.9835 | 0.9917 | 0.9997 |
| **XGBoost** | 0.9593 | 0.9702 | 0.9913 | 0.9500 | 0.9966 |
| **SVM** | 0.9302 | 0.9508 | 0.9355 | 0.9667 | 0.9864 |

> [!TIP]
> O SVM agora inclui **StandardScaler** no pipeline, o que melhorou drasticamente o desempenho: F1 subiu de **0.7477 → 0.9508** (+27 p.p.). Sem normalização, o kernel RBF não consegue medir distâncias corretamente entre features de escalas diferentes.

---

## 2. Cross-Validation (5-Fold Estratificado)

| Modelo | CV F1 (média ± std) | CV Accuracy | CV Precision | CV Recall | CV AUC-ROC |
|--------|:-------------------:|:-----------:|:------------:|:---------:|:----------:|
| **TabPFN** 🥇 | **0.9958 ± 0.0027** | **0.9942 ± 0.0037** | **0.9983 ± 0.0033** | 0.9933 ± 0.0062 | **0.9997 ± 0.0004** |
| **Reg. Logística** 🥈 | 0.9933 ± 0.0057 | 0.9907 ± 0.0079 | **1.0000 ± 0.0000** | 0.9867 ± 0.0113 | 0.9982 ± 0.0033 |
| **CatBoost** 🥉 | 0.9918 ± 0.0093 | 0.9884 ± 0.0133 | 0.9839 ± 0.0181 | **1.0000 ± 0.0000** | 0.9994 ± 0.0012 |
| **LightGBM** | 0.9910 ± 0.0070 | 0.9872 ± 0.0100 | 0.9838 ± 0.0143 | 0.9983 ± 0.0033 | 0.9988 ± 0.0016 |
| **SVM** | 0.9608 ± 0.0160 | 0.9453 ± 0.0225 | 0.9638 ± 0.0241 | 0.9584 ± 0.0211 | 0.9809 ± 0.0140 |
| **XGBoost** | 0.9742 ± 0.0060 | 0.9640 ± 0.0085 | 0.9770 ± 0.0158 | 0.9717 ± 0.0135 | 0.9926 ± 0.0061 |

> [!IMPORTANT]
> No **Cross-Validation**, TabPFN lidera com o maior F1 (0.9958) e menor variância (0.0027). Regressão Logística tem **100% de precisão** em todos os folds. CatBoost tem **100% de recall** em todos os folds.

---

## 3. Matriz de Confusão (Test Set)

| Modelo | TN | FP | FN | TP | Total Erros |
|--------|:--:|:--:|:--:|:--:|:-----------:|
| CatBoost | 52 | 0 | 0 | 120 | **0** |
| Reg. Logística | 52 | 0 | 0 | 120 | **0** |
| LightGBM | 51 | 1 | 0 | 120 | 1 |
| TabPFN | 51 | 1 | 1 | 119 | 2 |
| XGBoost | 51 | 1 | 6 | 114 | 7 |
| SVM | 44 | 8 | 4 | 116 | 12 |

---

## 4. Top 5 Features por Modelo

| # | XGBoost | CatBoost | LightGBM | TabPFN | Reg. Logística | SVM |
|---|---------|----------|----------|--------|----------------|-----|
| 1 | Nº Av | **Idade 22** | **Idade 22** | **Idade 22** | **Idade 22** | **Idade 22** |
| 2 | Idade 22 | Fase_encoded | Fase_encoded | INDE 22 | INDE 22 | Cf |
| 3 | Fase_encoded | Nº Av | Cf | Cf | Fase_encoded | INDE 22 |
| 4 | Indicado_flag | Cf | IPV | IEG | IDA | IEG |
| 5 | Cf | INDE 22 | INDE 22 | IPV | IEG | IAA |

> [!NOTE]
> **Consenso:** `Idade 22` é a feature #1 em **5 de 6 modelos** (incluindo SVM após normalização). `INDE 22` e `Cf` aparecem consistentemente no top 5.

---

## 5. Evolução do SVM com StandardScaler

| Métrica | Sem Scaler | Com Scaler | Melhoria |
|---------|:----------:|:----------:|:--------:|
| F1 Score | 0.7477 | **0.9508** | **+27.2%** |
| Accuracy | 0.6744 | **0.9302** | **+37.9%** |
| Recall | 0.6917 | **0.9667** | **+39.7%** |
| AUC-ROC | 0.7914 | **0.9864** | **+24.6%** |
| CV F1 | 0.7785 | **0.9608** | **+23.4%** |

> [!NOTE]
> O `StandardScaler` normaliza cada feature para média=0 e desvio=1, essencial para o kernel RBF do SVM que calcula distâncias euclidianas no espaço de features.

---

## 6. Ranking Final

| Pos | Modelo | Test F1 | CV F1 | Tipo | Pontos Fortes | Pontos Fracos |
|:---:|--------|:-------:|:-----:|------|---------------|---------------|
| 🥇 | **CatBoost** | 1.0000 | 0.9918 | Ensemble | 100% recall + precisão, robusto a NaN | Possível leve overfitting |
| 🥈 | **Reg. Logística** | 1.0000 | **0.9933** | Linear | 100% precisão CV, rápido, interpretável | Não suporta NaN |
| 🥉 | **TabPFN** | 0.9876 | **0.9958** | Transformer | **Melhor CV**, mais consistente | Lento, não suporta NaN |
| 4 | **LightGBM** | 0.9959 | 0.9910 | Ensemble | Rápido, 100% recall, robusto a NaN | 1 FP |
| 5 | **SVM** | 0.9508 | 0.9608 | Kernel | Melhoria com normalização | 12 erros no test set |
| 6 | **XGBoost** | 0.9702 | 0.9742 | Ensemble | Estabelecido, calibração | 6 FN, menor recall |

### Recomendação para Produção

- **Modelo principal:** **CatBoost** (perfeito no test, 100% recall no CV)
- **Modelo interpretável:** **Regressão Logística** (coeficientes explicáveis para stakeholders)
- **Referência de generalização:** **TabPFN** (melhor CV F1 = 0.9958, menor variância)

---

## 7. Modelos na API

```json
POST /train
{
  "model_type": "catboost"
}
```

| Tipo | Status | Model ID | Test F1 |
|------|--------|----------|:-------:|
| catboost | ✅ | `cat_20260216_112526` | 1.0000 |
| logistic_regression | ✅ | `lr_20260216_112934` | 1.0000 |
| lightgbm | ✅ | `lgb_20260216_112534` | 0.9959 |
| tabpfn | ✅ | `tpfn_20260216_112907` | 0.9876 |
| xgboost | ✅ | `xgb_20260216_112518` | 0.9702 |
| svm | ✅ | `svm_20260216_112942` | 0.9508 |
