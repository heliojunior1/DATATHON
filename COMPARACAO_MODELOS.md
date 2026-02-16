# Comparação de Modelos — Datathon Passos Mágicos

> **Data:** 16/02/2026 &nbsp;|&nbsp; **Dataset:** 860 alunos (688 treino / 172 teste) &nbsp;|&nbsp; **Features:** 35 &nbsp;|&nbsp; **Target:** Risco de evasão (69.9% positivo)

---

## 1. Métricas no Test Set

| Modelo | Accuracy | F1 Score | Precision | Recall | AUC-ROC |
|--------|:--------:|:--------:|:---------:|:------:|:-------:|
| **CatBoost** 🥇 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **LightGBM** 🥈 | 0.9942 | 0.9959 | 0.9917 | **1.0000** | **1.0000** |
| **TabPFN** 🥉 | 0.9826 | 0.9876 | 0.9835 | 0.9917 | 0.9997 |
| **XGBoost** | 0.9593 | 0.9702 | 0.9913 | 0.9500 | 0.9966 |

---

## 2. Cross-Validation (5-Fold Estratificado)

| Modelo | CV F1 (média ± std) | CV Accuracy | CV Precision | CV Recall | CV AUC-ROC |
|--------|:-------------------:|:-----------:|:------------:|:---------:|:----------:|
| **TabPFN** 🥇 | **0.9958 ± 0.0027** | **0.9942 ± 0.0037** | **0.9983 ± 0.0033** | 0.9933 ± 0.0062 | **0.9997 ± 0.0004** |
| **CatBoost** 🥈 | 0.9918 ± 0.0093 | 0.9884 ± 0.0133 | 0.9839 ± 0.0181 | **1.0000 ± 0.0000** | 0.9994 ± 0.0012 |
| **LightGBM** 🥉 | 0.9910 ± 0.0070 | 0.9872 ± 0.0100 | 0.9838 ± 0.0143 | 0.9983 ± 0.0033 | 0.9988 ± 0.0016 |
| **XGBoost** | 0.9742 ± 0.0060 | 0.9640 ± 0.0085 | 0.9770 ± 0.0158 | 0.9717 ± 0.0135 | 0.9926 ± 0.0061 |

> [!IMPORTANT]
> No **Cross-Validation**, o TabPFN lidera com o maior F1 e o menor desvio-padrão (0.0027), indicando a melhor **consistência** entre todos os modelos. CatBoost lidera no test set, mas TabPFN generaliza melhor.

---

## 3. Matriz de Confusão (Test Set)

| Modelo | TN | FP | FN | TP | Erros |
|--------|:--:|:--:|:--:|:--:|:-----:|
| CatBoost | 52 | 0 | 0 | 120 | 0 |
| LightGBM | 51 | 1 | 0 | 120 | 1 |
| TabPFN | 51 | 1 | 1 | 119 | 2 |
| XGBoost | 51 | 1 | 6 | 114 | 7 |

> [!WARNING]
> O XGBoost teve **6 falsos negativos** (alunos em risco classificados como sem risco). Para detecção de risco de evasão, CatBoost e LightGBM (100% recall) são preferíveis.

---

## 4. Top 10 Features Mais Importantes

### Por Modelo (Top 5)

| # | XGBoost | CatBoost | LightGBM | TabPFN |
|---|---------|----------|----------|--------|
| 1 | Nº Av | **Idade 22** | **Idade 22** | **Idade 22** |
| 2 | Idade 22 | **Fase_encoded** | **Fase_encoded** | INDE 22 |
| 3 | Fase_encoded | Nº Av | Cf | Cf |
| 4 | Indicado_flag | Cf | IPV | IEG |
| 5 | Cf | INDE 22 | INDE 22 | IPV |

> [!NOTE]
> **Consenso entre modelos:** `Idade 22` é a feature mais importante em 3 dos 4 modelos. `Cf`, `INDE 22` e `IPV` também aparecem consistentemente no top 5. As escalas de importância diferem por modelo (XGBoost=gain fraction, CatBoost=prediction value change, LightGBM=split count, TabPFN=permutation importance).

---

## 5. Análise e Recomendações

### Ranking Final

| Pos | Modelo | Test F1 | CV F1 | Pontos Fortes | Pontos Fracos |
|:---:|--------|:-------:|:-----:|---------------|---------------|
| 🥇 | **CatBoost** | 1.0000 | 0.9918 | Melhor test set, 100% recall, robusto a NaN | Possível leve overfitting |
| 🥈 | **TabPFN** | 0.9876 | **0.9958** | **Melhor CV**, mais consistente, sem tuning | Lento, não suporta NaN, limite de features |
| 🥉 | **LightGBM** | 0.9959 | 0.9910 | Rápido, 100% recall, robusto a NaN | 1 FP |
| 4 | **XGBoost** | 0.9702 | 0.9742 | Estabelecido, boa calibração | 6 FN, recall mais baixo |

### Recomendação

Para **detecção de risco de evasão escolar** (custo alto de falso negativo):

- **Produção:** **CatBoost** como modelo principal (100% recall + melhor precisão no test set)
- **Validação:** **TabPFN** como referência de generalização (melhor CV F1, menor variância)
- **Backup:** **LightGBM** como alternativa rápida com 100% recall

---

## 6. Modelos na API

```json
POST /train
{
  "model_type": "catboost"  // ou "xgboost", "lightgbm", "tabpfn"
}
```

| Tipo | Status | Model ID |
|------|--------|----------|
| xgboost | ✅ Operacional | `xgb_20260216_104712` |
| catboost | ✅ Operacional | `cat_20260216_104720` |
| lightgbm | ✅ Operacional | `lgb_20260216_104729` |
| tabpfn | ✅ Operacional | `tpfn_20260216_105151` |

> [!NOTE]
> TabPFN v1 (0.1.11) requer o patch `python scripts/patch_tabpfn.py` após instalação para compatibilidade com PyTorch ≥ 2.0.
