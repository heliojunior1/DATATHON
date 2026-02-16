# Comparação de Modelos — Datathon Passos Mágicos

> **Data:** 16/02/2026 &nbsp;|&nbsp; **Dataset:** 860 alunos (688 treino / 172 teste) &nbsp;|&nbsp; **Features:** 35 &nbsp;|&nbsp; **Target:** Risco de evasão (69.9% positivo)

---

## 1. Métricas no Test Set

| Modelo | Accuracy | F1 Score | Precision | Recall | AUC-ROC |
|--------|:--------:|:--------:|:---------:|:------:|:-------:|
| **CatBoost** 🥇 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **LightGBM** 🥈 | 0.9942 | 0.9959 | 0.9917 | **1.0000** | **1.0000** |
| **XGBoost** 🥉 | 0.9593 | 0.9702 | 0.9913 | 0.9500 | 0.9966 |
| TabPFN | — | — | — | — | — |

> [!NOTE]
> TabPFN não pôde ser avaliado nesta comparação pois o modelo pré-treinado requer autenticação no HuggingFace (modelo *gated*). Veja instruções em [docs.priorlabs.ai](https://docs.priorlabs.ai/how-to-access-gated-models).

---

## 2. Cross-Validation (5-Fold Estratificado)

| Modelo | CV F1 (média ± std) | CV Accuracy | CV Precision | CV Recall | CV AUC-ROC |
|--------|:-------------------:|:-----------:|:------------:|:---------:|:----------:|
| **CatBoost** 🥇 | **0.9918 ± 0.0093** | 0.9884 ± 0.0133 | 0.9839 ± 0.0181 | **1.0000 ± 0.0000** | **0.9994 ± 0.0012** |
| **LightGBM** 🥈 | 0.9910 ± 0.0070 | 0.9872 ± 0.0100 | 0.9838 ± 0.0143 | 0.9983 ± 0.0033 | 0.9988 ± 0.0016 |
| **XGBoost** 🥉 | 0.9742 ± 0.0060 | 0.9640 ± 0.0085 | 0.9770 ± 0.0158 | 0.9717 ± 0.0135 | 0.9926 ± 0.0061 |

> [!IMPORTANT]
> Os resultados de **Cross-Validation confirmam o ranking** do test set. CatBoost e LightGBM são muito próximos (diferença de 0.08% no F1), enquanto XGBoost fica ~1.7% abaixo.

---

## 3. Matriz de Confusão (Test Set)

```
                 CatBoost          LightGBM          XGBoost
               Pred 0  Pred 1    Pred 0  Pred 1    Pred 0  Pred 1
Real 0 (52)      52      0         51      1         51      1
Real 1 (120)      0    120          0    120          6    114
```

| Modelo | TN | FP | FN | TP |
|--------|:--:|:--:|:--:|:--:|
| CatBoost | 52 | 0 | 0 | 120 |
| LightGBM | 51 | 1 | 0 | 120 |
| XGBoost | 51 | 1 | 6 | 114 |

> [!WARNING]
> O XGBoost teve **6 falsos negativos** (alunos em risco classificados como sem risco). Para um sistema de detecção de risco de evasão, o Recall de 100% do CatBoost e LightGBM é preferível — nenhum aluno em risco deixa de ser identificado.

---

## 4. Top 10 Features Mais Importantes

### CatBoost
| # | Feature | Importância |
|---|---------|:-----------:|
| 1 | Idade 22 | 49.51 |
| 2 | Fase_encoded | 29.75 |
| 3 | Nº Av | 5.19 |
| 4 | Cf | 5.15 |
| 5 | INDE 22 | 1.67 |
| 6 | IPV | 1.01 |
| 7 | Portug | 0.84 |
| 8 | IDA | 0.80 |
| 9 | Rec_av2_encoded | 0.72 |
| 10 | Ratio_IDA_IEG | 0.69 |

### LightGBM
| # | Feature | Importância |
|---|---------|:-----------:|
| 1 | Idade 22 | 379 |
| 2 | Fase_encoded | 252 |
| 3 | Cf | 127 |
| 4 | IPV | 103 |
| 5 | INDE 22 | 91 |
| 6 | Ratio_IDA_IEG | 88 |
| 7 | Variancia_indicadores | 74 |
| 8 | IDA | 64 |
| 9 | Portug | 60 |
| 10 | Matem | 59 |

### XGBoost
| # | Feature | Importância |
|---|---------|:-----------:|
| 1 | Nº Av | 0.1113 |
| 2 | Idade 22 | 0.0754 |
| 3 | Fase_encoded | 0.0751 |
| 4 | Indicado_flag | 0.0707 |
| 5 | Cf | 0.0673 |
| 6 | Rec_av2_encoded | 0.0601 |
| 7 | Tem_nota_ingles | 0.0525 |
| 8 | Escola_encoded | 0.0497 |
| 9 | INDE 22 | 0.0458 |
| 10 | Variancia_indicadores | 0.0434 |

> [!NOTE]
> As escalas de importância são diferentes entre modelos (CatBoost usa *prediction value change*, LightGBM usa *split count*, XGBoost usa *gain fraction*), mas as **features mais relevantes são consistentes**:
> - **Idade 22** e **Fase_encoded** dominam em todos os modelos
> - **Cf**, **INDE 22** e **IPV** aparecem no top 6 de todos
> - **Nº Av** é mais valorizada pelo XGBoost do que pelos outros

---

## 5. Análise e Recomendações

### Ranking Final

| Posição | Modelo | Pontos Fortes | Pontos Fracos |
|:-------:|--------|---------------|---------------|
| 🥇 | **CatBoost** | Melhor desempenho geral, 100% recall, robusto a NaN, codificação categórica nativa | Pode indicar leve overfitting (100% test set), mais lento que LightGBM |
| 🥈 | **LightGBM** | Muito próximo do CatBoost, mais rápido, 100% recall | 1 falso positivo |
| 🥉 | **XGBoost** | Robusto e bem estabelecido, boa calibração | 6 falsos negativos, recall menor |
| — | **TabPFN** | Ideal para datasets pequenos, sem tuning necessário | Requer autenticação HuggingFace, dependência do PyTorch |

### Recomendação

Para o caso de uso de **detecção de risco de evasão escolar**, onde o custo de um falso negativo (não identificar um aluno em risco) é alto:

- **Produção:** Usar **CatBoost** como modelo principal (melhor recall + precisão)
- **Backup:** **LightGBM** como alternativa rápida com desempenho quase idêntico
- **Monitorar:** Ficar atento a overfitting do CatBoost conforme novos dados entram — o CV (F1=0.9918) confirma boa generalização

---

## 6. Modelos Disponíveis na API

Todos os modelos estão disponíveis para treinamento via API:

```json
POST /train
{
  "model_type": "catboost",  // ou "xgboost", "lightgbm", "tabpfn"
  "optimize": false
}
```

| Tipo | Status | Model ID |
|------|--------|----------|
| xgboost | ✅ Operacional | `xgb_20260216_101754` |
| catboost | ✅ Operacional | `cat_20260216_101759` |
| lightgbm | ✅ Operacional | `lgb_20260216_101808` |
| tabpfn | ⚠️ Requer HF Auth | — |
