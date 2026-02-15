# 🎯 Datathon Passos Mágicos — Previsão de Risco de Defasagem Escolar

API de Machine Learning para estimar o risco de **defasagem escolar** dos estudantes da Associação Passos Mágicos, construída com **XGBoost** e **FastAPI**.

---

## 📋 Visão Geral

### Problema de Negócio
A Associação Passos Mágicos transforma a vida de crianças e jovens em vulnerabilidade social por meio da educação. Este projeto prevê quais alunos estão **em risco de defasagem escolar**, permitindo intervenções educacionais preventivas.

### Solução Proposta
Pipeline completa de Machine Learning usando **XGBoost** como classificador binário:
- **Em Risco** (Defasagem ≤ -2): aluno atrasado 2+ fases
- **Sem Risco** (Defasagem ≥ -1): aluno no nível adequado ou levemente atrasado

### Resultados do Modelo

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 95.93% |
| **Precision** | 96.97% |
| **Recall** | 84.21% |
| **F1-Score** | 90.14% |
| **AUC-ROC** | 98.47% |

### Stack Tecnológica
- **Linguagem**: Python 3.11
- **ML**: scikit-learn, XGBoost, pandas, numpy
- **API**: FastAPI + Uvicorn
- **Serialização**: joblib
- **Testes**: pytest (91 testes, 84% cobertura)
- **Empacotamento**: Docker
- **Monitoramento**: drift detection (PSI, KS-test)

---

## 📁 Estrutura do Projeto

```
datathon/
├── app/
│   ├── api/
│   │   ├── routes.py          # Endpoints FastAPI
│   │   └── schemas.py         # Modelos Pydantic
│   ├── core/
│   │   └── config.py          # Configurações centrais
│   ├── ml/
│   │   ├── preprocessing.py   # Pré-processamento de dados
│   │   ├── feature_engineering.py  # Engenharia de features
│   │   ├── train.py           # Pipeline de treinamento
│   │   ├── evaluate.py        # Métricas de avaliação
│   │   └── predict.py         # Lógica de predição
│   ├── monitoring/
│   │   └── drift.py           # Detecção de data drift
│   ├── utils/
│   │   └── helpers.py         # Utilitários (logging)
│   └── main.py                # Entrada da aplicação FastAPI
├── data/                       # Dataset PEDE 2024
├── models/                     # Modelos serializados (.joblib)
├── tests/                      # Testes unitários (91 testes)
├── train_pipeline.py           # Script CLI de treinamento
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Instruções de Deploy

### Pré-requisitos
- Python 3.11+
- pip

### 1. Configurar Ambiente Virtual (Recomendado)

```bash
# Criar venv
python -m venv venv

# Ativar venv (Windows)
.\venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Treinar o Modelo

```bash
# Treinamento rápido (sem otimização de hiperparâmetros)
python train_pipeline.py --no-optimize

# Treinamento completo (com RandomizedSearchCV)
python train_pipeline.py

# Sem a feature IAN (evitar data leakage)
python train_pipeline.py --no-ian
```

### 3. Iniciar a API

```bash
Eu ```

A documentação interativa estará em: http://localhost:8000/docs


### 4. Analise dos dados
TIER 1 — Usar com confiança (< 30% nulos, alta relevância)
Feature	Tipo	Escala	Nulos	Normalização	Observação
INDE	Numérico	0–10	~20%	StandardScaler	Índice composto principal
IAA	Numérico	0–10	~20%	StandardScaler	Auto-avaliação do aluno
IEG	Numérico	0–10	~20%	StandardScaler	Engajamento (lições de casa)
IPS	Numérico	0–10	~20%	StandardScaler	Psicossocial
IDA	Numérico	0–10	~20%	StandardScaler	Desempenho acadêmico
IPP	Numérico	0–10	~20%	StandardScaler	Psicopedagógico
IPV	Numérico	0–10	~20%	StandardScaler	Avaliação de "ponto de virada"
PEDRA	Ordinal	4 classes	~20%	OrdinalEncoder (1-4)	Hierarquia natural
Idade	Numérico	~8–20	~25%	StandardScaler	Demográfica
Ano ingresso	Numérico	2016–2022	~25%	Derivar Anos_na_PM	Tempo no programa
Gênero	Binário	2 classes	~25%	LabelEncoder (0/1)	Demográfica
TIER 2 — Usar com cautela (30–60% nulos ou risco de leakage)
Feature	Tipo	Nulos	Problema	Recomendação
IAN	Numérico (discreto: 0, 5, 10)	~25%	DATA LEAKAGE — é praticamente sinônimo de defasagem. Domina com 58.4% de importância	REMOVER do modelo principal. IAN mede "adequação ao nível", que é o próprio target
DEFASAGEM	Numérico	~30%	É o target, não feature	Usar só para criar y
Ponto de Virada	Binário	~25%	Pode ser consequência, não causa	Usar com monitoramento
Rec Psicologia	Ordinal (5 níveis)	~40%	Muitos nulos	Imputar como "Não avaliado" (0)
Rec Avaliador 1/2	Ordinal (5 níveis)	~40%	Possível leakage — avaliadores podem ver defasagem	Testar modelo com e sem
TIER 3 — Evitar (> 60% nulos ou irrelevantes)
Feature	Nulos	Por que evitar
NOTA_PORT / NOTA_MAT / NOTA_ING	~70-80%	Quase inútil — tão poucos dados que a imputação por mediana distorce a realidade
Cg, Cf, Ct	~25%	Significado não documentado no dicionário. Rankings internos? Possível leakage
DESTAQUE_IEG/IDA/IPV	Texto livre	Não processável sem NLP
REC_EQUIPE_*	~50%	Muitas categorias, muitos nulos
TURMA	~30%	Identificador, sem valor preditivo
NOME	0%	Identificador pessoal
INDE_CONCEITO	~20%	Redundante — é apenas a faixa do INDE
Nº Av	~25%	Se reflete número de avaliações do mesmo período, pode ser leaker

### 4. Deploy com Docker

```bash
# Build
docker build -t datathon-passos .

# Run
docker run -p 8000:8000 datathon-passos

# Ou com docker-compose
docker-compose up -d
```

---

## 🔌 Exemplos de Chamadas à API

### Health Check

```bash
curl http://localhost:8000/health
```

**Resposta:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "xgboost_defasagem",
  "model_version": "1.0.0"
}
```

### Predição Individual

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "IAA": 7.5,
    "IEG": 8.0,
    "IPS": 6.5,
    "IDA": 7.0,
    "IPV": 5.5,
    "IAN": 5.0,
    "INDE 22": 7.2,
    "Matem": 7.5,
    "Portug": 6.8,
    "Inglês": 7.0,
    "Idade 22": 14,
    "Gênero": "Menina",
    "Instituição de ensino": "Escola Pública",
    "Ano ingresso": 2018,
    "Pedra 22": "Ametista",
    "Rec Psicologia": "Sem limitações",
    "Atingiu PV": "Não",
    "Indicado": "Não",
    "Cg": 300,
    "Cf": 50,
    "Ct": 5,
    "Nº Av": 3
  }'
```

**Resposta:**
```json
{
  "prediction": 0,
  "probability": 0.1234,
  "risk_level": "Muito Baixo",
  "label": "Sem Risco",
  "top_factors": [
    {"feature": "IAN", "importance": 0.5844},
    {"feature": "Nº Av", "importance": 0.0628}
  ]
}
```

### Predição em Lote

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"students": [<aluno1>, <aluno2>, ...]}'
```

### Informações do Modelo

```bash
curl http://localhost:8000/model-info
```

### Monitoramento de Drift

```bash
curl http://localhost:8000/monitoring/drift
curl http://localhost:8000/monitoring/stats
```

---

## 🔬 Pipeline de Machine Learning

### 1. Pré-processamento
- Carregamento do dataset PEDE 2024 (860 alunos × 42 colunas)
- Criação da variável target binária (Defas ≤ -2 → em risco)
- Codificação ordinal de categóricas (Pedras, Gênero, Escola, Rec. Psicologia)
- Tratamento de nulos (mediana para Math/Port, NaN nativo para XGBoost)

### 2. Engenharia de Features
- **31 features** selecionadas
- Evolução temporal das Pedras (2020→2022, 2021→2022)
- Anos na Passos Mágicos
- Flags de destaque (IEG, IDA, IPV)

### 3. Treinamento
- **XGBoost** com `scale_pos_weight` para balanceamento (22% em risco)
- Validação cruzada estratificada (5-fold)
- `RandomizedSearchCV` com 50 iterações

### 4. Avaliação
- Métrica primária: **F1-Score** (equilíbrio entre precisão e recall)
- Priorização do **Recall** (evitar falsos negativos — não perder alunos em risco)

### 5. Top Features

| # | Feature | Importância |
|---|---------|------------|
| 1 | IAN (Adequação ao Nível) | 58.44% |
| 2 | Nº Avaliações | 6.28% |
| 3 | Idade | 4.34% |
| 4 | Pedra 2020 | 3.53% |
| 5 | Rec. Avaliador 2 | 2.64% |

---

## 🧪 Testes

```bash
# Executar todos os testes
pytest tests/ -v

# Com cobertura
pytest tests/ --cov=app --cov-report=term-missing

# Verificar cobertura mínima de 80%
pytest tests/ --cov=app --cov-fail-under=80
```

**Resultado atual**: 91 testes, 84% de cobertura.

---

## 📊 Monitoramento

A API inclui monitoramento contínuo de **data drift**:

- **PSI (Population Stability Index)**: detecta mudanças na distribuição das features
- **KS-test**: teste estatístico contra distribuição de referência
- **Logs de predição**: todas as predições são registradas para análise

Endpoints:
- `GET /monitoring/drift` — Status de drift por feature
- `GET /monitoring/stats` — Estatísticas das predições

---

## 📄 Licença

Projeto desenvolvido para o Datathon PÓS TECH — Machine Learning Engineering.
