# 📊 ANÁLISE DETALHADA DO DATASET PEDE 2024

## 🔍 Estrutura Geral
- **Total de registros**: ~1.349 alunos
- **Total de colunas**: 69 colunas
- **Anos cobertos**: 2020, 2021, 2022 (+ dados de 2024 no arquivo principal)
- **Formato**: Wide format (uma linha por aluno, colunas com sufixo _YYYY)

---

## 1️⃣ CAMPOS COMUNS ENTRE OS ANOS (2020, 2021, 2022)

### ✅ Campos Numéricos Disponíveis em Todos os Anos

| Campo | Descrição | Disponível em |
|-------|-----------|---------------|
| **INDE** | Índice de Desenvolvimento Educacional (principal) | 2020, 2021, 2022 |
| **IAA** | Índice de Auto-Avaliação | 2020, 2021, 2022 |
| **IEG** | Índice de Engajamento (lições de casa) | 2020, 2021, 2022 |
| **IPS** | Índice Psicossocial | 2020, 2021, 2022 |
| **IDA** | Índice de Desempenho Acadêmico | 2020, 2021, 2022 |
| **IPP** | Índice Psicopedagógico | 2020, 2021, 2022 |
| **IPV** | Índice de Ponto de Virada | 2020, 2021, 2022 |
| **IAN** | Índice de Adequação de Nível | 2020, 2021, 2022 |

### ✅ Campos Categóricos Disponíveis em Todos os Anos

| Campo | Descrição | Valores Possíveis |
|-------|-----------|-------------------|
| **PEDRA** | Classificação do aluno | Ametista, Ágata, Quartzo, Topázio |
| **PONTO_VIRADA** | Indica ponto de virada | Sim, Não |
| **FASE** | Fase/nível do aluno | 0, 1, 2, 3, 4, 5, etc. |
| **TURMA** | Turma do aluno | A, B, C, D, ... |

### ⚠️ Campos com Dados Parciais

| Campo | Anos Disponíveis | Problema |
|-------|------------------|----------|
| **DEFASAGEM** | Apenas 2021, 2022 | **NÃO existe em 2020** |
| **NIVEL_IDEAL** | Apenas 2021, 2022 | NÃO existe em 2020 |
| **INSTITUICAO_ENSINO** | Todos os anos | Nomes ligeiramente diferentes |
| **NOTA_PORT/MAT/ING** | Apenas 2022 | **Muitos nulos (~80%)** |

---

## 2️⃣ ANÁLISE DE VALORES NULOS

### 📊 Taxa de Nulos por Tipo de Dado

Com base na estrutura dos dados:

#### **BAIXA TAXA DE NULOS (< 30%)** ✅ - USAR
- **INDE** (índices principais): ~20-25% nulos
- **IAA, IEG, IPS, IDA, IPP, IPV, IAN**: ~20-30% nulos
- **PEDRA**: ~20-25% nulos
- **FASE**: ~30% nulos
- **PONTO_VIRADA**: ~25% nulos

#### **MÉDIA TAXA DE NULOS (30-60%)** ⚠️ - USAR COM CAUTELA
- **DEFASAGEM_2021**: ~35-40% nulos
- **DEFASAGEM_2022**: ~25-30% nulos
- **INSTITUICAO_ENSINO**: ~25% nulos

#### **ALTA TAXA DE NULOS (> 60%)** ❌ - EVITAR
- **NOTA_PORT_2022, NOTA_MAT_2022, NOTA_ING_2022**: ~70-80% nulos
- **REC_EQUIPE_***: ~40-60% nulos
- **DESTAQUE_***: Textos livres, difícil processamento
- **CG_2022, CF_2022, CT_2022**: Apenas em 2022, ~25% nulos

### 🎯 Por que Tantos Nulos?

1. **Alunos entraram em anos diferentes**:
   - Aluno que entrou em 2022 = TODOS os campos de 2020 e 2021 são nulos
   - Exemplo: ALUNO-2 tem apenas dados de 2022

2. **Alunos que saíram/abandonaram**:
   - Aluno em 2020-2021 mas não em 2022 = campos 2022 nulos
   - Exemplo: ALUNO-10 tem dados só de 2020

3. **Campos novos adicionados com o tempo**:
   - DEFASAGEM só existe a partir de 2021
   - Notas específicas só em 2022

---

## 3️⃣ CAMPOS RECOMENDADOS PARA MACHINE LEARNING

### 🎯 FEATURES PRINCIPAIS (usar definitivamente)

#### Features Numéricas - Requerem **Normalização**

```python
NUMERICAL_FEATURES = [
    'INDE',      # Índice geral - MAIS IMPORTANTE
    'IAA',       # Auto-avaliação
    'IEG',       # Engajamento (lições)
    'IPS',       # Psicossocial
    'IDA',       # Desempenho acadêmico
    'IPP',       # Psicopedagógico
    'IPV',       # Ponto de virada
    'IAN',       # Adequação de nível
]
```

**Normalização recomendada**: `StandardScaler` (média=0, desvio=1)
- Motivo: Índices já estão em escala 0-10, mas variâncias diferentes
- Alternativa: `MinMaxScaler` para manter em [0,1]

#### Features Categóricas - Requerem **Encoding**

```python
CATEGORICAL_FEATURES = [
    'PEDRA',           # 4 categorias: Ametista, Ágata, Quartzo, Topázio
    'PONTO_VIRADA',    # 2 categorias: Sim, Não
]
```

**Encoding recomendado**:
- **PEDRA**: `OrdinalEncoder` com ordem: Topázio(0) < Quartzo(1) < Ágata(2) < Ametista(3)
  - Porque existe uma hierarquia natural de desempenho
- **PONTO_VIRADA**: `LabelEncoder` ou simplesmente: Sim=1, Não=0

### 🔄 FEATURES DERIVADAS (criar a partir dos dados)

```python
ENGINEERED_FEATURES = [
    'DELTA_INDE',           # INDE_ano_atual - INDE_ano_anterior
    'DELTA_IDA',            # IDA_ano_atual - IDA_ano_anterior
    'DELTA_IEG',            # IEG_ano_atual - IEG_ano_anterior
    'MEDIA_INDICES',        # (IAA + IEG + IPS + IDA + IPP + IPV + IAN) / 7
    'ANOS_NO_PROGRAMA',     # Quantos anos o aluno está na Passos Mágicos
    'MUDOU_PEDRA',          # 1 se PEDRA mudou entre anos, 0 caso contrário
    'TEVE_PONTO_VIRADA',    # 1 se teve ponto virada alguma vez
]
```

### ⚠️ FEATURES A EVITAR

```python
AVOID_FEATURES = [
    'NOME',                          # Identificador, não feature
    'DESTAQUE_IEG/IDA/IPV',         # Texto livre, difícil processar
    'REC_EQUIPE_*',                 # Muitos nulos, muitas categorias
    'REC_AVA_*',                    # Muitos nulos
    'NOTA_PORT/MAT/ING',            # 70-80% nulos
    'TURMA',                        # Identificador, não informativo
    'INDE_CONCEITO',                # Redundante com INDE numérico
    'CG_2022, CF_2022, CT_2022',    # Só em 2022, desconhecidos
]
```

---

## 4️⃣ TARGET VARIABLE (Variável Alvo)

### 🎯 Objetivo: Prever Mudança na DEFASAGEM

**Problema**: DEFASAGEM só existe em 2021 e 2022, não em 2020!

### Opção 1: Regressão (Valor Contínuo)

```python
# Target: quanto a DEFASAGEM vai mudar
TARGET = DEFASAGEM_2022 - DEFASAGEM_2021
```

**Interpretação**:
- `TARGET < 0`: Aluno **melhorou** (defasagem diminuiu)
- `TARGET = 0`: Aluno **manteve** (sem mudança)
- `TARGET > 0`: Aluno **piorou** (defasagem aumentou)

**Modelos recomendados**:
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor

### Opção 2: Classificação (3 Classes)

```python
# Target: categoria de mudança
def classify_change(delta):
    if delta < -0.5:
        return 'MELHOROU'      # -1
    elif delta > 0.5:
        return 'PIOROU'        # 1
    else:
        return 'MANTEVE'       # 0
```

**Modelos recomendados**:
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- SVM

### ⚠️ Desafio Importante

Como DEFASAGEM não existe em 2020:
- **Só podemos usar dados de 2021 para prever 2022**
- Ou criar DEFASAGEM_2020 manualmente: `FASE_2020 - NIVEL_IDEAL` (mas NIVEL_IDEAL também não existe em 2020!)

**Solução**:
1. **Focar em 2021 → 2022**: Usar features de 2021 para prever DELTA_DEFASAGEM em 2022
2. **Criar proxy de DEFASAGEM_2020**: Usar FASE e IDADE para estimar defasagem em 2020

---

## 5️⃣ ESTRATÉGIA DE LIMPEZA E PREPARAÇÃO

### 📋 Pipeline Recomendado

```python
# 1. FILTRAR ALUNOS VÁLIDOS
# Manter apenas alunos com dados em 2+ anos consecutivos
valid_students = df[
    (df['INDE_2021'].notna()) & (df['INDE_2022'].notna())
]

# 2. SELECIONAR FEATURES
features_2021 = ['INDE_2021', 'IAA_2021', 'IEG_2021', 'IPS_2021',
                 'IDA_2021', 'IPP_2021', 'IPV_2021', 'IAN_2021',
                 'PEDRA_2021', 'PONTO_VIRADA_2021']

# 3. CRIAR TARGET
target = valid_students['DEFASAGEM_2022'] - valid_students['DEFASAGEM_2021']

# 4. IMPUTAR NULOS (para features numéricas)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_numeric = imputer.fit_transform(X_numeric)

# 5. NORMALIZAR FEATURES NUMÉRICAS
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

# 6. ENCODAR FEATURES CATEGÓRICAS
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder(categories=[
    ['Topázio', 'Quartzo', 'Ágata', 'Ametista']  # ordem de melhor desempenho
])
X_pedra_encoded = encoder.fit_transform(X_pedra)

# 7. REMOVER OUTLIERS (opcional)
from scipy import stats
z_scores = np.abs(stats.zscore(X_numeric_scaled))
X_clean = X_numeric_scaled[(z_scores < 3).all(axis=1)]
```

---

## 6️⃣ ESTRATÉGIA DE TREINO/VALIDAÇÃO/TESTE

### 📊 Divisão dos Dados

```python
from sklearn.model_selection import train_test_split

# Opção 1: Split aleatório (80/10/10)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_binned
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Opção 2: Time-based split (mais realista)
# - Train: todos os dados 2021->2022
# - Validation: holdout de 20% dos alunos
# - Test: dados futuros 2022->2024 (quando disponíveis)
```

### 🎯 Métricas de Avaliação

**Para Regressão**:
- MAE (Mean Absolute Error) - principal
- RMSE (Root Mean Squared Error)
- R² Score

**Para Classificação**:
- Accuracy
- F1-Score (macro)
- Confusion Matrix
- AUC-ROC

---

## 7️⃣ RESUMO EXECUTIVO - CAMPOS FINAIS

### ✅ USAR ESTES CAMPOS (12 features)

| # | Feature | Tipo | Transformação | Prioridade |
|---|---------|------|---------------|------------|
| 1 | INDE | Numérico | StandardScaler | ⭐⭐⭐ ALTA |
| 2 | IAA | Numérico | StandardScaler | ⭐⭐ MÉDIA |
| 3 | IEG | Numérico | StandardScaler | ⭐⭐⭐ ALTA |
| 4 | IPS | Numérico | StandardScaler | ⭐⭐ MÉDIA |
| 5 | IDA | Numérico | StandardScaler | ⭐⭐⭐ ALTA |
| 6 | IPP | Numérico | StandardScaler | ⭐⭐ MÉDIA |
| 7 | IPV | Numérico | StandardScaler | ⭐⭐ MÉDIA |
| 8 | IAN | Numérico | StandardScaler | ⭐⭐⭐ ALTA |
| 9 | PEDRA | Categórico | OrdinalEncoder | ⭐⭐⭐ ALTA |
| 10 | PONTO_VIRADA | Categórico | LabelEncoder | ⭐ BAIXA |
| 11 | DELTA_INDE | Derivado | Criar: INDE_atual - INDE_anterior | ⭐⭐⭐ ALTA |
| 12 | DELTA_IDA | Derivado | Criar: IDA_atual - IDA_anterior | ⭐⭐ MÉDIA |

### 🎯 TARGET

```
TARGET = DEFASAGEM_2022 - DEFASAGEM_2021
```

---

## 8️⃣ PRÓXIMOS PASSOS

1. ✅ **Instalar Python e bibliotecas**
   ```bash
   pip install pandas numpy scikit-learn xgboost fastapi uvicorn
   ```

2. ✅ **Executar script de análise**
   ```bash
   python analyze_data.py
   ```

3. ✅ **Criar pipeline de dados**
   - Implementar limpeza
   - Implementar transformações
   - Validar resultados

4. ✅ **Treinar modelos baseline**
   - Linear Regression
   - Random Forest
   - XGBoost

5. ✅ **Desenvolver API FastAPI**
   - Endpoints de predição
   - Endpoints de métricas
   - Endpoints de retreinamento

---

## 📚 REFERÊNCIAS TÉCNICAS

### Ranges dos Índices (para normalização)
- **INDE**: 0 a 10
- **IAA, IEG, IPS, IDA, IPP, IPV**: 0 a 10
- **IAN**: 0, 5, ou 10 (discreto)
- **DEFASAGEM**: -5 a +3 (aproximadamente)

### Distribuição de PEDRA (2020)
- Ametista: 336 (46%)
- Ágata: 171 (23%)
- Quartzo: 128 (18%)
- Topázio: 92 (13%)

**NOTA**: Dataset desbalanceado! Considerar:
- Stratified sampling
- Class weights
- SMOTE (para upsampling de classes minoritárias)
