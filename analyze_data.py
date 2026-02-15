#!/usr/bin/env python3
"""
Análise detalhada do dataset PEDE 2024
Identifica: campos comuns, valores nulos, e recomendações
"""
import pandas as pd
import numpy as np

# Ler o CSV (vou usar o antigo primeiro para testar, depois adaptamos para o Excel 2024)
file_path = 'DATATHON-20260215T114121Z-3-001/DATATHON/Bases antigas/PEDE_PASSOS_DATASET_FIAP.csv'
df = pd.read_csv(file_path, delimiter=';')

print("="*80)
print("ANÁLISE DO DATASET PEDE - PASSOS MÁGICOS")
print("="*80)
print(f"\n📊 Total de registros: {len(df)}")
print(f"📊 Total de colunas: {len(df.columns)}")

# Separar colunas por ano
cols_2020 = [col for col in df.columns if '2020' in col]
cols_2021 = [col for col in df.columns if '2021' in col]
cols_2022 = [col for col in df.columns if '2022' in col]
cols_sem_ano = [col for col in df.columns if not any(year in col for year in ['2020', '2021', '2022'])]

print(f"\n📅 Colunas 2020: {len(cols_2020)}")
print(f"📅 Colunas 2021: {len(cols_2021)}")
print(f"📅 Colunas 2022: {len(cols_2022)}")
print(f"📅 Colunas sem ano (ex: NOME): {len(cols_sem_ano)}")

print("\n" + "="*80)
print("1️⃣ IDENTIFICANDO CAMPOS COMUNS ENTRE OS ANOS")
print("="*80)

# Extrair prefixos base (removendo o ano)
def get_base_field(col):
    """Remove o ano do nome da coluna para encontrar o campo base"""
    for year in ['_2020', '_2021', '_2022']:
        col = col.replace(year, '')
    return col

# Campos base de cada ano
base_2020 = set([get_base_field(col) for col in cols_2020])
base_2021 = set([get_base_field(col) for col in cols_2021])
base_2022 = set([get_base_field(col) for col in cols_2022])

# Campos comuns a todos os anos
common_fields = base_2020 & base_2021 & base_2022

print(f"\n✅ Campos comuns aos 3 anos ({len(common_fields)}):")
for field in sorted(common_fields):
    print(f"   - {field}")

print("\n" + "="*80)
print("2️⃣ ANÁLISE DE VALORES NULOS POR CAMPO COMUM")
print("="*80)

null_analysis = []
for field in sorted(common_fields):
    col_2020 = f"{field}_2020" if f"{field}_2020" in df.columns else None
    col_2021 = f"{field}_2021" if f"{field}_2021" in df.columns else None
    col_2022 = f"{field}_2022" if f"{field}_2022" in df.columns else None

    nulls_2020 = df[col_2020].isnull().sum() / len(df) * 100 if col_2020 else 0
    nulls_2021 = df[col_2021].isnull().sum() / len(df) * 100 if col_2021 else 0
    nulls_2022 = df[col_2022].isnull().sum() / len(df) * 100 if col_2022 else 0

    avg_nulls = (nulls_2020 + nulls_2021 + nulls_2022) / 3

    null_analysis.append({
        'Campo': field,
        'Nulls_2020_%': nulls_2020,
        'Nulls_2021_%': nulls_2021,
        'Nulls_2022_%': nulls_2022,
        'Media_Nulls_%': avg_nulls
    })

null_df = pd.DataFrame(null_analysis).sort_values('Media_Nulls_%')
print(null_df.to_string(index=False))

print("\n" + "="*80)
print("3️⃣ CAMPOS RECOMENDADOS (< 50% nulos)")
print("="*80)

recommended = null_df[null_df['Media_Nulls_%'] < 50]
print(f"\n✅ {len(recommended)} campos recomendados para ML:\n")
for idx, row in recommended.iterrows():
    print(f"   {row['Campo']:20s} - Média de nulos: {row['Media_Nulls_%']:5.1f}%")

print("\n" + "="*80)
print("4️⃣ ANÁLISE DE DEFASAGEM (TARGET VARIABLE)")
print("="*80)

# Análise da DEFASAGEM
print("\n📊 Estatísticas de DEFASAGEM por ano:\n")
for year in [2021, 2022]:  # DEFASAGEM não existe em 2020
    col = f'DEFASAGEM_{year}'
    if col in df.columns:
        data = df[col].dropna()
        print(f"  {year}:")
        print(f"    - Registros não-nulos: {len(data)}")
        print(f"    - Média: {data.mean():.2f}")
        print(f"    - Min: {data.min():.0f} | Max: {data.max():.0f}")
        print(f"    - Distribuição:")
        print(f"      Negativos (atrasados): {(data < 0).sum()} ({(data < 0).sum()/len(data)*100:.1f}%)")
        print(f"      Zero (no ritmo):       {(data == 0).sum()} ({(data == 0).sum()/len(data)*100:.1f}%)")
        print(f"      Positivos (adiantados):{(data > 0).sum()} ({(data > 0).sum()/len(data)*100:.1f}%)")
        print()

print("\n" + "="*80)
print("5️⃣ TIPOS DE DADOS E NECESSIDADE DE NORMALIZAÇÃO")
print("="*80)

# Analisar tipos de dados dos campos recomendados
print("\n📋 Análise dos campos recomendados:\n")

field_types = {}
for field in recommended['Campo']:
    col_2022 = f"{field}_2022" if f"{field}_2022" in df.columns else None
    if col_2022 and col_2022 in df.columns:
        sample_data = df[col_2022].dropna()
        if len(sample_data) > 0:
            dtype = sample_data.dtype
            if dtype in ['float64', 'int64']:
                print(f"  {field:20s}")
                print(f"    Tipo: NUMÉRICO (requer normalização)")
                print(f"    Range: {sample_data.min():.2f} - {sample_data.max():.2f}")
                print(f"    Desvio padrão: {sample_data.std():.2f}")
            else:
                unique_vals = sample_data.nunique()
                print(f"  {field:20s}")
                print(f"    Tipo: CATEGÓRICO (requer encoding)")
                print(f"    Valores únicos: {unique_vals}")
                if unique_vals < 10:
                    print(f"    Valores: {list(sample_data.unique()[:5])}")
            print()

print("\n" + "="*80)
print("6️⃣ RECOMENDAÇÕES FINAIS")
print("="*80)

print("""
✅ CAMPOS SELECIONADOS PARA O MODELO:

📈 FEATURES NUMÉRICAS (normalizar com StandardScaler ou MinMaxScaler):
   - INDE (Índice geral de desempenho)
   - IAA (Auto-avaliação)
   - IEG (Engajamento)
   - IPS (Psicossocial)
   - IDA (Desempenho acadêmico)
   - IPP (Psicopedagógico)
   - IPV (Ponto de virada)
   - IAN (Adequação de nível)

📊 FEATURES CATEGÓRICAS (encoding com LabelEncoder ou OneHotEncoder):
   - PEDRA (Ametista, Ágata, Quartzo, Topázio)
   - PONTO_VIRADA (Sim/Não)
   - Instituição (se disponível)

🎯 TARGET VARIABLE:
   - Criar: DELTA_DEFASAGEM = DEFASAGEM_2022 - DEFASAGEM_2021
   - Ou classificar em: Melhorou (-1), Manteve (0), Piorou (+1)

⚠️  CAMPOS A EVITAR (> 50% nulos ou não informativos):
   - DESTAQUE_* (texto livre, difícil de processar)
   - REC_EQUIPE_* (muitos nulos, categórico com muitas classes)
   - Notas específicas (NOTA_PORT, NOTA_MAT) se houver muitos nulos

🔄 ESTRATÉGIAS DE LIMPEZA:
   1. Filtrar apenas alunos com dados em 2+ anos consecutivos
   2. Imputar nulos com média/mediana para features numéricas
   3. Criar flag "missing" para features com muitos nulos
   4. Remover outliers extremos (> 3 desvios padrão)
""")

print("\n" + "="*80)
print("✅ Análise concluída!")
print("="*80)
