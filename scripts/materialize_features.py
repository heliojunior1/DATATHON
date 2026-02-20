#!/usr/bin/env python3
"""
Script CLI para materialização de features no Feature Store.

Uso:
    python scripts/materialize_features.py
    python scripts/materialize_features.py --dataset caminho/para/dados.xlsx
    python scripts/materialize_features.py --incremental
"""
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Garantir que o diretório raiz está no path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.preprocessing import preprocess_dataset
from app.ml.feature_engineering import run_feature_engineering
from feature_store.feature_store_manager import FeatureStoreManager


def main():
    parser = argparse.ArgumentParser(
        description="Materialização de features no Feature Store (Feast + SQLite)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Caminho do arquivo Excel (.xlsx). Se não fornecido, usa o padrão.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Materialização incremental (apenas dados novos).",
    )
    parser.add_argument(
        "--skip-materialize",
        action="store_true",
        help="Pular a materialização (apenas ingerir Parquet e aplicar registry).",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  📦 FEATURE STORE — MATERIALIZAÇÃO DE FEATURES")
    print("=" * 70)

    # 1. Pré-processamento
    print("\n📋 Etapa 1/4: Pré-processamento do dataset...")
    df = preprocess_dataset(args.dataset)
    print(f"   ✅ {df.shape[0]} registros pré-processados")

    # 2. Feature Engineering
    print("\n🔧 Etapa 2/4: Feature Engineering...")
    df = run_feature_engineering(df)
    print(f"   ✅ {df.shape[1]} colunas após feature engineering")

    # 3. Ingestão em Parquet
    print("\n💾 Etapa 3/4: Ingestão de features em Parquet...")
    manager = FeatureStoreManager()
    created_files = manager.ingest_features(df)
    for fv_name, path in created_files.items():
        size_kb = path.stat().st_size / 1024
        print(f"   ✅ {fv_name}: {path.name} ({size_kb:.1f} KB)")

    # 4. Apply + Materialização
    print("\n🏗️  Etapa 4/4: Registro e materialização...")
    manager.apply()
    print("   ✅ Feature Views registradas no Feast registry")

    if not args.skip_materialize:
        if args.incremental:
            manager.materialize_incremental()
            print("   ✅ Materialização incremental concluída (SQLite)")
        else:
            manager.materialize()
            print("   ✅ Materialização completa concluída (SQLite)")
    else:
        print("   ⏭️  Materialização pulada (--skip-materialize)")

    # Resumo
    status = manager.get_status()
    print("\n" + "=" * 70)
    print("  📊 RESUMO")
    print("=" * 70)
    print(f"  Registry:     {'✅' if status['registry_exists'] else '❌'}")
    print(f"  Online Store: {'✅' if status['online_store_exists'] else '❌'}")
    print(f"  Parquet:      {len(status['parquet_files'])} arquivos")
    print(f"  Feature Views: {len(status['feature_views'])} registradas")
    for fv in status.get("feature_views", []):
        print(f"    • {fv['name']}: {len(fv['features'])} features")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
