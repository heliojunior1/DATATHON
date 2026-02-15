"""
Script de treinamento do modelo.

Uso:
    python train_pipeline.py [--no-optimize] [--no-ian] [--n-iter 50]
"""
import sys
import argparse
from pathlib import Path

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ml.train import run_training_pipeline
from app.utils.helpers import setup_logger

logger = setup_logger(__name__, log_file="training.log")


def main():
    parser = argparse.ArgumentParser(
        description="Treina o modelo de previsão de defasagem escolar"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Caminho do dataset Excel (padrão: data/BASE DE DADOS PEDE 2024 - DATATHON.xlsx)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Desabilita busca de hiperparâmetros (treinamento rápido)",
    )
    parser.add_argument(
        "--no-ian",
        action="store_true",
        help="Exclui a feature IAN do modelo (evita data leakage)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=50,
        help="Número de iterações para RandomizedSearchCV (padrão: 50)",
    )

    args = parser.parse_args()

    results = run_training_pipeline(
        filepath=args.data,
        include_ian=not args.no_ian,
        optimize=not args.no_optimize,
        n_iter=args.n_iter,
    )

    # Resumo final
    print("\n" + "=" * 70)
    print("  RESUMO DO TREINAMENTO")
    print("=" * 70)
    print(f"  Modelo: {results['model_name']} v{results['model_version']}")
    print(f"  Amostras: {results['n_train']} treino + {results['n_test']} teste")
    print(f"  Features: {len(results['feature_names'])}")
    print()
    print("  Métricas (conjunto de teste):")
    for metric, value in results["metrics"].items():
        print(f"    {metric:15s}: {value:.4f}")
    print()
    print("  Top 5 Features:")
    for i, feat in enumerate(results["feature_importance"][:5], 1):
        print(f"    {i}. {feat['feature']:30s} — {feat['importance']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
