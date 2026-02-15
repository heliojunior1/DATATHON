"""
Script de treinamento do modelo.

Uso:
    python train_pipeline.py [--no-optimize] [--no-ian] [--n-iter 50] [--skip-cv] [--skip-learning-curves]
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
        help="Exclui a feature IAN do modelo (já é o padrão)",
    )
    parser.add_argument(
        "--include-ian",
        action="store_true",
        help="Inclui a feature IAN (para comparação — cuidado: data leakage)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=50,
        help="Número de iterações para RandomizedSearchCV (padrão: 50)",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Pula a validação cruzada K-Fold independente",
    )
    parser.add_argument(
        "--skip-learning-curves",
        action="store_true",
        help="Pula a geração do gráfico de learning curves",
    )

    args = parser.parse_args()

    # Determinar include_ian: --include-ian tem prioridade, senão usa config default (False)
    if args.include_ian:
        include_ian = True
    elif args.no_ian:
        include_ian = False
    else:
        include_ian = None  # Usa config.INCLUDE_IAN (default: False)

    results = run_training_pipeline(
        filepath=args.data,
        include_ian=include_ian,
        optimize=not args.no_optimize,
        n_iter=args.n_iter,
        run_cv=not args.skip_cv,
        run_learning_curves=not args.skip_learning_curves,
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

    # Métricas CV
    if results.get("cv_results"):
        print()
        print("  Métricas (Cross-Validation 5-Fold):")
        for metric, values in results["cv_results"].items():
            print(f"    {metric:15s}: {values['mean']:.4f} ± {values['std']:.4f}")

    print()
    print("  Top 5 Features:")
    for i, feat in enumerate(results["feature_importance"][:5], 1):
        print(f"    {i}. {feat['feature']:30s} — {feat['importance']:.4f}")

    # Learning curves
    if results.get("learning_curve_path"):
        print()
        print(f"  Learning Curves: {results['learning_curve_path']}")

    print("=" * 70)


if __name__ == "__main__":
    main()

