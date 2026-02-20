#!/usr/bin/env python3
"""
CLI para executar a pipeline de treinamento.

Uso:
    python train_pipeline.py
    python train_pipeline.py --no-optimize
    python train_pipeline.py --include-ian
    python train_pipeline.py --model-type xgboost --features IAA,IEG,IPS,IDA
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ml.train import run_training_pipeline
from app.core.config import AVAILABLE_FEATURES


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de treinamento — Datathon Passos Mágicos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Caminho do arquivo Excel (.xlsx). Se não fornecido, usa o padrão.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="xgboost",
        choices=["xgboost"],  # Fase 2 adicionará: lightgbm, logistic_regression, svm, stacking, tabnet
        help="Tipo de modelo a treinar (default: xgboost)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Features a usar, separadas por vírgula (default: todas). Ex: IAA,IEG,IPS,IDA",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Treinar sem busca de hiperparâmetros (mais rápido).",
    )
    parser.add_argument(
        "--include-ian",
        action="store_true",
        help="Incluir a feature IAN (⚠️ possível data leakage).",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=50,
        help="Número de iterações para RandomizedSearchCV (default: 50).",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Pular validação cruzada K-Fold.",
    )
    parser.add_argument(
        "--no-learning-curves",
        action="store_true",
        help="Pular geração de learning curves.",
    )
    parser.add_argument(
        "--use-feature-store",
        action="store_true",
        help="Ativar ingestão/materialização no Feature Store (Feast).",
    )
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="Listar todas as features disponíveis e sair.",
    )

    args = parser.parse_args()

    # Listar features disponíveis
    if args.list_features:
        print("\n📋 Features disponíveis para seleção:\n")
        categories = {}
        for name, meta in AVAILABLE_FEATURES.items():
            cat = meta["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((name, meta))
        for cat, feats in categories.items():
            print(f"  {cat}:")
            for name, meta in feats:
                default = "✅" if meta["default"] else "  "
                print(f"    {default} {name:30s} — {meta['description']}")
        print(f"\nTotal: {len(AVAILABLE_FEATURES)} features")
        return

    # Parsear features selecionadas
    selected_features = None
    if args.features:
        selected_features = [f.strip() for f in args.features.split(",") if f.strip()]

    print("\n" + "=" * 70)
    print("  🚀 PIPELINE DE TREINAMENTO — DATATHON PASSOS MÁGICOS")
    print("=" * 70)
    print(f"  Modelo:      {args.model_type}")
    print(f"  Features:    {len(selected_features) if selected_features else 'todas'}")
    print(f"  Otimização:  {'Sim' if not args.no_optimize else 'Não'}")
    print(f"  Include IAN: {'Sim' if args.include_ian else 'Não'}")
    print(f"  CV K-Fold:   {'Sim' if not args.no_cv else 'Não'}")
    print(f"  L. Curves:   {'Sim' if not args.no_learning_curves else 'Não'}")
    print(f"  Feat. Store: {'Sim' if args.use_feature_store else 'Não'}")
    print("=" * 70 + "\n")

    results = run_training_pipeline(
        filepath=args.dataset,
        model_type=args.model_type,
        selected_features=selected_features,
        include_ian=args.include_ian,
        optimize=not args.no_optimize,
        n_iter=args.n_iter,
        run_cv=not args.no_cv,
        run_learning_curves=not args.no_learning_curves,
        use_feature_store=args.use_feature_store,
    )

    # Resumo
    metrics = results.get("metrics", {})
    print("\n" + "=" * 70)
    print("  📊 RESULTADOS")
    print("=" * 70)
    print(f"  Model ID:    {results['model_id']}")
    print(f"  Model Type:  {results['model_type']}")
    print(f"  Features:    {len(results['feature_names'])}")
    print(f"  Treino:      {results['n_train']} amostras")
    print(f"  Teste:       {results['n_test']} amostras")
    print(f"  Accuracy:    {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision:   {metrics.get('precision', 0):.4f}")
    print(f"  Recall:      {metrics.get('recall', 0):.4f}")
    print(f"  F1-Score:    {metrics.get('f1_score', 0):.4f}")
    print(f"  AUC-ROC:     {metrics.get('auc_roc', 0):.4f}")

    if results.get("cv_results"):
        cv = results["cv_results"]
        if cv.get("f1_score"):
            print(f"  CV F1:       {cv['f1_score']['mean']:.4f} ± {cv['f1_score']['std']:.4f}")

    print("=" * 70)
    print(f"  ✅ Modelo salvo: {results['model_id']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
