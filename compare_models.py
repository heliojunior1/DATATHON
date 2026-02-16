"""
Script de comparação CatBoost vs XGBoost.

Treina ambos os modelos com os mesmos dados e compara métricas.
"""
import sys
import json
from pathlib import Path

# Adicionar raiz ao path
sys.path.insert(0, str(Path(__file__).parent))

from app.ml.train import run_training_pipeline


def main():
    print("=" * 70)
    print("  COMPARAÇÃO: XGBoost vs CatBoost")
    print("=" * 70)

    results = {}

    for model_type in ["xgboost", "catboost"]:
        print(f"\n{'─' * 70}")
        print(f"  Treinando: {model_type.upper()}")
        print(f"{'─' * 70}")

        result = run_training_pipeline(
            model_type=model_type,
            optimize=False,       # Sem busca de hiperparâmetros (para fair comparison)
            run_cv=True,          # Cross-validation
            run_learning_curves=False,  # Sem learning curves (mais rápido)
        )

        results[model_type] = result
        metrics = result["metrics"]

        print(f"\n  📊 Resultados {model_type.upper()}:")
        print(f"     Model ID:   {result['model_id']}")
        print(f"     Accuracy:   {metrics['accuracy']:.4f}")
        print(f"     F1 Score:   {metrics['f1_score']:.4f}")
        print(f"     Precision:  {metrics['precision']:.4f}")
        print(f"     Recall:     {metrics['recall']:.4f}")
        print(f"     ROC AUC:    {metrics.get('roc_auc', 'N/A')}")

        if result.get("cv_results"):
            cv = result["cv_results"]
            print(f"     CV F1 Mean: {cv.get('mean_f1', 'N/A')}")
            print(f"     CV F1 Std:  {cv.get('std_f1', 'N/A')}")

    # ── Resumo Comparativo ──
    print(f"\n{'=' * 70}")
    print("  RESUMO COMPARATIVO")
    print(f"{'=' * 70}")

    xgb = results["xgboost"]["metrics"]
    cat = results["catboost"]["metrics"]

    print(f"\n  {'Métrica':<20} {'XGBoost':>10} {'CatBoost':>10} {'Δ':>10} {'Vencedor':>10}")
    print(f"  {'─' * 60}")

    for metric_name in ["accuracy", "f1_score", "precision", "recall", "roc_auc"]:
        xgb_val = xgb.get(metric_name, 0)
        cat_val = cat.get(metric_name, 0)
        if isinstance(xgb_val, (int, float)) and isinstance(cat_val, (int, float)):
            delta = cat_val - xgb_val
            winner = "CatBoost" if delta > 0 else ("XGBoost" if delta < 0 else "Empate")
            print(f"  {metric_name:<20} {xgb_val:>10.4f} {cat_val:>10.4f} {delta:>+10.4f} {winner:>10}")

    # CV comparison
    xgb_cv = results["xgboost"].get("cv_results", {})
    cat_cv = results["catboost"].get("cv_results", {})
    if xgb_cv and cat_cv:
        xgb_cv_f1 = xgb_cv.get("mean_f1", 0)
        cat_cv_f1 = cat_cv.get("mean_f1", 0)
        delta = cat_cv_f1 - xgb_cv_f1
        winner = "CatBoost" if delta > 0 else ("XGBoost" if delta < 0 else "Empate")
        print(f"  {'CV F1 (mean)':<20} {xgb_cv_f1:>10.4f} {cat_cv_f1:>10.4f} {delta:>+10.4f} {winner:>10}")

    # Top features comparison
    print(f"\n  Top 5 Features mais importantes:")
    for model_type in ["xgboost", "catboost"]:
        importance = results[model_type].get("feature_importance", [])[:5]
        print(f"\n    {model_type.upper()}:")
        for i, feat in enumerate(importance, 1):
            print(f"      {i}. {feat['feature']:30s} — {feat['importance']:.4f}")

    print(f"\n{'=' * 70}")
    print("  COMPARAÇÃO CONCLUÍDA")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
