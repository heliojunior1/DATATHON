"""
Script de comparação de todos os modelos implementados.
Gera resultados em JSON para posterior criação do relatório MD.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.ml.train import run_training_pipeline


def main():
    models_to_test = ["xgboost", "catboost", "lightgbm", "tabpfn", "logistic_regression", "svm"]
    all_results = {}

    for model_type in models_to_test:
        print(f"\n{'='*70}")
        print(f"  TREINANDO: {model_type.upper()}")
        print(f"{'='*70}")

        try:
            result = run_training_pipeline(
                model_type=model_type,
                optimize=False,
                run_cv=True,
                run_learning_curves=False,
            )
            all_results[model_type] = {
                "model_id": result["model_id"],
                "metrics": result["metrics"],
                "cv_results": result.get("cv_results"),
                "feature_importance": result.get("feature_importance", [])[:10],
                "n_train": result["n_train"],
                "n_test": result["n_test"],
                "feature_names": result.get("feature_names", []),
            }
            print(f"  ✅ {model_type} concluído: F1={result['metrics']['f1_score']:.4f}")
        except Exception as e:
            print(f"  ❌ {model_type} falhou: {e}")
            all_results[model_type] = {"error": str(e)}

    # Salvar resultados
    output_path = Path("model_comparison_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n\nResultados salvos em {output_path}")

    # Imprimir tabela resumo
    print(f"\n{'='*70}")
    print("  RESUMO COMPARATIVO")
    print(f"{'='*70}")
    print(f"\n  {'Modelo':<15} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'AUC':>10}")
    print(f"  {'─'*65}")

    for model_type in models_to_test:
        r = all_results.get(model_type, {})
        if "error" in r:
            print(f"  {model_type:<15} {'ERRO':>10}")
            continue
        m = r["metrics"]
        print(f"  {model_type:<15} {m.get('accuracy',0):>10.4f} {m.get('f1_score',0):>10.4f} {m.get('precision',0):>10.4f} {m.get('recall',0):>10.4f} {m.get('auc_roc',0):>10.4f}")


if __name__ == "__main__":
    main()
