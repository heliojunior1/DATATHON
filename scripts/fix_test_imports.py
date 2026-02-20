import os
from pathlib import Path

def fix_imports():
    tests_dir = Path("tests")
    for p in tests_dir.rglob("*.py"):
        try:
            content = p.read_text(encoding="utf-8")
            if "app.ml" in content or "app.api" in content:
                print(f"Fixing {p}")
                new_content = content.replace("app.ml.model_storage", "app.services.model_storage")
                new_content = new_content.replace("app.api.routes.load_model", "app.routers.prediction.load_model")
                new_content = new_content.replace("app.api.routes.predict", "app.routers.prediction.predict")
                p.write_text(new_content, encoding="utf-8")
        except Exception as e:
            print(f"Error reading {p}: {e}")

if __name__ == "__main__":
    fix_imports()
