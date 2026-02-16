"""
Patch de compatibilidade: TabPFN v1 (0.1.11) + PyTorch >= 2.0

O TabPFN v1 importa símbolos de `torch.nn.modules.transformer` que foram
removidos no PyTorch 2.x. Este script corrige o import em `tabpfn/layer.py`.

Execute após `pip install tabpfn==0.1.11`:
    python scripts/patch_tabpfn.py
"""
import importlib
import sys
from pathlib import Path


def patch_tabpfn():
    """Aplica patch no tabpfn/layer.py para compatibilidade com torch>=2.0."""
    try:
        import tabpfn
    except ImportError:
        print("⚠️  tabpfn não está instalado, ignorando patch.")
        return False

    layer_path = Path(tabpfn.__file__).parent / "layer.py"

    if not layer_path.exists():
        print(f"⚠️  {layer_path} não encontrado.")
        return False

    content = layer_path.read_text(encoding="utf-8")

    old_import = (
        "from torch.nn.modules.transformer import "
        "_get_activation_fn, Module, Tensor, Optional, "
        "MultiheadAttention, Linear, Dropout, LayerNorm"
    )

    new_import = (
        "from torch.nn.modules.transformer import _get_activation_fn\n"
        "from typing import Optional\n"
        "from torch import Tensor\n"
        "from torch.nn import Module, MultiheadAttention, Linear, Dropout, LayerNorm"
    )

    if old_import in content:
        content = content.replace(old_import, new_import)
        layer_path.write_text(content, encoding="utf-8")
        print(f"✅ Patch aplicado em {layer_path}")
        return True
    elif "from typing import Optional" in content:
        print("ℹ️  Patch já foi aplicado anteriormente.")
        return True
    else:
        print("⚠️  Import não reconhecido, patch não aplicado.")
        return False


if __name__ == "__main__":
    success = patch_tabpfn()
    sys.exit(0 if success else 1)
