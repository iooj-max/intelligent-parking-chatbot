"""Pytest configuration: load .env and project root for imports."""

import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

# Suppress Pydantic serializer warnings from structured outputs (LangChain/LangGraph)
warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings.*",
    category=UserWarning,
)

# Add project root and src so `src.config` and `parking_agent` are importable
_root = Path(__file__).resolve().parents[1]
_src = _root / "src"
for path in (_root, _src):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

_env = _root / ".env"
if _env.exists():
    load_dotenv(dotenv_path=_env, override=False)
