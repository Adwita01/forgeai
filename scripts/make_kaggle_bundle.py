"""Zip the code + demo inputs for upload to Kaggle as a private dataset.

Usage:  python scripts/make_kaggle_bundle.py      -> dist/forgeai_kaggle_bundle.zip

Includes the working tree as it is now (uncommitted edits too). Leaves out caches, previous
run outputs, secrets (.env) and the assignment PDF.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["pyproject.toml", "README.md", ".env.example", "app.py", "src", "prompts", "ui", "scripts", "tests", "demos", "kaggle"]
SKIP_DIRS = {"__pycache__", ".forge_cache", "run", "verify_run", "baseline_comparison", "forge_out", "forge_eval", "comparison"}
SKIP_SUFFIX = {".pyc"}


def wanted(p: Path) -> bool:
    rel = p.relative_to(ROOT)
    if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in rel.parts[:-1]):
        return False
    return p.is_file() and p.suffix not in SKIP_SUFFIX and p.name != ".env"


def main() -> None:
    out = ROOT / "dist" / "forgeai_kaggle_bundle.zip"
    out.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            src = ROOT / item
            files = [src] if src.is_file() else sorted(src.rglob("*")) if src.exists() else []
            for f in files:
                if wanted(f):
                    z.write(f, f.relative_to(ROOT).as_posix())
                    n += 1
    print(f"{out}  ({n} files, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
