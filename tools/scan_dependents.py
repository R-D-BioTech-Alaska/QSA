from __future__ import annotations
import argparse
import re
from pathlib import Path

PATTERNS = {
    "canonical-python": re.compile(r"\b(?:from\s+qsa\s+import|import\s+qsa\b)"),
    "legacy-python": re.compile(r"\bqubit_native\b"),
    "register-api": re.compile(r"\b(?:QubitRegister|QRegister)\b"),
    "qsc": re.compile(r"\b(?:encode_qsc|decode_qsc|save_qsc|load_qsc)\b"),
    "native-env": re.compile(r"\b(?:QSA_NATIVE_LIB|QUBIT_NATIVE_LIB)\b"),
    "c-api": re.compile(r"\bqstate_[A-Za-z0-9_]+\b"),
}

SKIP_PARTS = {".git", "build", "dist", ".venv", "venv", "node_modules", "__pycache__"}
TEXT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".py", ".md", ".toml",
    ".yml", ".yaml", ".json", ".txt", ".ini", ".cfg", ".cmake", ".ps1", ".sh",
}

def scan(root: Path) -> int:
    matches = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != "CMakeLists.txt":
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, 1):
            labels = [label for label, pattern in PATTERNS.items() if pattern.search(line)]
            if labels:
                print(f"{root.name}:{path.relative_to(root)}:{line_number}:"
                      f"{','.join(labels)}:{line.strip()}")
                matches += 1
    return matches

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    total = 0
    for root in args.roots:
        if not root.is_dir():
            parser.error(f"not a directory: {root}")
        total += scan(root.resolve())
    print(f"QSA dependency scan complete: {total} matching lines")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
