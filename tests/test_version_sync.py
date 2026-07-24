"""Ensure every release-facing version source agrees."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def capture(path: str, pattern: str) -> str:
    text = (ROOT / path).read_text(encoding="utf-8")
    match = re.search(pattern, text, re.MULTILINE)
    assert match, f"version pattern missing in {path}"
    return match.group(1)


def main() -> None:
    expected = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    versions = {
        "pyproject.toml": capture("pyproject.toml", r'^version\s*=\s*"([^"]+)"'),
        "CMakeLists.txt": capture(
            "CMakeLists.txt", r"project\(QubitNativeStateEngine VERSION ([0-9.]+)"
        ),
        "include/qubit/version.h": capture(
            "include/qubit/version.h", r'#define QSTATE_VERSION_STRING "([^"]+)"'
        ),
        "python/qsa/__init__.py": capture(
            "python/qsa/__init__.py", r'__version__\s*=\s*"([^"]+)"'
        ),
        "README.md": capture("README.md", r"QSA\.git@v([0-9.]+)"),
        "CITATION.cff": capture("CITATION.cff", r'^version:\s*"([^"]+)"'),
    }
    for source, version in versions.items():
        assert version == expected, f"{source} has {version}, expected {expected}"
    print(f"QSA version sources agree: {expected}")


if __name__ == "__main__":
    main()
