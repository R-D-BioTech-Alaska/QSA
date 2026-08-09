# Release checklist

1. Update `VERSION`, `pyproject.toml`, CMake, native headers, Python version, and
   tagged-install documentation together.
2. Run the version-sync and compatibility-manifest tests.
3. Build portable Release on Linux, macOS, and Windows.
4. Run native, C ABI, Python, plan, Grover, symmetry, QSC, and adversarial tests.
5. Run NumPy, Grover, and symmetry differential validation.
6. Run ASan/UBSan on Linux.
7. Build an sdist and wheel, run metadata checks, install the wheel, and execute
   a canonical plus legacy import smoke test.
8. Install the CMake package and build the independent C and C++ consumer.
9. Rebuild and test from the exact tagged source archive.
10. Record benchmark hardware, compiler, build flags, medians, and raw data.
11. Verify that no build artifacts, caches, private data, credentials, or local
    QSC files are present.
12. Generate SHA-256 checksums for release artifacts.
13. Confirm the default-branch `QSA Build and Test` workflow is green before
    publishing the release tag.