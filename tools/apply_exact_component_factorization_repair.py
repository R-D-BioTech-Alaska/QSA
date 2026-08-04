from __future__ import annotations

from pathlib import Path

PATH = Path("src/qstate.cpp")
NEEDLE = """        rest_store = AmplitudeStore::from_entries(
            rest_dimension, std::move(rest_entries), config_);
    }

    StateComponent rest_component;
"""
REPLACEMENT = """        rest_store = AmplitudeStore::from_entries(
            rest_dimension, std::move(rest_entries), config_);
    }

    // The reduced-density determinant is only a fast candidate filter.  A
    // small nonzero determinant can represent weak but real entanglement, so
    // never replace the original state with a product approximation unless
    // the proposed factors reconstruct every amplitude within the engine's
    // exact-state epsilon.
    const double reconstruction_threshold = config_.epsilon * config_.epsilon;
    for (BasisIndex rest_index = 0; rest_index < rest_dimension; ++rest_index) {
        const BasisIndex zero_index = insert_zero_bit(rest_index, local_position_value);
        const BasisIndex one_index = zero_index | mask;
        const QComplex rest_value = rest_store.at(rest_index);
        const QComplex zero_residual =
            cell_amplitudes[0] * rest_value - store.at(zero_index);
        const QComplex one_residual =
            cell_amplitudes[1] * rest_value - store.at(one_index);
        const double zero_error = zero_residual.norm2();
        const double one_error = one_residual.norm2();
        if (!std::isfinite(zero_error) || !std::isfinite(one_error) ||
            zero_error > reconstruction_threshold ||
            one_error > reconstruction_threshold) {
            return std::nullopt;
        }
    }

    StateComponent rest_component;
"""


def main() -> None:
    source = PATH.read_text(encoding="utf-8")
    if REPLACEMENT in source:
        print("exact component-factorization repair is already present")
        return
    if source.count(NEEDLE) != 1:
        raise SystemExit("expected factor_singleton insertion point was not unique")
    PATH.write_text(source.replace(NEEDLE, REPLACEMENT), encoding="utf-8")
    print("inserted exact product-state reconstruction verification")


if __name__ == "__main__":
    main()
