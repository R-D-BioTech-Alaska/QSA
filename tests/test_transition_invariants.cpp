#include "qubit/qpauli.hpp"
#include "qubit/qstate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::QComplex;
using qubit::QMatrix2;
using qubit::QMatrix4;
using qubit::QRegister;
using qubit::QubitId;

struct SplitMix64 {
    std::uint64_t state;

    [[nodiscard]] std::uint64_t next() noexcept {
        std::uint64_t value = (state += 0x9E3779B97F4A7C15ULL);
        value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
        return value ^ (value >> 31U);
    }

    [[nodiscard]] double unit() noexcept {
        return static_cast<double>(next() >> 11U) * (1.0 / 9007199254740992.0);
    }
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class DenseReference {
public:
    explicit DenseReference(std::size_t qubit_count)
        : amplitudes_(std::size_t{1} << qubit_count) {
        amplitudes_.front() = {1.0, 0.0};
    }

    [[nodiscard]] const std::vector<QComplex>& amplitudes() const noexcept {
        return amplitudes_;
    }

    void apply_single(QubitId qubit, const QMatrix2& matrix) {
        const std::size_t mask = std::size_t{1} << qubit;
        for (std::size_t zero = 0; zero < amplitudes_.size(); ++zero) {
            if ((zero & mask) != 0U) {
                continue;
            }
            const std::size_t one = zero | mask;
            const QComplex zero_value = amplitudes_[zero];
            const QComplex one_value = amplitudes_[one];
            amplitudes_[zero] = matrix(0U, 0U) * zero_value + matrix(0U, 1U) * one_value;
            amplitudes_[one] = matrix(1U, 0U) * zero_value + matrix(1U, 1U) * one_value;
        }
    }

    void apply_two(QubitId first, QubitId second, const QMatrix4& matrix) {
        const std::size_t first_mask = std::size_t{1} << first;
        const std::size_t second_mask = std::size_t{1} << second;
        for (std::size_t base = 0; base < amplitudes_.size(); ++base) {
            if ((base & first_mask) != 0U || (base & second_mask) != 0U) {
                continue;
            }
            const std::array<std::size_t, 4> indices{
                base,
                base | second_mask,
                base | first_mask,
                base | first_mask | second_mask,
            };
            const std::array<QComplex, 4> input{
                amplitudes_[indices[0]],
                amplitudes_[indices[1]],
                amplitudes_[indices[2]],
                amplitudes_[indices[3]],
            };
            for (std::size_t row = 0; row < 4U; ++row) {
                QComplex value{};
                for (std::size_t column = 0; column < 4U; ++column) {
                    value += matrix(row, column) * input[column];
                }
                amplitudes_[indices[row]] = value;
            }
        }
    }

    [[nodiscard]] double probability_one(QubitId qubit) const {
        const std::size_t mask = std::size_t{1} << qubit;
        long double probability = 0.0L;
        for (std::size_t basis = 0; basis < amplitudes_.size(); ++basis) {
            if ((basis & mask) != 0U) {
                probability += static_cast<long double>(amplitudes_[basis].norm2());
            }
        }
        return std::clamp(static_cast<double>(probability), 0.0, 1.0);
    }

    int measure(QubitId qubit, double sample) {
        const int outcome = sample < probability_one(qubit) ? 1 : 0;
        const std::size_t mask = std::size_t{1} << qubit;
        for (std::size_t basis = 0; basis < amplitudes_.size(); ++basis) {
            const int bit = (basis & mask) == 0U ? 0 : 1;
            if (bit != outcome) {
                amplitudes_[basis] = {};
            }
        }
        normalize();
        return outcome;
    }

    void apply_bit_flip_trajectory(QubitId qubit, double probability, double sample) {
        if (sample < probability) {
            apply_single(qubit, qubit::gates::x());
        }
    }

    void apply_phase_flip_trajectory(QubitId qubit, double probability, double sample) {
        if (sample < probability) {
            apply_single(qubit, qubit::gates::z());
        }
    }

    void apply_depolarizing_trajectory(QubitId qubit, double probability, double sample) {
        if (sample >= probability || probability == 0.0) {
            return;
        }
        const double branch = sample / probability;
        if (branch < 1.0 / 3.0) {
            apply_single(qubit, qubit::gates::x());
        } else if (branch < 2.0 / 3.0) {
            apply_single(qubit, qubit::gates::y());
        } else {
            apply_single(qubit, qubit::gates::z());
        }
    }

    void apply_amplitude_damping_trajectory(QubitId qubit, double gamma, double sample) {
        const double jump_probability = gamma * probability_one(qubit);
        QMatrix2 matrix{};
        if (sample < jump_probability) {
            matrix.values[1] = {std::sqrt(gamma), 0.0};
        } else {
            matrix.values[0] = {1.0, 0.0};
            matrix.values[3] = {std::sqrt(1.0 - gamma), 0.0};
        }
        apply_single(qubit, matrix);
        normalize();
    }

    [[nodiscard]] QComplex pauli_expectation(std::span<const PauliFactor> factors) const {
        QComplex result{};
        for (std::size_t basis = 0; basis < amplitudes_.size(); ++basis) {
            std::size_t mapped = basis;
            QComplex phase{1.0, 0.0};
            for (const PauliFactor& factor : factors) {
                const std::size_t mask = std::size_t{1} << factor.qubit;
                const bool bit = (basis & mask) != 0U;
                switch (factor.axis) {
                    case PauliAxis::I:
                        break;
                    case PauliAxis::X:
                        mapped ^= mask;
                        break;
                    case PauliAxis::Y:
                        mapped ^= mask;
                        phase *= bit ? QComplex{0.0, -1.0} : QComplex{0.0, 1.0};
                        break;
                    case PauliAxis::Z:
                        if (bit) {
                            phase *= -1.0;
                        }
                        break;
                }
            }
            result += amplitudes_[mapped].conjugate() * phase * amplitudes_[basis];
        }
        return result;
    }

private:
    std::vector<QComplex> amplitudes_{};

    void normalize() {
        long double norm2 = 0.0L;
        for (const QComplex& value : amplitudes_) {
            norm2 += static_cast<long double>(value.norm2());
        }
        const double norm = std::sqrt(static_cast<double>(norm2));
        require(std::isfinite(norm) && norm > 1e-15, "dense reference normalization failed");
        for (QComplex& value : amplitudes_) {
            value /= norm;
        }
    }
};

void require_equivalent(
    const std::vector<QComplex>& actual,
    const std::vector<QComplex>& expected,
    double tolerance,
    const std::string& label) {
    require(actual.size() == expected.size(), label + ": state sizes differ");

    QComplex phase{1.0, 0.0};
    bool found = false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index].magnitude() > tolerance && expected[index].magnitude() > tolerance) {
            phase = actual[index] / expected[index];
            phase /= phase.magnitude();
            found = true;
            break;
        }
    }
    require(found, label + ": no nonzero amplitude found");

    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!qubit::almost_equal(actual[index], phase * expected[index], tolerance)) {
            throw std::runtime_error(label + ": amplitude mismatch at basis " + std::to_string(index));
        }
    }
}

void verify_state(const QRegister& state, const DenseReference& reference, const std::string& label) {
    std::string reason;
    require(state.validate(&reason), label + ": QRegister validation failed: " + reason);
    require_equivalent(state.materialize(12U), reference.amplitudes(), 2e-10, label);
    for (std::size_t qubit = 0; qubit < state.qubit_count(); ++qubit) {
        const double actual = state.probability_one(static_cast<QubitId>(qubit));
        const double expected = reference.probability_one(static_cast<QubitId>(qubit));
        require(std::abs(actual - expected) <= 2e-10,
                label + ": probability mismatch on qubit " + std::to_string(qubit));
    }
}

void verify_dense_copy_on_write() {
    std::vector<QComplex> amplitudes(64U);
    SplitMix64 generator{0xC0FFEE1234ULL};
    for (QComplex& value : amplitudes) {
        value = {generator.unit() - 0.5, generator.unit() - 0.5};
    }

    QRegister root = QRegister::from_amplitudes(amplitudes);
    const std::vector<QComplex> root_before = root.materialize(6U);
    QRegister branch = root;
    require(root.component_storage_owner_count(0U) >= 2L,
            "dense QRegister copy did not share storage");

    branch.apply_ry(0U, 0.317);
    require(root.component_storage_owner_count(0U) == 1L,
            "dense root did not detach after branch mutation");
    require(branch.component_storage_owner_count(0U) == 1L,
            "dense branch did not detach after mutation");
    require_equivalent(root.materialize(6U), root_before, 1e-12,
                       "dense copy-on-write root changed");
    require(branch.validate(), "dense copy-on-write branch failed validation");
}

void run_transition_sequence(std::uint64_t seed) {
    constexpr std::size_t qubit_count = 5U;
    QRegister state(qubit_count);
    DenseReference reference(qubit_count);
    SplitMix64 generator{seed};

    for (std::size_t step = 0; step < 180U; ++step) {
        const QubitId first = static_cast<QubitId>(generator.next() % qubit_count);
        QubitId second = static_cast<QubitId>(generator.next() % qubit_count);
        if (second == first) {
            second = static_cast<QubitId>((second + 1U) % qubit_count);
        }
        const double angle = (generator.unit() - 0.5) * 2.0;

        switch (generator.next() % 15U) {
            case 0U:
                state.apply_x(first);
                reference.apply_single(first, qubit::gates::x());
                break;
            case 1U:
                state.apply_y(first);
                reference.apply_single(first, qubit::gates::y());
                break;
            case 2U:
                state.apply_h(first);
                reference.apply_single(first, qubit::gates::h());
                break;
            case 3U:
                state.apply_t(first);
                reference.apply_single(first, qubit::gates::t());
                break;
            case 4U:
                state.apply_rx(first, angle);
                reference.apply_single(first, qubit::gates::rx(angle));
                break;
            case 5U:
                state.apply_ry(first, angle);
                reference.apply_single(first, qubit::gates::ry(angle));
                break;
            case 6U:
                state.apply_rz(first, angle);
                reference.apply_single(first, qubit::gates::rz(angle));
                break;
            case 7U:
                if ((generator.next() & 1U) == 0U) {
                    state.apply_cnot(first, second);
                } else {
                    state.apply_cnot_structured(first, second);
                }
                reference.apply_two(first, second, qubit::gates::cnot());
                break;
            case 8U:
                if ((generator.next() & 1U) == 0U) {
                    state.apply_cz(first, second);
                } else {
                    state.apply_cz_structured(first, second);
                }
                reference.apply_two(first, second, qubit::gates::cz());
                break;
            case 9U:
                if ((generator.next() & 1U) == 0U) {
                    state.apply_swap(first, second);
                } else {
                    state.apply_swap_structured(first, second);
                }
                reference.apply_two(first, second, qubit::gates::swap());
                break;
            case 10U: {
                const double sample = generator.unit();
                state.apply_bit_flip_trajectory(first, 0.35, sample);
                reference.apply_bit_flip_trajectory(first, 0.35, sample);
                break;
            }
            case 11U: {
                const double sample = generator.unit();
                state.apply_phase_flip_trajectory(first, 0.4, sample);
                reference.apply_phase_flip_trajectory(first, 0.4, sample);
                break;
            }
            case 12U: {
                const double sample = generator.unit();
                state.apply_depolarizing_trajectory(first, 0.3, sample);
                reference.apply_depolarizing_trajectory(first, 0.3, sample);
                break;
            }
            case 13U: {
                const double sample = generator.unit();
                state.apply_amplitude_damping_trajectory(first, 0.2, sample);
                reference.apply_amplitude_damping_trajectory(first, 0.2, sample);
                break;
            }
            case 14U: {
                const double sample = generator.unit();
                require(state.measure(first, sample) == reference.measure(first, sample),
                        "measurement outcome differs from dense reference");
                break;
            }
            default:
                throw std::runtime_error("unreachable transition operation");
        }

        const std::string label = "seed " + std::to_string(seed) + " step " + std::to_string(step);
        verify_state(state, reference, label);

        if (step % 17U == 0U) {
            QRegister restored = QRegister::decode_qsc(state.encode_qsc());
            verify_state(restored, reference, label + " QSC roundtrip");
            state = std::move(restored);
        }

        if (step % 23U == 0U) {
            QRegister branch = state;
            const std::vector<QComplex> root_before = state.materialize(12U);
            branch.apply_rz(first, 0.173);
            require_equivalent(state.materialize(12U), root_before, 1e-12,
                               label + " branch isolation");
            require(branch.validate(), label + ": branch validation failed");
        }

        if (step % 29U == 0U) {
            std::vector<PauliFactor> factors{{
                first,
                static_cast<PauliAxis>(1U + generator.next() % 3U),
            }};
            if ((generator.next() & 1U) != 0U) {
                factors.push_back({
                    second,
                    static_cast<PauliAxis>(1U + generator.next() % 3U),
                });
            }
            PauliObservable observable(qubit_count);
            observable.add_term({1.0, 0.0}, factors);
            require(qubit::almost_equal(
                        observable.expectation(state),
                        reference.pauli_expectation(factors),
                        2e-10),
                    label + ": Pauli expectation differs from dense reference");
        }
    }
}

}  // namespace

int main() {
    verify_dense_copy_on_write();
    for (std::uint64_t seed = 1U; seed <= 24U; ++seed) {
        run_transition_sequence(0xA5A5000000000000ULL + seed * 0x9E3779B97F4A7C15ULL);
    }
    std::cout << "state transition invariant tests passed\n";
    return 0;
}
