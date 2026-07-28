#include "qubit/qstabilizer.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace qubit {
namespace {

struct StabilizerSplitMix64 {
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

[[nodiscard]] std::uint8_t phase_mod4(int value) noexcept {
    value %= 4;
    if (value < 0) {
        value += 4;
    }
    return static_cast<std::uint8_t>(value);
}

}  // namespace

StabilizerState::StabilizerState(
    std::size_t qubit_count,
    StabilizerConfig config)
    : qubit_count_(qubit_count),
      word_count_((qubit_count + 63U) / 64U),
      config_(config) {
    if (qubit_count_ == 0U) {
        throw QStateError("StabilizerState requires at least one qubit");
    }
    if (qubit_count_ > static_cast<std::size_t>(std::numeric_limits<QubitId>::max())) {
        throw QStateError("StabilizerState qubit count exceeds QubitId capacity");
    }
    if (config_.max_tableau_bytes == 0U) {
        throw QStateError("StabilizerState max_tableau_bytes must be positive");
    }
    if (row_count() > std::numeric_limits<std::size_t>::max() / word_count_) {
        throw QStateError("Stabilizer tableau dimensions overflow");
    }
    const std::size_t words = row_count() * word_count_;
    if (words > std::numeric_limits<std::size_t>::max() / (2U * sizeof(std::uint64_t))) {
        throw QStateError("Stabilizer tableau byte count overflows");
    }
    const std::size_t required = words * 2U * sizeof(std::uint64_t) + row_count();
    if (required > config_.max_tableau_bytes) {
        throw QStateError("Stabilizer tableau exceeds configured memory limit");
    }

    x_.assign(words, 0U);
    z_.assign(words, 0U);
    phase_.assign(row_count(), 0U);
    for (std::size_t qubit = 0; qubit < qubit_count_; ++qubit) {
        set_bit(x_, qubit, qubit, true);
        set_bit(z_, qubit_count_ + qubit, qubit, true);
    }
}

bool StabilizerState::bit(
    const std::vector<std::uint64_t>& table,
    std::size_t row,
    std::size_t qubit) const noexcept {
    const std::size_t word = qubit >> 6U;
    const std::uint64_t mask = std::uint64_t{1} << (qubit & 63U);
    return (table[offset(row, word)] & mask) != 0U;
}

void StabilizerState::set_bit(
    std::vector<std::uint64_t>& table,
    std::size_t row,
    std::size_t qubit,
    bool value) noexcept {
    const std::size_t word = qubit >> 6U;
    const std::uint64_t mask = std::uint64_t{1} << (qubit & 63U);
    std::uint64_t& selected = table[offset(row, word)];
    if (value) {
        selected |= mask;
    } else {
        selected &= ~mask;
    }
}

void StabilizerState::validate_qubit(QubitId qubit) const {
    if (static_cast<std::size_t>(qubit) >= qubit_count_) {
        throw QStateError("Stabilizer qubit index is out of range");
    }
}

void StabilizerState::clear_row(std::size_t row) noexcept {
    for (std::size_t word = 0; word < word_count_; ++word) {
        x_[offset(row, word)] = 0U;
        z_[offset(row, word)] = 0U;
    }
    phase_[row] = 0U;
}

void StabilizerState::copy_row(std::size_t target, std::size_t source) noexcept {
    for (std::size_t word = 0; word < word_count_; ++word) {
        x_[offset(target, word)] = x_[offset(source, word)];
        z_[offset(target, word)] = z_[offset(source, word)];
    }
    phase_[target] = phase_[source];
}

void StabilizerState::multiply_row(std::size_t target, std::size_t source) noexcept {
    unsigned parity = 0U;
    for (std::size_t word = 0; word < word_count_; ++word) {
        parity ^= static_cast<unsigned>(
            std::popcount(z_[offset(target, word)] & x_[offset(source, word)]) & 1U);
    }
    phase_[target] = phase_mod4(
        static_cast<int>(phase_[target]) + static_cast<int>(phase_[source]) +
        static_cast<int>(2U * parity));
    for (std::size_t word = 0; word < word_count_; ++word) {
        x_[offset(target, word)] ^= x_[offset(source, word)];
        z_[offset(target, word)] ^= z_[offset(source, word)];
    }
}

void StabilizerState::multiply_scratch(
    std::vector<std::uint64_t>& scratch_x,
    std::vector<std::uint64_t>& scratch_z,
    std::uint8_t& scratch_phase,
    std::size_t source) const noexcept {
    unsigned parity = 0U;
    for (std::size_t word = 0; word < word_count_; ++word) {
        parity ^= static_cast<unsigned>(
            std::popcount(scratch_z[word] & x_[offset(source, word)]) & 1U);
    }
    scratch_phase = phase_mod4(
        static_cast<int>(scratch_phase) + static_cast<int>(phase_[source]) +
        static_cast<int>(2U * parity));
    for (std::size_t word = 0; word < word_count_; ++word) {
        scratch_x[word] ^= x_[offset(source, word)];
        scratch_z[word] ^= z_[offset(source, word)];
    }
}

bool StabilizerState::symplectic_anticommutes(
    std::size_t first,
    std::size_t second) const noexcept {
    unsigned parity = 0U;
    for (std::size_t word = 0; word < word_count_; ++word) {
        parity ^= static_cast<unsigned>(
            std::popcount(x_[offset(first, word)] & z_[offset(second, word)]) & 1U);
        parity ^= static_cast<unsigned>(
            std::popcount(z_[offset(first, word)] & x_[offset(second, word)]) & 1U);
    }
    return parity != 0U;
}

void StabilizerState::apply_x(QubitId qubit) {
    validate_qubit(qubit);
    for (std::size_t row = 0; row < row_count(); ++row) {
        if (bit(z_, row, qubit)) {
            phase_[row] = phase_mod4(static_cast<int>(phase_[row]) + 2);
        }
    }
}

void StabilizerState::apply_y(QubitId qubit) {
    validate_qubit(qubit);
    for (std::size_t row = 0; row < row_count(); ++row) {
        if (bit(x_, row, qubit) != bit(z_, row, qubit)) {
            phase_[row] = phase_mod4(static_cast<int>(phase_[row]) + 2);
        }
    }
}

void StabilizerState::apply_z(QubitId qubit) {
    validate_qubit(qubit);
    for (std::size_t row = 0; row < row_count(); ++row) {
        if (bit(x_, row, qubit)) {
            phase_[row] = phase_mod4(static_cast<int>(phase_[row]) + 2);
        }
    }
}

void StabilizerState::apply_h(QubitId qubit) {
    validate_qubit(qubit);
    for (std::size_t row = 0; row < row_count(); ++row) {
        const bool x = bit(x_, row, qubit);
        const bool z = bit(z_, row, qubit);
        if (x && z) {
            phase_[row] = phase_mod4(static_cast<int>(phase_[row]) + 2);
        }
        set_bit(x_, row, qubit, z);
        set_bit(z_, row, qubit, x);
    }
}

void StabilizerState::apply_s(QubitId qubit) {
    validate_qubit(qubit);
    for (std::size_t row = 0; row < row_count(); ++row) {
        const bool x = bit(x_, row, qubit);
        if (x) {
            phase_[row] = phase_mod4(static_cast<int>(phase_[row]) + 1);
            set_bit(z_, row, qubit, !bit(z_, row, qubit));
        }
    }
}

void StabilizerState::apply_sdg(QubitId qubit) {
    validate_qubit(qubit);
    for (std::size_t row = 0; row < row_count(); ++row) {
        const bool x = bit(x_, row, qubit);
        if (x) {
            phase_[row] = phase_mod4(static_cast<int>(phase_[row]) + 3);
            set_bit(z_, row, qubit, !bit(z_, row, qubit));
        }
    }
}

void StabilizerState::apply_cnot(QubitId control, QubitId target) {
    validate_qubit(control);
    validate_qubit(target);
    if (control == target) {
        throw QStateError("Stabilizer CNOT requires distinct qubits");
    }
    for (std::size_t row = 0; row < row_count(); ++row) {
        const bool xc = bit(x_, row, control);
        const bool zc = bit(z_, row, control);
        const bool xt = bit(x_, row, target);
        const bool zt = bit(z_, row, target);
        const int old_xz = static_cast<int>(xc && zc) + static_cast<int>(xt && zt);
        const bool next_xt = xt != xc;
        const bool next_zc = zc != zt;
        const int next_xz = static_cast<int>(xc && next_zc) +
                            static_cast<int>(next_xt && zt);
        const bool sign_flip = xc && zt && (xt == zc);
        phase_[row] = phase_mod4(
            static_cast<int>(phase_[row]) + next_xz - old_xz +
            (sign_flip ? 2 : 0));
        set_bit(x_, row, target, next_xt);
        set_bit(z_, row, control, next_zc);
    }
}

void StabilizerState::apply_cz(QubitId first, QubitId second) {
    validate_qubit(first);
    validate_qubit(second);
    if (first == second) {
        throw QStateError("Stabilizer CZ requires distinct qubits");
    }
    apply_h(second);
    apply_cnot(first, second);
    apply_h(second);
}

void StabilizerState::apply_swap(QubitId first, QubitId second) {
    validate_qubit(first);
    validate_qubit(second);
    if (first == second) {
        throw QStateError("Stabilizer SWAP requires distinct qubits");
    }
    for (std::size_t row = 0; row < row_count(); ++row) {
        const bool first_x = bit(x_, row, first);
        const bool second_x = bit(x_, row, second);
        const bool first_z = bit(z_, row, first);
        const bool second_z = bit(z_, row, second);
        set_bit(x_, row, first, second_x);
        set_bit(x_, row, second, first_x);
        set_bit(z_, row, first, second_z);
        set_bit(z_, row, second, first_z);
    }
}

std::size_t StabilizerState::random_measurement_row(QubitId qubit) const noexcept {
    for (std::size_t row = qubit_count_; row < row_count(); ++row) {
        if (bit(x_, row, qubit)) {
            return row;
        }
    }
    return row_count();
}

int StabilizerState::deterministic_z(QubitId qubit) const {
    std::vector<std::uint64_t> scratch_x(word_count_, 0U);
    std::vector<std::uint64_t> scratch_z(word_count_, 0U);
    std::uint8_t scratch_phase = 0U;
    for (std::size_t destabilizer = 0; destabilizer < qubit_count_; ++destabilizer) {
        if (bit(x_, destabilizer, qubit)) {
            multiply_scratch(
                scratch_x,
                scratch_z,
                scratch_phase,
                destabilizer + qubit_count_);
        }
    }
    if (scratch_phase == 0U) {
        return 0;
    }
    if (scratch_phase == 2U) {
        return 1;
    }
    throw QStateError("Stabilizer deterministic measurement produced a non-Hermitian phase");
}

double StabilizerState::probability_one(QubitId qubit) const {
    validate_qubit(qubit);
    if (random_measurement_row(qubit) != row_count()) {
        return 0.5;
    }
    return static_cast<double>(deterministic_z(qubit));
}

int StabilizerState::measure_z(QubitId qubit, double sample) {
    validate_qubit(qubit);
    if (!std::isfinite(sample) || sample < 0.0 || sample >= 1.0) {
        throw QStateError("Stabilizer measurement sample must be finite and in [0, 1)");
    }
    const std::size_t pivot = random_measurement_row(qubit);
    if (pivot == row_count()) {
        return deterministic_z(qubit);
    }

    const std::size_t paired_destabilizer = pivot - qubit_count_;
    for (std::size_t row = 0; row < row_count(); ++row) {
        if (row != pivot && row != paired_destabilizer && bit(x_, row, qubit)) {
            multiply_row(row, pivot);
        }
    }
    copy_row(paired_destabilizer, pivot);
    clear_row(pivot);
    set_bit(z_, pivot, qubit, true);
    const int outcome = sample < 0.5 ? 1 : 0;
    phase_[pivot] = outcome == 0 ? 0U : 2U;
    return outcome;
}

std::vector<int> StabilizerState::measure_all(std::uint64_t seed) {
    StabilizerSplitMix64 generator{seed};
    std::vector<int> result(qubit_count_);
    for (std::size_t qubit = 0; qubit < qubit_count_; ++qubit) {
        result[qubit] = measure_z(static_cast<QubitId>(qubit), generator.unit());
    }
    return result;
}

std::size_t StabilizerState::estimated_bytes() const noexcept {
    return sizeof(*this) + x_.capacity() * sizeof(std::uint64_t) +
           z_.capacity() * sizeof(std::uint64_t) +
           phase_.capacity() * sizeof(std::uint8_t);
}

bool StabilizerState::validate(std::string* reason) const {
    const auto fail = [reason](const char* message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };
    if (qubit_count_ == 0U || word_count_ != (qubit_count_ + 63U) / 64U) {
        return fail("stabilizer dimensions are invalid");
    }
    if (x_.size() != row_count() * word_count_ ||
        z_.size() != row_count() * word_count_ || phase_.size() != row_count()) {
        return fail("stabilizer tableau storage has the wrong size");
    }
    for (std::size_t row = 0; row < row_count(); ++row) {
        unsigned xz_parity = 0U;
        bool any = false;
        for (std::size_t word = 0; word < word_count_; ++word) {
            const std::uint64_t x = x_[offset(row, word)];
            const std::uint64_t z = z_[offset(row, word)];
            any = any || x != 0U || z != 0U;
            xz_parity ^= static_cast<unsigned>(std::popcount(x & z) & 1U);
        }
        if (!any) {
            return fail("stabilizer tableau contains an identity generator");
        }
        if ((phase_[row] & 1U) != xz_parity) {
            return fail("stabilizer generator has a non-Hermitian phase");
        }
        if (!symplectic_anticommutes(row, row % qubit_count_ +
                                             (row < qubit_count_ ? qubit_count_ : 0U))) {
            return fail("stabilizer paired generators do not anticommute");
        }
    }
    return true;
}

bool StabilizerState::validate_full(std::string* reason) const {
    if (!validate(reason)) {
        return false;
    }
    for (std::size_t first = 0; first < qubit_count_; ++first) {
        for (std::size_t second = first + 1U; second < qubit_count_; ++second) {
            if (symplectic_anticommutes(first, second)) {
                if (reason != nullptr) {
                    *reason = "destabilizer generators do not commute";
                }
                return false;
            }
            if (symplectic_anticommutes(
                    qubit_count_ + first,
                    qubit_count_ + second)) {
                if (reason != nullptr) {
                    *reason = "stabilizer generators do not commute";
                }
                return false;
            }
            if (symplectic_anticommutes(first, qubit_count_ + second) ||
                symplectic_anticommutes(second, qubit_count_ + first)) {
                if (reason != nullptr) {
                    *reason = "unpaired destabilizer and stabilizer generators anticommute";
                }
                return false;
            }
        }
    }
    return true;
}

std::string StabilizerState::describe() const {
    std::ostringstream stream;
    stream << "QSA stabilizer tableau\n"
           << "qubits: " << qubit_count_ << "\n"
           << "rows: " << row_count() << "\n"
           << "words per row: " << word_count_ << "\n"
           << "estimated engine bytes: " << estimated_bytes() << "\n";
    return stream.str();
}

}  // namespace qubit
