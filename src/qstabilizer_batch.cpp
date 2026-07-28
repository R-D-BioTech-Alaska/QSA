#include "qubit/qstabilizer.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>

namespace qubit {

void StabilizerState::apply_batch(
    std::span<const StabilizerOperation> operations) {
    if (operations.empty()) {
        return;
    }

    const auto has_second_qubit = [this](
                                       const StabilizerOperation& operation) {
        validate_qubit(operation.first);
        switch (operation.code) {
            case StabilizerOperationCode::X:
            case StabilizerOperationCode::Y:
            case StabilizerOperationCode::Z:
            case StabilizerOperationCode::H:
            case StabilizerOperationCode::S:
            case StabilizerOperationCode::Sdg:
                return false;
            case StabilizerOperationCode::Cnot:
            case StabilizerOperationCode::Cz:
            case StabilizerOperationCode::Swap:
                validate_qubit(operation.second);
                if (operation.first == operation.second) {
                    throw QStateError(
                        "Stabilizer batch two-qubit operation requires distinct qubits");
                }
                return true;
            default:
                throw QStateError("Stabilizer batch contains an unknown operation");
        }
    };

    std::vector<QubitId> active_qubits;
    const std::size_t dense_discovery_threshold =
        std::max<std::size_t>(1U, qubit_count_ / 64U);
    if (operations.size() >= dense_discovery_threshold) {
        std::vector<std::uint64_t> active_bits(word_count_, 0U);
        std::size_t active_bit_count = 0U;
        const auto record = [&active_bits, &active_bit_count](QubitId qubit) {
            std::uint64_t& word =
                active_bits[static_cast<std::size_t>(qubit) >> 6U];
            const std::uint64_t mask =
                std::uint64_t{1} << (qubit & 63U);
            if ((word & mask) == 0U) {
                word |= mask;
                ++active_bit_count;
            }
        };
        for (const StabilizerOperation& operation : operations) {
            const bool has_second = has_second_qubit(operation);
            record(operation.first);
            if (has_second) {
                record(operation.second);
            }
        }
        active_qubits.reserve(active_bit_count);
        for (std::size_t word = 0; word < active_bits.size(); ++word) {
            std::uint64_t bits = active_bits[word];
            while (bits != 0U) {
                const std::size_t within =
                    static_cast<std::size_t>(std::countr_zero(bits));
                const std::size_t qubit = (word << 6U) + within;
                if (qubit < qubit_count_) {
                    active_qubits.push_back(static_cast<QubitId>(qubit));
                }
                bits &= bits - 1U;
            }
        }
    } else {
        const std::size_t reserve_count = operations.size() > qubit_count_ / 2U
            ? qubit_count_
            : std::min(qubit_count_, operations.size() * 2U);
        active_qubits.reserve(reserve_count);
        for (const StabilizerOperation& operation : operations) {
            const bool has_second = has_second_qubit(operation);
            active_qubits.push_back(operation.first);
            if (has_second) {
                active_qubits.push_back(operation.second);
            }
        }
        std::sort(active_qubits.begin(), active_qubits.end());
        active_qubits.erase(
            std::unique(active_qubits.begin(), active_qubits.end()),
            active_qubits.end());
    }

    struct LocalOperation {
        StabilizerOperationCode code{StabilizerOperationCode::H};
        std::uint32_t first{0};
        std::uint32_t second{0};
    };
    struct ActiveWord {
        std::size_t word{0};
        std::uint64_t mask{0};
        std::array<std::uint32_t, 64> local{};
    };
    constexpr std::uint32_t kInactive =
        std::numeric_limits<std::uint32_t>::max();

    const std::size_t rows = row_count();
    const std::size_t row_word_count = (rows + 63U) / 64U;
    const std::size_t active_count = active_qubits.size();
    const std::size_t full_threshold =
        qubit_count_ - qubit_count_ / 4U;
    const bool full_candidate = active_count >= full_threshold;

    const auto scratch_requirement = [
        &active_qubits,
        &operations,
        row_word_count,
        this](std::size_t column_count,
              bool active_only,
              std::size_t* result) {
        if (column_count >
            std::numeric_limits<std::size_t>::max() / row_word_count) {
            return false;
        }
        const std::size_t column_words_value =
            column_count * row_word_count;
        std::size_t bytes = 0U;
        const auto add = [&bytes](std::size_t count, std::size_t element_size) {
            if (count >
                (std::numeric_limits<std::size_t>::max() - bytes) /
                    element_size) {
                return false;
            }
            bytes += count * element_size;
            return true;
        };
        if (!add(column_words_value, sizeof(std::uint64_t)) ||
            !add(column_words_value, sizeof(std::uint64_t)) ||
            !add(row_word_count, sizeof(std::uint64_t)) ||
            !add(row_word_count, sizeof(std::uint64_t)) ||
            !add(active_qubits.capacity(), sizeof(QubitId))) {
            return false;
        }
        if (active_only &&
            (!add(std::min(word_count_, active_qubits.size()),
                  sizeof(ActiveWord)) ||
             !add(operations.size(), sizeof(LocalOperation)))) {
            return false;
        }
        *result = bytes;
        return true;
    };

    std::size_t full_scratch_bytes = 0U;
    bool use_full_transpose = full_candidate &&
        scratch_requirement(qubit_count_, false, &full_scratch_bytes) &&
        full_scratch_bytes <= config_.max_batch_scratch_bytes;
    std::size_t scratch_bytes = full_scratch_bytes;
    if (!use_full_transpose) {
        if (!scratch_requirement(active_count, true, &scratch_bytes)) {
            throw QStateError("Stabilizer batch scratch byte count overflows");
        }
    }
    if (config_.max_batch_scratch_bytes == 0U ||
        scratch_bytes > config_.max_batch_scratch_bytes) {
        throw QStateError("Stabilizer batch exceeds configured scratch memory limit");
    }

    const std::size_t transposed_column_count = use_full_transpose
        ? qubit_count_
        : active_count;
    const std::size_t column_words =
        transposed_column_count * row_word_count;

    std::vector<ActiveWord> active_words;
    if (!use_full_transpose) {
        active_words.reserve(std::min(word_count_, active_count));
        for (std::size_t local = 0; local < active_count; ++local) {
            const std::size_t qubit = active_qubits[local];
            const std::size_t word = qubit >> 6U;
            if (active_words.empty() || active_words.back().word != word) {
                ActiveWord active_word;
                active_word.word = word;
                active_word.local.fill(kInactive);
                active_words.push_back(active_word);
            }
            ActiveWord& active_word = active_words.back();
            const std::size_t within = qubit & 63U;
            active_word.mask |= std::uint64_t{1} << within;
            active_word.local[within] = static_cast<std::uint32_t>(local);
        }
    }

    std::vector<LocalOperation> local_operations;
    if (!use_full_transpose) {
        const auto local_index = [&active_qubits](QubitId qubit) {
            const auto iterator = std::lower_bound(
                active_qubits.begin(), active_qubits.end(), qubit);
            return static_cast<std::uint32_t>(
                std::distance(active_qubits.begin(), iterator));
        };
        local_operations.reserve(operations.size());
        for (const StabilizerOperation& operation : operations) {
            local_operations.push_back(LocalOperation{
                operation.code,
                local_index(operation.first),
                local_index(operation.second),
            });
        }
    }

    std::vector<std::uint64_t> transposed_x(column_words, 0U);
    std::vector<std::uint64_t> transposed_z(column_words, 0U);
    std::vector<std::uint64_t> phase_low(row_word_count, 0U);
    std::vector<std::uint64_t> phase_high(row_word_count, 0U);
    const auto column_offset = [row_word_count](
                                   std::size_t local_qubit,
                                   std::size_t row_word) noexcept {
        return local_qubit * row_word_count + row_word;
    };

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t row_word = row >> 6U;
        const std::uint64_t row_mask = std::uint64_t{1} << (row & 63U);
        if ((phase_[row] & 1U) != 0U) {
            phase_low[row_word] |= row_mask;
        }
        if ((phase_[row] & 2U) != 0U) {
            phase_high[row_word] |= row_mask;
        }
    }

    if (use_full_transpose) {
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t row_word = row >> 6U;
            const std::uint64_t row_mask = std::uint64_t{1} << (row & 63U);
            for (std::size_t qubit_word = 0;
                 qubit_word < word_count_;
                 ++qubit_word) {
                std::uint64_t x_bits = x_[offset(row, qubit_word)];
                std::uint64_t z_bits = z_[offset(row, qubit_word)];
                while (x_bits != 0U) {
                    const std::size_t within =
                        static_cast<std::size_t>(std::countr_zero(x_bits));
                    const std::size_t qubit = (qubit_word << 6U) + within;
                    if (qubit < qubit_count_) {
                        transposed_x[column_offset(qubit, row_word)] |= row_mask;
                    }
                    x_bits &= x_bits - 1U;
                }
                while (z_bits != 0U) {
                    const std::size_t within =
                        static_cast<std::size_t>(std::countr_zero(z_bits));
                    const std::size_t qubit = (qubit_word << 6U) + within;
                    if (qubit < qubit_count_) {
                        transposed_z[column_offset(qubit, row_word)] |= row_mask;
                    }
                    z_bits &= z_bits - 1U;
                }
            }
        }
    } else {
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t row_word = row >> 6U;
            const std::uint64_t row_mask = std::uint64_t{1} << (row & 63U);
            for (const ActiveWord& active_word : active_words) {
                std::uint64_t x_bits =
                    x_[offset(row, active_word.word)] & active_word.mask;
                std::uint64_t z_bits =
                    z_[offset(row, active_word.word)] & active_word.mask;
                while (x_bits != 0U) {
                    const std::size_t within =
                        static_cast<std::size_t>(std::countr_zero(x_bits));
                    const std::uint32_t local = active_word.local[within];
                    transposed_x[column_offset(local, row_word)] |= row_mask;
                    x_bits &= x_bits - 1U;
                }
                while (z_bits != 0U) {
                    const std::size_t within =
                        static_cast<std::size_t>(std::countr_zero(z_bits));
                    const std::uint32_t local = active_word.local[within];
                    transposed_z[column_offset(local, row_word)] |= row_mask;
                    z_bits &= z_bits - 1U;
                }
            }
        }
    }

    const std::size_t remainder = rows & 63U;
    const std::uint64_t final_valid_mask = remainder == 0U
        ? std::numeric_limits<std::uint64_t>::max()
        : (std::uint64_t{1} << remainder) - 1U;
    const auto valid_mask = [row_word_count, final_valid_mask](
                                std::size_t row_word) noexcept {
        return row_word + 1U == row_word_count
            ? final_valid_mask
            : std::numeric_limits<std::uint64_t>::max();
    };
    const auto add_one = [&phase_low, &phase_high](
                             std::size_t word,
                             std::uint64_t mask) noexcept {
        const std::uint64_t carry = phase_low[word] & mask;
        phase_low[word] ^= mask;
        phase_high[word] ^= carry;
    };
    const auto add_three = [&phase_low, &phase_high](
                               std::size_t word,
                               std::uint64_t mask) noexcept {
        const std::uint64_t borrow = ~phase_low[word] & mask;
        phase_low[word] ^= mask;
        phase_high[word] ^= borrow;
    };

    const auto apply_h_column = [&](std::size_t local_qubit) {
        for (std::size_t word = 0; word < row_word_count; ++word) {
            std::uint64_t& x =
                transposed_x[column_offset(local_qubit, word)];
            std::uint64_t& z =
                transposed_z[column_offset(local_qubit, word)];
            phase_high[word] ^= x & z;
            std::swap(x, z);
        }
    };
    const auto apply_cnot_columns = [
        &transposed_x,
        &transposed_z,
        &phase_high,
        &add_one,
        &add_three,
        &column_offset,
        &valid_mask,
        row_word_count](std::size_t control, std::size_t target) {
        for (std::size_t word = 0; word < row_word_count; ++word) {
            const std::uint64_t xc =
                transposed_x[column_offset(control, word)];
            const std::uint64_t zc =
                transposed_z[column_offset(control, word)];
            const std::uint64_t xt =
                transposed_x[column_offset(target, word)];
            const std::uint64_t zt =
                transposed_z[column_offset(target, word)];
            const std::uint64_t next_xt = xt ^ xc;
            const std::uint64_t next_zc = zc ^ zt;
            const std::uint64_t old_control_xz = xc & zc;
            const std::uint64_t old_target_xz = xt & zt;
            const std::uint64_t next_control_xz = xc & next_zc;
            const std::uint64_t next_target_xz = next_xt & zt;
            const std::uint64_t sign_flip =
                xc & zt & ~(xt ^ zc) & valid_mask(word);
            add_one(word, next_control_xz);
            add_one(word, next_target_xz);
            add_three(word, old_control_xz);
            add_three(word, old_target_xz);
            phase_high[word] ^= sign_flip;
            transposed_x[column_offset(target, word)] = next_xt;
            transposed_z[column_offset(control, word)] = next_zc;
        }
    };

    const auto execute_operations = [&](const auto& execution_operations) {
        for (const auto& operation : execution_operations) {
            const std::size_t first =
                static_cast<std::size_t>(operation.first);
            const std::size_t second =
                static_cast<std::size_t>(operation.second);
            switch (operation.code) {
                case StabilizerOperationCode::X:
                    for (std::size_t word = 0; word < row_word_count; ++word) {
                        phase_high[word] ^=
                            transposed_z[column_offset(first, word)];
                    }
                    break;
                case StabilizerOperationCode::Y:
                    for (std::size_t word = 0; word < row_word_count; ++word) {
                        phase_high[word] ^=
                            transposed_x[column_offset(first, word)] ^
                            transposed_z[column_offset(first, word)];
                    }
                    break;
                case StabilizerOperationCode::Z:
                    for (std::size_t word = 0; word < row_word_count; ++word) {
                        phase_high[word] ^=
                            transposed_x[column_offset(first, word)];
                    }
                    break;
                case StabilizerOperationCode::H:
                    apply_h_column(first);
                    break;
                case StabilizerOperationCode::S:
                    for (std::size_t word = 0; word < row_word_count; ++word) {
                        const std::uint64_t x =
                            transposed_x[column_offset(first, word)];
                        add_one(word, x);
                        transposed_z[column_offset(first, word)] ^= x;
                    }
                    break;
                case StabilizerOperationCode::Sdg:
                    for (std::size_t word = 0; word < row_word_count; ++word) {
                        const std::uint64_t x =
                            transposed_x[column_offset(first, word)];
                        add_three(word, x);
                        transposed_z[column_offset(first, word)] ^= x;
                    }
                    break;
                case StabilizerOperationCode::Cnot:
                    apply_cnot_columns(first, second);
                    break;
                case StabilizerOperationCode::Cz:
                    apply_h_column(second);
                    apply_cnot_columns(first, second);
                    apply_h_column(second);
                    break;
                case StabilizerOperationCode::Swap:
                    for (std::size_t word = 0; word < row_word_count; ++word) {
                        std::swap(
                            transposed_x[column_offset(first, word)],
                            transposed_x[column_offset(second, word)]);
                        std::swap(
                            transposed_z[column_offset(first, word)],
                            transposed_z[column_offset(second, word)]);
                    }
                    break;
                default:
                    throw QStateError(
                        "Stabilizer batch contains an unknown operation");
            }
        }
    };

    if (use_full_transpose) {
        execute_operations(operations);
    } else {
        execute_operations(local_operations);
    }

    if (use_full_transpose) {
        std::fill(x_.begin(), x_.end(), 0U);
        std::fill(z_.begin(), z_.end(), 0U);
    } else {
        for (std::size_t row = 0; row < rows; ++row) {
            for (const ActiveWord& active_word : active_words) {
                x_[offset(row, active_word.word)] &= ~active_word.mask;
                z_[offset(row, active_word.word)] &= ~active_word.mask;
            }
        }
    }
    const std::size_t restore_count = use_full_transpose
        ? qubit_count_
        : active_count;
    for (std::size_t local = 0; local < restore_count; ++local) {
        const std::size_t qubit = use_full_transpose
            ? local
            : static_cast<std::size_t>(active_qubits[local]);
        const std::size_t qubit_word = qubit >> 6U;
        const std::uint64_t qubit_mask =
            std::uint64_t{1} << (qubit & 63U);
        for (std::size_t row_word = 0;
             row_word < row_word_count;
             ++row_word) {
            std::uint64_t x_bits =
                transposed_x[column_offset(local, row_word)];
            std::uint64_t z_bits =
                transposed_z[column_offset(local, row_word)];
            while (x_bits != 0U) {
                const std::size_t within =
                    static_cast<std::size_t>(std::countr_zero(x_bits));
                const std::size_t row = (row_word << 6U) + within;
                if (row < rows) {
                    x_[offset(row, qubit_word)] |= qubit_mask;
                }
                x_bits &= x_bits - 1U;
            }
            while (z_bits != 0U) {
                const std::size_t within =
                    static_cast<std::size_t>(std::countr_zero(z_bits));
                const std::size_t row = (row_word << 6U) + within;
                if (row < rows) {
                    z_[offset(row, qubit_word)] |= qubit_mask;
                }
                z_bits &= z_bits - 1U;
            }
        }
    }

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t word = row >> 6U;
        const std::uint64_t mask = std::uint64_t{1} << (row & 63U);
        phase_[row] = static_cast<std::uint8_t>(
            ((phase_low[word] & mask) != 0U ? 1U : 0U) |
            ((phase_high[word] & mask) != 0U ? 2U : 0U));
    }
}

}  // namespace qubit
