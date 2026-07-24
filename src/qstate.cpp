#include "qubit/qstate.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_map>

namespace qubit {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

[[nodiscard]] bool is_power_of_two(BasisIndex value) noexcept {
    return value != 0 && (value & (value - 1U)) == 0;
}

[[nodiscard]] std::size_t bit_count_for_dimension(BasisIndex dimension) {
    if (!is_power_of_two(dimension)) {
        throw QStateError("Amplitude dimension must be a nonzero power of two");
    }
    return static_cast<std::size_t>(std::countr_zero(dimension));
}

void validate_probability(double value, const char* label) {
    if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
        throw QStateError(std::string(label) + " must be finite and in [0, 1]");
    }
}

[[nodiscard]] BasisIndex remove_bit(BasisIndex value, std::size_t position) noexcept {
    const BasisIndex lower_mask = position == 0 ? 0 : ((BasisIndex{1} << position) - 1U);
    const BasisIndex lower = value & lower_mask;
    const BasisIndex upper = value >> (position + 1U);
    return lower | (upper << position);
}

[[nodiscard]] BasisIndex insert_zero_bit(BasisIndex value, std::size_t position) noexcept {
    const BasisIndex lower_mask = position == 0 ? 0 : ((BasisIndex{1} << position) - 1U);
    const BasisIndex lower = value & lower_mask;
    const BasisIndex upper = value >> position;
    return lower | (upper << (position + 1U));
}

[[nodiscard]] BasisIndex remap_index(
    BasisIndex index,
    std::span<const std::size_t> local_to_merged) noexcept {
    BasisIndex result = 0;
    for (std::size_t local = 0; local < local_to_merged.size(); ++local) {
        if (((index >> local) & 1U) != 0U) {
            result |= BasisIndex{1} << local_to_merged[local];
        }
    }
    return result;
}

[[nodiscard]] std::string storage_name(StorageMode mode) {
    return mode == StorageMode::Dense ? "dense patch" : "sparse patch";
}

struct SplitMix64 {
    std::uint64_t state;

    [[nodiscard]] std::uint64_t next() noexcept {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31U);
    }

    [[nodiscard]] double unit() noexcept {
        return static_cast<double>(next() >> 11U) * (1.0 / 9007199254740992.0);
    }
};

} 

namespace gates {

QMatrix2 identity() {
    return QMatrix2{{QComplex{1.0, 0.0}, QComplex{0.0, 0.0}, QComplex{0.0, 0.0}, QComplex{1.0, 0.0}}};
}

QMatrix2 x() {
    return QMatrix2{{QComplex{0.0, 0.0}, QComplex{1.0, 0.0}, QComplex{1.0, 0.0}, QComplex{0.0, 0.0}}};
}

QMatrix2 y() {
    return QMatrix2{{QComplex{0.0, 0.0}, QComplex{0.0, -1.0}, QComplex{0.0, 1.0}, QComplex{0.0, 0.0}}};
}

QMatrix2 z() {
    return QMatrix2{{QComplex{1.0, 0.0}, QComplex{0.0, 0.0}, QComplex{0.0, 0.0}, QComplex{-1.0, 0.0}}};
}

QMatrix2 h() {
    const double scale = 1.0 / std::sqrt(2.0);
    return QMatrix2{{QComplex{scale, 0.0}, QComplex{scale, 0.0}, QComplex{scale, 0.0}, QComplex{-scale, 0.0}}};
}

QMatrix2 s() {
    return QMatrix2{{QComplex{1.0, 0.0}, QComplex{0.0, 0.0}, QComplex{0.0, 0.0}, QComplex{0.0, 1.0}}};
}

QMatrix2 sdg() {
    return QMatrix2{{QComplex{1.0, 0.0}, QComplex{0.0, 0.0}, QComplex{0.0, 0.0}, QComplex{0.0, -1.0}}};
}

QMatrix2 t() {
    return QMatrix2{{QComplex{1.0, 0.0}, QComplex{0.0, 0.0}, QComplex{0.0, 0.0}, QComplex::from_polar(1.0, kPi / 4.0)}};
}

QMatrix2 tdg() {
    return QMatrix2{{QComplex{1.0, 0.0}, QComplex{0.0, 0.0}, QComplex{0.0, 0.0}, QComplex::from_polar(1.0, -kPi / 4.0)}};
}

QMatrix2 rx(double theta) {
    const double cosine = std::cos(theta / 2.0);
    const double sine = std::sin(theta / 2.0);
    return QMatrix2{{QComplex{cosine, 0.0}, QComplex{0.0, -sine}, QComplex{0.0, -sine}, QComplex{cosine, 0.0}}};
}

QMatrix2 ry(double theta) {
    const double cosine = std::cos(theta / 2.0);
    const double sine = std::sin(theta / 2.0);
    return QMatrix2{{QComplex{cosine, 0.0}, QComplex{-sine, 0.0}, QComplex{sine, 0.0}, QComplex{cosine, 0.0}}};
}

QMatrix2 rz(double theta) {
    return QMatrix2{{QComplex::from_polar(1.0, -theta / 2.0),
                     QComplex{0.0, 0.0},
                     QComplex{0.0, 0.0},
                     QComplex::from_polar(1.0, theta / 2.0)}};
}

QMatrix4 cnot() {
    QMatrix4 matrix{};
    matrix.values[0] = {1.0, 0.0};
    matrix.values[5] = {1.0, 0.0};
    matrix.values[11] = {1.0, 0.0};
    matrix.values[14] = {1.0, 0.0};
    return matrix;
}

QMatrix4 cz() {
    QMatrix4 matrix{};
    matrix.values[0] = {1.0, 0.0};
    matrix.values[5] = {1.0, 0.0};
    matrix.values[10] = {1.0, 0.0};
    matrix.values[15] = {-1.0, 0.0};
    return matrix;
}

QMatrix4 swap() {
    QMatrix4 matrix{};
    matrix.values[0] = {1.0, 0.0};
    matrix.values[6] = {1.0, 0.0};
    matrix.values[9] = {1.0, 0.0};
    matrix.values[15] = {1.0, 0.0};
    return matrix;
}

} 

void BlochCell::normalize(double epsilon) {
    const double length = std::sqrt(x * x + y * y + z * z);
    if (!std::isfinite(length) || length <= epsilon) {
        throw QStateError("Invalid Bloch cell: zero or non-finite vector");
    }
    x /= length;
    y /= length;
    z /= length;
    x = std::clamp(x, -1.0, 1.0);
    y = std::clamp(y, -1.0, 1.0);
    z = std::clamp(z, -1.0, 1.0);
}

double BlochCell::probability_one() const noexcept {
    return std::clamp((1.0 - z) * 0.5, 0.0, 1.0);
}

std::array<QComplex, 2> BlochCell::amplitudes(double epsilon) const {
    const double p_zero = std::clamp((1.0 + z) * 0.5, 0.0, 1.0);
    const double alpha = std::sqrt(p_zero);
    if (alpha > epsilon) {
        return {{{alpha, 0.0}, {x / (2.0 * alpha), y / (2.0 * alpha)}}};
    }
    return {{{0.0, 0.0}, {1.0, 0.0}}};
}

BlochCell BlochCell::from_amplitudes(QComplex zero, QComplex one, double epsilon) {
    const double norm = std::sqrt(zero.norm2() + one.norm2());
    if (!std::isfinite(norm) || norm <= epsilon) {
        throw QStateError("Cannot create Bloch cell from a zero state");
    }
    zero /= norm;
    one /= norm;
    const QComplex coherence = zero.conjugate() * one;
    BlochCell cell{
        2.0 * coherence.re,
        2.0 * coherence.im,
        zero.norm2() - one.norm2(),
    };
    cell.normalize(epsilon);
    return cell;
}

void BlochCell::apply_x() noexcept {
    y = -y;
    z = -z;
}

void BlochCell::apply_y() noexcept {
    x = -x;
    z = -z;
}

void BlochCell::apply_z() noexcept {
    x = -x;
    y = -y;
}

void BlochCell::apply_h() noexcept {
    const double previous_x = x;
    x = z;
    y = -y;
    z = previous_x;
}

void BlochCell::rotate_x(double theta) noexcept {
    const double cosine = std::cos(theta);
    const double sine = std::sin(theta);
    const double previous_y = y;
    y = cosine * y - sine * z;
    z = sine * previous_y + cosine * z;
}

void BlochCell::rotate_y(double theta) noexcept {
    const double cosine = std::cos(theta);
    const double sine = std::sin(theta);
    const double previous_x = x;
    x = cosine * x + sine * z;
    z = -sine * previous_x + cosine * z;
}

void BlochCell::rotate_z(double theta) noexcept {
    const double cosine = std::cos(theta);
    const double sine = std::sin(theta);
    const double previous_x = x;
    x = cosine * x - sine * y;
    y = sine * previous_x + cosine * y;
}

AmplitudeStore AmplitudeStore::from_entries(
    BasisIndex dimension,
    std::vector<SparseEntry> entries,
    const QStateConfig& config,
    bool normalize_values) {
    AmplitudeStore store;
    store.assign_entries(dimension, std::move(entries), config, normalize_values);
    return store;
}

AmplitudeStore AmplitudeStore::from_dense(
    std::vector<QComplex> amplitudes,
    const QStateConfig& config,
    bool normalize_values) {
    if (amplitudes.empty() || !is_power_of_two(static_cast<BasisIndex>(amplitudes.size()))) {
        throw QStateError("Dense amplitude storage must have a power-of-two size");
    }
    if (amplitudes.size() > config.max_dense_amplitudes) {
        throw QStateError("Dense amplitude allocation exceeds configured limit");
    }
    AmplitudeStore store;
    store.dimension_ = static_cast<BasisIndex>(amplitudes.size());
    store.mode_ = StorageMode::Dense;
    store.dense_ = std::move(amplitudes);
    if (normalize_values) {
        store.normalize(config.epsilon);
    }
    store.rebalance(config);
    return store;
}

std::size_t AmplitudeStore::nonzero_count() const noexcept {
    if (mode_ == StorageMode::Sparse) {
        return sparse_.size();
    }
    return static_cast<std::size_t>(std::count_if(
        dense_.begin(), dense_.end(), [](const QComplex& value) { return value.norm2() > 0.0; }));
}

QComplex AmplitudeStore::at(BasisIndex index) const {
    if (index >= dimension_) {
        throw QStateError("Amplitude index is out of range");
    }
    if (mode_ == StorageMode::Dense) {
        return dense_[static_cast<std::size_t>(index)];
    }
    const auto iterator = std::lower_bound(
        sparse_.begin(), sparse_.end(), index,
        [](const SparseEntry& entry, BasisIndex target) { return entry.first < target; });
    return iterator != sparse_.end() && iterator->first == index ? iterator->second : QComplex{};
}

std::vector<AmplitudeStore::SparseEntry> AmplitudeStore::entries(double epsilon) const {
    const double threshold = epsilon * epsilon;
    if (mode_ == StorageMode::Sparse) {
        if (epsilon <= 0.0) {
            return sparse_;
        }
        std::vector<SparseEntry> result;
        result.reserve(sparse_.size());
        for (const auto& entry : sparse_) {
            if (entry.second.norm2() > threshold) {
                result.push_back(entry);
            }
        }
        return result;
    }
    std::vector<SparseEntry> result;
    for (std::size_t index = 0; index < dense_.size(); ++index) {
        if (dense_[index].norm2() > threshold) {
            result.emplace_back(static_cast<BasisIndex>(index), dense_[index]);
        }
    }
    return result;
}

std::vector<QComplex> AmplitudeStore::dense_copy() const {
    if (dimension_ > static_cast<BasisIndex>(std::numeric_limits<std::size_t>::max())) {
        throw QStateError("Amplitude dimension cannot be materialized on this platform");
    }
    if (mode_ == StorageMode::Dense) {
        return dense_;
    }
    std::vector<QComplex> result(static_cast<std::size_t>(dimension_));
    for (const auto& [index, value] : sparse_) {
        result[static_cast<std::size_t>(index)] = value;
    }
    return result;
}

std::size_t AmplitudeStore::estimated_bytes() const noexcept {
    if (mode_ == StorageMode::Dense) {
        return dense_.capacity() * sizeof(QComplex);
    }
    return sparse_.capacity() * sizeof(SparseEntry);
}

void AmplitudeStore::normalize(double epsilon) {
    long double total = 0.0L;
    if (mode_ == StorageMode::Dense) {
        for (const QComplex& value : dense_) {
            total += static_cast<long double>(value.norm2());
        }
    } else {
        for (const auto& entry : sparse_) {
            total += static_cast<long double>(entry.second.norm2());
        }
    }
    const double norm = std::sqrt(static_cast<double>(total));
    if (!std::isfinite(norm) || norm <= epsilon) {
        throw QStateError("Quantum state has zero or non-finite norm");
    }
    if (mode_ == StorageMode::Dense) {
        for (QComplex& value : dense_) {
            value /= norm;
            if (value.norm2() <= epsilon * epsilon) {
                value = {};
            }
        }
    } else {
        for (auto& entry : sparse_) {
            entry.second /= norm;
        }
        std::erase_if(sparse_, [epsilon](const SparseEntry& entry) {
            return entry.second.norm2() <= epsilon * epsilon;
        });
    }
}

void AmplitudeStore::apply_single(
    std::size_t bit_position,
    const QMatrix2& matrix,
    const QStateConfig& config,
    bool renormalize) {
    const std::size_t bits = bit_count_for_dimension(dimension_);
    if (bit_position >= bits) {
        throw QStateError("Single-qubit gate bit position is out of range");
    }
    const BasisIndex mask = BasisIndex{1} << bit_position;
    if (mode_ == StorageMode::Dense) {
        for (BasisIndex base = 0; base < dimension_; ++base) {
            if ((base & mask) != 0U) {
                continue;
            }
            const BasisIndex one_index = base | mask;
            const QComplex zero = dense_[static_cast<std::size_t>(base)];
            const QComplex one = dense_[static_cast<std::size_t>(one_index)];
            dense_[static_cast<std::size_t>(base)] = matrix(0, 0) * zero + matrix(0, 1) * one;
            dense_[static_cast<std::size_t>(one_index)] = matrix(1, 0) * zero + matrix(1, 1) * one;
        }
        if (renormalize) {
            normalize(config.epsilon);
        }
        rebalance(config);
        return;
    }

    struct ReducedEntry {
        BasisIndex index;
        QComplex amplitude;
    };
    std::vector<ReducedEntry> zeros;
    std::vector<ReducedEntry> ones;
    zeros.reserve(sparse_.size());
    ones.reserve(sparse_.size());
    for (const auto& [index, amplitude] : sparse_) {
        const ReducedEntry reduced{remove_bit(index, bit_position), amplitude};
        ((index & mask) == 0U ? zeros : ones).push_back(reduced);
    }

    std::vector<SparseEntry> output_zero;
    std::vector<SparseEntry> output_one;
    output_zero.reserve(zeros.size() + ones.size());
    output_one.reserve(zeros.size() + ones.size());
    const double threshold = config.epsilon * config.epsilon;
    std::size_t zero_index = 0;
    std::size_t one_index = 0;
    while (zero_index < zeros.size() || one_index < ones.size()) {
        const BasisIndex reduced_index = one_index >= ones.size()
                                             ? zeros[zero_index].index
                                             : zero_index >= zeros.size()
                                                   ? ones[one_index].index
                                                   : std::min(
                                                         zeros[zero_index].index,
                                                         ones[one_index].index);
        QComplex zero{};
        QComplex one{};
        if (zero_index < zeros.size() && zeros[zero_index].index == reduced_index) {
            zero = zeros[zero_index++].amplitude;
        }
        if (one_index < ones.size() && ones[one_index].index == reduced_index) {
            one = ones[one_index++].amplitude;
        }
        const QComplex next_zero = matrix(0, 0) * zero + matrix(0, 1) * one;
        const QComplex next_one = matrix(1, 0) * zero + matrix(1, 1) * one;
        const BasisIndex base = insert_zero_bit(reduced_index, bit_position);
        if (next_zero.norm2() > threshold) {
            output_zero.emplace_back(base, next_zero);
        }
        if (next_one.norm2() > threshold) {
            output_one.emplace_back(base | mask, next_one);
        }
    }

    std::vector<SparseEntry> entries_out;
    entries_out.reserve(output_zero.size() + output_one.size());
    std::merge(
        output_zero.begin(),
        output_zero.end(),
        output_one.begin(),
        output_one.end(),
        std::back_inserter(entries_out),
        [](const SparseEntry& lhs, const SparseEntry& rhs) {
            return lhs.first < rhs.first;
        });
    assign_sorted_entries(dimension_, std::move(entries_out), config, renormalize);
}

void AmplitudeStore::apply_two(
    std::size_t first_bit,
    std::size_t second_bit,
    const QMatrix4& matrix,
    const QStateConfig& config,
    bool renormalize) {
    const std::size_t bits = bit_count_for_dimension(dimension_);
    if (first_bit >= bits || second_bit >= bits || first_bit == second_bit) {
        throw QStateError("Two-qubit gate bit positions are invalid");
    }
    const BasisIndex first_mask = BasisIndex{1} << first_bit;
    const BasisIndex second_mask = BasisIndex{1} << second_bit;

    if (mode_ == StorageMode::Dense) {
        for (BasisIndex base = 0; base < dimension_; ++base) {
            if ((base & first_mask) != 0U || (base & second_mask) != 0U) {
                continue;
            }
            const std::array<BasisIndex, 4> indices{
                base,
                base | second_mask,
                base | first_mask,
                base | first_mask | second_mask,
            };
            std::array<QComplex, 4> input{};
            for (std::size_t i = 0; i < 4; ++i) {
                input[i] = dense_[static_cast<std::size_t>(indices[i])];
            }
            for (std::size_t row = 0; row < 4; ++row) {
                QComplex value{};
                for (std::size_t column = 0; column < 4; ++column) {
                    value += matrix(row, column) * input[column];
                }
                dense_[static_cast<std::size_t>(indices[row])] = value;
            }
        }
        if (renormalize) {
            normalize(config.epsilon);
        }
        rebalance(config);
        return;
    }

    std::unordered_map<BasisIndex, QComplex> output;
    const std::size_t multiplier = sparse_.size() > config.max_sparse_entries / 4U ? 1U : 4U;
    output.reserve(std::min(config.max_sparse_entries, sparse_.size() * multiplier + 1U));
    for (const auto& [index, amplitude] : sparse_) {
        const std::size_t first_value = (index & first_mask) == 0U ? 0U : 1U;
        const std::size_t second_value = (index & second_mask) == 0U ? 0U : 1U;
        const std::size_t input_basis = (first_value << 1U) | second_value;
        const BasisIndex cleared = index & ~first_mask & ~second_mask;
        for (std::size_t output_basis = 0; output_basis < 4; ++output_basis) {
            const QComplex coefficient = matrix(output_basis, input_basis);
            if (coefficient.norm2() <= config.epsilon * config.epsilon) {
                continue;
            }
            BasisIndex output_index = cleared;
            if ((output_basis & 2U) != 0U) {
                output_index |= first_mask;
            }
            if ((output_basis & 1U) != 0U) {
                output_index |= second_mask;
            }
            output[output_index] += coefficient * amplitude;
        }
    }
    std::vector<SparseEntry> entries_out;
    entries_out.reserve(output.size());
    for (const auto& entry : output) {
        entries_out.push_back(entry);
    }
    assign_entries(dimension_, std::move(entries_out), config, renormalize);
}

void AmplitudeStore::sort_sparse() {
    std::sort(sparse_.begin(), sparse_.end(), [](const SparseEntry& lhs, const SparseEntry& rhs) {
        return lhs.first < rhs.first;
    });
}

void AmplitudeStore::apply_x(std::size_t bit_position) {
    const std::size_t bits = bit_count_for_dimension(dimension_);
    if (bit_position >= bits) {
        throw QStateError("X gate bit position is out of range");
    }
    const BasisIndex mask = BasisIndex{1} << bit_position;
    if (mode_ == StorageMode::Dense) {
        for (BasisIndex base = 0; base < dimension_; ++base) {
            if ((base & mask) == 0U) {
                std::swap(
                    dense_[static_cast<std::size_t>(base)],
                    dense_[static_cast<std::size_t>(base | mask)]);
            }
        }
        return;
    }
    for (auto& entry : sparse_) {
        entry.first ^= mask;
    }
    sort_sparse();
}

void AmplitudeStore::apply_y(std::size_t bit_position) {
    const std::size_t bits = bit_count_for_dimension(dimension_);
    if (bit_position >= bits) {
        throw QStateError("Y gate bit position is out of range");
    }
    const BasisIndex mask = BasisIndex{1} << bit_position;
    if (mode_ == StorageMode::Dense) {
        for (BasisIndex base = 0; base < dimension_; ++base) {
            if ((base & mask) != 0U) {
                continue;
            }
            const BasisIndex one_index = base | mask;
            const QComplex zero = dense_[static_cast<std::size_t>(base)];
            const QComplex one = dense_[static_cast<std::size_t>(one_index)];
            dense_[static_cast<std::size_t>(base)] = -QI * one;
            dense_[static_cast<std::size_t>(one_index)] = QI * zero;
        }
        return;
    }
    for (auto& [index, amplitude] : sparse_) {
        const bool one = (index & mask) != 0U;
        amplitude = one ? -QI * amplitude : QI * amplitude;
        index ^= mask;
    }
    sort_sparse();
}

void AmplitudeStore::apply_phase(
    std::size_t bit_position,
    QComplex phase_zero,
    QComplex phase_one) {
    const std::size_t bits = bit_count_for_dimension(dimension_);
    if (bit_position >= bits) {
        throw QStateError("Phase gate bit position is out of range");
    }
    const BasisIndex mask = BasisIndex{1} << bit_position;
    if (mode_ == StorageMode::Dense) {
        for (BasisIndex index = 0; index < dimension_; ++index) {
            dense_[static_cast<std::size_t>(index)] *=
                (index & mask) == 0U ? phase_zero : phase_one;
        }
        return;
    }
    for (auto& [index, amplitude] : sparse_) {
        amplitude *= (index & mask) == 0U ? phase_zero : phase_one;
    }
}

void AmplitudeStore::apply_z(std::size_t bit_position) {
    apply_phase(bit_position, QComplex{1.0, 0.0}, QComplex{-1.0, 0.0});
}

void AmplitudeStore::apply_cnot(std::size_t control_bit, std::size_t target_bit) {
    const std::size_t bits = bit_count_for_dimension(dimension_);
    if (control_bit >= bits || target_bit >= bits || control_bit == target_bit) {
        throw QStateError("CNOT bit positions are invalid");
    }
    const BasisIndex control_mask = BasisIndex{1} << control_bit;
    const BasisIndex target_mask = BasisIndex{1} << target_bit;
    if (mode_ == StorageMode::Dense) {
        for (BasisIndex index = 0; index < dimension_; ++index) {
            if ((index & control_mask) != 0U && (index & target_mask) == 0U) {
                std::swap(
                    dense_[static_cast<std::size_t>(index)],
                    dense_[static_cast<std::size_t>(index | target_mask)]);
            }
        }
        return;
    }
    bool reordered = false;
    for (auto& entry : sparse_) {
        if ((entry.first & control_mask) != 0U) {
            entry.first ^= target_mask;
            reordered = true;
        }
    }
    if (reordered) {
        sort_sparse();
    }
}

void AmplitudeStore::apply_cz(std::size_t first_bit, std::size_t second_bit) {
    const std::size_t bits = bit_count_for_dimension(dimension_);
    if (first_bit >= bits || second_bit >= bits || first_bit == second_bit) {
        throw QStateError("CZ bit positions are invalid");
    }
    const BasisIndex mask = (BasisIndex{1} << first_bit) | (BasisIndex{1} << second_bit);
    if (mode_ == StorageMode::Dense) {
        for (BasisIndex index = 0; index < dimension_; ++index) {
            if ((index & mask) == mask) {
                dense_[static_cast<std::size_t>(index)] =
                    -dense_[static_cast<std::size_t>(index)];
            }
        }
        return;
    }
    for (auto& [index, amplitude] : sparse_) {
        if ((index & mask) == mask) {
            amplitude = -amplitude;
        }
    }
}

void AmplitudeStore::apply_swap(std::size_t first_bit, std::size_t second_bit) {
    const std::size_t bits = bit_count_for_dimension(dimension_);
    if (first_bit >= bits || second_bit >= bits || first_bit == second_bit) {
        throw QStateError("SWAP bit positions are invalid");
    }
    const BasisIndex first_mask = BasisIndex{1} << first_bit;
    const BasisIndex second_mask = BasisIndex{1} << second_bit;
    const BasisIndex both = first_mask | second_mask;
    if (mode_ == StorageMode::Dense) {
        for (BasisIndex index = 0; index < dimension_; ++index) {
            if ((index & first_mask) == 0U && (index & second_mask) != 0U) {
                std::swap(
                    dense_[static_cast<std::size_t>(index)],
                    dense_[static_cast<std::size_t>(index ^ both)]);
            }
        }
        return;
    }
    bool reordered = false;
    for (auto& entry : sparse_) {
        const bool first = (entry.first & first_mask) != 0U;
        const bool second = (entry.first & second_mask) != 0U;
        if (first != second) {
            entry.first ^= both;
            reordered = true;
        }
    }
    if (reordered) {
        sort_sparse();
    }
}

void AmplitudeStore::assign_entries(
    BasisIndex dimension,
    std::vector<SparseEntry> entries_in,
    const QStateConfig& config,
    bool normalize_values) {
    if (!is_power_of_two(dimension)) {
        throw QStateError("Amplitude dimension must be a power of two");
    }
    std::sort(entries_in.begin(), entries_in.end(), [](const SparseEntry& lhs, const SparseEntry& rhs) {
        return lhs.first < rhs.first;
    });

    std::vector<SparseEntry> combined;
    combined.reserve(entries_in.size());
    for (const auto& [index, value] : entries_in) {
        if (index >= dimension) {
            throw QStateError("Sparse amplitude index exceeds its dimension");
        }
        if (!combined.empty() && combined.back().first == index) {
            combined.back().second += value;
        } else {
            combined.emplace_back(index, value);
        }
    }
    assign_sorted_entries(dimension, std::move(combined), config, normalize_values);
}

void AmplitudeStore::assign_sorted_entries(
    BasisIndex dimension,
    std::vector<SparseEntry> combined,
    const QStateConfig& config,
    bool normalize_values) {
    if (!is_power_of_two(dimension)) {
        throw QStateError("Amplitude dimension must be a power of two");
    }
    if (!std::is_sorted(
            combined.begin(),
            combined.end(),
            [](const SparseEntry& lhs, const SparseEntry& rhs) {
                return lhs.first < rhs.first;
            })) {
        throw QStateError("Sorted sparse assignment received unordered entries");
    }
    if (!combined.empty() && combined.back().first >= dimension) {
        throw QStateError("Sparse amplitude index exceeds its dimension");
    }
    std::erase_if(combined, [&config](const SparseEntry& entry) {
        return entry.second.norm2() <= config.epsilon * config.epsilon;
    });
    if (combined.empty()) {
        throw QStateError("Quantum state cannot contain zero amplitudes only");
    }

    dimension_ = dimension;
    const bool dense_possible = dimension <= config.max_dense_amplitudes &&
                                dimension <= static_cast<BasisIndex>(std::numeric_limits<std::size_t>::max());
    const bool dense_preferred = dense_possible &&
                                 static_cast<long double>(combined.size()) * 2.0L >
                                     static_cast<long double>(dimension);

    if (!dense_preferred && combined.size() > config.max_sparse_entries) {
        if (!dense_possible) {
            throw QStateError("Sparse state exceeds configured entry limit and cannot be promoted to dense storage");
        }
    }

    if (dense_preferred || combined.size() > config.max_sparse_entries) {
        mode_ = StorageMode::Dense;
        sparse_.clear();
        dense_.assign(static_cast<std::size_t>(dimension), QComplex{});
        for (const auto& [index, value] : combined) {
            dense_[static_cast<std::size_t>(index)] = value;
        }
    } else {
        mode_ = StorageMode::Sparse;
        dense_.clear();
        sparse_ = std::move(combined);
    }

    if (normalize_values) {
        normalize(config.epsilon);
    }
}

void AmplitudeStore::rebalance(const QStateConfig& config) {
    if (mode_ == StorageMode::Dense) {
        const double threshold = config.epsilon * config.epsilon;
        const std::size_t nonzero_count = static_cast<std::size_t>(std::count_if(
            dense_.begin(), dense_.end(), [threshold](const QComplex& value) {
                return value.norm2() > threshold;
            }));
        if (nonzero_count <= config.max_sparse_entries &&
            static_cast<long double>(nonzero_count) * 4.0L < static_cast<long double>(dimension_)) {
            std::vector<SparseEntry> nonzero;
            nonzero.reserve(nonzero_count);
            for (std::size_t index = 0; index < dense_.size(); ++index) {
                if (dense_[index].norm2() > threshold) {
                    nonzero.emplace_back(static_cast<BasisIndex>(index), dense_[index]);
                }
            }
            mode_ = StorageMode::Sparse;
            sparse_ = std::move(nonzero);
            dense_.clear();
            dense_.shrink_to_fit();
        }
        return;
    }

    const bool dense_possible = dimension_ <= config.max_dense_amplitudes &&
                                dimension_ <= static_cast<BasisIndex>(std::numeric_limits<std::size_t>::max());
    if (dense_possible &&
        static_cast<long double>(sparse_.size()) * 2.0L > static_cast<long double>(dimension_)) {
        mode_ = StorageMode::Dense;
        dense_.assign(static_cast<std::size_t>(dimension_), QComplex{});
        for (const auto& [index, value] : sparse_) {
            dense_[static_cast<std::size_t>(index)] = value;
        }
        sparse_.clear();
        sparse_.shrink_to_fit();
    }
}

QRegister::QRegister(std::size_t qubit_count, QStateConfig config)
    : qubit_count_(qubit_count), config_(config), qubit_component_(qubit_count) {
    if (qubit_count == 0) {
        throw QStateError("QRegister requires at least one qubit");
    }
    if (qubit_count > static_cast<std::size_t>(std::numeric_limits<QubitId>::max())) {
        throw QStateError("QRegister qubit count exceeds QubitId capacity");
    }
    if (config_.max_component_qubits == 0 || config_.max_component_qubits >= 63) {
        throw QStateError("max_component_qubits must be between 1 and 62");
    }
    components_.reserve(qubit_count);
    component_order_.reserve(qubit_count);
    for (std::size_t qubit = 0; qubit < qubit_count; ++qubit) {
        components_.push_back(StateComponent{{static_cast<QubitId>(qubit)}, BlochCell{}});
        component_order_.push_back(static_cast<std::uint32_t>(next_component_order_++));
    }
    reindex_components();
}


QRegister QRegister::from_amplitudes(
    std::vector<QComplex> amplitudes,
    QStateConfig config) {
    if (amplitudes.size() < 2U ||
        !is_power_of_two(static_cast<BasisIndex>(amplitudes.size()))) {
        throw QStateError("QRegister amplitude initialization requires a power-of-two size of at least two");
    }
    const std::size_t qubits = bit_count_for_dimension(
        static_cast<BasisIndex>(amplitudes.size()));
    if (qubits > config.max_component_qubits) {
        throw QStateError("QRegister amplitude initialization exceeds the configured component limit");
    }
    QRegister state(qubits, config);
    AmplitudeStore store = AmplitudeStore::from_dense(
        std::move(amplitudes), state.config_, true);
    state.components_.clear();
    state.component_order_.clear();
    state.next_component_order_ = 0;
    std::vector<QubitId> members(qubits);
    std::iota(members.begin(), members.end(), QubitId{0});
    if (qubits == 1U) {
        state.components_.push_back(StateComponent{
            std::move(members),
            BlochCell::from_amplitudes(store.at(0U), store.at(1U), state.config_.epsilon),
        });
    } else {
        state.components_.push_back(StateComponent{std::move(members), std::move(store)});
    }
    state.component_order_.push_back(0U);
    state.next_component_order_ = 1U;
    state.reindex_components();
    return state;
}

void QRegister::validate_qubit(QubitId qubit) const {
    if (static_cast<std::size_t>(qubit) >= qubit_count_) {
        throw QStateError("Qubit index is out of range");
    }
}

void QRegister::reindex_components() {
    if (component_order_.size() != components_.size()) {
        throw QStateError("Internal component-order table is inconsistent");
    }
    qubit_component_.assign(qubit_count_, std::numeric_limits<std::size_t>::max());
    for (std::size_t component = 0; component < components_.size(); ++component) {
        for (QubitId qubit : components_[component].qubits) {
            if (static_cast<std::size_t>(qubit) >= qubit_count_) {
                throw QStateError("Component contains an out-of-range qubit");
            }
            if (qubit_component_[qubit] != std::numeric_limits<std::size_t>::max()) {
                throw QStateError("Qubit appears in more than one state component");
            }
            qubit_component_[qubit] = component;
        }
    }
}

std::vector<std::size_t> QRegister::ordered_component_indices() const {
    if (component_order_.size() != components_.size()) {
        throw QStateError("Internal component-order table is inconsistent");
    }
    std::vector<std::size_t> indices(components_.size());
    std::iota(indices.begin(), indices.end(), std::size_t{0});
    std::sort(indices.begin(), indices.end(), [this](std::size_t lhs, std::size_t rhs) {
        return component_order_[lhs] < component_order_[rhs];
    });
    return indices;
}

void QRegister::renumber_component_order() {
    const auto ordered = ordered_component_indices();
    for (std::size_t logical = 0; logical < ordered.size(); ++logical) {
        component_order_[ordered[logical]] = static_cast<std::uint32_t>(logical);
    }
    next_component_order_ = ordered.size();
}

std::size_t QRegister::append_component(StateComponent component) {
    if (next_component_order_ > std::numeric_limits<std::uint32_t>::max()) {
        renumber_component_order();
    }
    const std::size_t index = components_.size();
    components_.push_back(std::move(component));
    component_order_.push_back(static_cast<std::uint32_t>(next_component_order_++));
    for (QubitId qubit : components_.back().qubits) {
        if (static_cast<std::size_t>(qubit) >= qubit_count_) {
            throw QStateError("Component contains an out-of-range qubit");
        }
        qubit_component_[qubit] = index;
    }
    return index;
}

void QRegister::remove_component(std::size_t component_index_value) {
    if (component_index_value >= components_.size()) {
        throw QStateError("Cannot remove an invalid state component");
    }
    const std::size_t last = components_.size() - 1U;
    if (component_index_value != last) {
        components_[component_index_value] = std::move(components_[last]);
        component_order_[component_index_value] = component_order_[last];
        for (QubitId qubit : components_[component_index_value].qubits) {
            qubit_component_[qubit] = component_index_value;
        }
    }
    components_.pop_back();
    component_order_.pop_back();
}

std::size_t QRegister::component_index(QubitId qubit) const {
    validate_qubit(qubit);
    const std::size_t index = qubit_component_[qubit];
    if (index >= components_.size()) {
        throw QStateError("Internal component index is invalid");
    }
    return index;
}

std::size_t QRegister::local_position(const StateComponent& component, QubitId qubit) const {
    const auto iterator = std::find(component.qubits.begin(), component.qubits.end(), qubit);
    if (iterator == component.qubits.end()) {
        throw QStateError("Qubit is not present in the selected component");
    }
    return static_cast<std::size_t>(std::distance(component.qubits.begin(), iterator));
}

void QRegister::apply_cell_matrix(BlochCell& cell, const QMatrix2& matrix) {
    const auto input = cell.amplitudes(config_.epsilon);
    const QComplex zero = matrix(0, 0) * input[0] + matrix(0, 1) * input[1];
    const QComplex one = matrix(1, 0) * input[0] + matrix(1, 1) * input[1];
    cell = BlochCell::from_amplitudes(zero, one, config_.epsilon);
}

void QRegister::apply_x(QubitId qubit) {
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        std::get<BlochCell>(components_[index].state).apply_x();
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_x(local_position(components_[index], qubit));
    }
}

void QRegister::apply_y(QubitId qubit) {
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        std::get<BlochCell>(components_[index].state).apply_y();
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_y(local_position(components_[index], qubit));
    }
}

void QRegister::apply_z(QubitId qubit) {
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        std::get<BlochCell>(components_[index].state).apply_z();
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_z(local_position(components_[index], qubit));
    }
}

void QRegister::apply_h(QubitId qubit) {
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        std::get<BlochCell>(components_[index].state).apply_h();
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_single(local_position(components_[index], qubit), gates::h(), config_, false);
    }
}

void QRegister::apply_s(QubitId qubit) {
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        std::get<BlochCell>(components_[index].state).rotate_z(kPi / 2.0);
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_phase(local_position(components_[index], qubit), QComplex{1.0, 0.0}, QI);
    }
}

void QRegister::apply_sdg(QubitId qubit) {
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        std::get<BlochCell>(components_[index].state).rotate_z(-kPi / 2.0);
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_phase(local_position(components_[index], qubit), QComplex{1.0, 0.0}, -QI);
    }
}

void QRegister::apply_t(QubitId qubit) {
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        std::get<BlochCell>(components_[index].state).rotate_z(kPi / 4.0);
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_phase(
                local_position(components_[index], qubit),
                QComplex{1.0, 0.0},
                QComplex::from_polar(1.0, kPi / 4.0));
    }
}

void QRegister::apply_tdg(QubitId qubit) {
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        std::get<BlochCell>(components_[index].state).rotate_z(-kPi / 4.0);
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_phase(
                local_position(components_[index], qubit),
                QComplex{1.0, 0.0},
                QComplex::from_polar(1.0, -kPi / 4.0));
    }
}

void QRegister::apply_rx(QubitId qubit, double theta) {
    if (!std::isfinite(theta)) {
        throw QStateError("Rotation angle must be finite");
    }
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        std::get<BlochCell>(components_[index].state).rotate_x(theta);
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_single(local_position(components_[index], qubit), gates::rx(theta), config_, false);
    }
}

void QRegister::apply_ry(QubitId qubit, double theta) {
    if (!std::isfinite(theta)) {
        throw QStateError("Rotation angle must be finite");
    }
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        std::get<BlochCell>(components_[index].state).rotate_y(theta);
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_single(local_position(components_[index], qubit), gates::ry(theta), config_, false);
    }
}

void QRegister::apply_rz(QubitId qubit, double theta) {
    if (!std::isfinite(theta)) {
        throw QStateError("Rotation angle must be finite");
    }
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        std::get<BlochCell>(components_[index].state).rotate_z(theta);
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_phase(
                local_position(components_[index], qubit),
                QComplex::from_polar(1.0, -theta / 2.0),
                QComplex::from_polar(1.0, theta / 2.0));
    }
}

void QRegister::apply_single(QubitId qubit, const QMatrix2& matrix) {
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        apply_cell_matrix(std::get<BlochCell>(components_[index].state), matrix);
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_single(local_position(components_[index], qubit), matrix, config_);
    }
}

void QRegister::apply_diagonal(std::span<const QDiagonalPhase> phases) {
    using LocalPhase = std::pair<std::size_t, std::array<QComplex, 2>>;
    std::vector<std::vector<LocalPhase>> grouped(components_.size());

    for (const QDiagonalPhase& phase : phases) {
        validate_qubit(phase.qubit);
        const double zero_norm = phase.zero.norm2();
        const double one_norm = phase.one.norm2();
        if (!std::isfinite(zero_norm) || !std::isfinite(one_norm) ||
            std::abs(zero_norm - 1.0) > 1e-10 || std::abs(one_norm - 1.0) > 1e-10) {
            throw QStateError("Diagonal phase entries must be finite unit-magnitude values");
        }

        const std::size_t index = component_index(phase.qubit);
        StateComponent& component = components_[index];
        if (component.is_cell()) {
            QMatrix2 matrix{};
            matrix.values[0] = phase.zero;
            matrix.values[3] = phase.one;
            apply_cell_matrix(std::get<BlochCell>(component.state), matrix);
        } else {
            grouped[index].push_back(LocalPhase{
                local_position(component, phase.qubit),
                std::array<QComplex, 2>{phase.zero, phase.one},
            });
        }
    }

    for (std::size_t index = 0; index < grouped.size(); ++index) {
        if (grouped[index].empty()) {
            continue;
        }
        AmplitudeStore& store = std::get<AmplitudeStore>(components_[index].state);
        const auto apply_to_value = [&grouped, index](BasisIndex basis, QComplex& value) {
            QComplex coefficient{1.0, 0.0};
            for (const auto& [position, values] : grouped[index]) {
                coefficient *= values[(basis >> position) & 1U];
            }
            value *= coefficient;
        };
        if (store.mode_ == StorageMode::Sparse) {
            for (auto& [basis, value] : store.sparse_) {
                apply_to_value(basis, value);
            }
        } else {
            for (std::size_t basis = 0; basis < store.dense_.size(); ++basis) {
                apply_to_value(static_cast<BasisIndex>(basis), store.dense_[basis]);
            }
        }
    }
}

std::size_t QRegister::merge_components(std::size_t first, std::size_t second) {
    if (first == second) {
        return first;
    }
    if (first >= components_.size() || second >= components_.size()) {
        throw QStateError("Cannot merge an invalid state component");
    }

    StateComponent left = std::move(components_[first]);
    StateComponent right = std::move(components_[second]);
    std::vector<QubitId> merged_qubits = left.qubits;
    merged_qubits.insert(merged_qubits.end(), right.qubits.begin(), right.qubits.end());
    std::sort(merged_qubits.begin(), merged_qubits.end());
    if (merged_qubits.size() > config_.max_component_qubits || merged_qubits.size() >= 63) {
        throw QStateError("Entangled component exceeds configured qubit limit");
    }

    std::vector<std::size_t> left_positions(left.qubits.size());
    std::vector<std::size_t> right_positions(right.qubits.size());
    std::size_t left_cursor = 0;
    std::size_t right_cursor = 0;
    for (std::size_t merged_position = 0; merged_position < merged_qubits.size(); ++merged_position) {
        const QubitId qubit = merged_qubits[merged_position];
        if (left_cursor < left.qubits.size() && left.qubits[left_cursor] == qubit) {
            left_positions[left_cursor++] = merged_position;
        } else if (right_cursor < right.qubits.size() && right.qubits[right_cursor] == qubit) {
            right_positions[right_cursor++] = merged_position;
        } else {
            throw QStateError("Internal error: component merge mapping is inconsistent");
        }
    }

    const auto component_entries = [this](const StateComponent& component) {
        if (component.is_cell()) {
            const auto amplitudes = std::get<BlochCell>(component.state).amplitudes(config_.epsilon);
            std::vector<AmplitudeStore::SparseEntry> result;
            if (amplitudes[0].norm2() > config_.epsilon * config_.epsilon) {
                result.emplace_back(0U, amplitudes[0]);
            }
            if (amplitudes[1].norm2() > config_.epsilon * config_.epsilon) {
                result.emplace_back(1U, amplitudes[1]);
            }
            return result;
        }
        return std::get<AmplitudeStore>(component.state).entries(config_.epsilon);
    };

    const auto left_entries = component_entries(left);
    const auto right_entries = component_entries(right);
    const long double product_size = static_cast<long double>(left_entries.size()) *
                                     static_cast<long double>(right_entries.size());
    const BasisIndex dimension = BasisIndex{1} << merged_qubits.size();
    const long double practical_limit = static_cast<long double>(
        std::max<std::uint64_t>(config_.max_dense_amplitudes, config_.max_sparse_entries));
    if (product_size > practical_limit && dimension > config_.max_dense_amplitudes) {
        throw QStateError("Component merge would exceed configured state capacity");
    }

    std::vector<AmplitudeStore::SparseEntry> merged_entries;
    merged_entries.reserve(static_cast<std::size_t>(product_size));
    for (const auto& [left_index, left_value] : left_entries) {
        const BasisIndex remapped_left = remap_index(left_index, left_positions);
        for (const auto& [right_index, right_value] : right_entries) {
            const BasisIndex remapped_right = remap_index(right_index, right_positions);
            merged_entries.emplace_back(remapped_left | remapped_right, left_value * right_value);
        }
    }

    StateComponent merged{
        std::move(merged_qubits),
        AmplitudeStore::from_entries(dimension, std::move(merged_entries), config_),
    };

    const std::size_t high = std::max(first, second);
    const std::size_t low = std::min(first, second);
    remove_component(high);
    remove_component(low);
    return append_component(std::move(merged));
}

void QRegister::apply_cnot(QubitId control, QubitId target) {
    validate_qubit(control);
    validate_qubit(target);
    if (control == target) {
        throw QStateError("A two-qubit gate requires two distinct qubits");
    }
    std::size_t component = component_index(control);
    const std::size_t other = component_index(target);
    if (component != other) {
        component = merge_components(component, other);
    }
    StateComponent& selected = components_[component];
    const std::size_t control_position = local_position(selected, control);
    const std::size_t target_position = local_position(selected, target);
    std::get<AmplitudeStore>(selected.state).apply_cnot(control_position, target_position);
    const std::array<QubitId, 2> candidates{control, target};
    compact_component_targets(component, candidates);
}

void QRegister::apply_cz(QubitId first, QubitId second) {
    validate_qubit(first);
    validate_qubit(second);
    if (first == second) {
        throw QStateError("A two-qubit gate requires two distinct qubits");
    }
    std::size_t component = component_index(first);
    const std::size_t other = component_index(second);
    if (component != other) {
        component = merge_components(component, other);
    }
    StateComponent& selected = components_[component];
    const std::size_t first_position = local_position(selected, first);
    const std::size_t second_position = local_position(selected, second);
    std::get<AmplitudeStore>(selected.state).apply_cz(first_position, second_position);
    const std::array<QubitId, 2> candidates{first, second};
    compact_component_targets(component, candidates);
}

void QRegister::apply_swap(QubitId first, QubitId second) {
    validate_qubit(first);
    validate_qubit(second);
    if (first == second) {
        throw QStateError("A two-qubit gate requires two distinct qubits");
    }
    std::size_t component = component_index(first);
    const std::size_t other = component_index(second);
    if (component != other) {
        component = merge_components(component, other);
    }
    StateComponent& selected = components_[component];
    const std::size_t first_position = local_position(selected, first);
    const std::size_t second_position = local_position(selected, second);
    std::get<AmplitudeStore>(selected.state).apply_swap(first_position, second_position);
    const std::array<QubitId, 2> candidates{first, second};
    compact_component_targets(component, candidates);
}

void QRegister::apply_two(QubitId first, QubitId second, const QMatrix4& matrix) {
    validate_qubit(first);
    validate_qubit(second);
    if (first == second) {
        throw QStateError("A two-qubit gate requires two distinct qubits");
    }
    std::size_t component = component_index(first);
    const std::size_t other = component_index(second);
    if (component != other) {
        component = merge_components(component, other);
    }
    StateComponent& selected = components_[component];
    if (selected.is_cell()) {
        throw QStateError("Internal error: merged two-qubit component is a cell");
    }
    const std::size_t first_position = local_position(selected, first);
    const std::size_t second_position = local_position(selected, second);
    std::get<AmplitudeStore>(selected.state)
        .apply_two(first_position, second_position, matrix, config_);
    const std::array<QubitId, 2> candidates{first, second};
    compact_component_targets(component, candidates);
}

void QRegister::apply_bit_flip_trajectory(QubitId qubit, double probability, double sample) {
    validate_probability(probability, "Bit-flip probability");
    validate_probability(sample, "Trajectory sample");
    if (sample < probability) {
        apply_x(qubit);
    }
}

void QRegister::apply_phase_flip_trajectory(QubitId qubit, double probability, double sample) {
    validate_probability(probability, "Phase-flip probability");
    validate_probability(sample, "Trajectory sample");
    if (sample < probability) {
        apply_z(qubit);
    }
}

void QRegister::apply_depolarizing_trajectory(QubitId qubit, double probability, double sample) {
    validate_probability(probability, "Depolarizing probability");
    validate_probability(sample, "Trajectory sample");
    if (sample >= probability || probability == 0.0) {
        return;
    }
    const double branch = sample / probability;
    if (branch < 1.0 / 3.0) {
        apply_x(qubit);
    } else if (branch < 2.0 / 3.0) {
        apply_y(qubit);
    } else {
        apply_z(qubit);
    }
}

void QRegister::apply_amplitude_damping_trajectory(QubitId qubit, double gamma, double sample) {
    validate_probability(gamma, "Amplitude-damping gamma");
    validate_probability(sample, "Trajectory sample");
    const double jump_probability = gamma * probability_one(qubit);
    if (sample < jump_probability) {
        QMatrix2 jump{};
        jump.values[1] = {std::sqrt(gamma), 0.0};
        apply_nonunitary_single(qubit, jump);
    } else {
        QMatrix2 no_jump{};
        no_jump.values[0] = {1.0, 0.0};
        no_jump.values[3] = {std::sqrt(1.0 - gamma), 0.0};
        apply_nonunitary_single(qubit, no_jump);
    }
}

void QRegister::apply_nonunitary_single(QubitId qubit, const QMatrix2& matrix) {
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        apply_cell_matrix(std::get<BlochCell>(components_[index].state), matrix);
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_single(local_position(components_[index], qubit), matrix, config_, true);
        compact_component(index);
    }
}

std::size_t QRegister::promote_global_dense() {
    if (qubit_count_ >= 63 || qubit_count_ > config_.max_component_qubits) {
        throw QStateError("Full-register Grover execution exceeds the configured component width");
    }
    const BasisIndex dimension = BasisIndex{1} << qubit_count_;
    if (dimension > config_.max_dense_amplitudes ||
        dimension > static_cast<BasisIndex>(std::numeric_limits<std::size_t>::max())) {
        throw QStateError("Full-register Grover execution exceeds the configured dense-state limit");
    }

    if (components_.size() == 1U && !components_.front().is_cell() &&
        components_.front().qubits.size() == qubit_count_) {
        AmplitudeStore& existing = std::get<AmplitudeStore>(components_.front().state);
        if (existing.mode_ != StorageMode::Dense) {
            existing.dense_ = existing.dense_copy();
            existing.sparse_.clear();
            existing.sparse_.shrink_to_fit();
            existing.mode_ = StorageMode::Dense;
        }
        return 0U;
    }

    std::vector<QComplex> dense = materialize(qubit_count_);
    AmplitudeStore store;
    store.dimension_ = dimension;
    store.mode_ = StorageMode::Dense;
    store.dense_ = std::move(dense);
    store.sparse_.clear();

    std::vector<QubitId> qubits(qubit_count_);
    std::iota(qubits.begin(), qubits.end(), QubitId{0});
    components_.clear();
    component_order_.clear();
    components_.push_back(StateComponent{std::move(qubits), std::move(store)});
    component_order_.push_back(0U);
    next_component_order_ = 1U;
    qubit_component_.assign(qubit_count_, 0U);
    return 0U;
}

void QRegister::apply_grover_oracle(std::span<const BasisIndex> marked_indices) {
    if (marked_indices.empty()) {
        throw QStateError("Grover oracle requires at least one marked basis state");
    }
    if (qubit_count_ >= 63) {
        throw QStateError("Integer Grover oracle supports at most 62 qubits");
    }
    const BasisIndex dimension = BasisIndex{1} << qubit_count_;
    std::vector<BasisIndex> sorted(marked_indices.begin(), marked_indices.end());
    std::sort(sorted.begin(), sorted.end());
    if (sorted.back() >= dimension) {
        throw QStateError("Grover marked basis state is outside the register");
    }
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        throw QStateError("Grover marked basis states must be unique");
    }

    const std::size_t component = promote_global_dense();
    AmplitudeStore& store = std::get<AmplitudeStore>(components_[component].state);
    for (BasisIndex index : sorted) {
        store.dense_[static_cast<std::size_t>(index)] =
            -store.dense_[static_cast<std::size_t>(index)];
    }
}

void QRegister::apply_grover_diffusion() {
    const std::size_t component = promote_global_dense();
    AmplitudeStore& store = std::get<AmplitudeStore>(components_[component].state);
    long double sum_re = 0.0L;
    long double sum_im = 0.0L;
    for (const QComplex& value : store.dense_) {
        sum_re += static_cast<long double>(value.re);
        sum_im += static_cast<long double>(value.im);
    }
    const long double inverse_dimension = 1.0L / static_cast<long double>(store.dimension_);
    const QComplex twice_mean{
        static_cast<double>(2.0L * sum_re * inverse_dimension),
        static_cast<double>(2.0L * sum_im * inverse_dimension),
    };
    for (QComplex& value : store.dense_) {
        value = twice_mean - value;
    }
}

void QRegister::apply_grover_iterations(
    std::span<const BasisIndex> marked_indices,
    std::uint64_t iteration_count) {
    for (std::uint64_t iteration = 0; iteration < iteration_count; ++iteration) {
        apply_grover_oracle(marked_indices);
        apply_grover_diffusion();
    }
}

double QRegister::probability_one(QubitId qubit) const {
    const std::size_t index = component_index(qubit);
    const StateComponent& component = components_[index];
    if (component.is_cell()) {
        return std::get<BlochCell>(component.state).probability_one();
    }
    const std::size_t position = local_position(component, qubit);
    const BasisIndex mask = BasisIndex{1} << position;
    long double probability = 0.0L;
    const AmplitudeStore& store = std::get<AmplitudeStore>(component.state);
    if (store.mode_ == StorageMode::Sparse) {
        for (const auto& [basis, amplitude_value] : store.sparse_) {
            if ((basis & mask) != 0U) {
                probability += static_cast<long double>(amplitude_value.norm2());
            }
        }
    } else {
        const std::size_t half_block = static_cast<std::size_t>(mask);
        const std::size_t block = half_block << 1U;
        for (std::size_t base = 0; base < store.dense_.size(); base += block) {
            const std::size_t end = base + block;
            for (std::size_t basis = base + half_block; basis < end; ++basis) {
                probability += static_cast<long double>(store.dense_[basis].norm2());
            }
        }
    }
    return std::clamp(static_cast<double>(probability), 0.0, 1.0);
}

std::vector<double> QRegister::probabilities_one() const {
    std::vector<double> result(qubit_count_, 0.0);
    for (const StateComponent& component : components_) {
        if (component.is_cell()) {
            result[component.qubits.front()] =
                std::get<BlochCell>(component.state).probability_one();
            continue;
        }

        const AmplitudeStore& store = std::get<AmplitudeStore>(component.state);
        std::vector<long double> local(component.qubits.size(), 0.0L);
        const auto accumulate = [&local](BasisIndex basis, const QComplex& amplitude_value) {
            const long double weight = static_cast<long double>(amplitude_value.norm2());
            BasisIndex remaining = basis;
            while (remaining != 0U) {
                const std::size_t position = static_cast<std::size_t>(std::countr_zero(remaining));
                local[position] += weight;
                remaining &= remaining - 1U;
            }
        };

        if (store.mode_ == StorageMode::Sparse) {
            for (const auto& [basis, amplitude_value] : store.sparse_) {
                accumulate(basis, amplitude_value);
            }
        } else {
            for (std::size_t position = 0; position < component.qubits.size(); ++position) {
                const std::size_t half_block = std::size_t{1} << position;
                const std::size_t block = half_block << 1U;
                long double probability = 0.0L;
                for (std::size_t base = 0; base < store.dense_.size(); base += block) {
                    const std::size_t end = base + block;
                    for (std::size_t basis = base + half_block; basis < end; ++basis) {
                        probability += static_cast<long double>(store.dense_[basis].norm2());
                    }
                }
                local[position] = probability;
            }
        }

        for (std::size_t position = 0; position < component.qubits.size(); ++position) {
            result[component.qubits[position]] =
                std::clamp(static_cast<double>(local[position]), 0.0, 1.0);
        }
    }
    return result;
}

int QRegister::measure(QubitId qubit, double sample) {
    validate_probability(sample, "Measurement sample");
    const double one_probability = probability_one(qubit);
    const int outcome = sample < one_probability ? 1 : 0;
    const std::size_t index = component_index(qubit);
    StateComponent& component = components_[index];
    if (component.is_cell()) {
        component.state = outcome == 0 ? BlochCell{0.0, 0.0, 1.0} : BlochCell{0.0, 0.0, -1.0};
    } else {
        collapse_patch(index, local_position(component, qubit), outcome);
    }
    return outcome;
}

std::vector<int> QRegister::measure_all(std::uint64_t seed) {
    SplitMix64 generator{seed};
    std::vector<int> result(qubit_count_);
    for (std::size_t qubit = 0; qubit < qubit_count_; ++qubit) {
        result[qubit] = measure(static_cast<QubitId>(qubit), generator.unit());
    }
    return result;
}

void QRegister::collapse_patch(
    std::size_t component_index_value,
    std::size_t local_position_value,
    int outcome) {
    StateComponent& component = components_[component_index_value];
    AmplitudeStore& store = std::get<AmplitudeStore>(component.state);
    const BasisIndex mask = BasisIndex{1} << local_position_value;
    std::vector<AmplitudeStore::SparseEntry> kept;
    for (const auto& entry : store.entries(config_.epsilon)) {
        const int bit = (entry.first & mask) == 0U ? 0 : 1;
        if (bit == outcome) {
            kept.push_back(entry);
        }
    }
    component.state = AmplitudeStore::from_entries(store.dimension(), std::move(kept), config_);
    compact_component(component_index_value);
}

QComplex QRegister::amplitude(BasisIndex global_basis_index) const {
    if (qubit_count_ > 63) {
        throw QStateError("Integer basis lookup supports at most 63 qubits; use amplitude_bits for larger registers");
    }
    const BasisIndex global_dimension = BasisIndex{1} << qubit_count_;
    if (global_basis_index >= global_dimension) {
        throw QStateError("Global basis index is out of range");
    }
    std::vector<std::uint8_t> bits(qubit_count_);
    for (std::size_t qubit = 0; qubit < qubit_count_; ++qubit) {
        bits[qubit] = static_cast<std::uint8_t>((global_basis_index >> qubit) & 1U);
    }
    return amplitude_bits(bits);
}

QComplex QRegister::amplitude_bits(std::span<const std::uint8_t> bits) const {
    if (bits.size() != qubit_count_) {
        throw QStateError("Bit-vector length does not match QRegister qubit count");
    }
    QComplex result{1.0, 0.0};
    for (const StateComponent& component : components_) {
        if (component.is_cell()) {
            const auto values = std::get<BlochCell>(component.state).amplitudes(config_.epsilon);
            const std::uint8_t bit = bits[component.qubits.front()];
            if (bit > 1U) {
                throw QStateError("Basis bit values must be 0 or 1");
            }
            result *= values[bit];
            continue;
        }
        BasisIndex local_index = 0;
        for (std::size_t position = 0; position < component.qubits.size(); ++position) {
            const std::uint8_t bit = bits[component.qubits[position]];
            if (bit > 1U) {
                throw QStateError("Basis bit values must be 0 or 1");
            }
            if (bit != 0U) {
                local_index |= BasisIndex{1} << position;
            }
        }
        result *= std::get<AmplitudeStore>(component.state).at(local_index);
    }
    return result;
}

std::vector<QComplex> QRegister::materialize(std::size_t max_qubits) const {
    if (qubit_count_ > max_qubits || qubit_count_ >= 63) {
        throw QStateError("Materialization refused: register exceeds the requested qubit limit");
    }
    const BasisIndex dimension = BasisIndex{1} << qubit_count_;
    std::vector<QComplex> result(static_cast<std::size_t>(dimension));

    using GlobalEntry = std::pair<BasisIndex, QComplex>;
    std::vector<std::vector<GlobalEntry>> component_entries;
    component_entries.reserve(components_.size());
    for (const StateComponent& component : components_) {
        std::vector<AmplitudeStore::SparseEntry> local_entries;
        if (component.is_cell()) {
            const auto amplitudes = std::get<BlochCell>(component.state).amplitudes(config_.epsilon);
            if (amplitudes[0].norm2() > config_.epsilon * config_.epsilon) {
                local_entries.emplace_back(0U, amplitudes[0]);
            }
            if (amplitudes[1].norm2() > config_.epsilon * config_.epsilon) {
                local_entries.emplace_back(1U, amplitudes[1]);
            }
        } else {
            local_entries = std::get<AmplitudeStore>(component.state).entries(config_.epsilon);
        }

        std::vector<GlobalEntry> global_entries;
        global_entries.reserve(local_entries.size());
        for (const auto& [local_index, value] : local_entries) {
            BasisIndex global_index = 0;
            for (std::size_t position = 0; position < component.qubits.size(); ++position) {
                if (((local_index >> position) & 1U) != 0U) {
                    global_index |= BasisIndex{1} << component.qubits[position];
                }
            }
            global_entries.emplace_back(global_index, value);
        }
        component_entries.push_back(std::move(global_entries));
    }

    const auto fill = [&](const auto& self,
                          std::size_t component_index_value,
                          BasisIndex global_index,
                          QComplex amplitude_value) -> void {
        if (component_index_value == component_entries.size()) {
            result[static_cast<std::size_t>(global_index)] = amplitude_value;
            return;
        }
        for (const auto& [entry_index, entry_value] : component_entries[component_index_value]) {
            self(
                self,
                component_index_value + 1U,
                global_index | entry_index,
                amplitude_value * entry_value);
        }
    };
    fill(fill, 0U, 0U, QComplex{1.0, 0.0});
    return result;
}

std::size_t QRegister::component_size(QubitId qubit) const {
    return components_[component_index(qubit)].qubits.size();
}

StorageMode QRegister::component_storage_mode(QubitId qubit) const {
    const StateComponent& component = components_[component_index(qubit)];
    if (component.is_cell()) {
        return StorageMode::Sparse;
    }
    return std::get<AmplitudeStore>(component.state).mode();
}

ComponentKind QRegister::component_kind(QubitId qubit) const {
    const StateComponent& component = components_[component_index(qubit)];
    if (component.is_cell()) {
        return ComponentKind::Cell;
    }
    return std::get<AmplitudeStore>(component.state).mode() == StorageMode::Sparse
               ? ComponentKind::Sparse
               : ComponentKind::Dense;
}

std::size_t QRegister::component_nonzero_count(QubitId qubit) const {
    const StateComponent& component = components_[component_index(qubit)];
    if (component.is_cell()) {
        const auto values = std::get<BlochCell>(component.state).amplitudes(config_.epsilon);
        return static_cast<std::size_t>(values[0].norm2() > config_.epsilon * config_.epsilon) +
               static_cast<std::size_t>(values[1].norm2() > config_.epsilon * config_.epsilon);
    }
    return std::get<AmplitudeStore>(component.state).nonzero_count();
}

std::size_t QRegister::estimated_bytes() const noexcept {
    std::size_t total = sizeof(*this) + qubit_component_.capacity() * sizeof(std::size_t) +
                        components_.capacity() * sizeof(StateComponent) +
                        component_order_.capacity() * sizeof(std::uint32_t);
    for (const StateComponent& component : components_) {
        total += component.qubits.capacity() * sizeof(QubitId);
        if (!component.is_cell()) {
            total += std::get<AmplitudeStore>(component.state).estimated_bytes();
        }
    }
    return total;
}

std::string QRegister::describe() const {
    std::ostringstream stream;
    stream << "Qubit Native State Engine\n"
           << "qubits: " << qubit_count_ << "\n"
           << "components: " << components_.size() << "\n"
           << "estimated engine bytes: " << estimated_bytes() << "\n";
    const auto ordered = ordered_component_indices();
    for (std::size_t logical_index = 0; logical_index < ordered.size(); ++logical_index) {
        const StateComponent& component = components_[ordered[logical_index]];
        stream << "  [" << logical_index << "] ";
        if (component.is_cell()) {
            const BlochCell& cell = std::get<BlochCell>(component.state);
            stream << "geometric cell q" << component.qubits.front() << "  r=("
                   << std::setprecision(7) << cell.x << ", " << cell.y << ", " << cell.z << ")\n";
        } else {
            const AmplitudeStore& store = std::get<AmplitudeStore>(component.state);
            stream << storage_name(store.mode()) << " qubits={";
            for (std::size_t q = 0; q < component.qubits.size(); ++q) {
                if (q != 0) {
                    stream << ',';
                }
                stream << component.qubits[q];
            }
            stream << "} support=" << store.nonzero_count() << " dimension=2^"
                   << component.qubits.size() << "\n";
        }
    }
    return stream.str();
}

bool QRegister::validate(std::string* reason) const {
    const auto fail = [reason](const std::string& message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };
    if (components_.empty()) {
        return fail("Register has no state components");
    }
    if (component_order_.size() != components_.size()) {
        return fail("Component order table does not match the component store");
    }
    std::vector<std::uint32_t> order = component_order_;
    std::sort(order.begin(), order.end());
    if (std::adjacent_find(order.begin(), order.end()) != order.end()) {
        return fail("Component order keys are not unique");
    }
    std::vector<bool> seen(qubit_count_, false);
    for (std::size_t component_index_value = 0;
         component_index_value < components_.size();
         ++component_index_value) {
        const StateComponent& component = components_[component_index_value];
        if (component.qubits.empty()) {
            return fail("A state component has no qubits");
        }
        if (component.is_cell() && component.qubits.size() != 1U) {
            return fail("A geometric cell must contain exactly one qubit");
        }
        for (QubitId qubit : component.qubits) {
            if (static_cast<std::size_t>(qubit) >= qubit_count_) {
                return fail("A component contains an out-of-range qubit");
            }
            if (seen[qubit]) {
                return fail("A qubit appears in multiple components");
            }
            if (qubit_component_[qubit] != component_index_value) {
                return fail("Qubit-to-component index does not match the component store");
            }
            seen[qubit] = true;
        }
        if (component.is_cell()) {
            const BlochCell& cell = std::get<BlochCell>(component.state);
            const double length = std::sqrt(cell.x * cell.x + cell.y * cell.y + cell.z * cell.z);
            if (!std::isfinite(length) || std::abs(length - 1.0) > 1e-8) {
                return fail("A geometric cell is not normalized");
            }
        } else {
            const AmplitudeStore& store = std::get<AmplitudeStore>(component.state);
            if (component.qubits.size() >= 63 ||
                store.dimension() != (BasisIndex{1} << component.qubits.size())) {
                return fail("Patch dimension does not match its qubit count");
            }
            long double norm = 0.0L;
            for (const auto& entry : store.entries()) {
                norm += static_cast<long double>(entry.second.norm2());
            }
            if (std::abs(static_cast<double>(norm) - 1.0) > 1e-8) {
                return fail("A state patch is not normalized");
            }
        }
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
        return fail("At least one qubit is missing from the component partition");
    }
    return true;
}

std::optional<std::pair<StateComponent, StateComponent>> QRegister::factor_singleton(
    const StateComponent& component,
    std::size_t local_position_value) const {
    if (component.is_cell() || component.qubits.size() <= 1U ||
        local_position_value >= component.qubits.size()) {
        return std::nullopt;
    }

    const AmplitudeStore& store = std::get<AmplitudeStore>(component.state);
    const BasisIndex mask = BasisIndex{1} << local_position_value;
    long double rho00 = 0.0L;
    long double rho11 = 0.0L;
    QComplex rho01{};

    struct Slice {
        BasisIndex index{0};
        QComplex zero{};
        QComplex one{};
    };
    std::vector<Slice> sparse_slices;
    if (store.mode_ == StorageMode::Dense) {
        const BasisIndex rest_dimension = store.dimension_ >> 1U;
        for (BasisIndex rest_index = 0; rest_index < rest_dimension; ++rest_index) {
            const BasisIndex zero_index = insert_zero_bit(rest_index, local_position_value);
            const QComplex& zero = store.dense_[static_cast<std::size_t>(zero_index)];
            const QComplex& one = store.dense_[static_cast<std::size_t>(zero_index | mask)];
            rho00 += static_cast<long double>(zero.norm2());
            rho11 += static_cast<long double>(one.norm2());
            rho01 += zero * one.conjugate();
        }
    } else {
        struct ReducedEntry {
            BasisIndex index;
            QComplex amplitude;
        };
        std::vector<ReducedEntry> zeros;
        std::vector<ReducedEntry> ones;
        zeros.reserve(store.sparse_.size());
        ones.reserve(store.sparse_.size());
        for (const auto& [index, value] : store.sparse_) {
            const ReducedEntry reduced{remove_bit(index, local_position_value), value};
            ((index & mask) == 0U ? zeros : ones).push_back(reduced);
        }
        sparse_slices.reserve(zeros.size() + ones.size());
        std::size_t zero_index = 0;
        std::size_t one_index = 0;
        while (zero_index < zeros.size() || one_index < ones.size()) {
            const BasisIndex reduced_index = one_index >= ones.size()
                                                 ? zeros[zero_index].index
                                                 : zero_index >= zeros.size()
                                                       ? ones[one_index].index
                                                       : std::min(
                                                             zeros[zero_index].index,
                                                             ones[one_index].index);
            Slice slice;
            slice.index = reduced_index;
            if (zero_index < zeros.size() && zeros[zero_index].index == reduced_index) {
                slice.zero = zeros[zero_index++].amplitude;
            }
            if (one_index < ones.size() && ones[one_index].index == reduced_index) {
                slice.one = ones[one_index++].amplitude;
            }
            rho00 += static_cast<long double>(slice.zero.norm2());
            rho11 += static_cast<long double>(slice.one.norm2());
            rho01 += slice.zero * slice.one.conjugate();
            sparse_slices.push_back(slice);
        }
    }
    const long double determinant = rho00 * rho11 - static_cast<long double>(rho01.norm2());
    const long double scale = std::max(1.0L, rho00 * rho11);
    if (std::abs(determinant) > static_cast<long double>(config_.factor_tolerance) * scale) {
        return std::nullopt;
    }

    BlochCell cell{
        2.0 * rho01.re,
        -2.0 * rho01.im,
        static_cast<double>(rho00 - rho11),
    };
    cell.normalize(config_.epsilon);
    const auto cell_amplitudes = cell.amplitudes(config_.epsilon);
    const bool use_zero = cell_amplitudes[0].norm2() >= cell_amplitudes[1].norm2();
    const QComplex divisor = use_zero ? cell_amplitudes[0] : cell_amplitudes[1];
    if (divisor.norm2() <= config_.epsilon * config_.epsilon) {
        return std::nullopt;
    }

    std::vector<QubitId> rest_qubits = component.qubits;
    const QubitId extracted_qubit = rest_qubits[local_position_value];
    rest_qubits.erase(rest_qubits.begin() + static_cast<std::ptrdiff_t>(local_position_value));
    const BasisIndex rest_dimension = BasisIndex{1} << rest_qubits.size();
    AmplitudeStore rest_store;
    if (store.mode_ == StorageMode::Dense) {
        std::vector<QComplex> rest_dense(static_cast<std::size_t>(rest_dimension));
        for (BasisIndex rest_index = 0; rest_index < rest_dimension; ++rest_index) {
            const BasisIndex zero_index = insert_zero_bit(rest_index, local_position_value);
            const QComplex& numerator = use_zero
                                            ? store.dense_[static_cast<std::size_t>(zero_index)]
                                            : store.dense_[static_cast<std::size_t>(zero_index | mask)];
            rest_dense[static_cast<std::size_t>(rest_index)] = numerator / divisor;
        }
        rest_store = AmplitudeStore::from_dense(std::move(rest_dense), config_);
    } else {
        std::vector<AmplitudeStore::SparseEntry> rest_entries;
        rest_entries.reserve(sparse_slices.size());
        for (const Slice& slice : sparse_slices) {
            const QComplex numerator = use_zero ? slice.zero : slice.one;
            const QComplex value = numerator / divisor;
            if (value.norm2() > config_.epsilon * config_.epsilon) {
                rest_entries.emplace_back(slice.index, value);
            }
        }
        rest_store = AmplitudeStore::from_entries(
            rest_dimension, std::move(rest_entries), config_);
    }

    StateComponent rest_component;
    rest_component.qubits = std::move(rest_qubits);
    if (rest_component.qubits.size() == 1U) {
        rest_component.state = BlochCell::from_amplitudes(
            rest_store.at(0U), rest_store.at(1U), config_.epsilon);
    } else {
        rest_component.state = std::move(rest_store);
    }
    StateComponent cell_component{{extracted_qubit}, cell};
    return std::make_optional(std::make_pair(std::move(rest_component), std::move(cell_component)));
}

void QRegister::compact_component(std::size_t component_index_value) {
    if (component_index_value >= components_.size()) {
        throw QStateError("Cannot compact an invalid component");
    }
    while (!components_[component_index_value].is_cell()) {
        StateComponent& component = components_[component_index_value];
        if (component.qubits.size() == 1U) {
            const AmplitudeStore& store = std::get<AmplitudeStore>(component.state);
            component.state = BlochCell::from_amplitudes(store.at(0U), store.at(1U), config_.epsilon);
            break;
        }

        bool extracted = false;
        for (std::size_t position = 0; position < component.qubits.size(); ++position) {
            auto factor = factor_singleton(component, position);
            if (!factor.has_value()) {
                continue;
            }
            components_[component_index_value] = std::move(factor->first);
            for (QubitId qubit : components_[component_index_value].qubits) {
                qubit_component_[qubit] = component_index_value;
            }
            (void)append_component(std::move(factor->second));
            extracted = true;
            break;
        }
        if (!extracted) {
            break;
        }
    }
}

void QRegister::compact_component_targets(
    std::size_t component_index_value,
    std::span<const QubitId> candidate_qubits) {
    if (component_index_value >= components_.size()) {
        throw QStateError("Cannot compact an invalid component");
    }
    for (QubitId qubit : candidate_qubits) {
        validate_qubit(qubit);
        const std::size_t current_index = component_index(qubit);
        StateComponent& component = components_[current_index];
        if (component.is_cell()) {
            continue;
        }
        if (component.qubits.size() == 1U) {
            const AmplitudeStore& store = std::get<AmplitudeStore>(component.state);
            component.state = BlochCell::from_amplitudes(
                store.at(0U), store.at(1U), config_.epsilon);
            continue;
        }
        const std::size_t position = local_position(component, qubit);
        auto factor = factor_singleton(component, position);
        if (!factor.has_value()) {
            continue;
        }
        components_[current_index] = std::move(factor->first);
        for (QubitId remaining : components_[current_index].qubits) {
            qubit_component_[remaining] = current_index;
        }
        (void)append_component(std::move(factor->second));
    }
}

std::vector<std::uint8_t> QRegister::encode_qsc() const {
    return QStateCodec::encode(*this);
}

QRegister QRegister::decode_qsc(std::span<const std::uint8_t> bytes) {
    return QStateCodec::decode(bytes);
}

} 
