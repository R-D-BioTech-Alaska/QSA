#include "qubit/qstate.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iomanip>
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

[[nodiscard]] BasisIndex remap_index(
    BasisIndex index,
    const std::vector<QubitId>& source_qubits,
    const std::vector<std::size_t>& merged_positions) noexcept {
    BasisIndex result = 0;
    for (std::size_t local = 0; local < source_qubits.size(); ++local) {
        if (((index >> local) & 1U) != 0U) {
            result |= BasisIndex{1} << merged_positions[source_qubits[local]];
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

}  // namespace

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

}  // namespace gates

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

    std::unordered_map<BasisIndex, QComplex> output;
    const std::size_t reserve_size = std::min(
        config.max_sparse_entries,
        sparse_.size() > config.max_sparse_entries / 2U
            ? config.max_sparse_entries
            : sparse_.size() * 2U + 1U);
    output.reserve(reserve_size);
    for (const auto& [index, amplitude] : sparse_) {
        const std::size_t input_bit = (index & mask) == 0U ? 0U : 1U;
        for (std::size_t output_bit = 0; output_bit < 2; ++output_bit) {
            const QComplex coefficient = matrix(output_bit, input_bit);
            if (coefficient.norm2() <= config.epsilon * config.epsilon) {
                continue;
            }
            const BasisIndex output_index = output_bit == 0 ? (index & ~mask) : (index | mask);
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
        std::vector<SparseEntry> nonzero;
        nonzero.reserve(dense_.size() / 4U + 1U);
        for (std::size_t index = 0; index < dense_.size(); ++index) {
            if (dense_[index].norm2() > config.epsilon * config.epsilon) {
                nonzero.emplace_back(static_cast<BasisIndex>(index), dense_[index]);
            }
        }
        if (nonzero.size() <= config.max_sparse_entries &&
            static_cast<long double>(nonzero.size()) * 4.0L < static_cast<long double>(dimension_)) {
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
    for (std::size_t qubit = 0; qubit < qubit_count; ++qubit) {
        components_.push_back(StateComponent{{static_cast<QubitId>(qubit)}, BlochCell{}});
    }
    reindex_components();
}

void QRegister::validate_qubit(QubitId qubit) const {
    if (static_cast<std::size_t>(qubit) >= qubit_count_) {
        throw QStateError("Qubit index is out of range");
    }
}

void QRegister::reindex_components() {
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
            .apply_single(local_position(components_[index], qubit), gates::x(), config_);
    }
}

void QRegister::apply_y(QubitId qubit) {
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        std::get<BlochCell>(components_[index].state).apply_y();
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_single(local_position(components_[index], qubit), gates::y(), config_);
    }
}

void QRegister::apply_z(QubitId qubit) {
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        std::get<BlochCell>(components_[index].state).apply_z();
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_single(local_position(components_[index], qubit), gates::z(), config_);
    }
}

void QRegister::apply_h(QubitId qubit) {
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        std::get<BlochCell>(components_[index].state).apply_h();
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_single(local_position(components_[index], qubit), gates::h(), config_);
    }
}

void QRegister::apply_s(QubitId qubit) {
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        std::get<BlochCell>(components_[index].state).rotate_z(kPi / 2.0);
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_single(local_position(components_[index], qubit), gates::s(), config_);
    }
}

void QRegister::apply_sdg(QubitId qubit) {
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        std::get<BlochCell>(components_[index].state).rotate_z(-kPi / 2.0);
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_single(local_position(components_[index], qubit), gates::sdg(), config_);
    }
}

void QRegister::apply_t(QubitId qubit) {
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        std::get<BlochCell>(components_[index].state).rotate_z(kPi / 4.0);
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_single(local_position(components_[index], qubit), gates::t(), config_);
    }
}

void QRegister::apply_tdg(QubitId qubit) {
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        std::get<BlochCell>(components_[index].state).rotate_z(-kPi / 4.0);
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_single(local_position(components_[index], qubit), gates::tdg(), config_);
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
            .apply_single(local_position(components_[index], qubit), gates::rx(theta), config_);
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
            .apply_single(local_position(components_[index], qubit), gates::ry(theta), config_);
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
            .apply_single(local_position(components_[index], qubit), gates::rz(theta), config_);
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

std::size_t QRegister::merge_components(std::size_t first, std::size_t second) {
    if (first == second) {
        return first;
    }
    if (first >= components_.size() || second >= components_.size()) {
        throw QStateError("Cannot merge an invalid state component");
    }

    StateComponent left = components_[first];
    StateComponent right = components_[second];
    std::vector<QubitId> merged_qubits = left.qubits;
    merged_qubits.insert(merged_qubits.end(), right.qubits.begin(), right.qubits.end());
    std::sort(merged_qubits.begin(), merged_qubits.end());
    if (merged_qubits.size() > config_.max_component_qubits || merged_qubits.size() >= 63) {
        throw QStateError("Entangled component exceeds configured qubit limit");
    }

    std::vector<std::size_t> merged_positions(qubit_count_, 0);
    for (std::size_t position = 0; position < merged_qubits.size(); ++position) {
        merged_positions[merged_qubits[position]] = position;
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
        const BasisIndex remapped_left = remap_index(left_index, left.qubits, merged_positions);
        for (const auto& [right_index, right_value] : right_entries) {
            const BasisIndex remapped_right = remap_index(right_index, right.qubits, merged_positions);
            merged_entries.emplace_back(remapped_left | remapped_right, left_value * right_value);
        }
    }

    StateComponent merged{
        std::move(merged_qubits),
        AmplitudeStore::from_entries(dimension, std::move(merged_entries), config_),
    };

    const std::size_t high = std::max(first, second);
    const std::size_t low = std::min(first, second);
    components_.erase(components_.begin() + static_cast<std::ptrdiff_t>(high));
    components_.erase(components_.begin() + static_cast<std::ptrdiff_t>(low));
    components_.push_back(std::move(merged));
    reindex_components();
    return components_.size() - 1U;
}

void QRegister::apply_cnot(QubitId control, QubitId target) {
    apply_two(control, target, gates::cnot());
}

void QRegister::apply_cz(QubitId first, QubitId second) {
    apply_two(first, second, gates::cz());
}

void QRegister::apply_swap(QubitId first, QubitId second) {
    apply_two(first, second, gates::swap());
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
    compact_component(component);
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

double QRegister::probability_one(QubitId qubit) const {
    const std::size_t index = component_index(qubit);
    const StateComponent& component = components_[index];
    if (component.is_cell()) {
        return std::get<BlochCell>(component.state).probability_one();
    }
    const std::size_t position = local_position(component, qubit);
    const BasisIndex mask = BasisIndex{1} << position;
    long double probability = 0.0L;
    for (const auto& [basis, amplitude_value] :
         std::get<AmplitudeStore>(component.state).entries(config_.epsilon)) {
        if ((basis & mask) != 0U) {
            probability += static_cast<long double>(amplitude_value.norm2());
        }
    }
    return std::clamp(static_cast<double>(probability), 0.0, 1.0);
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
    for (BasisIndex index = 0; index < dimension; ++index) {
        result[static_cast<std::size_t>(index)] = amplitude(index);
    }
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
                        components_.capacity() * sizeof(StateComponent);
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
    for (std::size_t index = 0; index < components_.size(); ++index) {
        const StateComponent& component = components_[index];
        stream << "  [" << index << "] ";
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
    std::vector<bool> seen(qubit_count_, false);
    for (const StateComponent& component : components_) {
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

    struct Slice {
        QComplex zero{};
        QComplex one{};
    };
    std::unordered_map<BasisIndex, Slice> slices;
    const AmplitudeStore& store = std::get<AmplitudeStore>(component.state);
    slices.reserve(store.nonzero_count() + 1U);
    const BasisIndex mask = BasisIndex{1} << local_position_value;
    for (const auto& [index, value] : store.entries(config_.epsilon)) {
        Slice& slice = slices[remove_bit(index, local_position_value)];
        if ((index & mask) == 0U) {
            slice.zero = value;
        } else {
            slice.one = value;
        }
    }

    long double rho00 = 0.0L;
    long double rho11 = 0.0L;
    QComplex rho01{};
    for (const auto& [unused, slice] : slices) {
        (void)unused;
        rho00 += static_cast<long double>(slice.zero.norm2());
        rho11 += static_cast<long double>(slice.one.norm2());
        rho01 += slice.zero * slice.one.conjugate();
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

    std::vector<AmplitudeStore::SparseEntry> rest_entries;
    rest_entries.reserve(slices.size());
    for (const auto& [rest_index, slice] : slices) {
        const QComplex numerator = use_zero ? slice.zero : slice.one;
        const QComplex value = numerator / divisor;
        if (value.norm2() > config_.epsilon * config_.epsilon) {
            rest_entries.emplace_back(rest_index, value);
        }
    }

    std::vector<QubitId> rest_qubits = component.qubits;
    const QubitId extracted_qubit = rest_qubits[local_position_value];
    rest_qubits.erase(rest_qubits.begin() + static_cast<std::ptrdiff_t>(local_position_value));
    const BasisIndex rest_dimension = BasisIndex{1} << rest_qubits.size();
    AmplitudeStore rest_store = AmplitudeStore::from_entries(
        rest_dimension, std::move(rest_entries), config_);

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
            components_.push_back(std::move(factor->second));
            reindex_components();
            extracted = true;
            break;
        }
        if (!extracted) {
            break;
        }
    }
}

std::vector<std::uint8_t> QRegister::encode_qsc() const {
    return QStateCodec::encode(*this);
}

QRegister QRegister::decode_qsc(std::span<const std::uint8_t> bytes) {
    return QStateCodec::decode(bytes);
}

}  // namespace qubit
