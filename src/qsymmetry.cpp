#include "qubit/qsymmetry.hpp"
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

void validate_sample(double sample, const char* label) {
    if (!std::isfinite(sample) || sample < 0.0 || sample >= 1.0) {
        throw QStateError(std::string(label) + " must be finite and in [0, 1)");
    }
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

[[nodiscard]] std::vector<QComplex> identity_matrix(std::size_t size) {
    std::vector<QComplex> result(size * size);
    for (std::size_t index = 0; index < size; ++index) {
        result[index * size + index] = {1.0, 0.0};
    }
    return result;
}

[[nodiscard]] std::vector<QComplex> multiply_matrix(
    std::span<const QComplex> left,
    std::span<const QComplex> right,
    std::size_t size) {
    std::vector<QComplex> result(size * size);
    for (std::size_t row = 0; row < size; ++row) {
        for (std::size_t inner = 0; inner < size; ++inner) {
            const QComplex coefficient = left[row * size + inner];
            if (coefficient.norm2() == 0.0) {
                continue;
            }
            for (std::size_t column = 0; column < size; ++column) {
                result[row * size + column] +=
                    coefficient * right[inner * size + column];
            }
        }
    }
    return result;
}

[[nodiscard]] std::vector<QComplex> multiply_vector(
    std::span<const QComplex> matrix,
    std::span<const QComplex> vector,
    std::size_t size) {
    std::vector<QComplex> result(size);
    for (std::size_t row = 0; row < size; ++row) {
        QComplex value{};
        for (std::size_t column = 0; column < size; ++column) {
            value += matrix[row * size + column] * vector[column];
        }
        result[row] = value;
    }
    return result;
}


struct ExactComplexKey {
    std::uint64_t real;
    std::uint64_t imag;

    [[nodiscard]] bool operator==(const ExactComplexKey&) const noexcept = default;
};

struct ExactComplexKeyHash {
    [[nodiscard]] std::size_t operator()(const ExactComplexKey& key) const noexcept {
        const std::uint64_t mixed = key.real ^
            (key.imag + 0x9E3779B97F4A7C15ULL + (key.real << 6U) + (key.real >> 2U));
        return static_cast<std::size_t>(mixed ^ (mixed >> 32U));
    }
};

struct GridKey {
    std::int64_t real;
    std::int64_t imag;

    [[nodiscard]] bool operator==(const GridKey&) const noexcept = default;
};

struct GridKeyHash {
    [[nodiscard]] std::size_t operator()(const GridKey& key) const noexcept {
        const std::uint64_t first = static_cast<std::uint64_t>(key.real);
        const std::uint64_t second = static_cast<std::uint64_t>(key.imag);
        const std::uint64_t mixed = first ^
            (second + 0x9E3779B97F4A7C15ULL + (first << 6U) + (first >> 2U));
        return static_cast<std::size_t>(mixed ^ (mixed >> 32U));
    }
};

[[nodiscard]] ExactComplexKey exact_key(QComplex value) noexcept {
    if (value.re == 0.0) {
        value.re = 0.0;
    }
    if (value.im == 0.0) {
        value.im = 0.0;
    }
    return {
        std::bit_cast<std::uint64_t>(value.re),
        std::bit_cast<std::uint64_t>(value.im),
    };
}


[[nodiscard]] BasisIndex binomial(std::size_t n, std::size_t k) {
    if (k > n) {
        return 0U;
    }
    k = std::min(k, n - k);
    BasisIndex result = 1U;
    for (std::size_t index = 1; index <= k; ++index) {
        BasisIndex numerator = static_cast<BasisIndex>(n - k + index);
        BasisIndex denominator = static_cast<BasisIndex>(index);
        const BasisIndex first_gcd = std::gcd(numerator, denominator);
        numerator /= first_gcd;
        denominator /= first_gcd;
        const BasisIndex second_gcd = std::gcd(result, denominator);
        result /= second_gcd;
        denominator /= second_gcd;
        if (denominator != 1U) {
            throw QStateError("Internal binomial reduction failed");
        }
        if (numerator != 0U && result > std::numeric_limits<BasisIndex>::max() / numerator) {
            throw QStateError("Binomial coefficient exceeds QSA basis-index capacity");
        }
        result *= numerator;
    }
    return result;
}

[[nodiscard]] BasisIndex unrank_hamming_weight(
    std::size_t qubits,
    std::size_t weight,
    BasisIndex rank) {
    if (rank >= binomial(qubits, weight)) {
        throw QStateError("Hamming-weight sample rank is outside the class");
    }
    BasisIndex basis = 0U;
    std::size_t remaining = weight;
    for (std::size_t position = 0; position < qubits && remaining != 0U; ++position) {
        const std::size_t positions_after = qubits - position - 1U;
        const BasisIndex without_current = binomial(positions_after, remaining);
        if (rank < without_current) {
            continue;
        }
        rank -= without_current;
        basis |= BasisIndex{1} << position;
        --remaining;
    }
    if (remaining != 0U) {
        throw QStateError("Hamming-weight unranking failed");
    }
    return basis;
}

[[nodiscard]] GridKey grid_key(QComplex value, double cell) {
    const long double real = std::floor(static_cast<long double>(value.re) / cell);
    const long double imag = std::floor(static_cast<long double>(value.im) / cell);
    const long double minimum = static_cast<long double>(std::numeric_limits<std::int64_t>::min());
    const long double maximum = static_cast<long double>(std::numeric_limits<std::int64_t>::max());
    if (real < minimum || real > maximum || imag < minimum || imag > maximum) {
        throw QStateError("Symmetry discovery tolerance is too small for amplitude quantization");
    }
    return {static_cast<std::int64_t>(real), static_cast<std::int64_t>(imag)};
}

[[nodiscard]] std::optional<std::int64_t> checked_offset(
    std::int64_t value,
    std::int64_t offset) noexcept {
    if ((offset > 0 && value > std::numeric_limits<std::int64_t>::max() - offset) ||
        (offset < 0 && value < std::numeric_limits<std::int64_t>::min() - offset)) {
        return std::nullopt;
    }
    return value + offset;
}

void validate_unitary(
    std::span<const QComplex> matrix,
    std::size_t size,
    double tolerance) {
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw QStateError("Symmetry unitary tolerance must be finite and positive");
    }
    if (matrix.size() != size * size) {
        throw QStateError("Symmetry class-unitary matrix has the wrong dimensions");
    }
    for (const QComplex& value : matrix) {
        if (!std::isfinite(value.re) || !std::isfinite(value.im)) {
            throw QStateError("Symmetry class-unitary matrix contains a non-finite value");
        }
    }
    for (std::size_t first = 0; first < size; ++first) {
        for (std::size_t second = 0; second < size; ++second) {
            QComplex inner{};
            for (std::size_t row = 0; row < size; ++row) {
                inner += matrix[row * size + first].conjugate() *
                         matrix[row * size + second];
            }
            const QComplex expected = first == second ? QComplex{1.0, 0.0} : QComplex{};
            if (!almost_equal(inner, expected, tolerance)) {
                throw QStateError("Symmetry class-space matrix is not unitary");
            }
        }
    }
}

} 

SymmetryState::SymmetryState(
    std::size_t qubit_count,
    std::span<const BasisIndex> class_counts)
    : membership_(SymmetryMembership::OrderedRanges) {
    initialize_space(qubit_count);
    initialize_counts(class_counts);
    reset_uniform();
}

SymmetryState::SymmetryState(
    std::size_t qubit_count,
    std::span<const BasisIndex> class_counts,
    CountOnlyTag)
    : membership_(SymmetryMembership::CountOnly) {
    initialize_space(qubit_count);
    initialize_counts(class_counts);
    reset_uniform();
}

SymmetryState::SymmetryState(
    std::size_t qubit_count,
    std::span<const std::uint32_t> labels,
    ExplicitLabelsTag)
    : membership_(SymmetryMembership::ExplicitLabels) {
    initialize_space(qubit_count);
    if (labels.size() != static_cast<std::size_t>(space_size_)) {
        throw QStateError("Symmetry explicit-label count must equal the logical state count");
    }
    if (labels.empty()) {
        throw QStateError("Symmetry explicit labels cannot be empty");
    }
    const auto maximum = *std::max_element(labels.begin(), labels.end());
    counts_.assign(static_cast<std::size_t>(maximum) + 1U, 0U);
    labels_.assign(labels.begin(), labels.end());
    for (std::uint32_t label : labels_) {
        ++counts_[label];
    }
    if (std::find(counts_.begin(), counts_.end(), BasisIndex{0}) != counts_.end()) {
        throw QStateError("Symmetry explicit labels must use every class index");
    }
    amplitudes_.resize(counts_.size());
    reset_uniform();
}

SymmetryState SymmetryState::from_counts(
    std::size_t qubit_count,
    std::span<const BasisIndex> class_counts) {
    return SymmetryState(qubit_count, class_counts, CountOnlyTag{});
}

SymmetryState SymmetryState::from_labels(
    std::size_t qubit_count,
    std::span<const std::uint32_t> labels) {
    return SymmetryState(qubit_count, labels, ExplicitLabelsTag{});
}


SymmetryState SymmetryState::hamming_weight(std::size_t qubit_count) {
    if (qubit_count == 0 || qubit_count >= 63) {
        throw QStateError("Hamming-weight symmetry requires between 1 and 62 qubits");
    }
    std::vector<BasisIndex> counts(qubit_count + 1U);
    for (std::size_t weight = 0; weight <= qubit_count; ++weight) {
        counts[weight] = binomial(qubit_count, weight);
    }
    SymmetryState result = SymmetryState::from_counts(qubit_count, counts);
    result.membership_ = SymmetryMembership::HammingWeight;
    return result;
}


SymmetryState SymmetryState::discover(
    const QRegister& state,
    std::size_t max_qubits,
    double tolerance,
    std::size_t max_classes) {
    if (!std::isfinite(tolerance) || tolerance < 0.0) {
        throw QStateError("Symmetry discovery tolerance must be finite and nonnegative");
    }
    if (max_classes == 0U) {
        throw QStateError("Symmetry discovery max_classes must be positive");
    }
    const std::vector<QComplex> source = state.materialize(max_qubits);
    std::vector<std::uint32_t> labels(source.size());
    std::vector<BasisIndex> counts;
    std::vector<QComplex> representatives;
    std::vector<QComplex> sums;
    counts.reserve(std::min(max_classes, source.size()));
    representatives.reserve(std::min(max_classes, source.size()));
    sums.reserve(std::min(max_classes, source.size()));

    const auto add_class = [&](QComplex value) -> std::uint32_t {
        if (counts.size() >= max_classes ||
            counts.size() >= static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw QStateError("Symmetry discovery exceeded the configured class limit");
        }
        const auto index = static_cast<std::uint32_t>(counts.size());
        counts.push_back(0U);
        representatives.push_back(value);
        sums.push_back({});
        return index;
    };

    if (tolerance == 0.0) {
        std::unordered_map<ExactComplexKey, std::uint32_t, ExactComplexKeyHash> classes;
        classes.reserve(std::min(max_classes, source.size()));
        for (std::size_t basis = 0; basis < source.size(); ++basis) {
            const QComplex value = source[basis];
            if (!std::isfinite(value.re) || !std::isfinite(value.im)) {
                throw QStateError("Symmetry discovery encountered a non-finite amplitude");
            }
            const ExactComplexKey key = exact_key(value);
            auto iterator = classes.find(key);
            std::uint32_t class_index = 0;
            if (iterator == classes.end()) {
                class_index = add_class(value);
                classes.emplace(key, class_index);
            } else {
                class_index = iterator->second;
            }
            labels[basis] = class_index;
            ++counts[class_index];
            sums[class_index] += value;
        }
    } else {
        const double cell = tolerance * 2.0;
        if (!std::isfinite(cell) || cell <= 0.0) {
            throw QStateError("Symmetry discovery tolerance is below numeric capacity");
        }
        std::unordered_map<GridKey, std::vector<std::uint32_t>, GridKeyHash> bins;
        bins.reserve(std::min(max_classes, source.size()));
        for (std::size_t basis = 0; basis < source.size(); ++basis) {
            const QComplex value = source[basis];
            if (!std::isfinite(value.re) || !std::isfinite(value.im)) {
                throw QStateError("Symmetry discovery encountered a non-finite amplitude");
            }
            const GridKey center = grid_key(value, cell);
            std::optional<std::uint32_t> matched;
            for (std::int64_t real_offset = -1; real_offset <= 1 && !matched.has_value(); ++real_offset) {
                for (std::int64_t imag_offset = -1; imag_offset <= 1 && !matched.has_value(); ++imag_offset) {
                    const auto real = checked_offset(center.real, real_offset);
                    const auto imag = checked_offset(center.imag, imag_offset);
                    if (!real.has_value() || !imag.has_value()) {
                        continue;
                    }
                    const GridKey key{*real, *imag};
                    const auto iterator = bins.find(key);
                    if (iterator == bins.end()) {
                        continue;
                    }
                    for (std::uint32_t candidate : iterator->second) {
                        if (almost_equal(value, representatives[candidate], tolerance)) {
                            matched = candidate;
                            break;
                        }
                    }
                }
            }
            std::uint32_t class_index = 0;
            if (matched.has_value()) {
                class_index = *matched;
            } else {
                class_index = add_class(value);
                bins[center].push_back(class_index);
            }
            labels[basis] = class_index;
            ++counts[class_index];
            sums[class_index] += value;
            representatives[class_index] =
                sums[class_index] / static_cast<double>(counts[class_index]);
        }
    }

    std::vector<QComplex> class_amplitudes(counts.size());
    for (std::size_t index = 0; index < counts.size(); ++index) {
        class_amplitudes[index] = tolerance == 0.0
            ? representatives[index]
            : sums[index] / static_cast<double>(counts[index]);
    }

    bool ordered = true;
    std::uint32_t current = 0U;
    for (std::uint32_t label : labels) {
        if (label == current) {
            continue;
        }
        if (label == current + 1U) {
            current = label;
            continue;
        }
        ordered = false;
        break;
    }
    SymmetryState result = ordered
        ? SymmetryState(state.qubit_count(), counts)
        : SymmetryState::from_labels(state.qubit_count(), labels);
    result.set_class_amplitudes(class_amplitudes, tolerance != 0.0);
    double maximum_error = 0.0;
    for (std::size_t basis = 0; basis < source.size(); ++basis) {
        maximum_error = std::max(
            maximum_error,
            (source[basis] - result.amplitudes_[labels[basis]]).magnitude());
    }
    result.discovery_error_ = maximum_error;
    return result;
}

void SymmetryState::initialize_space(std::size_t qubit_count) {
    if (qubit_count == 0 || qubit_count >= 63) {
        throw QStateError("Symmetry state qubit count must be between 1 and 62");
    }
    qubit_count_ = qubit_count;
    space_size_ = BasisIndex{1} << qubit_count_;
}

void SymmetryState::initialize_counts(std::span<const BasisIndex> class_counts) {
    if (class_counts.empty()) {
        throw QStateError("Symmetry state requires at least one amplitude class");
    }
    counts_.assign(class_counts.begin(), class_counts.end());
    if (std::find(counts_.begin(), counts_.end(), BasisIndex{0}) != counts_.end()) {
        throw QStateError("Symmetry amplitude classes cannot be empty");
    }
    BasisIndex total = 0;
    for (BasisIndex count : counts_) {
        if (count > space_size_ - total) {
            throw QStateError("Symmetry class sizes exceed the logical state count");
        }
        total += count;
    }
    if (total != space_size_) {
        throw QStateError("Symmetry class sizes must sum to the logical state count");
    }
    amplitudes_.resize(counts_.size());
    if (membership_ == SymmetryMembership::OrderedRanges) {
        offsets_.resize(counts_.size() + 1U);
        for (std::size_t index = 0; index < counts_.size(); ++index) {
            offsets_[index + 1U] = offsets_[index] + counts_[index];
        }
    }
}

void SymmetryState::validate_class(std::size_t class_index) const {
    if (class_index >= counts_.size()) {
        throw QStateError("Symmetry class index is out of range");
    }
}

BasisIndex SymmetryState::class_size(std::size_t class_index) const {
    validate_class(class_index);
    return counts_[class_index];
}

QComplex SymmetryState::class_amplitude(std::size_t class_index) const {
    validate_class(class_index);
    return amplitudes_[class_index];
}

double SymmetryState::class_probability(std::size_t class_index) const {
    validate_class(class_index);
    const long double value = static_cast<long double>(counts_[class_index]) *
                              static_cast<long double>(amplitudes_[class_index].norm2());
    return std::clamp(static_cast<double>(value), 0.0, 1.0);
}

std::size_t SymmetryState::class_for_basis(BasisIndex basis_index) const {
    if (basis_index >= space_size_) {
        throw QStateError("Symmetry basis index is outside the logical state space");
    }
    if (membership_ == SymmetryMembership::CountOnly) {
        throw QStateError("Basis membership is unavailable for a count-only symmetry state");
    }
    if (membership_ == SymmetryMembership::ExplicitLabels) {
        return labels_[static_cast<std::size_t>(basis_index)];
    }
    if (membership_ == SymmetryMembership::HammingWeight) {
        return static_cast<std::size_t>(std::popcount(basis_index));
    }
    const auto iterator = std::upper_bound(offsets_.begin(), offsets_.end(), basis_index);
    return static_cast<std::size_t>(iterator - offsets_.begin() - 1);
}

QComplex SymmetryState::amplitude(BasisIndex basis_index) const {
    return amplitudes_[class_for_basis(basis_index)];
}

void SymmetryState::reset_uniform() {
    const double value = 1.0 / std::sqrt(static_cast<double>(space_size_));
    std::fill(amplitudes_.begin(), amplitudes_.end(), QComplex{value, 0.0});
}

void SymmetryState::normalize() {
    long double norm2 = 0.0L;
    for (std::size_t index = 0; index < counts_.size(); ++index) {
        norm2 += static_cast<long double>(counts_[index]) *
                 static_cast<long double>(amplitudes_[index].norm2());
    }
    const double norm = std::sqrt(static_cast<double>(norm2));
    if (!std::isfinite(norm) || norm <= std::numeric_limits<double>::min()) {
        throw QStateError("Symmetry amplitude classes have zero or non-finite norm");
    }
    for (QComplex& amplitude : amplitudes_) {
        amplitude /= norm;
    }
}

void SymmetryState::set_class_amplitudes(
    std::span<const QComplex> amplitudes,
    bool normalize_values) {
    if (amplitudes.size() != amplitudes_.size()) {
        throw QStateError("Symmetry amplitude count does not match the class count");
    }
    for (const QComplex& value : amplitudes) {
        if (!std::isfinite(value.re) || !std::isfinite(value.im)) {
            throw QStateError("Symmetry amplitude contains a non-finite value");
        }
    }
    amplitudes_.assign(amplitudes.begin(), amplitudes.end());
    if (normalize_values) {
        normalize();
    }
}

void SymmetryState::apply_class_phase(std::size_t class_index, double angle) {
    validate_class(class_index);
    if (!std::isfinite(angle)) {
        throw QStateError("Symmetry class phase must be finite");
    }
    amplitudes_[class_index] *= QComplex::from_polar(1.0, angle);
}

void SymmetryState::apply_class_phases(std::span<const double> angles) {
    if (angles.size() != class_count()) {
        throw QStateError("Symmetry phase count does not match the class count");
    }
    for (std::size_t index = 0; index < angles.size(); ++index) {
        apply_class_phase(index, angles[index]);
    }
}

void SymmetryState::apply_weighted_reflection() {
    QComplex mean{};
    const long double total = static_cast<long double>(space_size_);
    for (std::size_t index = 0; index < counts_.size(); ++index) {
        mean += amplitudes_[index] *
                static_cast<double>(static_cast<long double>(counts_[index]) / total);
    }
    for (QComplex& amplitude : amplitudes_) {
        amplitude = mean * 2.0 - amplitude;
    }
}


std::size_t SymmetryState::split_class(
    std::size_t class_index,
    BasisIndex first_count) {
    validate_class(class_index);
    if (membership_ == SymmetryMembership::HammingWeight) {
        throw QStateError("Hamming-weight classes cannot be split without changing the symbolic partition");
    }
    const BasisIndex original_count = counts_[class_index];
    if (first_count == 0U || first_count >= original_count) {
        throw QStateError("Symmetry class split must leave both partitions nonempty");
    }
    const BasisIndex second_count = original_count - first_count;
    const std::size_t new_index = class_index + 1U;

    counts_[class_index] = first_count;
    counts_.insert(counts_.begin() + static_cast<std::ptrdiff_t>(new_index), second_count);
    amplitudes_.insert(
        amplitudes_.begin() + static_cast<std::ptrdiff_t>(new_index),
        amplitudes_[class_index]);

    if (membership_ == SymmetryMembership::OrderedRanges) {
        offsets_.resize(counts_.size() + 1U);
        offsets_[0] = 0U;
        for (std::size_t index = 0; index < counts_.size(); ++index) {
            offsets_[index + 1U] = offsets_[index] + counts_[index];
        }
    } else if (membership_ == SymmetryMembership::ExplicitLabels) {
        BasisIndex kept = 0U;
        for (std::uint32_t& label : labels_) {
            if (label > class_index) {
                ++label;
            } else if (label == class_index) {
                if (kept < first_count) {
                    ++kept;
                } else {
                    label = static_cast<std::uint32_t>(new_index);
                }
            }
        }
        if (kept != first_count) {
            throw QStateError("Internal symmetry split failed to preserve explicit membership");
        }
    }
    return new_index;
}

std::size_t SymmetryState::merge_equivalent(double tolerance) {
    if (!std::isfinite(tolerance) || tolerance < 0.0) {
        throw QStateError("Symmetry merge tolerance must be finite and nonnegative");
    }
    if (membership_ == SymmetryMembership::HammingWeight) {
        throw QStateError("Hamming-weight classes cannot be merged without changing the symbolic partition");
    }
    std::size_t removed = 0U;
    std::size_t first = 0U;
    while (first < class_count()) {
        std::size_t second = first + 1U;
        while (second < class_count()) {
            if (membership_ == SymmetryMembership::OrderedRanges && second != first + 1U) {
                break;
            }
            if (!almost_equal(amplitudes_[first], amplitudes_[second], tolerance)) {
                ++second;
                continue;
            }
            counts_[first] += counts_[second];
            counts_.erase(counts_.begin() + static_cast<std::ptrdiff_t>(second));
            amplitudes_.erase(amplitudes_.begin() + static_cast<std::ptrdiff_t>(second));
            if (membership_ == SymmetryMembership::ExplicitLabels) {
                for (std::uint32_t& label : labels_) {
                    if (label == second) {
                        label = static_cast<std::uint32_t>(first);
                    } else if (label > second) {
                        --label;
                    }
                }
            }
            ++removed;
            if (membership_ == SymmetryMembership::OrderedRanges) {
                second = first + 1U;
            }
        }
        ++first;
    }
    if (membership_ == SymmetryMembership::OrderedRanges) {
        offsets_.resize(counts_.size() + 1U);
        offsets_[0] = 0U;
        for (std::size_t index = 0; index < counts_.size(); ++index) {
            offsets_[index + 1U] = offsets_[index] + counts_[index];
        }
    }
    return removed;
}

std::vector<QComplex> SymmetryState::normalized_coefficients() const {
    std::vector<QComplex> result(class_count());
    for (std::size_t index = 0; index < class_count(); ++index) {
        result[index] = amplitudes_[index] * std::sqrt(static_cast<double>(counts_[index]));
    }
    return result;
}

void SymmetryState::assign_normalized_coefficients(
    std::span<const QComplex> coefficients) {
    if (coefficients.size() != class_count()) {
        throw QStateError("Symmetry coefficient count does not match the class count");
    }
    for (std::size_t index = 0; index < class_count(); ++index) {
        amplitudes_[index] =
            coefficients[index] / std::sqrt(static_cast<double>(counts_[index]));
    }
    normalize();
}

void SymmetryState::apply_class_unitary(
    std::span<const QComplex> matrix,
    double tolerance) {
    validate_unitary(matrix, class_count(), tolerance);
    const auto coefficients = normalized_coefficients();
    const auto transformed = multiply_vector(matrix, coefficients, class_count());
    assign_normalized_coefficients(transformed);
}

void SymmetryState::iterate_class_unitary(
    std::span<const QComplex> matrix,
    std::uint64_t count,
    double tolerance) {
    validate_unitary(matrix, class_count(), tolerance);
    if (count == 0U) {
        return;
    }
    std::vector<QComplex> power(matrix.begin(), matrix.end());
    std::vector<QComplex> result = identity_matrix(class_count());
    std::uint64_t exponent = count;
    while (exponent != 0U) {
        if ((exponent & 1U) != 0U) {
            result = multiply_matrix(result, power, class_count());
        }
        exponent >>= 1U;
        if (exponent != 0U) {
            power = multiply_matrix(power, power, class_count());
        }
    }
    const auto coefficients = normalized_coefficients();
    const auto transformed = multiply_vector(result, coefficients, class_count());
    assign_normalized_coefficients(transformed);
}

std::size_t SymmetryState::sample_class(double sample) const {
    validate_sample(sample, "Symmetry class sample");
    long double cumulative = 0.0L;
    for (std::size_t index = 0; index < class_count(); ++index) {
        cumulative += static_cast<long double>(class_probability(index));
        if (static_cast<long double>(sample) < cumulative || index + 1U == class_count()) {
            return index;
        }
    }
    return class_count() - 1U;
}

BasisIndex SymmetryState::sample_basis(double class_sample, double index_sample) const {
    validate_sample(index_sample, "Symmetry basis-index sample");
    const std::size_t selected_class = sample_class(class_sample);
    const BasisIndex rank = std::min<BasisIndex>(
        counts_[selected_class] - 1U,
        static_cast<BasisIndex>(index_sample * static_cast<double>(counts_[selected_class])));
    if (membership_ == SymmetryMembership::CountOnly) {
        throw QStateError("Basis sampling is unavailable for a count-only symmetry state");
    }
    if (membership_ == SymmetryMembership::OrderedRanges) {
        return offsets_[selected_class] + rank;
    }
    if (membership_ == SymmetryMembership::HammingWeight) {
        return unrank_hamming_weight(qubit_count_, selected_class, rank);
    }
    BasisIndex seen = 0;
    for (BasisIndex basis = 0; basis < space_size_; ++basis) {
        if (labels_[static_cast<std::size_t>(basis)] == selected_class) {
            if (seen == rank) {
                return basis;
            }
            ++seen;
        }
    }
    throw QStateError("Symmetry explicit-label sampling failed to locate a class member");
}

BasisIndex SymmetryState::sample_basis(std::uint64_t seed) const {
    SplitMix64 generator{seed};
    return sample_basis(generator.unit(), generator.unit());
}

std::vector<QComplex> SymmetryState::materialize(std::size_t max_qubits) const {
    if (!has_basis_membership()) {
        throw QStateError("Materialization requires symmetry basis membership");
    }
    if (qubit_count_ > max_qubits) {
        throw QStateError("Symmetry materialization exceeds the requested qubit limit");
    }
    std::vector<QComplex> result(static_cast<std::size_t>(space_size_));
    if (membership_ == SymmetryMembership::OrderedRanges) {
        for (std::size_t class_index = 0; class_index < class_count(); ++class_index) {
            std::fill(
                result.begin() + static_cast<std::ptrdiff_t>(offsets_[class_index]),
                result.begin() + static_cast<std::ptrdiff_t>(offsets_[class_index + 1U]),
                amplitudes_[class_index]);
        }
    } else if (membership_ == SymmetryMembership::HammingWeight) {
        for (BasisIndex basis = 0; basis < space_size_; ++basis) {
            result[static_cast<std::size_t>(basis)] =
                amplitudes_[static_cast<std::size_t>(std::popcount(basis))];
        }
    } else {
        for (std::size_t basis = 0; basis < result.size(); ++basis) {
            result[basis] = amplitudes_[labels_[basis]];
        }
    }
    return result;
}

QRegister SymmetryState::to_register(std::size_t max_qubits) const {
    return QRegister::from_amplitudes(materialize(max_qubits));
}

std::size_t SymmetryState::estimated_bytes() const noexcept {
    return sizeof(*this) + counts_.capacity() * sizeof(BasisIndex) +
           offsets_.capacity() * sizeof(BasisIndex) +
           labels_.capacity() * sizeof(std::uint32_t) +
           amplitudes_.capacity() * sizeof(QComplex);
}

bool SymmetryState::validate(std::string* reason) const {
    const auto fail = [reason](const std::string& message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };
    if (qubit_count_ == 0 || qubit_count_ >= 63 ||
        space_size_ != (BasisIndex{1} << qubit_count_)) {
        return fail("Symmetry logical space is invalid");
    }
    if (counts_.empty() || counts_.size() != amplitudes_.size()) {
        return fail("Symmetry class storage is inconsistent");
    }
    if (!std::isfinite(discovery_error_) || discovery_error_ < 0.0) {
        return fail("Symmetry discovery error is invalid");
    }
    BasisIndex total = 0;
    for (BasisIndex count : counts_) {
        if (count == 0U) {
            return fail("Symmetry state contains an empty class");
        }
        if (count > space_size_ - total) {
            return fail("Symmetry class sizes exceed the logical space");
        }
        total += count;
    }
    if (total != space_size_) {
        return fail("Symmetry class sizes do not cover the logical space");
    }
    if (membership_ == SymmetryMembership::OrderedRanges) {
        if (offsets_.size() != counts_.size() + 1U || offsets_.front() != 0U ||
            offsets_.back() != space_size_) {
            return fail("Symmetry ordered-range membership is inconsistent");
        }
    } else if (membership_ == SymmetryMembership::ExplicitLabels) {
        if (labels_.size() != static_cast<std::size_t>(space_size_)) {
            return fail("Symmetry explicit-label membership is inconsistent");
        }
    } else if (membership_ == SymmetryMembership::HammingWeight) {
        if (counts_.size() != qubit_count_ + 1U) {
            return fail("Hamming-weight class count is inconsistent");
        }
        for (std::size_t weight = 0; weight <= qubit_count_; ++weight) {
            if (counts_[weight] != binomial(qubit_count_, weight)) {
                return fail("Hamming-weight class size is inconsistent");
            }
        }
    }
    long double norm = 0.0L;
    for (std::size_t index = 0; index < counts_.size(); ++index) {
        const QComplex value = amplitudes_[index];
        if (!std::isfinite(value.re) || !std::isfinite(value.im)) {
            return fail("Symmetry state contains a non-finite amplitude");
        }
        norm += static_cast<long double>(counts_[index]) * value.norm2();
    }
    if (std::abs(norm - 1.0L) > 1e-9L) {
        return fail("Symmetry amplitude classes are not normalized");
    }
    return true;
}

std::string SymmetryState::describe() const {
    std::ostringstream stream;
    const char* mode = membership_ == SymmetryMembership::CountOnly
                           ? "count-only symbolic"
                           : membership_ == SymmetryMembership::OrderedRanges
                                 ? "ordered ranges"
                                 : membership_ == SymmetryMembership::ExplicitLabels
                                       ? "explicit labels"
                                       : "Hamming weight";
    stream << "QSA Symmetry Algebra\n"
           << "qubits: " << qubit_count_ << "\n"
           << "logical states: " << space_size_ << "\n"
           << "amplitude classes: " << class_count() << "\n"
           << "membership: " << mode << "\n"
           << "discovery error: " << std::setprecision(12) << discovery_error_ << "\n"
           << "estimated engine bytes: " << estimated_bytes() << "\n";
    for (std::size_t index = 0; index < class_count(); ++index) {
        stream << "  [" << index << "] count=" << counts_[index]
               << " probability=" << std::setprecision(12)
               << class_probability(index) << " amplitude=("
               << amplitudes_[index].re << ", " << amplitudes_[index].im << ")\n";
    }
    return stream.str();
}

} 
