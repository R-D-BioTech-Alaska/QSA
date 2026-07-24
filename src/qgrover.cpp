#include "qubit/qgrover.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace qubit {
namespace {

constexpr long double kPi = 3.141592653589793238462643383279502884L;

void validate_unit_sample(double sample, const char* label) {
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

[[nodiscard]] double clamp_probability(long double value) noexcept {
    return std::clamp(static_cast<double>(value), 0.0, 1.0);
}

} 

GroverSearch::GroverSearch(
    std::size_t qubit_count,
    std::span<const BasisIndex> marked_indices) {
    if (marked_indices.empty()) {
        throw QStateError("Grover search requires at least one marked basis state");
    }
    marked_indices_.assign(marked_indices.begin(), marked_indices.end());
    std::sort(marked_indices_.begin(), marked_indices_.end());
    if (std::adjacent_find(marked_indices_.begin(), marked_indices_.end()) !=
        marked_indices_.end()) {
        throw QStateError("Grover marked basis states must be unique");
    }
    initialize_space(qubit_count, static_cast<BasisIndex>(marked_indices_.size()));
    if (marked_indices_.back() >= space_size_) {
        throw QStateError("Grover marked basis state is outside the search space");
    }
    reset();
}

GroverSearch::GroverSearch(
    std::size_t qubit_count,
    BasisIndex marked_count,
    CountOnlyTag) {
    initialize_space(qubit_count, marked_count);
    reset();
}

GroverSearch GroverSearch::from_marked_count(
    std::size_t qubit_count,
    BasisIndex marked_count) {
    return GroverSearch(qubit_count, marked_count, CountOnlyTag{});
}

void GroverSearch::initialize_space(std::size_t qubit_count, BasisIndex marked_count) {
    if (qubit_count == 0 || qubit_count >= 63) {
        throw QStateError("Grover search qubit count must be between 1 and 62");
    }
    qubit_count_ = qubit_count;
    space_size_ = BasisIndex{1} << qubit_count_;
    if (marked_count == 0 || marked_count >= space_size_) {
        throw QStateError("Grover marked count must be between 1 and search-space-size - 1");
    }
    marked_count_ = marked_count;
}

void GroverSearch::reset() {
    const double amplitude = 1.0 / std::sqrt(static_cast<double>(space_size_));
    marked_amplitude_ = {amplitude, 0.0};
    unmarked_amplitude_ = {amplitude, 0.0};
    iteration_count_ = 0;
}

void GroverSearch::apply_oracle() {
    marked_amplitude_ = -marked_amplitude_;
}

void GroverSearch::apply_diffusion() {
    const long double marked_weight = static_cast<long double>(marked_count_);
    const long double unmarked_weight = static_cast<long double>(unmarked_count());
    const long double total = static_cast<long double>(space_size_);
    const QComplex mean =
        marked_amplitude_ * static_cast<double>(marked_weight / total) +
        unmarked_amplitude_ * static_cast<double>(unmarked_weight / total);
    marked_amplitude_ = mean * 2.0 - marked_amplitude_;
    unmarked_amplitude_ = mean * 2.0 - unmarked_amplitude_;
}

void GroverSearch::iterate(std::uint64_t count) {
    if (count == 0) {
        return;
    }

    const long double ratio =
        static_cast<long double>(marked_count_) / static_cast<long double>(space_size_);
    const long double theta = std::asin(std::sqrt(ratio));
    long double base_cos = std::cos(2.0L * theta);
    long double base_sin = std::sin(2.0L * theta);
    long double result_cos = 1.0L;
    long double result_sin = 0.0L;
    std::uint64_t exponent = count;

    while (exponent != 0U) {
        if ((exponent & 1U) != 0U) {
            const long double next_cos = result_cos * base_cos - result_sin * base_sin;
            const long double next_sin = result_sin * base_cos + result_cos * base_sin;
            result_cos = next_cos;
            result_sin = next_sin;
        }
        exponent >>= 1U;
        if (exponent != 0U) {
            const long double next_cos = base_cos * base_cos - base_sin * base_sin;
            const long double next_sin = 2.0L * base_sin * base_cos;
            base_cos = next_cos;
            base_sin = next_sin;
        }
    }

    const double sqrt_marked = std::sqrt(static_cast<double>(marked_count_));
    const double sqrt_unmarked = std::sqrt(static_cast<double>(unmarked_count()));
    const QComplex marked_coefficient = marked_amplitude_ * sqrt_marked;
    const QComplex unmarked_coefficient = unmarked_amplitude_ * sqrt_unmarked;
    const double c = static_cast<double>(result_cos);
    const double s = static_cast<double>(result_sin);
    const QComplex next_marked = marked_coefficient * c + unmarked_coefficient * s;
    const QComplex next_unmarked = -marked_coefficient * s + unmarked_coefficient * c;
    marked_amplitude_ = next_marked / sqrt_marked;
    unmarked_amplitude_ = next_unmarked / sqrt_unmarked;
    iteration_count_ += count;
}

void GroverSearch::run_optimal() {
    iterate(optimal_iterations());
}

std::uint64_t GroverSearch::optimal_iterations() const noexcept {
    const long double ratio =
        static_cast<long double>(marked_count_) / static_cast<long double>(space_size_);
    const long double theta = std::asin(std::sqrt(ratio));
    const long double ideal = kPi / (4.0L * theta) - 0.5L;
    if (ideal <= 0.0L) {
        return 0;
    }

    const auto probability_for = [theta](std::uint64_t count) {
        const long double angle = (2.0L * static_cast<long double>(count) + 1.0L) * theta;
        const long double sine = std::sin(angle);
        return sine * sine;
    };

    const auto lower = static_cast<std::uint64_t>(std::floor(ideal));
    const auto upper = lower + 1U;
    return probability_for(upper) > probability_for(lower) ? upper : lower;
}

double GroverSearch::success_probability() const noexcept {
    const long double probability =
        static_cast<long double>(marked_count_) *
        static_cast<long double>(marked_amplitude_.norm2());
    return clamp_probability(probability);
}

bool GroverSearch::is_marked(BasisIndex basis_index) const {
    if (!has_explicit_marked_indices()) {
        throw QStateError("Marked membership is unavailable for a count-only Grover search");
    }
    return std::binary_search(marked_indices_.begin(), marked_indices_.end(), basis_index);
}

QComplex GroverSearch::amplitude(BasisIndex basis_index) const {
    if (basis_index >= space_size_) {
        throw QStateError("Grover basis index is outside the search space");
    }
    return is_marked(basis_index) ? marked_amplitude_ : unmarked_amplitude_;
}

bool GroverSearch::sample_is_marked(double sample) const {
    validate_unit_sample(sample, "Grover branch sample");
    return sample < success_probability();
}

BasisIndex GroverSearch::select_unmarked(BasisIndex rank) const {
    if (!has_explicit_marked_indices()) {
        throw QStateError("Basis sampling requires explicit Grover marked indices");
    }
    if (rank >= unmarked_count()) {
        throw QStateError("Unmarked Grover sample rank is outside the search space");
    }

    BasisIndex low = 0;
    BasisIndex high = space_size_ - 1U;
    while (low < high) {
        const BasisIndex midpoint = low + (high - low) / 2U;
        const BasisIndex marked_through_midpoint = static_cast<BasisIndex>(
            std::upper_bound(marked_indices_.begin(), marked_indices_.end(), midpoint) -
            marked_indices_.begin());
        const BasisIndex unmarked_through_midpoint =
            (midpoint + 1U) - marked_through_midpoint;
        if (unmarked_through_midpoint > rank) {
            high = midpoint;
        } else {
            low = midpoint + 1U;
        }
    }
    return low;
}

BasisIndex GroverSearch::sample_basis(double branch_sample, double index_sample) const {
    validate_unit_sample(branch_sample, "Grover branch sample");
    validate_unit_sample(index_sample, "Grover index sample");
    if (!has_explicit_marked_indices()) {
        throw QStateError("Basis sampling requires explicit Grover marked indices");
    }

    if (sample_is_marked(branch_sample)) {
        const BasisIndex rank = std::min<BasisIndex>(
            marked_count_ - 1U,
            static_cast<BasisIndex>(index_sample * static_cast<double>(marked_count_)));
        return marked_indices_[static_cast<std::size_t>(rank)];
    }

    const BasisIndex count = unmarked_count();
    const BasisIndex rank = std::min<BasisIndex>(
        count - 1U,
        static_cast<BasisIndex>(index_sample * static_cast<double>(count)));
    return select_unmarked(rank);
}

BasisIndex GroverSearch::sample_basis(std::uint64_t seed) const {
    SplitMix64 generator{seed};
    return sample_basis(generator.unit(), generator.unit());
}

std::vector<QComplex> GroverSearch::materialize(std::size_t max_qubits) const {
    if (!has_explicit_marked_indices()) {
        throw QStateError("Materialization requires explicit Grover marked indices");
    }
    if (qubit_count_ > max_qubits) {
        throw QStateError("Grover materialization exceeds the requested qubit limit");
    }
    std::vector<QComplex> amplitudes(static_cast<std::size_t>(space_size_), unmarked_amplitude_);
    for (BasisIndex index : marked_indices_) {
        amplitudes[static_cast<std::size_t>(index)] = marked_amplitude_;
    }
    return amplitudes;
}

std::size_t GroverSearch::estimated_bytes() const noexcept {
    return sizeof(*this) + marked_indices_.capacity() * sizeof(BasisIndex);
}

bool GroverSearch::validate(std::string* reason) const {
    const auto fail = [reason](const std::string& message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };
    if (qubit_count_ == 0 || qubit_count_ >= 63) {
        return fail("Grover qubit count is invalid");
    }
    if (space_size_ != (BasisIndex{1} << qubit_count_)) {
        return fail("Grover search-space size is inconsistent");
    }
    if (marked_count_ == 0 || marked_count_ >= space_size_) {
        return fail("Grover marked count is invalid");
    }
    if (!marked_indices_.empty()) {
        if (marked_indices_.size() != static_cast<std::size_t>(marked_count_)) {
            return fail("Grover explicit marked-index count is inconsistent");
        }
        if (!std::is_sorted(marked_indices_.begin(), marked_indices_.end()) ||
            std::adjacent_find(marked_indices_.begin(), marked_indices_.end()) !=
                marked_indices_.end() ||
            marked_indices_.back() >= space_size_) {
            return fail("Grover explicit marked indices are invalid");
        }
    }
    const long double norm =
        static_cast<long double>(marked_count_) * marked_amplitude_.norm2() +
        static_cast<long double>(unmarked_count()) * unmarked_amplitude_.norm2();
    if (!std::isfinite(static_cast<double>(norm)) ||
        std::abs(norm - 1.0L) > 1e-9L) {
        return fail("Grover amplitude classes are not normalized");
    }
    return true;
}

std::string GroverSearch::describe() const {
    std::ostringstream stream;
    stream << "QSA Grover Search\n"
           << "qubits: " << qubit_count_ << "\n"
           << "logical states: " << space_size_ << "\n"
           << "marked states: " << marked_count_ << "\n"
           << "representation: two exact amplitude classes\n"
           << "iterations: " << iteration_count_ << "\n"
           << "success probability: " << std::setprecision(12)
           << success_probability() << "\n"
           << "estimated engine bytes: " << estimated_bytes() << "\n";
    return stream.str();
}

}
