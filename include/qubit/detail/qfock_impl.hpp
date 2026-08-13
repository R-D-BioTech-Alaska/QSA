#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>

namespace qubit {

inline ExactSparseFockState ExactSparseFockState::from_terms(
    std::size_t modes, std::vector<SparseFockTerm> terms, SparseFockConfig config) {
    return ExactSparseFockState(modes, std::move(terms), config);
}

inline ExactSparseFockState ExactSparseFockState::basis(
    std::size_t modes, FockOccupation occupation, SparseFockConfig config) {
    return from_terms(modes, {SparseFockTerm{std::move(occupation), QComplex{1.0}}}, config);
}

inline std::optional<std::size_t> ExactSparseFockState::fixed_particle_number() const noexcept {
    return stats_.fixed_particle_number ? std::optional<std::size_t>{stats_.particles} : std::nullopt;
}

inline double ExactSparseFockState::norm_squared() const noexcept {
    double result = 0.0;
    for (const auto& term : terms_) result += term.amplitude.norm2();
    return result;
}

inline QComplex ExactSparseFockState::amplitude(const FockOccupation& occupation) const {
    const FockOccupation key = canonical(occupation, mode_count_, config_.max_particles);
    const auto it = std::lower_bound(
        terms_.begin(), terms_.end(), key,
        [](const SparseFockTerm& term, const FockOccupation& value) { return term.occupation < value; });
    return it != terms_.end() && it->occupation == key ? it->amplitude : QComplex{};
}

inline double ExactSparseFockState::mean_number(std::size_t mode) const {
    validate_mode(mode);
    const double norm = norm_squared();
    if (!std::isfinite(norm) || norm <= 0.0) throw QStateError("Fock mean number requires positive finite norm");
    double numerator = 0.0;
    for (const auto& term : terms_)
        numerator += term.amplitude.norm2() * static_cast<double>(occupation_of(term.occupation, mode));
    return numerator / norm;
}

inline double ExactSparseFockState::sector_log2_dimension() const {
    if (!stats_.fixed_particle_number)
        throw QStateError("Fock sector dimension requires fixed total particle number");
    const long double m = static_cast<long double>(mode_count_);
    const long double n = static_cast<long double>(stats_.particles);
    const long double value = (std::lgamma(n + m) - std::lgamma(n + 1.0L) - std::lgamma(m)) / std::log(2.0L);
    if (!std::isfinite(static_cast<double>(value))) throw QStateError("Fock sector dimension became non-finite");
    return static_cast<double>(value);
}

inline ExactSparseFockState ExactSparseFockState::scaled(QComplex scalar) const {
    if (!finite(scalar)) throw QStateError("Fock scale must be finite");
    auto next = terms_;
    for (auto& term : next) term.amplitude *= scalar;
    return ExactSparseFockState(mode_count_, std::move(next), config_);
}

inline ExactSparseFockState ExactSparseFockState::add(const ExactSparseFockState& other) const {
    require_shape(other, "Fock addition");
    SparseFockConfig config{
        std::min(config_.max_modes, other.config_.max_modes),
        std::min(config_.max_particles, other.config_.max_particles),
        std::min(config_.max_terms, other.config_.max_terms),
        std::min(config_.max_occupied_entries, other.config_.max_occupied_entries),
        std::min(config_.max_branch_products, other.config_.max_branch_products)};
    auto next = terms_;
    next.insert(next.end(), other.terms_.begin(), other.terms_.end());
    return ExactSparseFockState(mode_count_, std::move(next), config);
}

inline ExactSparseFockState ExactSparseFockState::apply_hopping(
    std::span<const FockHoppingTerm> hopping) const {
    for (const auto& hop : hopping) {
        validate_mode(hop.create_mode);
        validate_mode(hop.annihilate_mode);
        if (!finite(hop.coefficient)) throw QStateError("Fock hopping coefficient must be finite");
    }
    const std::size_t products = checked_product(terms_.size(), hopping.size(), "Fock hopping branch count overflowed");
    if (products > config_.max_branch_products)
        throw QStateError("Fock hopping exceeds configured branch-product cap");
    std::vector<SparseFockTerm> output;
    output.reserve(products);
    for (const auto& state : terms_) {
        for (const auto& hop : hopping) {
            const std::size_t source = occupation_of(state.occupation, hop.annihilate_mode);
            if (source == 0U || hop.coefficient == QComplex{}) continue;
            FockOccupation next = state.occupation;
            double ladder;
            if (hop.create_mode == hop.annihilate_mode) {
                ladder = static_cast<double>(source);
            } else {
                const std::size_t target = occupation_of(next, hop.create_mode);
                ladder = std::sqrt(static_cast<double>(source) * static_cast<double>(target + 1U));
                decrement(next, hop.annihilate_mode);
                increment(next, hop.create_mode);
            }
            output.push_back({std::move(next), state.amplitude * hop.coefficient * ladder});
        }
    }
    return ExactSparseFockState(mode_count_, std::move(output), config_);
}

inline ExactSparseFockState ExactSparseFockState::apply_bose_hubbard(
    std::span<const double> onsite,
    std::span<const double> interaction,
    std::span<const FockHoppingTerm> hopping) const {
    if (onsite.size() != mode_count_ || interaction.size() != mode_count_)
        throw QStateError("Bose-Hubbard diagonal arrays must match mode count");
    for (std::size_t i = 0U; i < mode_count_; ++i)
        if (!std::isfinite(onsite[i]) || !std::isfinite(interaction[i]))
            throw QStateError("Bose-Hubbard diagonal coefficients must be finite");
    std::vector<SparseFockTerm> diagonal;
    diagonal.reserve(terms_.size());
    for (const auto& state : terms_) {
        double energy = 0.0;
        for (const auto& [mode, count] : state.occupation) {
            const double n = static_cast<double>(count);
            energy += onsite[mode] * n + 0.5 * interaction[mode] * n * (n - 1.0);
        }
        if (!std::isfinite(energy)) throw QStateError("Bose-Hubbard diagonal energy became non-finite");
        if (energy != 0.0) diagonal.push_back({state.occupation, state.amplitude * energy});
    }
    return ExactSparseFockState(mode_count_, std::move(diagonal), config_).add(apply_hopping(hopping));
}

inline ExactSparseFockState::ExactSparseFockState(
    std::size_t modes, std::vector<SparseFockTerm> terms, SparseFockConfig config)
    : mode_count_(modes), config_(config) {
    if (mode_count_ == 0U || mode_count_ > config_.max_modes)
        throw QStateError("Fock mode count is zero or exceeds configured cap");
    if (config_.max_particles == 0U || config_.max_terms == 0U ||
        config_.max_occupied_entries == 0U || config_.max_branch_products == 0U)
        throw QStateError("Fock configuration contains a zero resource cap");
    if (terms.size() > config_.max_terms) throw QStateError("Fock state exceeds configured input term cap");

    std::map<FockOccupation, QComplex> merged;
    std::size_t entries = 0U;
    for (auto& term : terms) {
        if (!finite(term.amplitude)) throw QStateError("Fock amplitude must be finite");
        term.occupation = canonical(std::move(term.occupation), mode_count_, config_.max_particles);
        entries = checked_sum(entries, term.occupation.size(), "Fock occupied-entry count overflowed");
        if (entries > config_.max_occupied_entries)
            throw QStateError("Fock state exceeds configured occupied-entry cap");
        merged[term.occupation] += term.amplitude;
    }

    entries = 0U;
    std::size_t maximum_particles = 0U;
    terms_.reserve(merged.size());
    for (auto& [occupation, amplitude] : merged) {
        if (!finite(amplitude)) throw QStateError("Fock merged amplitude became non-finite");
        if (amplitude == QComplex{}) continue;
        maximum_particles = std::max(maximum_particles, particle_count(occupation));
        entries = checked_sum(entries, occupation.size(), "Fock canonical occupied-entry count overflowed");
        if (entries > config_.max_occupied_entries)
            throw QStateError("Fock canonical state exceeds configured occupied-entry cap");
        terms_.push_back({std::move(occupation), amplitude});
    }
    if (terms_.size() > config_.max_terms) throw QStateError("Fock canonical state exceeds configured term cap");

    bool fixed = false;
    std::size_t particles = 0U;
    if (!terms_.empty()) {
        particles = particle_count(terms_.front().occupation);
        fixed = std::all_of(terms_.begin(), terms_.end(), [particles](const auto& term) {
            return particle_count(term.occupation) == particles;
        });
    }
    stats_ = {mode_count_, terms_.size(), entries, maximum_particles, fixed, fixed ? particles : 0U};
}

inline bool ExactSparseFockState::finite(QComplex value) noexcept {
    return std::isfinite(value.re) && std::isfinite(value.im);
}

inline std::size_t ExactSparseFockState::checked_product(
    std::size_t left, std::size_t right, const char* message) {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) throw QStateError(message);
    return left * right;
}

inline std::size_t ExactSparseFockState::checked_sum(
    std::size_t left, std::size_t right, const char* message) {
    if (right > std::numeric_limits<std::size_t>::max() - left) throw QStateError(message);
    return left + right;
}

inline std::size_t ExactSparseFockState::particle_count(const FockOccupation& occupation) {
    std::size_t total = 0U;
    for (const auto& entry : occupation) total = checked_sum(total, entry.second, "Fock particle count overflowed");
    return total;
}

inline FockOccupation ExactSparseFockState::canonical(
    FockOccupation occupation, std::size_t modes, std::size_t max_particles) {
    std::sort(occupation.begin(), occupation.end());
    std::size_t particles = 0U;
    std::size_t previous = modes;
    for (const auto& [mode, count] : occupation) {
        if (mode >= modes) throw QStateError("Fock occupation contains a mode outside logical shape");
        if (count == 0U) throw QStateError("Fock occupation entries must have positive count");
        if (previous == mode) throw QStateError("Fock occupation contains a duplicate mode");
        previous = mode;
        particles = checked_sum(particles, count, "Fock particle count overflowed");
        if (particles > max_particles) throw QStateError("Fock occupation exceeds configured particle cap");
    }
    return occupation;
}

inline std::size_t ExactSparseFockState::occupation_of(const FockOccupation& occupation, std::size_t mode) noexcept {
    const auto it = std::lower_bound(
        occupation.begin(), occupation.end(), mode,
        [](const auto& entry, std::size_t value) { return entry.first < value; });
    return it != occupation.end() && it->first == mode ? it->second : 0U;
}

inline void ExactSparseFockState::decrement(FockOccupation& occupation, std::size_t mode) {
    const auto it = std::lower_bound(
        occupation.begin(), occupation.end(), mode,
        [](const auto& entry, std::size_t value) { return entry.first < value; });
    if (it == occupation.end() || it->first != mode || it->second == 0U)
        throw QStateError("Fock annihilation encountered an empty mode");
    if (it->second == 1U) occupation.erase(it); else --it->second;
}

inline void ExactSparseFockState::increment(FockOccupation& occupation, std::size_t mode) {
    const auto it = std::lower_bound(
        occupation.begin(), occupation.end(), mode,
        [](const auto& entry, std::size_t value) { return entry.first < value; });
    if (it == occupation.end() || it->first != mode) occupation.insert(it, {mode, 1U});
    else it->second = checked_sum(it->second, 1U, "Fock occupation count overflowed");
}

inline void ExactSparseFockState::validate_mode(std::size_t mode) const {
    if (mode >= mode_count_) throw QStateError("Fock mode lies outside logical shape");
}

inline void ExactSparseFockState::require_shape(const ExactSparseFockState& other, const char* label) const {
    if (mode_count_ != other.mode_count_) throw QStateError(std::string(label) + " requires equal mode counts");
}

}  // namespace qubit
