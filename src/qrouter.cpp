#include "qubit/qrouter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace qubit {
namespace {

void validate_config(const RepresentationAdvisorConfig& config) {
    if (!std::isfinite(config.prior_success) || config.prior_success <= 0.0 ||
        !std::isfinite(config.prior_failure) || config.prior_failure <= 0.0) {
        throw QStateError("Representation advisor priors must be finite and positive");
    }
}

[[nodiscard]] double safe_product(double first, double second) noexcept {
    if (first == 0.0 || second == 0.0) {
        return 0.0;
    }
    if (first > std::numeric_limits<double>::max() / second) {
        return std::numeric_limits<double>::max();
    }
    return first * second;
}

}  // namespace

const char* representation_name(RepresentationKind kind) noexcept {
    switch (kind) {
        case RepresentationKind::Register:
            return "QRegister";
        case RepresentationKind::Symmetry:
            return "SymmetryState";
        case RepresentationKind::QuantumDot:
            return "QuantumDotPocket";
        case RepresentationKind::Stabilizer:
            return "StabilizerState";
        case RepresentationKind::PhaseGraph:
            return "PhaseGraphState";
        case RepresentationKind::Pauli:
            return "PauliObservable";
    }
    return "unknown";
}

RepresentationAdvisor::RepresentationAdvisor(RepresentationAdvisorConfig config)
    : config_(config) {
    validate_config(config_);
}

RepresentationFeatures RepresentationAdvisor::inspect(
    const QRegister& state,
    std::uint64_t repeated_steps,
    std::size_t exact_symmetry_classes,
    bool quantum_dot_declared,
    bool clifford_only,
    bool uniform_phase_graph,
    std::size_t phase_graph_edges) {
    if (repeated_steps == 0U) {
        throw QStateError("Representation advisor repeated_steps must be positive");
    }

    RepresentationFeatures features;
    features.qubit_count = state.qubit_count();
    features.component_count = state.component_count();
    features.repeated_steps = repeated_steps;
    features.exact_symmetry_classes = exact_symmetry_classes;
    features.quantum_dot_declared = quantum_dot_declared;
    features.clifford_only = clifford_only;
    features.uniform_phase_graph = uniform_phase_graph;
    features.phase_graph_edges = phase_graph_edges;
    for (std::size_t qubit = 0; qubit < state.qubit_count(); ++qubit) {
        const QubitId id = static_cast<QubitId>(qubit);
        features.largest_component =
            std::max(features.largest_component, state.component_size(id));
        features.support_entries += state.component_nonzero_count(id);
    }
    return features;
}

RepresentationFeatures RepresentationAdvisor::inspect_pauli(
    const QRegister& state,
    const PauliObservable& observable,
    std::uint64_t repeated_steps) {
    if (observable.qubit_count() != state.qubit_count()) {
        throw QStateError("Pauli observable width does not match QRegister");
    }
    if (!observable.validate()) {
        throw QStateError("Pauli observable failed exact validation");
    }

    RepresentationFeatures features = inspect(state, repeated_steps);
    features.pauli_observable = true;
    features.pauli_term_count = observable.term_count();
    features.pauli_support_qubits = observable.support_size();
    return features;
}

std::size_t RepresentationAdvisor::context(
    const RepresentationFeatures& features) const noexcept {
    if (features.pauli_observable) {
        return 6U;
    }
    if (features.quantum_dot_declared) {
        return 5U;
    }
    if (features.clifford_only) {
        return 4U;
    }
    if (features.uniform_phase_graph) {
        return 3U;
    }
    if (features.exact_symmetry_classes != 0U &&
        features.exact_symmetry_classes <= 256U) {
        return 2U;
    }
    if (features.component_count > 1U &&
        features.largest_component * 2U < features.qubit_count) {
        return 1U;
    }
    return 0U;
}

std::size_t RepresentationAdvisor::index(
    const RepresentationFeatures& features,
    RepresentationKind kind) const noexcept {
    return context(features) * 6U + static_cast<std::size_t>(kind);
}

double RepresentationAdvisor::posterior_mean(
    const RepresentationFeatures& features,
    RepresentationKind kind) const noexcept {
    const Posterior& posterior = posteriors_[index(features, kind)];
    const double success = config_.prior_success + static_cast<double>(posterior.successes);
    const double failure = config_.prior_failure + static_cast<double>(posterior.failures);
    return success / (success + failure);
}

std::vector<RepresentationScore> RepresentationAdvisor::rank(
    const RepresentationFeatures& features) const {
    if (features.qubit_count == 0U || features.component_count == 0U ||
        features.repeated_steps == 0U) {
        throw QStateError("Representation advisor features are incomplete");
    }

    const double steps = static_cast<double>(features.repeated_steps);
    const double support = static_cast<double>(std::max<std::size_t>(1U, features.support_entries));
    const double qubits = static_cast<double>(features.qubit_count);

    std::vector<RepresentationScore> scores;
    scores.reserve(6U);

    RepresentationScore register_score;
    register_score.kind = RepresentationKind::Register;
    register_score.eligible = true;
    register_score.estimated_work = safe_product(steps, support);
    register_score.posterior_success = posterior_mean(features, register_score.kind);
    register_score.adjusted_score =
        register_score.estimated_work / register_score.posterior_success;
    register_score.reason = "general exact component engine";
    scores.push_back(register_score);

    RepresentationScore symmetry_score;
    symmetry_score.kind = RepresentationKind::Symmetry;
    symmetry_score.eligible = features.exact_symmetry_classes != 0U;
    symmetry_score.estimated_work = symmetry_score.eligible
        ? safe_product(steps, static_cast<double>(features.exact_symmetry_classes))
        : std::numeric_limits<double>::infinity();
    symmetry_score.posterior_success = posterior_mean(features, symmetry_score.kind);
    symmetry_score.adjusted_score = symmetry_score.eligible
        ? symmetry_score.estimated_work / symmetry_score.posterior_success
        : std::numeric_limits<double>::infinity();
    symmetry_score.reason = symmetry_score.eligible
        ? "exact amplitude classes are available"
        : "no exact symmetry-class evidence supplied";
    scores.push_back(symmetry_score);

    RepresentationScore dot_score;
    dot_score.kind = RepresentationKind::QuantumDot;
    dot_score.eligible = features.quantum_dot_declared;
    dot_score.estimated_work = dot_score.eligible
        ? safe_product(steps, std::max(1.0, qubits * 0.5))
        : std::numeric_limits<double>::infinity();
    dot_score.posterior_success = posterior_mean(features, dot_score.kind);
    dot_score.adjusted_score = dot_score.eligible
        ? dot_score.estimated_work / dot_score.posterior_success
        : std::numeric_limits<double>::infinity();
    dot_score.reason = dot_score.eligible
        ? "workload explicitly declares fixed quantum-dot structure"
        : "workload does not declare quantum-dot structure";
    scores.push_back(dot_score);

    RepresentationScore stabilizer_score;
    stabilizer_score.kind = RepresentationKind::Stabilizer;
    stabilizer_score.eligible = features.clifford_only;
    const double stabilizer_width = std::max(1.0, std::ceil(qubits / 64.0));
    stabilizer_score.estimated_work = stabilizer_score.eligible
        ? safe_product(steps, safe_product(8.0, stabilizer_width))
        : std::numeric_limits<double>::infinity();
    stabilizer_score.posterior_success = posterior_mean(features, stabilizer_score.kind);
    stabilizer_score.adjusted_score = stabilizer_score.eligible
        ? stabilizer_score.estimated_work / stabilizer_score.posterior_success
        : std::numeric_limits<double>::infinity();
    stabilizer_score.reason = stabilizer_score.eligible
        ? "workload is explicitly restricted to Clifford operations"
        : "workload is not declared Clifford-only";
    scores.push_back(stabilizer_score);

    RepresentationScore phase_graph_score;
    phase_graph_score.kind = RepresentationKind::PhaseGraph;
    phase_graph_score.eligible = features.uniform_phase_graph;
    const double phase_graph_size = static_cast<double>(
        features.qubit_count + features.phase_graph_edges + 1U);
    phase_graph_score.estimated_work = phase_graph_score.eligible
        ? safe_product(steps, std::max(1.0, phase_graph_size * 0.02))
        : std::numeric_limits<double>::infinity();
    phase_graph_score.posterior_success = posterior_mean(features, phase_graph_score.kind);
    phase_graph_score.adjusted_score = phase_graph_score.eligible
        ? phase_graph_score.estimated_work / phase_graph_score.posterior_success
        : std::numeric_limits<double>::infinity();
    phase_graph_score.reason = phase_graph_score.eligible
        ? "workload is explicitly restricted to the uniform phase-graph gate family"
        : "workload is not declared uniform phase-graph";
    scores.push_back(phase_graph_score);

    RepresentationScore pauli_score;
    pauli_score.kind = RepresentationKind::Pauli;
    pauli_score.eligible = features.pauli_observable &&
                           features.pauli_support_qubits <= features.qubit_count;
    const double pauli_terms = static_cast<double>(
        std::max<std::size_t>(1U, features.pauli_term_count));
    const double pauli_support = static_cast<double>(
        std::max<std::size_t>(1U, features.pauli_support_qubits));
    pauli_score.estimated_work = pauli_score.eligible
        ? safe_product(steps, safe_product(pauli_terms, pauli_support))
        : std::numeric_limits<double>::infinity();
    pauli_score.posterior_success = posterior_mean(features, pauli_score.kind);
    pauli_score.adjusted_score = pauli_score.eligible
        ? pauli_score.estimated_work / pauli_score.posterior_success
        : std::numeric_limits<double>::infinity();
    pauli_score.reason = pauli_score.eligible
        ? "exact bounded Pauli observable is available"
        : "no exact bounded Pauli observable supplied";
    scores.push_back(pauli_score);

    std::stable_sort(scores.begin(), scores.end(), [](const RepresentationScore& left,
                                                       const RepresentationScore& right) {
        return left.adjusted_score < right.adjusted_score;
    });
    return scores;
}

RepresentationScore RepresentationAdvisor::recommend(
    const RepresentationFeatures& features) const {
    const auto scores = rank(features);
    if (scores.empty() || !scores.front().eligible) {
        throw QStateError("Representation advisor found no eligible exact backend");
    }
    return scores.front();
}

void RepresentationAdvisor::observe(
    const RepresentationFeatures& features,
    RepresentationKind kind,
    bool fastest) {
    Posterior& posterior = posteriors_[index(features, kind)];
    if (fastest) {
        ++posterior.successes;
    } else {
        ++posterior.failures;
    }
}

void RepresentationAdvisor::reset() noexcept {
    posteriors_.fill(Posterior{});
}

}  // namespace qubit
