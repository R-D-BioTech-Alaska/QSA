#pragma once

#include "qubit/qstate.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qubit {

enum class RepresentationKind : std::uint8_t {
    Register = 0,
    Symmetry = 1,
    QuantumDot = 2,
    Stabilizer = 3,
    PhaseGraph = 4,
};

struct RepresentationFeatures {
    std::size_t qubit_count{0};
    std::size_t component_count{0};
    std::size_t largest_component{0};
    std::size_t support_entries{0};
    std::uint64_t repeated_steps{1};
    std::size_t exact_symmetry_classes{0};
    bool quantum_dot_declared{false};
    bool clifford_only{false};
    bool uniform_phase_graph{false};
    std::size_t phase_graph_edges{0};
};

struct RepresentationScore {
    RepresentationKind kind{RepresentationKind::Register};
    bool eligible{false};
    double estimated_work{0.0};
    double posterior_success{0.0};
    double adjusted_score{0.0};
    std::string reason{};
};

struct RepresentationAdvisorConfig {
    double prior_success{1.0};
    double prior_failure{1.0};
};

class RepresentationAdvisor {
public:
    explicit RepresentationAdvisor(RepresentationAdvisorConfig config = {});

    [[nodiscard]] static RepresentationFeatures inspect(
        const QRegister& state,
        std::uint64_t repeated_steps = 1U,
        std::size_t exact_symmetry_classes = 0U,
        bool quantum_dot_declared = false,
        bool clifford_only = false,
        bool uniform_phase_graph = false,
        std::size_t phase_graph_edges = 0U);

    [[nodiscard]] std::vector<RepresentationScore> rank(
        const RepresentationFeatures& features) const;
    [[nodiscard]] RepresentationScore recommend(
        const RepresentationFeatures& features) const;

    void observe(
        const RepresentationFeatures& features,
        RepresentationKind kind,
        bool fastest);
    void reset() noexcept;

private:
    struct Posterior {
        std::uint64_t successes{0};
        std::uint64_t failures{0};
    };

    RepresentationAdvisorConfig config_{};
    std::array<Posterior, 30> posteriors_{};

    [[nodiscard]] std::size_t context(const RepresentationFeatures& features) const noexcept;
    [[nodiscard]] std::size_t index(
        const RepresentationFeatures& features,
        RepresentationKind kind) const noexcept;
    [[nodiscard]] double posterior_mean(
        const RepresentationFeatures& features,
        RepresentationKind kind) const noexcept;
};

[[nodiscard]] const char* representation_name(RepresentationKind kind) noexcept;

}  // namespace qubit
