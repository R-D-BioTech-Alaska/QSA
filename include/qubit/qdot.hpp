#pragma once

#include "qubit/qstate.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qubit::qdot {

struct DotInput {
    double theta{0.0};
    double phi{0.0};
    double strength{1.0};
};

struct DotConfig {
    double e0{0.0};
    double eup{0.72};
    double edown{0.93};
    double ex{2.05};
    double detuning_gain{0.42};
    double exciton_gain{0.28};
    double charge_drive{1.10};
    double spin_drive{0.74};
    double intra_dot_bias{0.16};
};

struct CouplingConfig {
    double capacitive{0.18};
    double spin_exchange{0.12};
    double charge_tunneling{0.08};
    double scale{1.15};
};

enum class Topology {
    Chain,
    PairBlocks,
    PairPlusContext,
};

struct PocketConfig {
    std::size_t dot_count{3};
    DotConfig dot{};
    CouplingConfig coupling{};
    double dt{0.16};
    std::size_t trotter_steps{2};
    Topology topology{Topology::PairPlusContext};
};

class QuantumDotPocket {
public:
    explicit QuantumDotPocket(PocketConfig config = {});

    void reset();
    void step(std::span<const DotInput> inputs);

    [[nodiscard]] std::size_t dot_count() const noexcept { return config_.dot_count; }
    [[nodiscard]] std::size_t logical_qubit_count() const noexcept { return config_.dot_count * 2U; }
    [[nodiscard]] std::size_t component_count() const noexcept { return blocks_.size(); }
    [[nodiscard]] std::size_t max_component_qubits() const noexcept;
    [[nodiscard]] std::size_t estimated_bytes() const noexcept;
    [[nodiscard]] std::vector<double> probabilities_one() const;
    [[nodiscard]] std::vector<QComplex> materialize(std::size_t max_qubits = 20) const;
    [[nodiscard]] bool validate(std::string* reason = nullptr) const;
    [[nodiscard]] double current_max_norm_error() const noexcept;
    [[nodiscard]] const PocketConfig& config() const noexcept { return config_; }

private:
    struct Block {
        std::vector<std::size_t> dots{};
        std::vector<QComplex> amplitudes{};
    };

    PocketConfig config_{};
    std::vector<Block> blocks_{};
    std::vector<std::pair<std::size_t, std::size_t>> edges_{};
    std::vector<std::size_t> dot_block_{};
    std::vector<std::size_t> dot_local_{};

    void build_blocks();
    [[nodiscard]] std::pair<std::size_t, std::size_t> locate(std::size_t dot) const;
    void apply_single(std::size_t dot, std::size_t within_dot, const QMatrix2& matrix);
    void apply_cnot(std::size_t first_dot, std::size_t first_within,
                    std::size_t second_dot, std::size_t second_within);
    void apply_h(std::size_t dot, std::size_t within);
    void apply_s(std::size_t dot, std::size_t within);
    void apply_sdg(std::size_t dot, std::size_t within);
    void apply_rx(std::size_t dot, std::size_t within, double angle);
    void apply_ry(std::size_t dot, std::size_t within, double angle);
    void apply_rz(std::size_t dot, std::size_t within, double angle);
    void apply_zz(std::size_t first_dot, std::size_t first_within,
                  std::size_t second_dot, std::size_t second_within, double angle);
    void apply_exchange(std::size_t first_dot, std::size_t first_within,
                        std::size_t second_dot, std::size_t second_within, double angle);
    void apply_xx(std::size_t first_dot, std::size_t first_within,
                  std::size_t second_dot, std::size_t second_within, double angle);
    void apply_yy(std::size_t first_dot, std::size_t first_within,
                  std::size_t second_dot, std::size_t second_within, double angle);
    void apply_local(std::size_t dot, const DotInput& input, double duration);
    void apply_couplings(double duration);
};

[[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> topology_edges(
    std::size_t dot_count, Topology topology);

void apply_reference_step(
    QRegister& state,
    const PocketConfig& config,
    std::span<const DotInput> inputs);

}  // namespace qubit::qdot
