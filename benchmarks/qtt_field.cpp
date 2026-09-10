#include "qubit/qqtt_field.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace qubit;

int main() {
    constexpr std::size_t logical_bits = 4096U;
    constexpr double kinetic = 0.25;
    constexpr double nonlinear = 0.125;
    std::vector<double> laplacian_weights(logical_bits);
    std::vector<QComplex> potential_weights(logical_bits);
    for (std::size_t i = 0U; i < logical_bits; ++i) {
        laplacian_weights[i] = 0.5 + 0.0625 * static_cast<double>(i % 5U);
        potential_weights[i] = QComplex{0.00025 * static_cast<double>((i % 7U) + 1U)};
    }

    QTTConfig field_config;
    field_config.max_rank = 32U;
    field_config.max_core_scalars = 8192U;
    field_config.max_total_scalars = 1U << 22U;
    QTTOperatorConfig operator_config;
    operator_config.max_rank = 32U;
    operator_config.max_core_scalars = 8192U;
    operator_config.max_total_scalars = 1U << 22U;
    operator_config.max_apply_rank = 32U;
    operator_config.max_apply_core_scalars = 8192U;
    operator_config.max_apply_total_scalars = 1U << 22U;

    const ExactQTTFunction field = ExactQTTFunction::complex_exponential(
        logical_bits, 1.0e-6, QComplex{1.0}, field_config);
    const ExactQTTFunction potential = ExactQTTFunction::weighted_bit_sum(
        potential_weights, QComplex{0.1}, field_config);

    const auto build_start = std::chrono::steady_clock::now();
    const ExactQTTFieldHamiltonian hamiltonian = ExactQTTFieldHamiltonian::hypercube(
        laplacian_weights, potential, kinetic, nonlinear, operator_config);
    const auto build_end = std::chrono::steady_clock::now();

    const auto apply_start = std::chrono::steady_clock::now();
    const ExactQTTFunction rhs = hamiltonian.rhs(field);
    const auto apply_end = std::chrono::steady_clock::now();

    std::vector<std::uint8_t> bits(logical_bits);
    for (std::size_t i = 0U; i < logical_bits; ++i) {
        bits[i] = static_cast<std::uint8_t>(((i * 13U + 5U) % 17U) < 8U ? 1U : 0U);
    }
    const auto query_start = std::chrono::steady_clock::now();
    const QComplex observed = rhs.value_bits(bits);
    const auto query_end = std::chrono::steady_clock::now();

    const QComplex center = field.value_bits(bits);
    QComplex laplacian{};
    double potential_value = 0.1;
    for (std::size_t i = 0U; i < logical_bits; ++i) {
        const QTTCore& core = field.cores()[i];
        const QComplex zero = core.zero[0];
        const QComplex one = core.one[0];
        const QComplex ratio = bits[i] == 0U ? one / zero : zero / one;
        laplacian += center * (ratio - QComplex{1.0}) * laplacian_weights[i];
        if (bits[i] != 0U) {
            potential_value += potential_weights[i].re;
        }
    }
    const QComplex h_value = laplacian * -kinetic + center * potential_value +
                             center * (nonlinear * center.norm2());
    const QComplex expected = h_value * QComplex{0.0, -1.0};
    const double error = (observed - expected).magnitude();

    const auto milliseconds = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    std::cout << std::setprecision(17);
    std::cout << "qtt_field_bits=" << logical_bits << '\n';
    std::cout << "qtt_field_logical_domain_exponent=" << logical_bits << '\n';
    std::cout << "qtt_field_linear_operator_rank=" << hamiltonian.stats().linear_operator_rank << '\n';
    std::cout << "qtt_field_linear_operator_scalars=" << hamiltonian.stats().linear_operator_scalars << '\n';
    std::cout << "qtt_field_rhs_rank=" << rhs.stats().maximum_rank << '\n';
    std::cout << "qtt_field_rhs_descriptor_scalars=" << rhs.stats().descriptor_scalars << '\n';
    std::cout << "qtt_field_build_ms=" << milliseconds(build_start, build_end) << '\n';
    std::cout << "qtt_field_apply_ms=" << milliseconds(apply_start, apply_end) << '\n';
    std::cout << "qtt_field_query_ms=" << milliseconds(query_start, query_end) << '\n';
    std::cout << "qtt_field_selected_error=" << error << '\n';
    return error <= 1e-8 ? 0 : 1;
}
