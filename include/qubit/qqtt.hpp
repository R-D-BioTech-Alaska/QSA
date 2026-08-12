#pragma once

#include "qubit/qstate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qubit {

struct QTTCore {
    std::size_t left_rank{0U};
    std::size_t right_rank{0U};
    std::vector<QComplex> zero{};
    std::vector<QComplex> one{};
};

struct QTTConfig {
    std::size_t max_rank{1024U};
    std::size_t max_core_scalars{1U << 20U};
    std::size_t max_total_scalars{1U << 24U};
    std::size_t max_environment_scalars{1U << 20U};
    std::size_t max_materialize_bits{24U};
};

struct QTTStats {
    std::size_t logical_bits{0U};
    std::size_t maximum_rank{0U};
    std::size_t descriptor_scalars{0U};
};

class ExactQTTFunction {
public:
    [[nodiscard]] static ExactQTTFunction from_certified_cores(
        std::vector<QTTCore> cores,
        QTTConfig config = {}) {
        return ExactQTTFunction(std::move(cores), config);
    }

    [[nodiscard]] static ExactQTTFunction product(
        std::span<const std::array<QComplex, 2>> factors,
        QTTConfig config = {}) {
        if (factors.empty()) {
            throw QStateError("QTT product requires at least one binary factor");
        }
        std::vector<QTTCore> cores;
        cores.reserve(factors.size());
        for (const auto& factor : factors) {
            cores.push_back(QTTCore{
                1U,
                1U,
                std::vector<QComplex>{factor[0]},
                std::vector<QComplex>{factor[1]},
            });
        }
        return ExactQTTFunction(std::move(cores), config);
    }

    [[nodiscard]] static ExactQTTFunction weighted_bit_sum(
        std::span<const QComplex> weights,
        QComplex offset = {},
        QTTConfig config = {}) {
        if (weights.empty()) {
            throw QStateError("QTT weighted bit sum requires at least one binary mode");
        }
        if (!finite(offset)) {
            throw QStateError("QTT weighted bit sum offset must be finite");
        }
        for (const QComplex& weight : weights) {
            if (!finite(weight)) {
                throw QStateError("QTT weighted bit sum weights must be finite");
            }
        }

        if (weights.size() == 1U) {
            return ExactQTTFunction(
                std::vector<QTTCore>{QTTCore{
                    1U,
                    1U,
                    std::vector<QComplex>{offset},
                    std::vector<QComplex>{offset + weights.front()},
                }},
                config);
        }

        std::vector<QTTCore> cores;
        cores.reserve(weights.size());
        cores.push_back(QTTCore{
            1U,
            2U,
            std::vector<QComplex>{QComplex{1.0}, offset},
            std::vector<QComplex>{QComplex{1.0}, offset + weights.front()},
        });
        for (std::size_t index = 1U; index + 1U < weights.size(); ++index) {
            cores.push_back(QTTCore{
                2U,
                2U,
                std::vector<QComplex>{
                    QComplex{1.0}, QComplex{}, QComplex{}, QComplex{1.0},
                },
                std::vector<QComplex>{
                    QComplex{1.0}, weights[index], QComplex{}, QComplex{1.0},
                },
            });
        }
        cores.push_back(QTTCore{
            2U,
            1U,
            std::vector<QComplex>{QComplex{}, QComplex{1.0}},
            std::vector<QComplex>{weights.back(), QComplex{1.0}},
        });
        return ExactQTTFunction(std::move(cores), config);
    }

    [[nodiscard]] static ExactQTTFunction hamming_weight(
        std::size_t logical_bits,
        QTTConfig config = {}) {
        if (logical_bits == 0U) {
            throw QStateError("QTT hamming-weight field requires at least one binary mode");
        }
        std::vector<QComplex> weights(logical_bits, QComplex{1.0});
        return weighted_bit_sum(weights, QComplex{}, config);
    }

    [[nodiscard]] static ExactQTTFunction affine_index(
        std::size_t logical_bits,
        QComplex offset = {},
        QComplex slope = QComplex{1.0},
        QTTConfig config = {}) {
        if (logical_bits == 0U) {
            throw QStateError("QTT affine-index field requires at least one binary mode");
        }
        if (!finite(offset) || !finite(slope)) {
            throw QStateError("QTT affine-index parameters must be finite");
        }
        std::vector<QComplex> weights(logical_bits);
        QComplex weight = slope;
        for (std::size_t reverse = logical_bits; reverse-- > 0U;) {
            weights[reverse] = weight;
            if (reverse != 0U) {
                weight *= 2.0;
                if (!finite(weight)) {
                    throw QStateError("QTT affine-index weights exceed finite scalar range");
                }
            }
        }
        return weighted_bit_sum(weights, offset, config);
    }

    [[nodiscard]] static ExactQTTFunction complex_exponential(
        std::size_t logical_bits,
        double angular_frequency,
        QComplex amplitude = QComplex{1.0},
        QTTConfig config = {}) {
        if (logical_bits == 0U) {
            throw QStateError("QTT complex exponential requires at least one binary mode");
        }
        if (!std::isfinite(angular_frequency) || !finite(amplitude)) {
            throw QStateError("QTT complex exponential parameters must be finite");
        }
        const double two_pi = 2.0 * std::acos(-1.0);
        double phase_angle = std::remainder(angular_frequency, two_pi);
        std::vector<std::array<QComplex, 2>> factors(logical_bits);
        for (std::size_t reverse = logical_bits; reverse-- > 0U;) {
            factors[reverse] = {
                QComplex{1.0},
                QComplex::from_polar(1.0, phase_angle),
            };
            if (reverse != 0U) {
                phase_angle = std::remainder(2.0 * phase_angle, two_pi);
            }
        }
        return product(factors, config).scaled(amplitude);
    }

    [[nodiscard]] std::size_t logical_bits() const noexcept { return cores_.size(); }
    [[nodiscard]] const QTTConfig& config() const noexcept { return config_; }
    [[nodiscard]] const QTTStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const std::vector<QTTCore>& cores() const noexcept { return cores_; }

    [[nodiscard]] std::vector<std::size_t> ranks() const {
        std::vector<std::size_t> result;
        result.reserve(cores_.size() + 1U);
        result.push_back(cores_.front().left_rank);
        for (const QTTCore& core : cores_) {
            result.push_back(core.right_rank);
        }
        return result;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) + cores_.capacity() * sizeof(QTTCore);
        for (const QTTCore& core : cores_) {
            bytes += core.zero.capacity() * sizeof(QComplex);
            bytes += core.one.capacity() * sizeof(QComplex);
        }
        return bytes;
    }

    [[nodiscard]] QComplex value_bits(std::span<const std::uint8_t> bits) const {
        if (bits.size() != cores_.size()) {
            throw QStateError("QTT bit string does not match logical shape");
        }
        std::vector<QComplex> row(1U, QComplex{1.0});
        for (std::size_t index = 0U; index < cores_.size(); ++index) {
            const std::uint8_t bit = bits[index];
            if (bit > 1U) {
                throw QStateError("QTT physical bits must be 0 or 1");
            }
            row = contract_row(row, cores_[index], bit);
        }
        return row.front();
    }

    [[nodiscard]] QComplex value(BasisIndex index) const {
        if (cores_.size() > std::numeric_limits<BasisIndex>::digits) {
            throw QStateError("QTT BasisIndex query cannot address more than 64 binary modes");
        }
        if (cores_.size() < std::numeric_limits<BasisIndex>::digits &&
            index >= (BasisIndex{1} << cores_.size())) {
            throw QStateError("QTT BasisIndex lies outside logical range");
        }
        std::vector<std::uint8_t> bits(cores_.size());
        for (std::size_t position = 0U; position < cores_.size(); ++position) {
            const std::size_t shift = cores_.size() - 1U - position;
            bits[position] = static_cast<std::uint8_t>((index >> shift) & BasisIndex{1});
        }
        return value_bits(bits);
    }

    [[nodiscard]] QComplex sum_all() const {
        std::vector<std::int8_t> assignment(cores_.size(), -1);
        return reduce_assignment(assignment);
    }

    [[nodiscard]] QComplex conditioned_sum(
        std::span<const std::size_t> positions,
        std::span<const std::uint8_t> bits) const {
        if (positions.size() != bits.size()) {
            throw QStateError("QTT conditioned reduction position/bit counts differ");
        }
        std::vector<std::int8_t> assignment(cores_.size(), -1);
        for (std::size_t index = 0U; index < positions.size(); ++index) {
            if (positions[index] >= cores_.size()) {
                throw QStateError("QTT conditioned reduction position is outside logical shape");
            }
            if (bits[index] > 1U) {
                throw QStateError("QTT conditioned reduction bits must be 0 or 1");
            }
            if (assignment[positions[index]] != -1) {
                throw QStateError("QTT conditioned reduction positions must be unique");
            }
            assignment[positions[index]] = static_cast<std::int8_t>(bits[index]);
        }
        return reduce_assignment(assignment);
    }

    [[nodiscard]] QComplex inner_product(const ExactQTTFunction& other) const {
        if (cores_.size() != other.cores_.size()) {
            throw QStateError("QTT inner product requires equal logical shapes");
        }
        const std::size_t environment_cap =
            std::min(config_.max_environment_scalars, other.config_.max_environment_scalars);
        std::vector<QComplex> environment(1U, QComplex{1.0});
        std::size_t left_rows = 1U;
        std::size_t right_rows = 1U;

        for (std::size_t index = 0U; index < cores_.size(); ++index) {
            const QTTCore& left = cores_[index];
            const QTTCore& right = other.cores_[index];
            if (left.left_rank != left_rows || right.left_rank != right_rows) {
                throw QStateError("QTT inner-product environment rank mismatch");
            }
            const std::size_t next_size = checked_product(
                left.right_rank,
                right.right_rank,
                "QTT inner-product environment size overflowed");
            if (next_size > environment_cap) {
                throw QStateError("QTT inner-product environment exceeds configured scalar cap");
            }
            std::vector<QComplex> next(next_size);
            for (std::size_t left_in = 0U; left_in < left.left_rank; ++left_in) {
                for (std::size_t right_in = 0U; right_in < right.left_rank; ++right_in) {
                    const QComplex prefix =
                        environment[left_in * right.left_rank + right_in];
                    if (prefix == QComplex{}) {
                        continue;
                    }
                    for (std::uint8_t bit = 0U; bit < 2U; ++bit) {
                        const std::vector<QComplex>& left_slice = bit == 0U ? left.zero : left.one;
                        const std::vector<QComplex>& right_slice = bit == 0U ? right.zero : right.one;
                        for (std::size_t left_out = 0U; left_out < left.right_rank; ++left_out) {
                            const QComplex left_value =
                                left_slice[left_in * left.right_rank + left_out].conjugate();
                            for (std::size_t right_out = 0U;
                                 right_out < right.right_rank;
                                 ++right_out) {
                                next[left_out * right.right_rank + right_out] +=
                                    prefix * left_value *
                                    right_slice[right_in * right.right_rank + right_out];
                            }
                        }
                    }
                }
            }
            environment = std::move(next);
            left_rows = left.right_rank;
            right_rows = right.right_rank;
        }
        return environment.front();
    }

    [[nodiscard]] double norm_squared() const {
        const QComplex result = inner_product(*this);
        const double scale = 1.0 + std::abs(result.re);
        if (std::abs(result.im) > 1e-10 * scale) {
            throw QStateError("QTT norm acquired a non-negligible imaginary component");
        }
        if (result.re < 0.0 && std::abs(result.re) > 1e-12 * scale) {
            throw QStateError("QTT norm acquired a negative real component");
        }
        return result.re < 0.0 ? 0.0 : result.re;
    }

    [[nodiscard]] ExactQTTFunction scaled(QComplex scalar) const {
        if (!finite(scalar)) {
            throw QStateError("QTT scale must be finite");
        }
        std::vector<QTTCore> next = cores_;
        for (QComplex& value : next.front().zero) {
            value *= scalar;
        }
        for (QComplex& value : next.front().one) {
            value *= scalar;
        }
        return ExactQTTFunction(std::move(next), config_);
    }

    [[nodiscard]] ExactQTTFunction add(const ExactQTTFunction& other) const {
        require_equal_shape(other, "QTT addition");
        const QTTConfig output_config = combined_config(other);
        if (cores_.size() == 1U) {
            QTTCore core{1U, 1U, std::vector<QComplex>(1U), std::vector<QComplex>(1U)};
            core.zero[0] = cores_[0].zero[0] + other.cores_[0].zero[0];
            core.one[0] = cores_[0].one[0] + other.cores_[0].one[0];
            return ExactQTTFunction(std::vector<QTTCore>{std::move(core)}, output_config);
        }

        std::vector<std::size_t> output_ranks(cores_.size() + 1U, 1U);
        for (std::size_t bond = 1U; bond < cores_.size(); ++bond) {
            output_ranks[bond] = checked_sum(
                cores_[bond - 1U].right_rank,
                other.cores_[bond - 1U].right_rank,
                "QTT addition rank overflowed");
        }
        preflight_ranks(output_ranks, output_config, "QTT addition");

        std::vector<QTTCore> next;
        next.reserve(cores_.size());
        for (std::size_t index = 0U; index < cores_.size(); ++index) {
            const QTTCore& left = cores_[index];
            const QTTCore& right = other.cores_[index];
            const std::size_t out_left = output_ranks[index];
            const std::size_t out_right = output_ranks[index + 1U];
            QTTCore core{
                out_left,
                out_right,
                std::vector<QComplex>(out_left * out_right),
                std::vector<QComplex>(out_left * out_right),
            };
            for (std::uint8_t bit = 0U; bit < 2U; ++bit) {
                const std::vector<QComplex>& left_slice = bit == 0U ? left.zero : left.one;
                const std::vector<QComplex>& right_slice = bit == 0U ? right.zero : right.one;
                std::vector<QComplex>& output = bit == 0U ? core.zero : core.one;
                if (index == 0U) {
                    std::copy(left_slice.begin(), left_slice.end(), output.begin());
                    std::copy(
                        right_slice.begin(),
                        right_slice.end(),
                        output.begin() + static_cast<std::ptrdiff_t>(left.right_rank));
                } else if (index + 1U == cores_.size()) {
                    for (std::size_t row = 0U; row < left.left_rank; ++row) {
                        output[row] = left_slice[row];
                    }
                    for (std::size_t row = 0U; row < right.left_rank; ++row) {
                        output[left.left_rank + row] = right_slice[row];
                    }
                } else {
                    for (std::size_t row = 0U; row < left.left_rank; ++row) {
                        for (std::size_t column = 0U; column < left.right_rank; ++column) {
                            output[row * out_right + column] =
                                left_slice[row * left.right_rank + column];
                        }
                    }
                    for (std::size_t row = 0U; row < right.left_rank; ++row) {
                        for (std::size_t column = 0U; column < right.right_rank; ++column) {
                            output[(left.left_rank + row) * out_right +
                                   left.right_rank + column] =
                                right_slice[row * right.right_rank + column];
                        }
                    }
                }
            }
            next.push_back(std::move(core));
        }
        return ExactQTTFunction(std::move(next), output_config);
    }

    [[nodiscard]] ExactQTTFunction hadamard(const ExactQTTFunction& other) const {
        require_equal_shape(other, "QTT Hadamard product");
        const QTTConfig output_config = combined_config(other);
        std::vector<std::size_t> output_ranks(cores_.size() + 1U, 1U);
        for (std::size_t bond = 1U; bond < cores_.size(); ++bond) {
            output_ranks[bond] = checked_product(
                cores_[bond - 1U].right_rank,
                other.cores_[bond - 1U].right_rank,
                "QTT Hadamard rank overflowed");
        }
        preflight_ranks(output_ranks, output_config, "QTT Hadamard product");

        std::vector<QTTCore> next;
        next.reserve(cores_.size());
        for (std::size_t index = 0U; index < cores_.size(); ++index) {
            const QTTCore& left = cores_[index];
            const QTTCore& right = other.cores_[index];
            const std::size_t out_left = output_ranks[index];
            const std::size_t out_right = output_ranks[index + 1U];
            QTTCore core{
                out_left,
                out_right,
                std::vector<QComplex>(out_left * out_right),
                std::vector<QComplex>(out_left * out_right),
            };
            for (std::uint8_t bit = 0U; bit < 2U; ++bit) {
                const std::vector<QComplex>& left_slice = bit == 0U ? left.zero : left.one;
                const std::vector<QComplex>& right_slice = bit == 0U ? right.zero : right.one;
                std::vector<QComplex>& output = bit == 0U ? core.zero : core.one;
                for (std::size_t left_row = 0U; left_row < left.left_rank; ++left_row) {
                    for (std::size_t right_row = 0U; right_row < right.left_rank; ++right_row) {
                        const std::size_t output_row =
                            left_row * right.left_rank + right_row;
                        for (std::size_t left_column = 0U;
                             left_column < left.right_rank;
                             ++left_column) {
                            const QComplex left_value =
                                left_slice[left_row * left.right_rank + left_column];
                            for (std::size_t right_column = 0U;
                                 right_column < right.right_rank;
                                 ++right_column) {
                                const std::size_t output_column =
                                    left_column * right.right_rank + right_column;
                                output[output_row * out_right + output_column] =
                                    left_value *
                                    right_slice[right_row * right.right_rank + right_column];
                            }
                        }
                    }
                }
            }
            next.push_back(std::move(core));
        }
        return ExactQTTFunction(std::move(next), output_config);
    }

    [[nodiscard]] std::vector<QComplex> materialize() const {
        if (cores_.size() > config_.max_materialize_bits) {
            throw QStateError("QTT dense materialization exceeds configured logical-bit cap");
        }
        if (cores_.size() >= std::numeric_limits<std::size_t>::digits) {
            throw QStateError("QTT dense materialization size overflowed");
        }
        const std::size_t count = std::size_t{1} << cores_.size();
        std::vector<QComplex> values(count);
        for (std::size_t index = 0U; index < count; ++index) {
            values[index] = value(static_cast<BasisIndex>(index));
        }
        return values;
    }

private:
    std::vector<QTTCore> cores_{};
    QTTConfig config_{};
    QTTStats stats_{};

    explicit ExactQTTFunction(std::vector<QTTCore> cores, QTTConfig config)
        : cores_(std::move(cores)), config_(config) {
        validate();
    }

    [[nodiscard]] static bool finite(const QComplex& value) noexcept {
        return std::isfinite(value.re) && std::isfinite(value.im);
    }

    [[nodiscard]] static std::size_t checked_product(
        std::size_t left,
        std::size_t right,
        const char* message) {
        if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
            throw QStateError(message);
        }
        return left * right;
    }

    [[nodiscard]] static std::size_t checked_sum(
        std::size_t left,
        std::size_t right,
        const char* message) {
        if (right > std::numeric_limits<std::size_t>::max() - left) {
            throw QStateError(message);
        }
        return left + right;
    }

    static void preflight_ranks(
        std::span<const std::size_t> ranks,
        const QTTConfig& config,
        const char* label) {
        std::size_t total = 0U;
        for (std::size_t rank : ranks) {
            if (rank == 0U || rank > config.max_rank) {
                throw QStateError(std::string(label) + " exceeds configured QTT rank cap");
            }
        }
        for (std::size_t index = 0U; index + 1U < ranks.size(); ++index) {
            const std::size_t matrix = checked_product(
                ranks[index], ranks[index + 1U], "QTT composition core size overflowed");
            const std::size_t scalars = checked_product(
                matrix, 2U, "QTT composition core scalar count overflowed");
            if (scalars > config.max_core_scalars) {
                throw QStateError(std::string(label) + " exceeds configured QTT core scalar cap");
            }
            total = checked_sum(total, scalars, "QTT composition total scalar count overflowed");
            if (total > config.max_total_scalars) {
                throw QStateError(std::string(label) + " exceeds configured QTT total scalar cap");
            }
        }
    }

    [[nodiscard]] QTTConfig combined_config(const ExactQTTFunction& other) const noexcept {
        return QTTConfig{
            std::min(config_.max_rank, other.config_.max_rank),
            std::min(config_.max_core_scalars, other.config_.max_core_scalars),
            std::min(config_.max_total_scalars, other.config_.max_total_scalars),
            std::min(config_.max_environment_scalars, other.config_.max_environment_scalars),
            std::min(config_.max_materialize_bits, other.config_.max_materialize_bits),
        };
    }

    void require_equal_shape(const ExactQTTFunction& other, const char* label) const {
        if (cores_.size() != other.cores_.size()) {
            throw QStateError(std::string(label) + " requires equal logical shapes");
        }
    }

    [[nodiscard]] static std::vector<QComplex> contract_row(
        std::span<const QComplex> row,
        const QTTCore& core,
        std::uint8_t bit) {
        if (row.size() != core.left_rank) {
            throw QStateError("QTT row/core rank mismatch");
        }
        const std::vector<QComplex>& matrix = bit == 0U ? core.zero : core.one;
        std::vector<QComplex> next(core.right_rank);
        for (std::size_t left = 0U; left < core.left_rank; ++left) {
            if (row[left] == QComplex{}) {
                continue;
            }
            for (std::size_t right = 0U; right < core.right_rank; ++right) {
                next[right] += row[left] * matrix[left * core.right_rank + right];
            }
        }
        return next;
    }

    [[nodiscard]] QComplex reduce_assignment(
        std::span<const std::int8_t> assignment) const {
        std::vector<QComplex> row(1U, QComplex{1.0});
        for (std::size_t index = 0U; index < cores_.size(); ++index) {
            const QTTCore& core = cores_[index];
            if (assignment[index] >= 0) {
                row = contract_row(
                    row,
                    core,
                    static_cast<std::uint8_t>(assignment[index]));
                continue;
            }
            if (row.size() != core.left_rank) {
                throw QStateError("QTT reduction row/core rank mismatch");
            }
            std::vector<QComplex> next(core.right_rank);
            for (std::size_t left = 0U; left < core.left_rank; ++left) {
                for (std::size_t right = 0U; right < core.right_rank; ++right) {
                    const std::size_t offset = left * core.right_rank + right;
                    next[right] += row[left] * (core.zero[offset] + core.one[offset]);
                }
            }
            row = std::move(next);
        }
        return row.front();
    }

    void validate() {
        if (cores_.empty()) {
            throw QStateError("QTT function requires at least one binary core");
        }
        if (config_.max_rank == 0U || config_.max_core_scalars == 0U ||
            config_.max_total_scalars == 0U || config_.max_environment_scalars == 0U) {
            throw QStateError("QTT configuration contains a zero resource cap");
        }

        std::size_t previous_right = 0U;
        std::size_t maximum_rank = 1U;
        std::size_t total_scalars = 0U;
        for (std::size_t index = 0U; index < cores_.size(); ++index) {
            const QTTCore& core = cores_[index];
            if (core.left_rank == 0U || core.right_rank == 0U) {
                throw QStateError("QTT ranks must be positive");
            }
            if (index == 0U && core.left_rank != 1U) {
                throw QStateError("QTT first left rank must be one");
            }
            if (index != 0U && core.left_rank != previous_right) {
                throw QStateError("QTT adjacent bond ranks do not match");
            }
            if (core.left_rank > config_.max_rank || core.right_rank > config_.max_rank) {
                throw QStateError("QTT core exceeds configured rank cap");
            }
            const std::size_t matrix = checked_product(
                core.left_rank, core.right_rank, "QTT core shape overflowed");
            if (core.zero.size() != matrix || core.one.size() != matrix) {
                throw QStateError("QTT physical slices do not match declared core ranks");
            }
            const std::size_t core_scalars = checked_product(
                matrix, 2U, "QTT core scalar count overflowed");
            if (core_scalars > config_.max_core_scalars) {
                throw QStateError("QTT core exceeds configured scalar cap");
            }
            total_scalars = checked_sum(
                total_scalars, core_scalars, "QTT total scalar count overflowed");
            if (total_scalars > config_.max_total_scalars) {
                throw QStateError("QTT function exceeds configured total scalar cap");
            }
            for (const QComplex& value : core.zero) {
                if (!finite(value)) {
                    throw QStateError("QTT core contains a non-finite scalar");
                }
            }
            for (const QComplex& value : core.one) {
                if (!finite(value)) {
                    throw QStateError("QTT core contains a non-finite scalar");
                }
            }
            maximum_rank = std::max(maximum_rank, std::max(core.left_rank, core.right_rank));
            previous_right = core.right_rank;
        }
        if (previous_right != 1U) {
            throw QStateError("QTT final right rank must be one");
        }
        stats_.logical_bits = cores_.size();
        stats_.maximum_rank = maximum_rank;
        stats_.descriptor_scalars = total_scalars;
    }
};

}  // namespace qubit
