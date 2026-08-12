#pragma once

#include "qubit/qqtt.hpp"

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

struct QTTOperatorCore {
    std::size_t left_rank{0U};
    std::size_t right_rank{0U};
    std::array<std::vector<QComplex>, 4> blocks{};
};

struct QTTOperatorConfig {
    std::size_t max_rank{1024U};
    std::size_t max_core_scalars{1U << 20U};
    std::size_t max_total_scalars{1U << 24U};
    std::size_t max_apply_rank{1024U};
    std::size_t max_apply_core_scalars{1U << 20U};
    std::size_t max_apply_total_scalars{1U << 24U};
};

struct QTTOperatorStats {
    std::size_t logical_bits{0U};
    std::size_t maximum_rank{0U};
    std::size_t descriptor_scalars{0U};
};

class ExactQTTOperator {
public:
    [[nodiscard]] static ExactQTTOperator from_certified_cores(
        std::vector<QTTOperatorCore> cores, QTTOperatorConfig config = {}) {
        return ExactQTTOperator(std::move(cores), config);
    }

    [[nodiscard]] static ExactQTTOperator identity(
        std::size_t logical_bits, QTTOperatorConfig config = {}) {
        if (logical_bits == 0U) {
            throw QStateError("QTT operator identity requires at least one binary mode");
        }
        preflight_uniform(logical_bits, 1U, config, "QTT operator identity");
        std::vector<QTTOperatorCore> cores(logical_bits);
        for (QTTOperatorCore& core : cores) {
            core = make_core(1U, 1U);
            put_identity(core, 0U, 0U);
        }
        return ExactQTTOperator(std::move(cores), config);
    }

    [[nodiscard]] static ExactQTTOperator diagonal(
        const ExactQTTFunction& field, QTTOperatorConfig config = {}) {
        const std::vector<std::size_t> ranks = field.ranks();
        preflight(ranks, config, "QTT diagonal operator");
        std::vector<QTTOperatorCore> cores;
        cores.reserve(field.logical_bits());
        for (const QTTCore& source : field.cores()) {
            QTTOperatorCore core = make_core(source.left_rank, source.right_rank);
            core.blocks[0] = source.zero;
            core.blocks[3] = source.one;
            cores.push_back(std::move(core));
        }
        return ExactQTTOperator(std::move(cores), config);
    }

    [[nodiscard]] static ExactQTTOperator weighted_hypercube_laplacian(
        std::span<const double> weights, QTTOperatorConfig config = {}) {
        if (weights.empty()) {
            throw QStateError("QTT hypercube Laplacian requires at least one binary mode");
        }
        for (double weight : weights) {
            if (!std::isfinite(weight)) {
                throw QStateError("QTT hypercube Laplacian weights must be finite");
            }
        }
        preflight_uniform(
            weights.size(), weights.size() == 1U ? 1U : 2U, config, "QTT hypercube Laplacian");
        if (weights.size() == 1U) {
            QTTOperatorCore core = make_core(1U, 1U);
            put_laplacian(core, 0U, 0U, weights.front());
            return ExactQTTOperator({std::move(core)}, config);
        }

        std::vector<QTTOperatorCore> cores;
        cores.reserve(weights.size());
        QTTOperatorCore first = make_core(1U, 2U);
        put_identity(first, 0U, 0U);
        put_laplacian(first, 0U, 1U, weights.front());
        cores.push_back(std::move(first));
        for (std::size_t i = 1U; i + 1U < weights.size(); ++i) {
            QTTOperatorCore core = make_core(2U, 2U);
            put_identity(core, 0U, 0U);
            put_laplacian(core, 0U, 1U, weights[i]);
            put_identity(core, 1U, 1U);
            cores.push_back(std::move(core));
        }
        QTTOperatorCore last = make_core(2U, 1U);
        put_laplacian(last, 0U, 0U, weights.back());
        put_identity(last, 1U, 0U);
        cores.push_back(std::move(last));
        return ExactQTTOperator(std::move(cores), config);
    }

    [[nodiscard]] std::size_t logical_bits() const noexcept { return cores_.size(); }
    [[nodiscard]] const QTTOperatorConfig& config() const noexcept { return config_; }
    [[nodiscard]] const QTTOperatorStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const std::vector<QTTOperatorCore>& cores() const noexcept { return cores_; }

    [[nodiscard]] QComplex matrix_element_bits(
        std::span<const std::uint8_t> output_bits,
        std::span<const std::uint8_t> input_bits) const {
        if (output_bits.size() != cores_.size() || input_bits.size() != cores_.size()) {
            throw QStateError("QTT operator bit strings do not match logical shape");
        }
        std::vector<QComplex> row(1U, QComplex{1.0});
        for (std::size_t i = 0U; i < cores_.size(); ++i) {
            if (output_bits[i] > 1U || input_bits[i] > 1U) {
                throw QStateError("QTT operator physical bits must be 0 or 1");
            }
            row = contract(row, cores_[i], output_bits[i], input_bits[i]);
        }
        return row.front();
    }

    [[nodiscard]] ExactQTTOperator scaled(QComplex scalar) const {
        if (!finite(scalar)) {
            throw QStateError("QTT operator scale must be finite");
        }
        std::vector<QTTOperatorCore> next = cores_;
        for (auto& block : next.front().blocks) {
            for (QComplex& value : block) {
                value *= scalar;
            }
        }
        return ExactQTTOperator(std::move(next), config_);
    }

    [[nodiscard]] ExactQTTOperator add(const ExactQTTOperator& other) const {
        if (cores_.size() != other.cores_.size()) {
            throw QStateError("QTT operator addition requires equal logical shapes");
        }
        const QTTOperatorConfig output_config = combined_config(other);
        if (cores_.size() == 1U) {
            QTTOperatorCore core = make_core(1U, 1U);
            for (std::size_t block = 0U; block < 4U; ++block) {
                core.blocks[block][0] = cores_[0].blocks[block][0] + other.cores_[0].blocks[block][0];
            }
            return ExactQTTOperator({std::move(core)}, output_config);
        }

        std::vector<std::size_t> ranks(cores_.size() + 1U, 1U);
        for (std::size_t bond = 1U; bond < cores_.size(); ++bond) {
            ranks[bond] = checked_sum(
                cores_[bond - 1U].right_rank, other.cores_[bond - 1U].right_rank,
                "QTT operator addition rank overflowed");
        }
        preflight(ranks, output_config, "QTT operator addition");

        std::vector<QTTOperatorCore> next;
        next.reserve(cores_.size());
        for (std::size_t i = 0U; i < cores_.size(); ++i) {
            const QTTOperatorCore& a = cores_[i];
            const QTTOperatorCore& b = other.cores_[i];
            const std::size_t out_left = ranks[i];
            const std::size_t out_right = ranks[i + 1U];
            QTTOperatorCore core = make_core(out_left, out_right);
            for (std::size_t block = 0U; block < 4U; ++block) {
                auto& dst = core.blocks[block];
                const auto& av = a.blocks[block];
                const auto& bv = b.blocks[block];
                if (i == 0U) {
                    std::copy(av.begin(), av.end(), dst.begin());
                    std::copy(bv.begin(), bv.end(), dst.begin() + static_cast<std::ptrdiff_t>(a.right_rank));
                } else if (i + 1U == cores_.size()) {
                    std::copy(av.begin(), av.end(), dst.begin());
                    std::copy(bv.begin(), bv.end(), dst.begin() + static_cast<std::ptrdiff_t>(a.left_rank));
                } else {
                    copy_block(av, a.left_rank, a.right_rank, dst, out_right, 0U, 0U);
                    copy_block(bv, b.left_rank, b.right_rank, dst, out_right, a.left_rank, a.right_rank);
                }
            }
            next.push_back(std::move(core));
        }
        return ExactQTTOperator(std::move(next), output_config);
    }

    [[nodiscard]] ExactQTTFunction apply(const ExactQTTFunction& state) const {
        if (state.logical_bits() != cores_.size()) {
            throw QStateError("QTT operator application requires equal logical shapes");
        }
        std::vector<std::size_t> ranks(cores_.size() + 1U, 1U);
        for (std::size_t bond = 1U; bond < cores_.size(); ++bond) {
            ranks[bond] = checked_product(
                cores_[bond - 1U].right_rank, state.cores()[bond - 1U].right_rank,
                "QTT operator application rank overflowed");
        }
        preflight_apply(ranks, state);

        std::vector<QTTCore> output;
        output.reserve(cores_.size());
        for (std::size_t i = 0U; i < cores_.size(); ++i) {
            const QTTOperatorCore& op = cores_[i];
            const QTTCore& input = state.cores()[i];
            const std::size_t out_left = ranks[i];
            const std::size_t out_right = ranks[i + 1U];
            QTTCore core{out_left, out_right,
                         std::vector<QComplex>(out_left * out_right),
                         std::vector<QComplex>(out_left * out_right)};
            for (std::uint8_t out_bit = 0U; out_bit < 2U; ++out_bit) {
                auto& dst = out_bit == 0U ? core.zero : core.one;
                for (std::size_t ol = 0U; ol < op.left_rank; ++ol) {
                    for (std::size_t sl = 0U; sl < input.left_rank; ++sl) {
                        const std::size_t dl = ol * input.left_rank + sl;
                        for (std::size_t orank = 0U; orank < op.right_rank; ++orank) {
                            for (std::size_t sr = 0U; sr < input.right_rank; ++sr) {
                                const std::size_t dr = orank * input.right_rank + sr;
                                QComplex value{};
                                for (std::uint8_t in_bit = 0U; in_bit < 2U; ++in_bit) {
                                    const auto& block = op.blocks[out_bit * 2U + in_bit];
                                    const auto& slice = in_bit == 0U ? input.zero : input.one;
                                    value += block[ol * op.right_rank + orank] *
                                             slice[sl * input.right_rank + sr];
                                }
                                dst[dl * out_right + dr] = value;
                            }
                        }
                    }
                }
            }
            output.push_back(std::move(core));
        }

        QTTConfig output_config = state.config();
        output_config.max_rank = std::min(output_config.max_rank, config_.max_apply_rank);
        output_config.max_core_scalars =
            std::min(output_config.max_core_scalars, config_.max_apply_core_scalars);
        output_config.max_total_scalars =
            std::min(output_config.max_total_scalars, config_.max_apply_total_scalars);
        return ExactQTTFunction::from_certified_cores(std::move(output), output_config);
    }

private:
    std::vector<QTTOperatorCore> cores_{};
    QTTOperatorConfig config_{};
    QTTOperatorStats stats_{};

    ExactQTTOperator(std::vector<QTTOperatorCore> cores, QTTOperatorConfig config)
        : cores_(std::move(cores)), config_(config) { validate(); }

    [[nodiscard]] static bool finite(const QComplex& value) noexcept {
        return std::isfinite(value.re) && std::isfinite(value.im);
    }

    [[nodiscard]] static std::size_t checked_product(
        std::size_t left, std::size_t right, const char* message) {
        if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
            throw QStateError(message);
        }
        return left * right;
    }

    [[nodiscard]] static std::size_t checked_sum(
        std::size_t left, std::size_t right, const char* message) {
        if (right > std::numeric_limits<std::size_t>::max() - left) {
            throw QStateError(message);
        }
        return left + right;
    }

    [[nodiscard]] static QTTOperatorCore make_core(std::size_t left, std::size_t right) {
        const std::size_t count = checked_product(left, right, "QTT operator core shape overflowed");
        return QTTOperatorCore{
            left, right,
            std::array<std::vector<QComplex>, 4>{
                std::vector<QComplex>(count), std::vector<QComplex>(count),
                std::vector<QComplex>(count), std::vector<QComplex>(count)}};
    }

    static void put_identity(QTTOperatorCore& core, std::size_t left, std::size_t right) {
        const std::size_t p = left * core.right_rank + right;
        core.blocks[0][p] = QComplex{1.0};
        core.blocks[3][p] = QComplex{1.0};
    }

    static void put_laplacian(
        QTTOperatorCore& core, std::size_t left, std::size_t right, double weight) {
        const std::size_t p = left * core.right_rank + right;
        core.blocks[0][p] = QComplex{-weight};
        core.blocks[1][p] = QComplex{weight};
        core.blocks[2][p] = QComplex{weight};
        core.blocks[3][p] = QComplex{-weight};
    }

    static void copy_block(
        std::span<const QComplex> source, std::size_t rows, std::size_t columns,
        std::vector<QComplex>& target, std::size_t target_columns,
        std::size_t row_offset, std::size_t column_offset) {
        for (std::size_t row = 0U; row < rows; ++row) {
            for (std::size_t column = 0U; column < columns; ++column) {
                target[(row_offset + row) * target_columns + column_offset + column] =
                    source[row * columns + column];
            }
        }
    }

    static void preflight_uniform(
        std::size_t logical_bits, std::size_t rank,
        const QTTOperatorConfig& config, const char* label) {
        std::vector<std::size_t> ranks(logical_bits + 1U, rank);
        ranks.front() = 1U;
        ranks.back() = 1U;
        preflight(ranks, config, label);
    }

    static void preflight(
        std::span<const std::size_t> ranks,
        const QTTOperatorConfig& config, const char* label) {
        std::size_t total = 0U;
        for (std::size_t rank : ranks) {
            if (rank == 0U || rank > config.max_rank) {
                throw QStateError(std::string(label) + " exceeds configured operator rank cap");
            }
        }
        for (std::size_t i = 0U; i + 1U < ranks.size(); ++i) {
            const std::size_t matrix = checked_product(
                ranks[i], ranks[i + 1U], "QTT operator preflight core size overflowed");
            const std::size_t scalars = checked_product(
                matrix, 4U, "QTT operator preflight scalar count overflowed");
            if (scalars > config.max_core_scalars) {
                throw QStateError(std::string(label) + " exceeds configured operator core scalar cap");
            }
            total = checked_sum(total, scalars, "QTT operator preflight total overflowed");
            if (total > config.max_total_scalars) {
                throw QStateError(std::string(label) + " exceeds configured operator total scalar cap");
            }
        }
    }

    void preflight_apply(
        std::span<const std::size_t> ranks, const ExactQTTFunction& state) const {
        const std::size_t rank_cap = std::min(config_.max_apply_rank, state.config().max_rank);
        const std::size_t core_cap =
            std::min(config_.max_apply_core_scalars, state.config().max_core_scalars);
        const std::size_t total_cap =
            std::min(config_.max_apply_total_scalars, state.config().max_total_scalars);
        std::size_t total = 0U;
        for (std::size_t rank : ranks) {
            if (rank == 0U || rank > rank_cap) {
                throw QStateError("QTT operator application exceeds configured output rank cap");
            }
        }
        for (std::size_t i = 0U; i + 1U < ranks.size(); ++i) {
            const std::size_t matrix = checked_product(
                ranks[i], ranks[i + 1U], "QTT operator application core size overflowed");
            const std::size_t scalars = checked_product(
                matrix, 2U, "QTT operator application scalar count overflowed");
            if (scalars > core_cap) {
                throw QStateError("QTT operator application exceeds configured output core scalar cap");
            }
            total = checked_sum(total, scalars, "QTT operator application total overflowed");
            if (total > total_cap) {
                throw QStateError("QTT operator application exceeds configured output total scalar cap");
            }
        }
    }

    [[nodiscard]] QTTOperatorConfig combined_config(const ExactQTTOperator& other) const noexcept {
        return QTTOperatorConfig{
            std::min(config_.max_rank, other.config_.max_rank),
            std::min(config_.max_core_scalars, other.config_.max_core_scalars),
            std::min(config_.max_total_scalars, other.config_.max_total_scalars),
            std::min(config_.max_apply_rank, other.config_.max_apply_rank),
            std::min(config_.max_apply_core_scalars, other.config_.max_apply_core_scalars),
            std::min(config_.max_apply_total_scalars, other.config_.max_apply_total_scalars)};
    }

    [[nodiscard]] static std::vector<QComplex> contract(
        std::span<const QComplex> row, const QTTOperatorCore& core,
        std::uint8_t output, std::uint8_t input) {
        if (row.size() != core.left_rank) {
            throw QStateError("QTT operator row/core rank mismatch");
        }
        const auto& block = core.blocks[output * 2U + input];
        std::vector<QComplex> next(core.right_rank);
        for (std::size_t left = 0U; left < core.left_rank; ++left) {
            for (std::size_t right = 0U; right < core.right_rank; ++right) {
                next[right] += row[left] * block[left * core.right_rank + right];
            }
        }
        return next;
    }

    void validate() {
        if (cores_.empty()) {
            throw QStateError("QTT operator requires at least one binary core");
        }
        if (config_.max_rank == 0U || config_.max_core_scalars == 0U ||
            config_.max_total_scalars == 0U || config_.max_apply_rank == 0U ||
            config_.max_apply_core_scalars == 0U || config_.max_apply_total_scalars == 0U) {
            throw QStateError("QTT operator configuration contains a zero resource cap");
        }
        std::size_t previous = 0U;
        std::size_t maximum = 1U;
        std::size_t total = 0U;
        for (std::size_t i = 0U; i < cores_.size(); ++i) {
            const QTTOperatorCore& core = cores_[i];
            if (core.left_rank == 0U || core.right_rank == 0U ||
                (i == 0U && core.left_rank != 1U) || (i != 0U && core.left_rank != previous)) {
                throw QStateError("QTT operator has invalid bond ranks");
            }
            if (core.left_rank > config_.max_rank || core.right_rank > config_.max_rank) {
                throw QStateError("QTT operator core exceeds configured rank cap");
            }
            const std::size_t matrix = checked_product(
                core.left_rank, core.right_rank, "QTT operator core shape overflowed");
            const std::size_t scalars = checked_product(
                matrix, 4U, "QTT operator core scalar count overflowed");
            if (scalars > config_.max_core_scalars) {
                throw QStateError("QTT operator core exceeds configured scalar cap");
            }
            total = checked_sum(total, scalars, "QTT operator total scalar count overflowed");
            if (total > config_.max_total_scalars) {
                throw QStateError("QTT operator exceeds configured total scalar cap");
            }
            for (const auto& block : core.blocks) {
                if (block.size() != matrix) {
                    throw QStateError("QTT operator block size does not match declared ranks");
                }
                for (const QComplex& value : block) {
                    if (!finite(value)) {
                        throw QStateError("QTT operator contains a non-finite scalar");
                    }
                }
            }
            maximum = std::max(maximum, std::max(core.left_rank, core.right_rank));
            previous = core.right_rank;
        }
        if (previous != 1U) {
            throw QStateError("QTT operator final right rank must be one");
        }
        stats_ = QTTOperatorStats{cores_.size(), maximum, total};
    }
};

}  // namespace qubit
