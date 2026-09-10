#pragma once

#include "qubit/qcompiler.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace qubit {

struct ExactCompiledMarginalProgramConfig {
    ExactCausalIndexConfig index{};
    ExactMarginalCompilerConfig marginal{};
    std::size_t max_cached_plans{1024U};
    std::size_t max_cached_bytes{1U << 30U};
};

struct ExactCompiledMarginalProgramStats {
    std::size_t qubits{0U};
    std::size_t operations{0U};
    std::size_t index_estimated_bytes{0U};
    std::size_t cached_plans{0U};
    std::size_t cached_bytes{0U};
    std::size_t compile_count{0U};
    std::size_t cache_hits{0U};
    std::size_t cache_misses{0U};
};

class ExactCompiledMarginalProgram {
public:
    ExactCompiledMarginalProgram(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        ExactCompiledMarginalProgramConfig config = {})
        : config_(config), index_(qubit_count, operations, config_.index) {
        if (config_.max_cached_plans == 0U || config_.max_cached_bytes == 0U) {
            throw QStateError("Compiled marginal program cache caps must be positive");
        }
        stats_.qubits = qubit_count;
        stats_.operations = operations.size();
        stats_.index_estimated_bytes = index_.stats().estimated_bytes;
    }

    [[nodiscard]] const ExactIndexedMarginalCompilerPlan& prepare(
        std::span<const QubitId> query_qubits) {
        for (CacheEntry& entry : cache_) {
            if (equal_query(entry.query, query_qubits)) {
                ++stats_.cache_hits;
                return *entry.plan;
            }
        }

        if (cache_.size() >= config_.max_cached_plans) {
            throw QStateError("Compiled marginal program reached its cached-plan cap");
        }

        auto plan = std::make_unique<ExactIndexedMarginalCompilerPlan>(
            index_, query_qubits, config_.marginal);
        std::vector<QubitId> query(query_qubits.begin(), query_qubits.end());
        std::size_t entry_bytes = sizeof(CacheEntry);
        entry_bytes = checked_sum(
            entry_bytes,
            checked_product(query.capacity(), sizeof(QubitId),
                "Compiled marginal program query cache storage overflowed"),
            "Compiled marginal program cache storage overflowed");
        entry_bytes = checked_sum(
            entry_bytes,
            plan->stats().estimated_bytes,
            "Compiled marginal program cache storage overflowed");
        const std::size_t next_cached_bytes = checked_sum(
            stats_.cached_bytes,
            entry_bytes,
            "Compiled marginal program cache storage overflowed");
        if (next_cached_bytes > config_.max_cached_bytes) {
            throw QStateError("Compiled marginal program reached its cached-byte cap");
        }

        cache_.push_back(CacheEntry{std::move(query), std::move(plan), entry_bytes});
        stats_.cached_plans = cache_.size();
        stats_.cached_bytes = next_cached_bytes;
        ++stats_.compile_count;
        ++stats_.cache_misses;
        return *cache_.back().plan;
    }

    [[nodiscard]] double probability(
        std::span<const QubitId> query_qubits,
        std::span<const std::uint8_t> bits) {
        return prepare(query_qubits).probability(bits);
    }

    void clear_cache() noexcept {
        cache_.clear();
        stats_.cached_plans = 0U;
        stats_.cached_bytes = 0U;
    }

    [[nodiscard]] const ExactCompiledMarginalProgramStats& stats() const noexcept {
        return stats_;
    }
    [[nodiscard]] const ExactCausalOperationIndex& causal_index() const noexcept {
        return index_;
    }
    [[nodiscard]] const ExactCompiledMarginalProgramConfig& config() const noexcept {
        return config_;
    }

private:
    struct CacheEntry {
        std::vector<QubitId> query{};
        std::unique_ptr<ExactIndexedMarginalCompilerPlan> plan{};
        std::size_t estimated_bytes{0U};
    };

    ExactCompiledMarginalProgramConfig config_{};
    ExactCausalOperationIndex index_;
    std::vector<CacheEntry> cache_{};
    ExactCompiledMarginalProgramStats stats_{};

    [[nodiscard]] static bool equal_query(
        const std::vector<QubitId>& cached,
        std::span<const QubitId> query) noexcept {
        if (cached.size() != query.size()) {
            return false;
        }
        for (std::size_t index = 0U; index < cached.size(); ++index) {
            if (cached[index] != query[index]) {
                return false;
            }
        }
        return true;
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
};

}  // namespace qubit
