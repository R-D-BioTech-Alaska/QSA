#pragma once

#include "qubit/qlct_qwen35.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace qubit {

struct Qwen35RuntimeMetric {
    std::size_t elements{0U};
    double max_abs{0.0};
    double rms{0.0};
    double reference_rms{0.0};
    double nrmse{0.0};
};

struct Qwen35TokenizerOutputMetric {
    std::vector<Qwen35RuntimeMetric> contexts{};
    std::size_t top_token_matches{0U};
    std::size_t top_token_total{0U};
};

namespace qwen35_runtime_detail {

inline float half_to_float(std::uint16_t value) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
    const std::uint32_t exponent = (value >> 10U) & 0x1fU;
    const std::uint32_t mantissa = value & 0x03ffU;
    std::uint32_t bits = 0U;
    if (exponent == 0U) {
        if (mantissa == 0U) {
            bits = sign;
        } else {
            std::uint32_t m = mantissa;
            std::uint32_t shift = 0U;
            while ((m & 0x0400U) == 0U) {
                m <<= 1U;
                ++shift;
            }
            m &= 0x03ffU;
            bits = sign | ((127U - 15U - shift) << 23U) | (m << 13U);
        }
    } else if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + (127U - 15U)) << 23U) | (mantissa << 13U);
    }
    return std::bit_cast<float>(bits);
}

inline std::uint16_t float_to_half(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 16U) & 0x8000U;
    const std::uint32_t exponent = (bits >> 23U) & 0xffU;
    const std::uint32_t mantissa = bits & 0x7fffffU;
    if (exponent == 0xffU) {
        return static_cast<std::uint16_t>(sign | 0x7c00U | (mantissa == 0U ? 0U : 0x0200U));
    }
    const int adjusted = static_cast<int>(exponent) - 127 + 15;
    if (adjusted >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
    if (adjusted <= 0) {
        if (adjusted < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        std::uint32_t m = mantissa | 0x800000U;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - adjusted);
        std::uint32_t rounded = m >> shift;
        const std::uint32_t remainder = m & ((1U << shift) - 1U);
        const std::uint32_t halfway = 1U << (shift - 1U);
        if (remainder > halfway || (remainder == halfway && (rounded & 1U))) {
            ++rounded;
        }
        return static_cast<std::uint16_t>(sign | rounded);
    }
    std::uint32_t rounded = mantissa >> 13U;
    const std::uint32_t remainder = mantissa & 0x1fffU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (rounded & 1U))) {
        ++rounded;
        if (rounded == 0x0400U) {
            rounded = 0U;
            if (adjusted + 1 >= 31) {
                return static_cast<std::uint16_t>(sign | 0x7c00U);
            }
            return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(adjusted + 1) << 10U));
        }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(adjusted) << 10U) | rounded);
}

inline float load_half(const std::uint8_t* data) noexcept {
    std::uint16_t value = 0U;
    std::memcpy(&value, data, sizeof(value));
    return half_to_float(value);
}

inline float roundf_away(float value) noexcept {
    return std::copysign(std::floor(std::abs(value) + 0.5F), value);
}

inline std::size_t q6_scale_index(std::size_t group, std::size_t lane) noexcept {
    return group * 2U + (lane >= 16U ? 1U : 0U);
}

}  // namespace qwen35_runtime_detail

class Qwen35LctRuntime {
public:
    static constexpr std::size_t SsmHeadDim = 128U;
    static constexpr std::size_t SsmQkHeads = 16U;
    static constexpr std::size_t SsmValueHeads = 48U;
    static constexpr std::size_t AttentionHeadDim = 256U;
    static constexpr std::size_t AttentionQueryHeads = 24U;
    static constexpr std::size_t AttentionKvHeads = 4U;
    static constexpr std::size_t EmbeddingWidth = 5120U;
    static constexpr std::size_t Vocabulary = 248320U;
    static constexpr std::size_t QkK = 256U;
    static constexpr std::size_t Q4kBlockBytes = 144U;
    static constexpr std::size_t Q6kBlockBytes = 210U;
    static constexpr std::size_t Q4kRowBytes = 2880U;
    static constexpr std::size_t Q6kRowBytes = 4200U;

    static void recurrent(
        std::span<const float> q,
        std::span<const float> k,
        std::span<const float> v,
        std::span<const float> gate,
        std::span<const float> beta,
        std::span<const float> state_before,
        std::size_t tokens,
        std::span<float> state_after,
        std::span<float> output) {
        const std::size_t qk_size = SsmHeadDim * SsmQkHeads * tokens;
        const std::size_t value_size = SsmHeadDim * SsmValueHeads * tokens;
        const std::size_t state_size = SsmHeadDim * SsmHeadDim * SsmValueHeads;
        if (q.size() != qk_size || k.size() != qk_size || v.size() != value_size ||
            gate.size() != SsmValueHeads * tokens || beta.size() != SsmValueHeads * tokens ||
            state_before.size() != state_size || state_after.size() != state_size || output.size() != value_size) {
            throw QStateError("Qwen35 recurrent runtime shape mismatch");
        }
        std::copy(state_before.begin(), state_before.end(), state_after.begin());
        const float scale = 1.0F / std::sqrt(static_cast<float>(SsmHeadDim));
        std::vector<float> predicted(SsmHeadDim);
        std::vector<float> delta(SsmHeadDim);
        for (std::size_t token = 0U; token < tokens; ++token) {
            for (std::size_t head = 0U; head < SsmValueHeads; ++head) {
                const std::size_t qk_head = head % SsmQkHeads;
                const float decay = std::exp(gate[head + SsmValueHeads * token]);
                const float b = 1.0F / (1.0F + std::exp(-beta[head + SsmValueHeads * token]));
                const float* kt = k.data() + SsmHeadDim * (qk_head + SsmQkHeads * token);
                const float* qt = q.data() + SsmHeadDim * (qk_head + SsmQkHeads * token);
                const float* vt = v.data() + SsmHeadDim * (head + SsmValueHeads * token);
                const std::size_t state_head = SsmHeadDim * SsmHeadDim * head;
                for (std::size_t column = 0U; column < SsmHeadDim; ++column) {
                    float sum = 0.0F;
                    const std::size_t column_base = state_head + SsmHeadDim * column;
                    for (std::size_t row = 0U; row < SsmHeadDim; ++row) {
                        sum += state_after[column_base + row] * kt[row];
                    }
                    predicted[column] = sum;
                    delta[column] = (vt[column] - decay * sum) * b;
                }
                for (std::size_t column = 0U; column < SsmHeadDim; ++column) {
                    const std::size_t column_base = state_head + SsmHeadDim * column;
                    for (std::size_t row = 0U; row < SsmHeadDim; ++row) {
                        state_after[column_base + row] = decay * state_after[column_base + row] + kt[row] * delta[column];
                    }
                }
                float* out = output.data() + SsmHeadDim * (head + SsmValueHeads * token);
                for (std::size_t column = 0U; column < SsmHeadDim; ++column) {
                    float sum = 0.0F;
                    const std::size_t column_base = state_head + SsmHeadDim * column;
                    for (std::size_t row = 0U; row < SsmHeadDim; ++row) {
                        sum += state_after[column_base + row] * qt[row];
                    }
                    out[column] = sum * scale;
                }
            }
        }
    }

    static void causal_grouped_attention(
        std::span<const float> q,
        std::span<const float> k,
        std::span<const float> v,
        std::size_t tokens,
        std::span<float> output) {
        const std::size_t q_size = AttentionHeadDim * AttentionQueryHeads * tokens;
        const std::size_t kv_size = AttentionHeadDim * AttentionKvHeads * tokens;
        if (q.size() != q_size || k.size() != kv_size || v.size() != kv_size || output.size() != q_size) {
            throw QStateError("Qwen35 grouped-attention runtime shape mismatch");
        }
        const float scale = 1.0F / std::sqrt(static_cast<float>(AttentionHeadDim));
        const std::size_t group = AttentionQueryHeads / AttentionKvHeads;
        std::vector<float> scores(tokens);
        std::vector<float> weights(tokens);
        for (std::size_t token = 0U; token < tokens; ++token) {
            for (std::size_t q_head = 0U; q_head < AttentionQueryHeads; ++q_head) {
                const std::size_t kv_head = q_head / group;
                const float* qt = q.data() + AttentionHeadDim * (q_head + AttentionQueryHeads * token);
                float max_score = -std::numeric_limits<float>::infinity();
                for (std::size_t key_token = 0U; key_token <= token; ++key_token) {
                    const float* kt = k.data() + AttentionHeadDim * (kv_head + AttentionKvHeads * key_token);
                    float score = 0.0F;
                    for (std::size_t i = 0U; i < AttentionHeadDim; ++i) {
                        score += qt[i] * kt[i];
                    }
                    score *= scale;
                    scores[key_token] = score;
                    max_score = std::max(max_score, score);
                }
                float denominator = 0.0F;
                for (std::size_t key_token = 0U; key_token <= token; ++key_token) {
                    const float value = std::exp(scores[key_token] - max_score);
                    weights[key_token] = value;
                    denominator += value;
                }
                float* out = output.data() + AttentionHeadDim * (q_head + AttentionQueryHeads * token);
                std::fill(out, out + AttentionHeadDim, 0.0F);
                for (std::size_t key_token = 0U; key_token <= token; ++key_token) {
                    const float weight = weights[key_token] / denominator;
                    const float* vt = v.data() + AttentionHeadDim * (kv_head + AttentionKvHeads * key_token);
                    for (std::size_t i = 0U; i < AttentionHeadDim; ++i) {
                        out[i] += vt[i] * weight;
                    }
                }
            }
        }
    }

    static void add(std::span<const float> first, std::span<const float> second, std::span<float> output) {
        if (first.size() != second.size() || first.size() != output.size()) {
            throw QStateError("Qwen35 residual-add runtime shape mismatch");
        }
        for (std::size_t i = 0U; i < output.size(); ++i) output[i] = first[i] + second[i];
    }

    static void multiply(std::span<const float> first, std::span<const float> second, std::span<float> output) {
        if (first.size() != second.size() || first.size() != output.size()) {
            throw QStateError("Qwen35 elementwise runtime shape mismatch");
        }
        for (std::size_t i = 0U; i < output.size(); ++i) output[i] = first[i] * second[i];
    }

    static Qwen35RuntimeMetric metric(std::span<const float> actual, std::span<const float> expected) {
        if (actual.size() != expected.size()) throw QStateError("Qwen35 runtime metric shape mismatch");
        double sum_sq = 0.0;
        double ref_sq = 0.0;
        double max_abs = 0.0;
        for (std::size_t i = 0U; i < actual.size(); ++i) {
            const double a = static_cast<double>(actual[i]);
            const double e = static_cast<double>(expected[i]);
            const double d = a - e;
            sum_sq += d * d;
            ref_sq += e * e;
            max_abs = std::max(max_abs, std::abs(d));
        }
        Qwen35RuntimeMetric out;
        out.elements = actual.size();
        out.max_abs = max_abs;
        out.rms = std::sqrt(sum_sq / static_cast<double>(std::max<std::size_t>(actual.size(), 1U)));
        out.reference_rms = std::sqrt(ref_sq / static_cast<double>(std::max<std::size_t>(actual.size(), 1U)));
        out.nrmse = out.rms / std::max(out.reference_rms, 1.0e-30);
        return out;
    }

    static void decode_q4_k_row(std::span<const std::uint8_t> raw, std::span<float> output) {
        if (raw.size() != Q4kRowBytes || output.size() != EmbeddingWidth) {
            throw QStateError("Qwen35 Q4_K embedding row contract changed");
        }
        std::size_t source = 0U;
        std::size_t target = 0U;
        for (std::size_t block_index = 0U; block_index < EmbeddingWidth / QkK; ++block_index) {
            const std::uint8_t* block = raw.data() + source;
            const float d = qwen35_runtime_detail::load_half(block);
            const float dmin = qwen35_runtime_detail::load_half(block + 2U);
            const std::uint8_t* scales = block + 4U;
            const std::uint8_t* qs = block + 16U;
            for (std::size_t group64 = 0U; group64 < 4U; ++group64) {
                const auto scale_min = [&](std::size_t index) {
                    if (index < 4U) {
                        return std::pair<unsigned, unsigned>{scales[index] & 0x3fU, scales[index + 4U] & 0x3fU};
                    }
                    const unsigned ds = (scales[index + 4U] & 0x0fU) | ((scales[index - 4U] & 0xc0U) >> 2U);
                    const unsigned ms = ((scales[index + 4U] >> 4U) & 0x0fU) | ((scales[index] & 0xc0U) >> 2U);
                    return std::pair<unsigned, unsigned>{ds, ms};
                };
                const auto low = scale_min(2U * group64);
                const auto high = scale_min(2U * group64 + 1U);
                const float low_d = d * static_cast<float>(low.first);
                const float low_m = dmin * static_cast<float>(low.second);
                const float high_d = d * static_cast<float>(high.first);
                const float high_m = dmin * static_cast<float>(high.second);
                const std::uint8_t* packed = qs + 32U * group64;
                for (std::size_t lane = 0U; lane < 32U; ++lane) {
                    output[target + lane] = low_d * static_cast<float>(packed[lane] & 0x0fU) - low_m;
                    output[target + 32U + lane] = high_d * static_cast<float>(packed[lane] >> 4U) - high_m;
                }
                target += 64U;
            }
            source += Q4kBlockBytes;
        }
    }

    static void q8_1_cuda_runtime(std::span<const float> input, std::span<float> output) {
        if (input.size() != output.size() || input.size() % 32U != 0U) {
            throw QStateError("Qwen35 CUDA Q8_1 runtime shape mismatch");
        }
        for (std::size_t base = 0U; base < input.size(); base += 32U) {
            float amax = 0.0F;
            for (std::size_t lane = 0U; lane < 32U; ++lane) amax = std::max(amax, std::abs(input[base + lane]));
            if (amax == 0.0F) {
                std::fill(output.begin() + static_cast<std::ptrdiff_t>(base), output.begin() + static_cast<std::ptrdiff_t>(base + 32U), 0.0F);
                continue;
            }
            const float d = amax / 127.0F;
            const std::uint16_t half = qwen35_runtime_detail::float_to_half(d);
            const float runtime_d = qwen35_runtime_detail::half_to_float(half);
            for (std::size_t lane = 0U; lane < 32U; ++lane) {
                const float scaled = input[base + lane] / d;
                const float rounded = qwen35_runtime_detail::roundf_away(scaled);
                const int q = std::clamp(static_cast<int>(rounded), -128, 127);
                output[base + lane] = static_cast<float>(q) * runtime_d;
            }
        }
    }

    static void decode_q6_k_block(const std::uint8_t* block, std::span<float, QkK> output) {
        const std::uint8_t* ql = block;
        const std::uint8_t* qh = block + 128U;
        const auto* scales = reinterpret_cast<const std::int8_t*>(block + 192U);
        const float d = qwen35_runtime_detail::load_half(block + 208U);
        for (std::size_t half = 0U; half < 2U; ++half) {
            const std::uint8_t* ql0 = ql + half * 64U;
            const std::uint8_t* ql1 = ql0 + 32U;
            const std::uint8_t* qh0 = qh + half * 32U;
            const std::int8_t* local = scales + half * 8U;
            const std::size_t base = half * 128U;
            for (std::size_t lane = 0U; lane < 32U; ++lane) {
                const std::size_t pair = lane >= 16U ? 1U : 0U;
                const int q1 = static_cast<int>((ql0[lane] & 0x0fU) | (((qh0[lane] >> 0U) & 0x03U) << 4U)) - 32;
                const int q2 = static_cast<int>((ql1[lane] & 0x0fU) | (((qh0[lane] >> 2U) & 0x03U) << 4U)) - 32;
                const int q3 = static_cast<int>((ql0[lane] >> 4U) | (((qh0[lane] >> 4U) & 0x03U) << 4U)) - 32;
                const int q4 = static_cast<int>((ql1[lane] >> 4U) | (((qh0[lane] >> 6U) & 0x03U) << 4U)) - 32;
                output[base + lane] = d * static_cast<float>(local[pair + 0U]) * static_cast<float>(q1);
                output[base + 32U + lane] = d * static_cast<float>(local[pair + 2U]) * static_cast<float>(q2);
                output[base + 64U + lane] = d * static_cast<float>(local[pair + 4U]) * static_cast<float>(q3);
                output[base + 96U + lane] = d * static_cast<float>(local[pair + 6U]) * static_cast<float>(q4);
            }
        }
    }

    static void q6_k_output(
        std::span<const std::uint8_t> weights,
        std::span<const float> hidden_context_major,
        std::size_t context_count,
        std::span<float> logits_context_major,
        std::size_t worker_count = 0U) {
        if (weights.size() != Vocabulary * Q6kRowBytes ||
            hidden_context_major.size() != context_count * EmbeddingWidth ||
            logits_context_major.size() != context_count * Vocabulary || context_count == 0U) {
            throw QStateError("Qwen35 Q6_K output runtime shape mismatch");
        }
        std::vector<float> hidden_runtime(hidden_context_major.size());
        for (std::size_t context = 0U; context < context_count; ++context) {
            q8_1_cuda_runtime(
                hidden_context_major.subspan(context * EmbeddingWidth, EmbeddingWidth),
                std::span<float>(hidden_runtime).subspan(context * EmbeddingWidth, EmbeddingWidth));
        }
        std::vector<float> transposed(EmbeddingWidth * context_count);
        for (std::size_t i = 0U; i < EmbeddingWidth; ++i) {
            for (std::size_t context = 0U; context < context_count; ++context) {
                transposed[i * context_count + context] = hidden_runtime[context * EmbeddingWidth + i];
            }
        }

        std::size_t workers = worker_count == 0U ? static_cast<std::size_t>(std::thread::hardware_concurrency()) : worker_count;
        workers = std::clamp<std::size_t>(workers, 1U, std::min<std::size_t>(Vocabulary, 32U));
        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (std::size_t worker = 0U; worker < workers; ++worker) {
            const std::size_t row_begin = Vocabulary * worker / workers;
            const std::size_t row_end = Vocabulary * (worker + 1U) / workers;
            threads.emplace_back([&, row_begin, row_end]() {
                std::vector<float> accum(context_count);
                std::array<float, QkK> decoded{};
                for (std::size_t row = row_begin; row < row_end; ++row) {
                    std::fill(accum.begin(), accum.end(), 0.0F);
                    const std::uint8_t* row_data = weights.data() + row * Q6kRowBytes;
                    for (std::size_t block = 0U; block < EmbeddingWidth / QkK; ++block) {
                        decode_q6_k_block(row_data + block * Q6kBlockBytes, decoded);
                        const std::size_t dim_base = block * QkK;
                        for (std::size_t lane = 0U; lane < QkK; ++lane) {
                            const float weight = decoded[lane];
                            const float* hidden = transposed.data() + (dim_base + lane) * context_count;
                            for (std::size_t context = 0U; context < context_count; ++context) {
                                accum[context] += weight * hidden[context];
                            }
                        }
                    }
                    for (std::size_t context = 0U; context < context_count; ++context) {
                        logits_context_major[context * Vocabulary + row] = accum[context];
                    }
                }
            });
        }
        for (auto& thread : threads) thread.join();
    }
};

}  // namespace qubit
