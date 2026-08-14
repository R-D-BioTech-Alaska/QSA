#include "qubit/qlct_qwen35_runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void recurrent_case() {
    constexpr std::size_t tokens = 1U;
    std::vector<float> q(qubit::Qwen35LctRuntime::SsmHeadDim * qubit::Qwen35LctRuntime::SsmQkHeads * tokens, 0.0F);
    std::vector<float> k(q.size(), 0.0F);
    std::vector<float> v(qubit::Qwen35LctRuntime::SsmHeadDim * qubit::Qwen35LctRuntime::SsmValueHeads * tokens, 0.0F);
    std::vector<float> gate(qubit::Qwen35LctRuntime::SsmValueHeads * tokens, 0.0F);
    std::vector<float> beta(gate.size(), 0.0F);
    std::vector<float> state(qubit::Qwen35LctRuntime::SsmHeadDim * qubit::Qwen35LctRuntime::SsmHeadDim * qubit::Qwen35LctRuntime::SsmValueHeads, 0.0F);
    std::vector<float> next(state.size(), 0.0F);
    std::vector<float> output(v.size(), 0.0F);
    q[0] = 1.0F;
    k[0] = 1.0F;
    v[0] = 1.0F;
    qubit::Qwen35LctRuntime::recurrent(q, k, v, gate, beta, state, tokens, next, output);
    require(std::abs(next[0] - 0.5F) < 1e-7F, "Qwen35 recurrent state update changed");
    const float expected = 0.5F / std::sqrt(128.0F);
    require(std::abs(output[0] - expected) < 1e-7F, "Qwen35 recurrent output changed");
}

void attention_case() {
    constexpr std::size_t tokens = 1U;
    std::vector<float> q(qubit::Qwen35LctRuntime::AttentionHeadDim * qubit::Qwen35LctRuntime::AttentionQueryHeads * tokens, 0.0F);
    std::vector<float> k(qubit::Qwen35LctRuntime::AttentionHeadDim * qubit::Qwen35LctRuntime::AttentionKvHeads * tokens, 0.0F);
    std::vector<float> v(k.size(), 0.0F);
    std::vector<float> output(q.size(), 0.0F);
    for (std::size_t kv = 0U; kv < qubit::Qwen35LctRuntime::AttentionKvHeads; ++kv) {
        v[qubit::Qwen35LctRuntime::AttentionHeadDim * kv] = static_cast<float>(kv + 1U);
    }
    qubit::Qwen35LctRuntime::causal_grouped_attention(q, k, v, tokens, output);
    for (std::size_t head = 0U; head < qubit::Qwen35LctRuntime::AttentionQueryHeads; ++head) {
        const float expected = static_cast<float>(head / 6U + 1U);
        require(std::abs(output[qubit::Qwen35LctRuntime::AttentionHeadDim * head] - expected) < 1e-7F,
                "Qwen35 grouped-query head mapping changed");
    }
}

void q8_case() {
    std::array<float, 32> input{};
    std::array<float, 32> output{};
    for (std::size_t i = 0U; i < input.size(); ++i) input[i] = static_cast<float>(static_cast<int>(i) - 16) / 7.0F;
    qubit::Qwen35LctRuntime::q8_1_cuda_runtime(input, output);
    require(output[16] == 0.0F, "Qwen35 Q8_1 zero changed");
    require(std::isfinite(output.front()) && std::isfinite(output.back()), "Qwen35 Q8_1 produced nonfinite output");
}

void q6_case() {
    std::array<std::uint8_t, qubit::Qwen35LctRuntime::Q6kBlockBytes> block{};
    block[192] = 1U;
    block[193] = 1U;
    block[194] = 1U;
    block[195] = 1U;
    block[196] = 1U;
    block[197] = 1U;
    block[198] = 1U;
    block[199] = 1U;
    block[200] = 1U;
    block[201] = 1U;
    block[202] = 1U;
    block[203] = 1U;
    block[204] = 1U;
    block[205] = 1U;
    block[206] = 1U;
    block[207] = 1U;
    block[208] = 0x00U;
    block[209] = 0x3cU;
    std::array<float, qubit::Qwen35LctRuntime::QkK> decoded{};
    qubit::Qwen35LctRuntime::decode_q6_k_block(block.data(), decoded);
    require(std::all_of(decoded.begin(), decoded.end(), [](float value) { return value == -32.0F; }),
            "Qwen35 Q6_K signed minimum changed");
    std::fill(block.begin(), block.begin() + 192, 0xffU);
    qubit::Qwen35LctRuntime::decode_q6_k_block(block.data(), decoded);
    require(std::all_of(decoded.begin(), decoded.end(), [](float value) { return value == 31.0F; }),
            "Qwen35 Q6_K signed maximum changed");
}

void q4_case() {
    std::vector<std::uint8_t> row(qubit::Qwen35LctRuntime::Q4kRowBytes, 0U);
    for (std::size_t block = 0U; block < qubit::Qwen35LctRuntime::EmbeddingWidth / qubit::Qwen35LctRuntime::QkK; ++block) {
        std::uint8_t* data = row.data() + block * qubit::Qwen35LctRuntime::Q4kBlockBytes;
        data[0] = 0x00U;
        data[1] = 0x3cU;
        data[2] = 0x00U;
        data[3] = 0x00U;
        data[4] = data[5] = data[6] = data[7] = 1U;
        data[12] = data[13] = data[14] = data[15] = 1U;
        std::fill(data + 16U, data + qubit::Qwen35LctRuntime::Q4kBlockBytes, 0xffU);
    }
    std::vector<float> decoded(qubit::Qwen35LctRuntime::EmbeddingWidth);
    qubit::Qwen35LctRuntime::decode_q4_k_row(row, decoded);
    require(std::all_of(decoded.begin(), decoded.end(), [](float value) { return value == 15.0F; }),
            "Qwen35 Q4_K row decoder changed");
}

}  // namespace

int main() {
    recurrent_case();
    attention_case();
    q8_case();
    q6_case();
    q4_case();
    return 0;
}
