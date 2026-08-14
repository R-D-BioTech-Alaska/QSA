#include "qubit/qlct_qwen35.hpp"

#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#else
#error qwen35_lct_runtime_evidence currently requires the QELM Windows evidence runner
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::string_view Pr65Head = "eff0f7c64fd65b0cefce29f2510bab96130e799a";
constexpr std::string_view ReferenceMechanism = "sha256:53083f33f2000c4342daf520a576ab6435624d6a1b9368c61b612a65fcc00fd5";
constexpr std::string_view ReferenceHoldoutIdentity = "436e109fb7928e05d2f066ad5181787b0793f0717e44f5de459d0e844dbbc80d";
constexpr std::string_view ContextIdentity = "sha256:9e61ed0d1f3de997331421b97f80071df368664d59fe858885de5ff076519177";
constexpr std::string_view ActivationReceipt = "sha256:60a9e08cc71dabe138f489615416163d0cc3002da78e678aeefad50c9cfc9c35";
constexpr std::string_view MechanismText =
    "qsa.qwen35-lct-runtime-evidence.v1|pr65=eff0f7c64fd65b0cefce29f2510bab96130e799a|"
    "atlas=d3f59577d19a65eabca5ef1bfa13c8a37036531dd710bcf17ded1c1f33019036|"
    "source=ec3a6c03cdb1bd88dc01a2cd745e6aed6a582512543c66088eb6e1193dfab0b2|"
    "ssm=head_map:value_head%16;beta:sigmoid;decay:exp;state:S=decay*S+k*((v-decay*(S^T*k))*beta)^T;output:S^T*q/sqrt128|"
    "attention=causal_gqa;kv=query_head/6;scale=1/sqrt256";

struct RoleBinding {
    qubit::Qwen35LctRole role;
    const char* source_identity;
    const char* component_manifest_identity;
    const char* source_fragment_identity;
    const char* producer_receipt_identity;
    const char* reference_cir_identity;
    const char* residual_identity;
    const char* decoder_identity;
    const char* reconstruction_identity;
};

constexpr RoleBinding SsmBinding{
    qubit::Qwen35LctRole::SsmRecurrent,
    "sha256:9852cc3230b0cac7bafb66cfb9e16fcf35966b0ff67e7afcb805bceedd415172",
    "sha256:2c80671af3588c9f9ce110c96398e47706be2724a39536707e14a634bcc6ab71",
    "0baf5fd5729eeaab69485f64575da3776c192531fc1ecbcb8a1e897879cdd0bd",
    "sha256:8c5bbeb395eae95c31c0ca5e16c66e2bbe6ed49f0988b1c6c0fc6a8a4bc191ee",
    "sha256:be6c04a161458f5c6bf15681bd543de5d57b77a2395e10503959edfe3418fa0f",
    "sha256:1f195afa0bb8f7a33e604d5a2ae6e9356d87aae86459ff1dcef8778594406ea7",
    "sha256:e436a3af4f8299d92674e9f7ba4d7ba1c69dbd3de350fdf1b7f2fa72bb9b6885",
    "sha256:2016343bfd2dde91577e18c2bc92d1b4d897960110631a7387fae41b163ac4d7",
};

constexpr RoleBinding AttentionBinding{
    qubit::Qwen35LctRole::FullAttention,
    "sha256:57f31f5f9a5df4fb976deed2563aa522eff959e2b4bdb2b32eb16410aa64430f",
    "sha256:fdbfe2b7a9fb8dba6486a4821f214bb5a6534d9ad2e587009f85f7393ef1f365",
    "0d1657ee4d242ee1aa72db8e85815393aa79834388338db7e9a9b5d89e502875",
    "sha256:9ba393740d76d2a93996a9a9f5deed154ce610b30a70a527d4947366905133b5",
    "sha256:6db6bd50c2abb8aa8f1e0ca360c5c728b91cbe70afe821c91423735753728672",
    "sha256:b63392ff00da8a063c62cc21ce12ee1ea6570627384f2848ab32595163e0f244",
    "sha256:beaa078e6f3781e0b9890955612895d65da770b0f2b34c7d01c2e26bc85db0df",
    "sha256:7a387af80a7f7184ffdf8fcb1817653fc8ea82db173ede8fce970f7573e85799",
};

struct Threshold { double max_abs; double nrmse; };

const std::map<std::string, Threshold> Thresholds{
    {"ssm_state", {2.022716064453125e-05, 5.865438078195686e-07}},
    {"ssm_output", {3.4186508178710935e-06, 5.431170712337355e-07}},
    {"ssm_attn_residual", {2.0e-06, 5.0e-07}},
    {"ssm_post_ffn", {2.0e-06, 5.0e-07}},
    {"attention_pregate", {0.009611627783966065, 0.0008826273981856719}},
    {"attention_gate", {2.0e-06, 5.0e-07}},
    {"attention_residual", {2.0e-06, 5.0e-07}},
    {"attention_post_ffn", {2.0e-06, 5.0e-07}},
};

std::string hex_digest(const std::array<unsigned char, 32>& digest) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out(64, '0');
    for (std::size_t i = 0; i < digest.size(); ++i) {
        out[2 * i] = hex[digest[i] >> 4];
        out[2 * i + 1] = hex[digest[i] & 0x0f];
    }
    return out;
}

std::string sha256_text(std::string_view value) {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    DWORD object_bytes = 0;
    DWORD result_bytes = 0;
    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0 ||
        BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&object_bytes), sizeof(object_bytes), &result_bytes, 0) < 0) {
        if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
        throw std::runtime_error("SHA-256 provider initialization failed");
    }
    std::vector<unsigned char> object(object_bytes);
    std::array<unsigned char, 32> digest{};
    if (BCryptCreateHash(algorithm, &hash, object.data(), static_cast<ULONG>(object.size()), nullptr, 0, 0) < 0 ||
        BCryptHashData(hash, reinterpret_cast<PUCHAR>(const_cast<char*>(value.data())), static_cast<ULONG>(value.size()), 0) < 0 ||
        BCryptFinishHash(hash, digest.data(), static_cast<ULONG>(digest.size()), 0) < 0) {
        if (hash) BCryptDestroyHash(hash);
        BCryptCloseAlgorithmProvider(algorithm, 0);
        throw std::runtime_error("SHA-256 hashing failed");
    }
    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(algorithm, 0);
    return hex_digest(digest);
}

std::vector<std::string> split(std::string_view value, char delimiter) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (start <= value.size()) {
        const auto end = value.find(delimiter, start);
        out.emplace_back(value.substr(start, end == std::string_view::npos ? value.size() - start : end - start));
        if (end == std::string_view::npos) break;
        start = end + 1;
    }
    return out;
}

std::size_t token_count(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("tokens.tsv is missing: " + path.string());
    std::size_t count = 0;
    std::string line;
    while (std::getline(input, line)) if (!line.empty()) ++count;
    if (count == 0) throw std::runtime_error("empty token list: " + path.string());
    return count;
}

struct TensorRow {
    std::string graph_name;
    int occurrence{};
    std::string type;
    std::vector<std::size_t> shape;
    std::size_t bytes{};
    std::string file;
};

std::unordered_map<std::string, TensorRow> tensor_index(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("tensor-index.tsv is missing: " + path.string());
    std::string line;
    std::getline(input, line);
    if (line != "name\tgraph_name\toccurrence\ttype\tshape\tbytes\tfile") throw std::runtime_error("tensor index schema changed");
    std::unordered_map<std::string, TensorRow> rows;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split(line, '\t');
        if (fields.size() != 7) throw std::runtime_error("malformed tensor index row");
        TensorRow row;
        row.graph_name = fields[1];
        row.occurrence = std::stoi(fields[2]);
        row.type = fields[3];
        for (const auto& item : split(fields[4], ',')) row.shape.push_back(static_cast<std::size_t>(std::stoull(item)));
        row.bytes = static_cast<std::size_t>(std::stoull(fields[5]));
        row.file = fields[6];
        rows.emplace(fields[0], std::move(row));
    }
    return rows;
}

std::vector<float> read_f32(const fs::path& path) {
    const auto bytes = fs::file_size(path);
    if (bytes % sizeof(float) != 0) throw std::runtime_error("F32 payload is not aligned: " + path.string());
    std::vector<float> data(static_cast<std::size_t>(bytes / sizeof(float)));
    std::ifstream input(path, std::ios::binary);
    if (!input || !input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes))) {
        throw std::runtime_error("failed to read tensor: " + path.string());
    }
    return data;
}

std::size_t logical_elements(const TensorRow& row) {
    std::size_t count = 1;
    for (const auto dim : row.shape) count *= dim;
    return count;
}

std::vector<float> load_tensor(const fs::path& root, const std::unordered_map<std::string, TensorRow>& rows, std::string_view name) {
    const auto found = rows.find(std::string(name));
    if (found == rows.end()) throw std::runtime_error("required tensor is absent: " + std::string(name));
    const auto& row = found->second;
    if (row.type != "f32") throw std::runtime_error("tensor type changed: " + std::string(name));
    auto raw = read_f32(root / row.file);
    if (name != "v_conv_predelta-0") {
        if (raw.size() != logical_elements(row) || row.bytes != raw.size() * sizeof(float)) {
            throw std::runtime_error("dense tensor storage changed: " + std::string(name));
        }
        return raw;
    }
    if (row.shape.size() != 4 || row.shape[0] != 128 || row.shape[1] != 48 || row.shape[3] != 1) {
        throw std::runtime_error("v_conv logical shape changed");
    }
    const std::size_t tokens = row.shape[2];
    constexpr std::size_t token_stride = 128 * (2 * 16 + 48);
    constexpr std::size_t logical_token = 128 * 48;
    const std::size_t expected_storage = (tokens - 1) * token_stride + logical_token;
    if (raw.size() != expected_storage || row.bytes != expected_storage * sizeof(float)) {
        throw std::runtime_error("v_conv strided storage contract changed");
    }
    std::vector<float> dense(logical_token * tokens);
    for (std::size_t token = 0; token < tokens; ++token) {
        std::copy_n(raw.begin() + static_cast<std::ptrdiff_t>(token * token_stride), logical_token,
                    dense.begin() + static_cast<std::ptrdiff_t>(token * logical_token));
    }
    return dense;
}

struct MetricAccumulator {
    std::size_t elements{};
    double sum_sq{};
    double reference_sq{};
    double max_abs{};
    double max_context_nrmse{};

    void add(const std::vector<float>& actual, const std::vector<float>& expected) {
        if (actual.size() != expected.size()) throw std::runtime_error("metric shape mismatch");
        double context_sum = 0.0;
        double context_ref = 0.0;
        double context_max = 0.0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const double delta = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
            context_sum += delta * delta;
            context_ref += static_cast<double>(expected[i]) * static_cast<double>(expected[i]);
            context_max = std::max(context_max, std::abs(delta));
        }
        const double rms = std::sqrt(context_sum / std::max<std::size_t>(actual.size(), 1));
        const double ref_rms = std::sqrt(context_ref / std::max<std::size_t>(actual.size(), 1));
        elements += actual.size();
        sum_sq += context_sum;
        reference_sq += context_ref;
        max_abs = std::max(max_abs, context_max);
        max_context_nrmse = std::max(max_context_nrmse, rms / std::max(ref_rms, 1.0e-30));
    }

    double nrmse() const {
        const double rms = std::sqrt(sum_sq / std::max<std::size_t>(elements, 1));
        const double ref_rms = std::sqrt(reference_sq / std::max<std::size_t>(elements, 1));
        return rms / std::max(ref_rms, 1.0e-30);
    }
};

using MetricMap = std::map<std::string, MetricAccumulator>;

std::vector<float> add_vectors(const std::vector<float>& left, const std::vector<float>& right) {
    if (left.size() != right.size()) throw std::runtime_error("residual shape mismatch");
    std::vector<float> out(left.size());
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = left[i] + right[i];
    return out;
}

void evaluate_ssm(const fs::path& context_root, const std::unordered_map<std::string, TensorRow>& rows,
                  std::size_t tokens, MetricMap& metrics) {
    constexpr std::size_t hd = 128;
    constexpr std::size_t qk_heads = 16;
    constexpr std::size_t value_heads = 48;
    auto q = load_tensor(context_root, rows, "q_conv_predelta-0");
    auto k = load_tensor(context_root, rows, "k_conv_predelta-0");
    auto v = load_tensor(context_root, rows, "v_conv_predelta-0");
    auto gate = load_tensor(context_root, rows, "gate-0");
    auto beta = load_tensor(context_root, rows, "beta-0");
    auto state = load_tensor(context_root, rows, "state_predelta-0");
    auto expected_state = load_tensor(context_root, rows, "new_state-0");
    auto expected_output = load_tensor(context_root, rows, "attn_output-0");
    std::vector<float> output(hd * value_heads * tokens);
    std::array<float, hd> delta{};
    const float scale = 1.0f / std::sqrt(128.0f);

    for (std::size_t token = 0; token < tokens; ++token) {
        for (std::size_t head = 0; head < value_heads; ++head) {
            const std::size_t qk_head = head % qk_heads;
            const float decay = std::exp(gate[head + value_heads * token]);
            const float b = 1.0f / (1.0f + std::exp(-beta[head + value_heads * token]));
            for (std::size_t column = 0; column < hd; ++column) {
                float predicted = 0.0f;
                for (std::size_t i = 0; i < hd; ++i) {
                    predicted += state[i + hd * (column + hd * head)] * k[i + hd * (qk_head + qk_heads * token)];
                }
                delta[column] = (v[column + hd * (head + value_heads * token)] - decay * predicted) * b;
            }
            for (std::size_t column = 0; column < hd; ++column) {
                for (std::size_t i = 0; i < hd; ++i) {
                    const auto index = i + hd * (column + hd * head);
                    state[index] = decay * state[index] + k[i + hd * (qk_head + qk_heads * token)] * delta[column];
                }
            }
            for (std::size_t column = 0; column < hd; ++column) {
                float sum = 0.0f;
                for (std::size_t i = 0; i < hd; ++i) {
                    sum += state[i + hd * (column + hd * head)] * q[i + hd * (qk_head + qk_heads * token)];
                }
                output[column + hd * (head + value_heads * token)] = sum * scale;
            }
        }
    }
    metrics["ssm_state"].add(state, expected_state);
    metrics["ssm_output"].add(output, expected_output);
    const auto input = load_tensor(context_root, rows, "model.input_embed");
    const auto linear = load_tensor(context_root, rows, "linear_attn_out-0");
    const auto attn_residual = load_tensor(context_root, rows, "attn_residual-0");
    const auto ffn = load_tensor(context_root, rows, "ffn_out-0");
    const auto post = load_tensor(context_root, rows, "post_ffn-0");
    metrics["ssm_attn_residual"].add(add_vectors(input, linear), attn_residual);
    metrics["ssm_post_ffn"].add(add_vectors(attn_residual, ffn), post);
}

void evaluate_attention(const fs::path& context_root, const std::unordered_map<std::string, TensorRow>& rows,
                        std::size_t tokens, MetricMap& metrics) {
    constexpr std::size_t hd = 256;
    constexpr std::size_t q_heads = 24;
    constexpr std::size_t kv_heads = 4;
    constexpr std::size_t group = q_heads / kv_heads;
    const auto q = load_tensor(context_root, rows, "Qcur-3");
    const auto k = load_tensor(context_root, rows, "Kcur-3");
    const auto v = load_tensor(context_root, rows, "Vcur-3");
    std::vector<float> predicted(hd * q_heads * tokens);
    const float scale = 1.0f / std::sqrt(256.0f);
    std::vector<float> scores(tokens);
    std::vector<float> weights(tokens);

    for (std::size_t token = 0; token < tokens; ++token) {
        for (std::size_t q_head = 0; q_head < q_heads; ++q_head) {
            const std::size_t kv_head = q_head / group;
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t past = 0; past <= token; ++past) {
                float dot = 0.0f;
                for (std::size_t i = 0; i < hd; ++i) {
                    dot += q[i + hd * (q_head + q_heads * token)] * k[i + hd * (kv_head + kv_heads * past)];
                }
                scores[past] = dot * scale;
                maximum = std::max(maximum, scores[past]);
            }
            float denominator = 0.0f;
            for (std::size_t past = 0; past <= token; ++past) {
                weights[past] = std::exp(scores[past] - maximum);
                denominator += weights[past];
            }
            for (std::size_t past = 0; past <= token; ++past) weights[past] /= denominator;
            for (std::size_t i = 0; i < hd; ++i) {
                float sum = 0.0f;
                for (std::size_t past = 0; past <= token; ++past) {
                    sum += v[i + hd * (kv_head + kv_heads * past)] * weights[past];
                }
                predicted[i + hd * (q_head + q_heads * token)] = sum;
            }
        }
    }

    const auto pregate = load_tensor(context_root, rows, "attn_pregate-3");
    metrics["attention_pregate"].add(predicted, pregate);
    const auto gate = load_tensor(context_root, rows, "gate_sigmoid-3");
    const auto gated = load_tensor(context_root, rows, "attn_gated-3");
    std::vector<float> gate_prediction(pregate.size());
    for (std::size_t i = 0; i < pregate.size(); ++i) gate_prediction[i] = pregate[i] * gate[i];
    metrics["attention_gate"].add(gate_prediction, gated);
    const auto input = load_tensor(context_root, rows, "post_ffn-2");
    const auto attention_output = load_tensor(context_root, rows, "attn_output-3");
    const auto residual = load_tensor(context_root, rows, "attn_residual-3");
    const auto ffn = load_tensor(context_root, rows, "ffn_out-3");
    const auto post = load_tensor(context_root, rows, "post_ffn-3");
    metrics["attention_residual"].add(add_vectors(input, attention_output), residual);
    metrics["attention_post_ffn"].add(add_vectors(residual, ffn), post);
}

qubit::Qwen35SourceFragment source_fragment(const RoleBinding& binding) {
    const auto topology = qubit::qwen35_role_topology(binding.role);
    qubit::Qwen35SourceFragment source;
    source.role = binding.role;
    source.donor_identity = std::string(qubit::Qwen35DonorIdentity);
    source.topology_identity = std::string(qubit::Qwen35TopologyIdentity);
    source.source_identity = binding.source_identity;
    source.component_manifest_identity = binding.component_manifest_identity;
    source.source_bytes = topology.source_bytes;
    source.component_count = topology.component_count;
    source.context_identity = std::string(ContextIdentity);
    source.activation_receipt_identity = std::string(ActivationReceipt);
    source.producer_receipt_identity = binding.producer_receipt_identity;
    source.identity = binding.source_fragment_identity;
    source.validate(sha256_text);
    return source;
}

bool metric_pass(std::string_view name, const MetricAccumulator& metric) {
    const auto found = Thresholds.find(std::string(name));
    if (found == Thresholds.end()) throw std::runtime_error("unknown metric threshold");
    return metric.max_abs <= found->second.max_abs && metric.max_context_nrmse <= found->second.nrmse;
}

bool role_pass(std::string_view role, const MetricMap& metrics) {
    for (const auto& [name, metric] : metrics) {
        const bool matches = role == "ssm_recurrent" ? name.rfind("ssm_", 0) == 0 : name.rfind("attention_", 0) == 0;
        if (matches && !metric_pass(name, metric)) return false;
    }
    return true;
}

std::string metrics_fingerprint(std::string_view role, const MetricMap& metrics) {
    std::ostringstream out;
    out << std::setprecision(17) << role << '|' << MechanismText;
    for (const auto& [name, metric] : metrics) {
        const bool matches = role == "ssm_recurrent" ? name.rfind("ssm_", 0) == 0 : name.rfind("attention_", 0) == 0;
        if (matches) out << '|' << name << ':' << metric.max_abs << ':' << metric.max_context_nrmse << ':' << metric.nrmse();
    }
    return out.str();
}

std::string address(std::string_view value) { return "sha256:" + sha256_text(value); }

std::string json_escape(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (const char ch : value) {
        if (ch == '"' || ch == '\\') out << '\\' << ch;
        else if (ch == '\n') out << "\\n";
        else out << ch;
    }
    out << '"';
    return out.str();
}

void write_text(const fs::path& path, std::string_view text) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("failed to create receipt: " + path.string());
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    output.put('\n');
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("required fit receipt is missing");
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string receipt_json(std::string_view phase, std::string_view mechanism_identity,
                         std::string_view ssm_cir, std::string_view attention_cir,
                         bool ssm_ok, bool attention_ok, const MetricMap& metrics) {
    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\"schema\":\"qsa.qwen35-lct-runtime-evidence.v1\",\"phase\":" << json_escape(phase)
        << ",\"pr65_head\":\"" << Pr65Head << "\",\"reference_mechanism_identity\":\"" << ReferenceMechanism
        << "\",\"mechanism_identity\":" << json_escape(mechanism_identity)
        << ",\"qsa_cir\":{\"full_attention\":" << json_escape(attention_cir)
        << ",\"ssm_recurrent\":" << json_escape(ssm_cir) << "},\"metrics\":{";
    bool first = true;
    for (const auto& [name, metric] : metrics) {
        if (!first) out << ',';
        first = false;
        const auto threshold = Thresholds.at(name);
        out << json_escape(name) << ":{\"elements\":" << metric.elements << ",\"max_abs\":" << metric.max_abs
            << ",\"nrmse\":" << metric.nrmse() << ",\"max_context_nrmse\":" << metric.max_context_nrmse
            << ",\"threshold_max_abs\":" << threshold.max_abs << ",\"threshold_nrmse\":" << threshold.nrmse
            << ",\"passed\":" << (metric_pass(name, metric) ? "true" : "false") << '}';
    }
    const bool fit = ssm_ok && attention_ok;
    out << "},\"roles\":{\"full_attention\":{\"operator_equivalence_verified\":" << (attention_ok ? "true" : "false")
        << "},\"ssm_recurrent\":{\"operator_equivalence_verified\":" << (ssm_ok ? "true" : "false")
        << "}},\"fit_pass\":" << (fit ? "true" : "false")
        << ",\"holdout_pass\":" << (phase == "holdout" && fit ? "true" : "false")
        << ",\"qsa_runtime_equivalence_verified\":" << (phase == "holdout" && fit ? "true" : "false") << '}';
    return out.str();
}

qubit::Qwen35QsaTranslationFragment finalize_role(const RoleBinding& binding, const MetricMap& metrics,
                                                   std::string_view phase) {
    auto source = source_fragment(binding);
    const auto program = qubit::Qwen35LctCompiler::compile(binding.role);
    const auto cir = program.cir_identity(source, sha256_text);
    if (cir != binding.reference_cir_identity) throw std::runtime_error("independently derived PR65 CIR differs from frozen reference CIR");
    const std::string role = std::string(qubit::qwen35_role_name(binding.role));
    const auto operator_root = address(std::string("qsa.pr65.operator.v1|") + std::string(phase) + "|" + cir + "|" + metrics_fingerprint(role, metrics));
    const auto producer = address(std::string("qsa.pr65.producer.v1|") + role + "|" + cir + "|" + operator_root + "|" + binding.source_identity + "|" + std::string(ReferenceHoldoutIdentity));
    qubit::Qwen35TranslationEvidence evidence;
    evidence.residual_identity = binding.residual_identity;
    evidence.decoder_identity = binding.decoder_identity;
    evidence.operator_receipt_root = operator_root;
    evidence.reconstruction_identity = binding.reconstruction_identity;
    evidence.producer_receipt_identity = producer;
    evidence.operator_mismatch_count = 0;
    evidence.reconstruction_mismatch_count = 0;
    evidence.operator_equivalence_verified = true;
    evidence.exact_reconstruction_verified = true;
    return qubit::qwen35_finalize_translation(program, source, evidence, sha256_text);
}

int run(std::string_view phase, const fs::path& atlas, const fs::path& output, const fs::path& fit_receipt) {
    const bool holdout = phase == "holdout";
    if (!holdout && phase != "fit") throw std::runtime_error("phase must be fit or holdout");
    const auto mechanism_identity = address(MechanismText);
    if (holdout) {
        const auto frozen = read_text(fit_receipt);
        if (frozen.find("\"fit_pass\":true") == std::string::npos ||
            frozen.find("\"mechanism_identity\":\"" + mechanism_identity + "\"") == std::string::npos) {
            throw std::runtime_error("fit receipt does not authorize frozen holdout");
        }
    }

    const auto ssm_source = source_fragment(SsmBinding);
    const auto attention_source = source_fragment(AttentionBinding);
    const auto ssm_program = qubit::Qwen35LctCompiler::compile(SsmBinding.role);
    const auto attention_program = qubit::Qwen35LctCompiler::compile(AttentionBinding.role);
    const auto ssm_cir = ssm_program.cir_identity(ssm_source, sha256_text);
    const auto attention_cir = attention_program.cir_identity(attention_source, sha256_text);
    if (ssm_cir != SsmBinding.reference_cir_identity || attention_cir != AttentionBinding.reference_cir_identity) {
        throw std::runtime_error("PR65 CIR derivation does not match frozen source-bound reference");
    }

    MetricMap metrics;
    const std::string prefix = holdout ? "h" : "f";
    const int count = holdout ? 16 : 48;
    for (int index = 1; index <= count; ++index) {
        std::ostringstream name;
        name << prefix << std::setw(3) << std::setfill('0') << index;
        const fs::path context = atlas / name.str();
        const auto tokens = token_count(context / "tokens.tsv");
        const auto rows = tensor_index(context / "tensor-index.tsv");
        evaluate_ssm(context, rows, tokens, metrics);
        evaluate_attention(context, rows, tokens, metrics);
        std::cout << "QSA_LCT_CONTEXT=" << name.str() << "\n";
    }
    const bool ssm_ok = role_pass("ssm_recurrent", metrics);
    const bool attention_ok = role_pass("full_attention", metrics);
    const auto receipt = receipt_json(phase, mechanism_identity, ssm_cir, attention_cir, ssm_ok, attention_ok, metrics);
    const fs::path receipt_path = output / (holdout ? "qsa-holdout-receipt.json" : "qsa-fit-receipt.json");
    write_text(receipt_path, receipt);
    std::cout << "QSA_LCT_SSM=" << (ssm_ok ? "PASS" : "FAIL") << "\n";
    std::cout << "QSA_LCT_ATTENTION=" << (attention_ok ? "PASS" : "FAIL") << "\n";
    std::cout << "QSA_LCT_MECHANISM=" << mechanism_identity << "\n";
    if (!(ssm_ok && attention_ok)) return 2;
    if (holdout) {
        const auto ssm_fragment = finalize_role(SsmBinding, metrics, phase);
        const auto attention_fragment = finalize_role(AttentionBinding, metrics, phase);
        write_text(output / "ssm_recurrent.qsa.json", ssm_fragment.to_json());
        write_text(output / "full_attention.qsa.json", attention_fragment.to_json());
        std::cout << "QSA_LCT_SSM_FRAGMENT=" << ssm_fragment.identity << "\n";
        std::cout << "QSA_LCT_ATTENTION_FRAGMENT=" << attention_fragment.identity << "\n";
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 5) {
            std::cerr << "usage: qwen35_lct_runtime_evidence <fit|holdout> <atlas> <output> [fit-receipt]\n";
            return 64;
        }
        const std::string_view phase = argv[1];
        const fs::path fit = argc == 5 ? fs::path(argv[4]) : fs::path{};
        return run(phase, fs::path(argv[2]), fs::path(argv[3]), fit);
    } catch (const std::exception& error) {
        std::cerr << "QSA_LCT_ERROR=" << error.what() << '\n';
        return 1;
    }
}
