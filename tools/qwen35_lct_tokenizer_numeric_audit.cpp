#include "qubit/qnumeric.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::size_t Width = 5120;
constexpr std::size_t Vocab = 248320;
constexpr std::size_t Qk = 256;
constexpr std::size_t Q6BlockBytes = 210;
constexpr std::size_t Q6BlocksPerRow = Width / Qk;
constexpr std::size_t Q6RowBytes = Q6BlockBytes * Q6BlocksPerRow;
constexpr std::size_t Q4BlockBytes = 144;
constexpr std::size_t Q4RowBytes = Q4BlockBytes * (Width / Qk);
constexpr std::uint64_t OutputStart = 10992928ULL;
constexpr std::uint64_t EmbedStart = 1053957408ULL;
constexpr double OutputMaxAbs = 5.0e-4;
constexpr double OutputNrmse = 1.0e-6;
constexpr double EmbedMaxAbs = 2.0e-6;
constexpr double EmbedNrmse = 5.0e-7;
constexpr std::string_view MechanismIdentity = "sha256:3edc832511409b54362332496386432a4ed7d0ff2f47e237b6ad93f7d8fa8abc";

struct Metric {
    std::size_t elements{};
    double sum_sq{};
    double ref_sq{};
    double max_abs{};
    double max_context_nrmse{};

    void add(double actual, double expected, double& local_sq, double& local_ref, double& local_max) {
        const double delta = actual - expected;
        sum_sq += delta * delta;
        ref_sq += expected * expected;
        max_abs = std::max(max_abs, std::abs(delta));
        ++elements;
        local_sq += delta * delta;
        local_ref += expected * expected;
        local_max = std::max(local_max, std::abs(delta));
    }

    void finish_context(double local_sq, double local_ref, std::size_t count) {
        if (count == 0) throw std::runtime_error("empty numeric audit context");
        const double rms = std::sqrt(local_sq / static_cast<double>(count));
        const double ref = std::sqrt(local_ref / static_cast<double>(count));
        max_context_nrmse = std::max(max_context_nrmse, rms / std::max(ref, 1.0e-30));
    }

    double nrmse() const {
        const double rms = std::sqrt(sum_sq / static_cast<double>(std::max<std::size_t>(elements, 1)));
        const double ref = std::sqrt(ref_sq / static_cast<double>(std::max<std::size_t>(elements, 1)));
        return rms / std::max(ref, 1.0e-30);
    }
};

float half_to_float(std::uint16_t value) {
    return _cvtsh_ss(value);
}

float half_roundtrip(float value) {
    return _cvtsh_ss(_cvtss_sh(value, 0));
}

std::vector<float> read_f32(const fs::path& path) {
    const auto bytes = fs::file_size(path);
    if (bytes % sizeof(float)) throw std::runtime_error("unaligned F32 payload: " + path.string());
    std::vector<float> values(static_cast<std::size_t>(bytes / sizeof(float)));
    std::ifstream input(path, std::ios::binary);
    if (!input || !input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes))) {
        throw std::runtime_error("failed to read F32 payload: " + path.string());
    }
    return values;
}

std::vector<int> tokens(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("tokens.tsv missing");
    std::vector<int> out;
    std::string line;
    while (std::getline(input, line)) {
        const auto tab = line.find('\t');
        if (tab == std::string::npos || std::stoi(line.substr(0, tab)) != static_cast<int>(out.size())) {
            throw std::runtime_error("token position changed");
        }
        const int token = std::stoi(line.substr(tab + 1));
        if (token < 0 || token >= static_cast<int>(Vocab)) throw std::runtime_error("token id outside vocabulary");
        out.push_back(token);
    }
    if (out.empty()) throw std::runtime_error("empty token list");
    return out;
}

fs::path tensor_path(const fs::path& context, std::string_view name) {
    std::ifstream input(context / "tensor-index.tsv");
    if (!input) throw std::runtime_error("tensor-index.tsv missing");
    std::string line;
    std::getline(input, line);
    if (line != "name\tgraph_name\toccurrence\ttype\tshape\tbytes\tfile") throw std::runtime_error("tensor index schema changed");
    fs::path found;
    int count = 0;
    while (std::getline(input, line)) {
        std::array<std::string, 7> fields;
        std::size_t start = 0;
        bool valid = true;
        for (std::size_t i = 0; i < fields.size(); ++i) {
            const auto end = i + 1 == fields.size() ? std::string::npos : line.find('\t', start);
            if (end == std::string::npos && i + 1 != fields.size()) { valid = false; break; }
            fields[i] = line.substr(start, end == std::string::npos ? end : end - start);
            start = end == std::string::npos ? line.size() : end + 1;
        }
        if (valid && fields[0] == name) {
            if (fields[3] != "f32") throw std::runtime_error("audit tensor type changed");
            found = context / fields[6];
            if (fs::file_size(found) != static_cast<std::uint64_t>(std::stoull(fields[5]))) throw std::runtime_error("audit tensor bytes changed");
            ++count;
        }
    }
    if (count != 1) throw std::runtime_error("audit tensor binding is not unique");
    return found;
}

std::pair<int, int> q4_scale_min(int index, const std::uint8_t* scales) {
    if (index < 4) return {scales[index] & 0x3f, scales[index + 4] & 0x3f};
    return {
        (scales[index + 4] & 0x0f) | ((scales[index - 4] & 0xc0) >> 2),
        ((scales[index + 4] >> 4) & 0x0f) | ((scales[index] & 0xc0) >> 2),
    };
}

std::vector<double> q4_row(const std::array<std::uint8_t, Q4RowBytes>& raw) {
    std::vector<double> out(Width);
    std::size_t target = 0;
    for (std::size_t block_index = 0; block_index < Width / Qk; ++block_index) {
        const auto* block = raw.data() + block_index * Q4BlockBytes;
        const std::uint16_t d_bits = static_cast<std::uint16_t>(block[0] | (block[1] << 8));
        const std::uint16_t min_bits = static_cast<std::uint16_t>(block[2] | (block[3] << 8));
        const double d = half_to_float(d_bits);
        const double dmin = half_to_float(min_bits);
        const auto* scales = block + 4;
        const auto* qs = block + 16;
        for (int group = 0; group < 4; ++group) {
            const auto [lo_scale, lo_min] = q4_scale_min(2 * group, scales);
            const auto [hi_scale, hi_min] = q4_scale_min(2 * group + 1, scales);
            for (int j = 0; j < 32; ++j) {
                const std::uint8_t packed = qs[group * 32 + j];
                out[target + j] = d * lo_scale * (packed & 0x0f) - dmin * lo_min;
                out[target + 32 + j] = d * hi_scale * (packed >> 4) - dmin * hi_min;
            }
            target += 64;
        }
    }
    return out;
}

std::vector<double> q6_row(const std::array<std::uint8_t, Q6RowBytes>& raw) {
    std::vector<double> out(Width);
    std::size_t target = 0;
    for (std::size_t block_index = 0; block_index < Q6BlocksPerRow; ++block_index) {
        const auto* block = raw.data() + block_index * Q6BlockBytes;
        const auto* ql = block;
        const auto* qh = block + 128;
        const auto* scales = reinterpret_cast<const std::int8_t*>(block + 192);
        const std::uint16_t d_bits = static_cast<std::uint16_t>(block[208] | (block[209] << 8));
        const double d = half_to_float(d_bits);
        for (int half = 0; half < 2; ++half) {
            for (int j = 0; j < 32; ++j) {
                const std::uint8_t lo0 = ql[half * 64 + j];
                const std::uint8_t lo1 = ql[half * 64 + 32 + j];
                const std::uint8_t hi = qh[half * 32 + j];
                const int scale_pair = j < 16 ? 0 : 1;
                const int local = half * 8;
                const int q1 = static_cast<int>((lo0 & 0x0f) | (((hi >> 0) & 0x03) << 4)) - 32;
                const int q2 = static_cast<int>((lo1 & 0x0f) | (((hi >> 2) & 0x03) << 4)) - 32;
                const int q3 = static_cast<int>((lo0 >> 4) | (((hi >> 4) & 0x03) << 4)) - 32;
                const int q4 = static_cast<int>((lo1 >> 4) | (((hi >> 6) & 0x03) << 4)) - 32;
                out[target + j] = d * scales[local + scale_pair + 0] * q1;
                out[target + 32 + j] = d * scales[local + scale_pair + 2] * q2;
                out[target + 64 + j] = d * scales[local + scale_pair + 4] * q3;
                out[target + 96 + j] = d * scales[local + scale_pair + 6] * q4;
            }
            target += 128;
        }
    }
    return out;
}

std::vector<double> q8_1(const std::vector<float>& values) {
    if (values.size() != Width) throw std::runtime_error("result_norm width changed");
    std::vector<double> out(Width);
    for (std::size_t block = 0; block < Width / 32; ++block) {
        float maximum = 0.0f;
        for (std::size_t j = 0; j < 32; ++j) maximum = std::max(maximum, std::abs(values[block * 32 + j]));
        const float d = maximum / 127.0f;
        const float stored = half_roundtrip(d);
        for (std::size_t j = 0; j < 32; ++j) {
            const float value = values[block * 32 + j];
            const int quantized = d == 0.0f ? 0 : static_cast<int>(std::round(value / d));
            out[block * 32 + j] = static_cast<double>(quantized) * stored;
        }
    }
    return out;
}

std::vector<std::size_t> output_rows(const std::vector<float>& logits) {
    if (logits.size() != Vocab) throw std::runtime_error("logit vocabulary changed");
    std::set<std::size_t> rows;
    for (std::size_t i = 0; i < 64; ++i) rows.insert(i * (Vocab - 1) / 63);
    rows.insert(static_cast<std::size_t>(std::max_element(logits.begin(), logits.end()) - logits.begin()));
    return {rows.begin(), rows.end()};
}

std::string receipt(std::string_view phase, const qubit::NumericExecutor& numeric, const Metric& embed, const Metric& output, bool passed) {
    std::ostringstream text;
    text << std::setprecision(17)
         << "{\"schema\":\"qsa.qwen35-lct-tokenizer-numeric-audit.v1\",\"phase\":\"" << phase
         << "\",\"mechanism_identity\":\"" << MechanismIdentity
         << "\",\"numeric_backend\":\"" << numeric.backend_name()
         << "\",\"numeric_workers\":" << numeric.worker_count()
         << ",\"embedding\":{\"elements\":" << embed.elements << ",\"max_abs\":" << embed.max_abs
         << ",\"nrmse\":" << embed.nrmse() << ",\"max_context_nrmse\":" << embed.max_context_nrmse << "}"
         << ",\"output_sample\":{\"elements\":" << output.elements << ",\"max_abs\":" << output.max_abs
         << ",\"nrmse\":" << output.nrmse() << ",\"max_context_nrmse\":" << output.max_context_nrmse << "}"
         << ",\"passed\":" << (passed ? "true" : "false") << '}';
    return text.str();
}

bool fit_receipt_allows_holdout(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    const std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    return text.find("\"passed\":true") != std::string::npos && text.find(std::string(MechanismIdentity)) != std::string::npos;
}

int run(std::string_view phase, const fs::path& donor, const fs::path& atlas, const fs::path& supplement,
        const fs::path& output_path, const fs::path& fit_receipt) {
    const bool holdout = phase == "holdout";
    if (!holdout && phase != "fit") throw std::runtime_error("phase must be fit or holdout");
    if (holdout && !fit_receipt_allows_holdout(fit_receipt)) throw std::runtime_error("numeric fit receipt does not authorize holdout");
    qubit::NumericConfig config;
    config.worker_count = 0;
    config.grain_size = 512;
    config.enable_simd = true;
    qubit::NumericExecutor numeric(config);
    Metric embedding;
    Metric output;
    const char prefix = holdout ? 'h' : 'f';
    const int count = holdout ? 16 : 48;
    std::ifstream donor_file(donor, std::ios::binary);
    if (!donor_file) throw std::runtime_error("donor is unavailable");

    for (int index = 1; index <= count; ++index) {
        std::ostringstream name;
        name << prefix << std::setw(3) << std::setfill('0') << index;
        const fs::path context = atlas / name.str();
        const auto token_ids = tokens(context / "tokens.tsv");
        const auto expected_embedding = read_f32(tensor_path(context, "model.input_embed"));
        if (expected_embedding.size() != token_ids.size() * Width) throw std::runtime_error("embedding tensor shape changed");
        double local_embed_sq = 0.0, local_embed_ref = 0.0, local_embed_max = 0.0;
        std::array<std::uint8_t, Q4RowBytes> q4{};
        std::size_t embed_count = 0;
        for (std::size_t position = 0; position < token_ids.size(); ++position) {
            donor_file.seekg(static_cast<std::streamoff>(EmbedStart + static_cast<std::uint64_t>(token_ids[position]) * Q4RowBytes));
            donor_file.read(reinterpret_cast<char*>(q4.data()), static_cast<std::streamsize>(q4.size()));
            if (!donor_file) throw std::runtime_error("failed to read Q4_K embedding row");
            const auto decoded = q4_row(q4);
            for (std::size_t element = 0; element < Width; ++element) {
                embedding.add(decoded[element], expected_embedding[element + Width * position], local_embed_sq, local_embed_ref, local_embed_max);
                ++embed_count;
            }
        }
        embedding.finish_context(local_embed_sq, local_embed_ref, embed_count);

        const auto norm = read_f32(supplement / name.str() / "result_norm.f32");
        const auto quantized_norm = q8_1(norm);
        const auto logits = read_f32(context / "logits.f32");
        const auto rows = output_rows(logits);
        double local_output_sq = 0.0, local_output_ref = 0.0, local_output_max = 0.0;
        std::array<std::uint8_t, Q6RowBytes> q6{};
        for (const auto row : rows) {
            donor_file.clear();
            donor_file.seekg(static_cast<std::streamoff>(OutputStart + static_cast<std::uint64_t>(row) * Q6RowBytes));
            donor_file.read(reinterpret_cast<char*>(q6.data()), static_cast<std::streamsize>(q6.size()));
            if (!donor_file) throw std::runtime_error("failed to read Q6_K output row");
            const auto decoded = q6_row(q6);
            const double value = numeric.dot(decoded, quantized_norm);
            output.add(value, logits[row], local_output_sq, local_output_ref, local_output_max);
        }
        output.finish_context(local_output_sq, local_output_ref, rows.size());
        std::cout << "QSA_TOKENIZER_NUMERIC_CONTEXT=" << name.str() << '\n';
    }

    const bool passed = embedding.max_abs <= EmbedMaxAbs && embedding.max_context_nrmse <= EmbedNrmse &&
                        output.max_abs <= OutputMaxAbs && output.max_context_nrmse <= OutputNrmse;
    if (fs::exists(output_path)) throw std::runtime_error("refusing to replace numeric audit receipt");
    fs::create_directories(output_path.parent_path());
    std::ofstream out(output_path, std::ios::binary);
    const auto text = receipt(phase, numeric, embedding, output, passed);
    out << text << '\n';
    if (!out) throw std::runtime_error("failed to write numeric audit receipt");
    std::cout << "QSA_TOKENIZER_NUMERIC_BACKEND=" << numeric.backend_name() << '\n';
    std::cout << "QSA_TOKENIZER_NUMERIC_EMBED_MAX=" << embedding.max_abs << '\n';
    std::cout << "QSA_TOKENIZER_NUMERIC_OUTPUT_MAX=" << output.max_abs << '\n';
    return passed ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6 && argc != 7) {
            std::cerr << "usage: qwen35_lct_tokenizer_numeric_audit <fit|holdout> <donor> <atlas> <supplement> <output> [fit-receipt]\n";
            return 64;
        }
        return run(argv[1], fs::path(argv[2]), fs::path(argv[3]), fs::path(argv[4]), fs::path(argv[5]), argc == 7 ? fs::path(argv[6]) : fs::path{});
    } catch (const std::exception& error) {
        std::cerr << "QSA_TOKENIZER_NUMERIC_ERROR=" << error.what() << '\n';
        return 1;
    }
}
