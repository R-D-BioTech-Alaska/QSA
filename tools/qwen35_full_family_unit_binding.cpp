#include "qubit/qlct_qwen35.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#else
#error qwen35_full_family_unit_binding requires the QELM Windows runner
#endif

#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::string hex_digest(const std::array<unsigned char, 32>& digest) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out(64, '0');
    for (std::size_t i = 0; i < digest.size(); ++i) {
        out[2 * i] = hex[digest[i] >> 4];
        out[2 * i + 1] = hex[digest[i] & 0x0f];
    }
    return out;
}

std::string sha256(std::string_view value) {
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

bool address(std::string_view value) {
    if (value.size() != 71 || value.substr(0, 7) != "sha256:") return false;
    for (std::size_t i = 7; i < value.size(); ++i) {
        const char ch = value[i];
        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) return false;
    }
    return true;
}

std::string quote(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (const char ch : value) {
        if (ch == '"' || ch == '\\') out << '\\';
        out << ch;
    }
    out << '"';
    return out.str();
}

qubit::Qwen35LctRole role_from(std::string_view value) {
    if (value == "ssm_recurrent") return qubit::Qwen35LctRole::SsmRecurrent;
    if (value == "full_attention") return qubit::Qwen35LctRole::FullAttention;
    if (value == "tokenizer_embedding") return qubit::Qwen35LctRole::TokenizerEmbedding;
    throw std::runtime_error("unsupported Qwen35 LCT role");
}

std::string_view accepted_cir(qubit::Qwen35LctRole role) {
    switch (role) {
        case qubit::Qwen35LctRole::SsmRecurrent:
            return "sha256:be6c04a161458f5c6bf15681bd543de5d57b77a2395e10503959edfe3418fa0f";
        case qubit::Qwen35LctRole::FullAttention:
            return "sha256:6db6bd50c2abb8aa8f1e0ca360c5c728b91cbe70afe821c91423735753728672";
        case qubit::Qwen35LctRole::TokenizerEmbedding:
            return "sha256:f3b244d5be92439281f9999344e6ebfbe5c30eabdd5f17f7b622f3b86027ae74";
    }
    throw std::runtime_error("unsupported Qwen35 LCT role");
}

std::string_view accepted_operator_root(qubit::Qwen35LctRole role) {
    switch (role) {
        case qubit::Qwen35LctRole::SsmRecurrent:
            return "sha256:941c40d882a3269a5f7cc9d66d55fd46334514455d516b5df53a1362491e059f";
        case qubit::Qwen35LctRole::FullAttention:
            return "sha256:2bf551a33df9fe8821054f28aae3ed89a8a445fcf5836046ade651aefd334d22";
        case qubit::Qwen35LctRole::TokenizerEmbedding:
            return "sha256:5b9bb1b1cb6063c42f6618f591e8aa65d70f659cc07587379c82aee1be7342fb";
    }
    throw std::runtime_error("unsupported Qwen35 LCT role");
}

std::string program_identity(const qubit::Qwen35OperatorProgram& program) {
    std::ostringstream text;
    text << "qsa.qwen35-full-family-program.v1|" << qubit::qwen35_role_name(program.role)
         << '|' << program.step_count << '|' << program.source_trace;
    for (const auto& state : program.states) {
        text << "|state:" << state.state << ':' << state.before << ':' << state.update << ':' << state.after;
    }
    for (const auto& residual : program.residuals) {
        text << "|residual:" << residual.output << ':' << residual.left << ':' << residual.right;
    }
    return "sha256:" + sha256(text.str());
}

void write_receipt(const fs::path& path, std::string_view text) {
    if (fs::exists(path)) throw std::runtime_error("refusing to replace QSA unit operator receipt");
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("failed to create QSA unit operator receipt");
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    output.put('\n');
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 10) {
            std::cerr << "usage: qwen35_full_family_unit_binding <role> <family_key> <planned_unit_identity> <unit_source_identity> <family_cir_identity> <representative_operator_root> <component_count> <source_bytes> <output>\n";
            return 2;
        }
        const std::string role_name = argv[1];
        const std::string family_key = argv[2];
        const std::string planned_unit_identity = argv[3];
        const std::string unit_source_identity = argv[4];
        const std::string family_cir_identity = argv[5];
        const std::string representative_operator_root = argv[6];
        const auto component_count = static_cast<std::size_t>(std::stoull(argv[7]));
        const auto source_bytes = static_cast<std::size_t>(std::stoull(argv[8]));
        const fs::path output = argv[9];
        if (!address(planned_unit_identity) || !address(unit_source_identity) || !address(family_cir_identity) || !address(representative_operator_root)) {
            throw std::runtime_error("QSA unit binding requires content-addressed inputs");
        }

        const auto role = role_from(role_name);
        const auto topology = qubit::qwen35_role_topology(role);
        if (component_count != topology.component_count || source_bytes != topology.source_bytes) {
            throw std::runtime_error("QSA unit source topology differs from the verified family topology");
        }
        if (family_cir_identity != accepted_cir(role)) {
            throw std::runtime_error("QSA unit CIR differs from the accepted role-specific CIR");
        }
        if (representative_operator_root != accepted_operator_root(role)) {
            throw std::runtime_error("QSA unit operator parent differs from the accepted representative root");
        }
        const auto program = qubit::Qwen35LctCompiler::compile(role);
        const auto program_id = program_identity(program);
        const std::string operator_text =
            "qsa.qwen35-full-family-unit-operator.v1|" + role_name + '|' + family_key + '|' +
            planned_unit_identity + '|' + unit_source_identity + '|' + family_cir_identity + '|' +
            representative_operator_root + '|' + program_id + '|' + std::to_string(component_count) + '|' +
            std::to_string(source_bytes);
        const std::string operator_identity = "sha256:" + sha256(operator_text);
        const std::string producer_identity = "sha256:" + sha256("qsa.qwen35-full-family-unit-producer.v1|" + operator_identity);

        std::ostringstream json;
        json << "{\"schema\":\"qsa.qwen35-full-family-unit-operator.v1\""
             << ",\"role\":" << quote(role_name)
             << ",\"family_key\":" << quote(family_key)
             << ",\"planned_unit_identity\":" << quote(planned_unit_identity)
             << ",\"unit_source_identity\":" << quote(unit_source_identity)
             << ",\"family_cir_identity\":" << quote(family_cir_identity)
             << ",\"operator_program_identity\":" << quote(program_id)
             << ",\"representative_operator_receipt_root\":" << quote(representative_operator_root)
             << ",\"component_count\":" << component_count
             << ",\"source_bytes\":" << source_bytes
             << ",\"accepted_role_cir_verified\":true"
             << ",\"accepted_operator_root_verified\":true"
             << ",\"operator_program_instantiated\":true"
             << ",\"representative_numerical_equivalence_parent_bound\":true"
             << ",\"unit_numerical_holdout_reexecuted\":false"
             << ",\"operator_binding_mismatch_count\":0"
             << ",\"operator_mismatch_count\":0"
             << ",\"operator_receipt_identity\":" << quote(operator_identity)
             << ",\"producer_receipt_identity\":" << quote(producer_identity) << '}';
        write_receipt(output, json.str());
        std::cout << "QSA_UNIT_OPERATOR_RECEIPT=" << output.string() << '\n';
        std::cout << "QSA_UNIT_OPERATOR_IDENTITY=" << operator_identity << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
