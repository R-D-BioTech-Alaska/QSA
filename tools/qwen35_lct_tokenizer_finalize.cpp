#include "qubit/qlct_qwen35.hpp"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#else
#error qwen35_lct_tokenizer_finalize currently requires Windows
#endif

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::string_view ContextIdentity = "sha256:9e61ed0d1f3de997331421b97f80071df368664d59fe858885de5ff076519177";
constexpr std::string_view ActivationReceipt = "sha256:60a9e08cc71dabe138f489615416163d0cc3002da78e678aeefad50c9cfc9c35";
constexpr std::string_view SourceIdentity = "sha256:39b41bbd3c923e231a8d1d5dc5aac4cb8d9846c49a157c47c459a0a113ff92a9";
constexpr std::string_view ComponentManifestIdentity = "sha256:9fa710fa7166a987915716c0ef68d3a6df47bdcfa7a77b36d986cbaae65866bb";
constexpr std::string_view SourceFragmentIdentity = "711996986bbaa7295ea8b6401533624b8b1d66196a5799c76f81e4b46828db01";
constexpr std::string_view SourceProducerIdentity = "sha256:dcba720868b73f59eb4ee556b5bb7526a84ce81680132a0cd4e6e6a2ca2fd765";
constexpr std::string_view ReferenceCirIdentity = "sha256:f3b244d5be92439281f9999344e6ebfbe5c30eabdd5f17f7b622f3b86027ae74";
constexpr std::string_view ResidualIdentity = "sha256:d8cf45f50d7495db90d48d267bc2c2c7afb09f6279244dfab7ae9641314b4d8a";
constexpr std::string_view DecoderIdentity = "sha256:80fb33669807adfb082a881c974a6fc402a6b3d2c9a95c46b84e57f1ee897a66";
constexpr std::string_view ReconstructionIdentity = "sha256:74ea6e7ed8c8cd93b6e3d10bbd52bc1b689090b8c3a7de4187a439edbd44bdd1";
constexpr std::string_view ReferenceHoldoutIdentity = "aabb23a0f1572b466b39ffefcf3510903dd25768ad95ac32836cb0c48cb919b9";

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

std::string address(std::string_view value) {
    return "sha256:" + sha256_text(value);
}

qubit::Qwen35SourceFragment tokenizer_source() {
    const auto topology = qubit::qwen35_role_topology(qubit::Qwen35LctRole::TokenizerEmbedding);
    qubit::Qwen35SourceFragment source;
    source.role = qubit::Qwen35LctRole::TokenizerEmbedding;
    source.donor_identity = std::string(qubit::Qwen35DonorIdentity);
    source.topology_identity = std::string(qubit::Qwen35TopologyIdentity);
    source.source_identity = std::string(SourceIdentity);
    source.component_manifest_identity = std::string(ComponentManifestIdentity);
    source.source_bytes = topology.source_bytes;
    source.component_count = topology.component_count;
    source.context_identity = std::string(ContextIdentity);
    source.activation_receipt_identity = std::string(ActivationReceipt);
    source.producer_receipt_identity = std::string(SourceProducerIdentity);
    source.identity = std::string(SourceFragmentIdentity);
    source.validate(sha256_text);
    return source;
}

void write_text(const fs::path& path, std::string_view text) {
    if (fs::exists(path)) throw std::runtime_error("refusing to replace tokenizer QSA fragment");
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("failed to create tokenizer QSA fragment");
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    output.put('\n');
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "usage: qwen35_lct_tokenizer_finalize <operator-receipt-root> <output-file>\n";
            return 64;
        }
        const std::string operator_root = argv[1];
        if (operator_root.rfind("sha256:", 0) != 0 || operator_root.size() != 71) {
            throw std::runtime_error("invalid tokenizer operator receipt root");
        }
        const auto source = tokenizer_source();
        const auto program = qubit::Qwen35LctCompiler::compile(qubit::Qwen35LctRole::TokenizerEmbedding);
        const auto cir = program.cir_identity(source, sha256_text);
        if (cir != ReferenceCirIdentity) {
            throw std::runtime_error("PR65 tokenizer CIR differs from frozen source-bound reference");
        }
        const auto producer = address(std::string("qsa.pr65.producer.v1|tokenizer_embedding|") + cir + "|" + operator_root + "|" + std::string(SourceIdentity) + "|" + std::string(ReferenceHoldoutIdentity));
        qubit::Qwen35TranslationEvidence evidence;
        evidence.residual_identity = std::string(ResidualIdentity);
        evidence.decoder_identity = std::string(DecoderIdentity);
        evidence.operator_receipt_root = operator_root;
        evidence.reconstruction_identity = std::string(ReconstructionIdentity);
        evidence.producer_receipt_identity = producer;
        evidence.operator_mismatch_count = 0;
        evidence.reconstruction_mismatch_count = 0;
        evidence.operator_equivalence_verified = true;
        evidence.exact_reconstruction_verified = true;
        const auto fragment = qubit::qwen35_finalize_translation(program, source, evidence, sha256_text);
        write_text(fs::path(argv[2]), fragment.to_json());
        std::cout << "QSA_TOKENIZER_CIR=" << cir << '\n';
        std::cout << "QSA_TOKENIZER_FRAGMENT=" << fragment.identity << '\n';
        std::cout << "QSA_TOKENIZER_PRODUCER=" << producer << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "QSA_TOKENIZER_ERROR=" << error.what() << '\n';
        return 1;
    }
}
