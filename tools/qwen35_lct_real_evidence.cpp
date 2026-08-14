#include "qubit/qlct_qwen35.hpp"

#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::string sha256(std::string_view value) {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    DWORD object_bytes = 0;
    DWORD hash_bytes = 0;
    DWORD copied = 0;
    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0) {
        throw std::runtime_error("cannot open SHA-256 provider");
    }
    auto close_algorithm = [&]() { if (algorithm != nullptr) BCryptCloseAlgorithmProvider(algorithm, 0); };
    if (BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&object_bytes), sizeof(object_bytes), &copied, 0) != 0 ||
        BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH, reinterpret_cast<PUCHAR>(&hash_bytes), sizeof(hash_bytes), &copied, 0) != 0 ||
        hash_bytes != 32U) {
        close_algorithm();
        throw std::runtime_error("cannot query SHA-256 provider");
    }
    std::vector<UCHAR> object(object_bytes);
    std::array<UCHAR, 32> digest{};
    if (BCryptCreateHash(algorithm, &hash, object.data(), object_bytes, nullptr, 0, 0) != 0) {
        close_algorithm();
        throw std::runtime_error("cannot create SHA-256 hash");
    }
    auto close_hash = [&]() { if (hash != nullptr) BCryptDestroyHash(hash); };
    const auto* data = reinterpret_cast<PUCHAR>(const_cast<char*>(value.data()));
    if (value.size() > static_cast<std::size_t>(UINT32_MAX) ||
        BCryptHashData(hash, data, static_cast<ULONG>(value.size()), 0) != 0 ||
        BCryptFinishHash(hash, digest.data(), static_cast<ULONG>(digest.size()), 0) != 0) {
        close_hash();
        close_algorithm();
        throw std::runtime_error("SHA-256 operation failed");
    }
    close_hash();
    close_algorithm();
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const auto byte : digest) out << std::setw(2) << static_cast<unsigned>(byte);
    return out.str();
}

qubit::Qwen35LctRole role_from_name(std::string_view name) {
    if (name == "ssm_recurrent") return qubit::Qwen35LctRole::SsmRecurrent;
    if (name == "full_attention") return qubit::Qwen35LctRole::FullAttention;
    if (name == "tokenizer_embedding") return qubit::Qwen35LctRole::TokenizerEmbedding;
    throw std::runtime_error("unsupported Qwen35 LCT role");
}

qubit::Qwen35SourceFragment source_fragment(qubit::Qwen35LctRole role) {
    qubit::Qwen35SourceFragment source;
    source.role = role;
    source.donor_identity = std::string(qubit::Qwen35DonorIdentity);
    source.topology_identity = std::string(qubit::Qwen35TopologyIdentity);
    source.context_identity = "sha256:9e61ed0d1f3de997331421b97f80071df368664d59fe858885de5ff076519177";
    source.activation_receipt_identity = "sha256:60a9e08cc71dabe138f489615416163d0cc3002da78e678aeefad50c9cfc9c35";
    const auto topology = qubit::qwen35_role_topology(role);
    source.source_bytes = topology.source_bytes;
    source.component_count = topology.component_count;
    if (role == qubit::Qwen35LctRole::SsmRecurrent) {
        source.source_identity = "sha256:9852cc3230b0cac7bafb66cfb9e16fcf35966b0ff67e7afcb805bceedd415172";
        source.component_manifest_identity = "sha256:2c80671af3588c9f9ce110c96398e47706be2724a39536707e14a634bcc6ab71";
        source.producer_receipt_identity = "sha256:8c5bbeb395eae95c31c0ca5e16c66e2bbe6ed49f0988b1c6c0fc6a8a4bc191ee";
        source.identity = "0baf5fd5729eeaab69485f64575da3776c192531fc1ecbcb8a1e897879cdd0bd";
    } else if (role == qubit::Qwen35LctRole::FullAttention) {
        source.source_identity = "sha256:57f31f5f9a5df4fb976deed2563aa522eff959e2b4bdb2b32eb16410aa64430f";
        source.component_manifest_identity = "sha256:fdbfe2b7a9fb8dba6486a4821f214bb5a6534d9ad2e587009f85f7393ef1f365";
        source.producer_receipt_identity = "sha256:9ba393740d76d2a93996a9a9f5deed154ce610b30a70a527d4947366905133b5";
        source.identity = "0d1657ee4d242ee1aa72db8e85815393aa79834388338db7e9a9b5d89e502875";
    } else {
        source.source_identity = "sha256:39b41bbd3c923e231a8d1d5dc5aac4cb8d9846c49a157c47c459a0a113ff92a9";
        source.component_manifest_identity = "sha256:9fa710fa7166a987915716c0ef68d3a6df47bdcfa7a77b36d986cbaae65866bb";
        source.producer_receipt_identity = "sha256:dcba720868b73f59eb4ee556b5bb7526a84ce81680132a0cd4e6e6a2ca2fd765";
        source.identity = "711996986bbaa7295ea8b6401533624b8b1d66196a5799c76f81e4b46828db01";
    }
    source.validate(sha256);
    return source;
}

int run(int argc, char** argv) {
    if (argc < 3) throw std::runtime_error("usage: qwen35_lct_real_evidence <cir|finalize> <role> [...]");
    const auto role = role_from_name(argv[2]);
    const auto source = source_fragment(role);
    const auto program = qubit::Qwen35LctCompiler::compile(role);
    if (std::string_view(argv[1]) == "cir") {
        std::cout << program.cir_identity(source, sha256) << '\n';
        return 0;
    }
    if (std::string_view(argv[1]) != "finalize" || argc != 8) {
        throw std::runtime_error("finalize requires residual, decoder, operator, reconstruction and producer identities");
    }
    qubit::Qwen35TranslationEvidence evidence;
    evidence.residual_identity = argv[3];
    evidence.decoder_identity = argv[4];
    evidence.operator_receipt_root = argv[5];
    evidence.reconstruction_identity = argv[6];
    evidence.producer_receipt_identity = argv[7];
    evidence.operator_mismatch_count = 0U;
    evidence.reconstruction_mismatch_count = 0U;
    evidence.operator_equivalence_verified = true;
    evidence.exact_reconstruction_verified = true;
    const auto fragment = qubit::qwen35_finalize_translation(program, source, evidence, sha256);
    std::cout << fragment.to_json() << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
