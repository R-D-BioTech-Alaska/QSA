#include "qubit/qlct_qwen35.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr std::string_view qwen_core =
    "{\"activation_receipt_identity\":\"sha256:6ad639df855da05828019298f012d3e7ddeb0039445cf09a448ac70d5fb4fd8e\","
    "\"component_count\":11,\"component_manifest_identity\":\"sha256:00d9ae881718982c67ac6f748a7b97eab8f9194a15129807d263c0762a11a80e\","
    "\"context_identity\":\"sha256:d27adb7d42ccdadd6b6a3a76b887cc3a6807da99450cee4f3cb00db235a1b8c6\","
    "\"donor_identity\":\"sha256:65b753ea835627f7b511143c6ceb976525c7f21f5df8c664bc0a9c23d1c49921\","
    "\"producer_receipt_identity\":\"sha256:e3348c459404513e46fc30b673c73193aff50efc58675e223a61d7fc7a6e51ed\","
    "\"role\":\"full_attention\",\"schema\":\"qelm.lct-qwen-source-fragment.v1\",\"source_bytes\":261859328,"
    "\"source_identity\":\"sha256:7c0c1fbf249c793d20a5b50c708da6b227ddcd775f134b9bec7c7cb66236dac0\","
    "\"topology_identity\":\"sha256:e2b88b52a3f7ecbb0aed0d88c48e1af0e212bc7cdead4ea11c858fe3d718d108\"}";
constexpr std::string_view qwen_hash = "a12ac2501cd254df8bda1c7178656ebeeaa485505751bdb65e73d845534f9861";

constexpr std::string_view qsa_core =
    "{\"cir_identity\":\"sha256:72fe2aa03a474bc315d28e0e655873b73e572acbc2719995447a141b8ffef0fd\","
    "\"component_manifest_identity\":\"sha256:00d9ae881718982c67ac6f748a7b97eab8f9194a15129807d263c0762a11a80e\","
    "\"decoder_identity\":\"sha256:237fee8cba9d1eaad4cd01d1037632ee124806cdffb26ca2855c9d472e56f62e\","
    "\"exact_reconstruction_verified\":true,\"operator_equivalence_verified\":true,"
    "\"operator_receipt_root\":\"sha256:cbce1b43a7b876bf39ce5e0b49662eda6b13543f50bc51313ffcd4ea7cc99daf\","
    "\"producer_receipt_identity\":\"sha256:25c300605c92bfc4609848e2b0dda60e7c8fb852ae11e859855b76126bb9ad9f\","
    "\"reconstruction_identity\":\"sha256:fbb0372cb00a68081e4dd705044d51644cbc96f208d7c426e050921805ea72a3\","
    "\"reconstruction_mismatch_count\":0,\"residual_identity\":\"sha256:3c883b99bf17685193e8228142f9ed60782a2d8ef882aad770e6abd9312033ca\","
    "\"role\":\"full_attention\",\"schema\":\"qelm.lct-qsa-translation-fragment.v1\","
    "\"source_identity\":\"sha256:7c0c1fbf249c793d20a5b50c708da6b227ddcd775f134b9bec7c7cb66236dac0\"}";
constexpr std::string_view qsa_hash = "479b4142b0c312f6d6751de85bafe24910eea1d4912b06abd44df916d0ee921a";

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::string fixture_sha(std::string_view value) {
    if (value == qwen_core) return std::string(qwen_hash);
    if (value == qsa_core) return std::string(qsa_hash);
    std::uint64_t state = 1469598103934665603ULL;
    for (const unsigned char ch : value) {
        state ^= ch;
        state *= 1099511628211ULL;
    }
    static constexpr char hex[] = "0123456789abcdef";
    std::string out(64U, '0');
    for (std::size_t i = 0U; i < out.size(); ++i) {
        out[i] = hex[(state >> ((i % 16U) * 4U)) & 0x0fU];
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    }
    return out;
}

std::string address(std::string_view label) { return "sha256:" + fixture_sha(label); }

qubit::Qwen35SourceFragment source(qubit::Qwen35LctRole role) {
    const auto topology = qubit::qwen35_role_topology(role);
    const std::string name(qubit::qwen35_role_name(role));
    qubit::Qwen35SourceFragment value;
    value.role = role;
    value.donor_identity = std::string(qubit::Qwen35DonorIdentity);
    value.topology_identity = std::string(qubit::Qwen35TopologyIdentity);
    if (role == qubit::Qwen35LctRole::FullAttention) {
        value.source_identity = "sha256:7c0c1fbf249c793d20a5b50c708da6b227ddcd775f134b9bec7c7cb66236dac0";
        value.component_manifest_identity = "sha256:00d9ae881718982c67ac6f748a7b97eab8f9194a15129807d263c0762a11a80e";
        value.context_identity = "sha256:d27adb7d42ccdadd6b6a3a76b887cc3a6807da99450cee4f3cb00db235a1b8c6";
        value.activation_receipt_identity = "sha256:6ad639df855da05828019298f012d3e7ddeb0039445cf09a448ac70d5fb4fd8e";
        value.producer_receipt_identity = "sha256:e3348c459404513e46fc30b673c73193aff50efc58675e223a61d7fc7a6e51ed";
    } else {
        value.source_identity = address(name + "-source");
        value.component_manifest_identity = address(name + "-components");
        value.context_identity = address(name + "-context");
        value.activation_receipt_identity = address(name + "-activations");
        value.producer_receipt_identity = address(name + "-qwen-receipt");
    }
    value.source_bytes = topology.source_bytes;
    value.component_count = topology.component_count;
    value.identity = fixture_sha(value.canonical_core_json());
    value.validate(fixture_sha);
    return value;
}

void source_contract() {
    const auto value = source(qubit::Qwen35LctRole::FullAttention);
    require(value.canonical_core_json() == qwen_core, "Qwen source canonical JSON differs from PR163");
    require(value.identity == qwen_hash, "Qwen source signing vector differs from PR163");
    require(qubit::Qwen35LlamaCommit == "74ade52741203e5c8f81eaf06a96cb1cfe15f2a3",
            "Qwen35 source commit changed");
    require(qubit::Qwen35SourceBlob == "6783d98ec204885caba96184fc5b9c6bd47e071b",
            "Qwen35 source blob changed");
    auto tampered = value;
    --tampered.source_bytes;
    bool rejected = false;
    try { tampered.validate(fixture_sha); } catch (const qubit::QStateError&) { rejected = true; }
    require(rejected, "Qwen source topology tamper did not fail closed");
}

void operator_contract() {
    const auto attention = qubit::Qwen35LctCompiler::compile(qubit::Qwen35LctRole::FullAttention);
    const auto recurrent = qubit::Qwen35LctCompiler::compile(qubit::Qwen35LctRole::SsmRecurrent);
    const auto tokenizer = qubit::Qwen35LctCompiler::compile(qubit::Qwen35LctRole::TokenizerEmbedding);
    require(attention.step_count == 25U && attention.states.empty() && attention.residuals.size() == 2U,
            "full-attention source trace changed");
    require(recurrent.step_count == 40U && recurrent.states.size() == 2U && recurrent.residuals.size() == 2U,
            "recurrent source trace/state boundary changed");
    require(tokenizer.step_count == 4U && tokenizer.states.empty() && tokenizer.residuals.empty(),
            "tokenizer/output source trace changed");
    require(recurrent.states[0].before == "conv_before" && recurrent.states[0].after == "conv_after" &&
            recurrent.states[1].before == "state_before" && recurrent.states[1].after == "state_after",
            "recurrent state-before/state-after proof changed");
    require(attention.source_trace.find("f_attention_scale==0?1/sqrt(n_embd_head)") != std::string_view::npos,
            "full-attention exact scale rule is missing");
    require(recurrent.source_trace.find("build_recurrent_attn(q,k,v,g,beta,state_before)") != std::string_view::npos,
            "recurrent gated-delta update is missing");
    const auto ssm = qubit::qwen35_role_topology(qubit::Qwen35LctRole::SsmRecurrent);
    const auto full = qubit::qwen35_role_topology(qubit::Qwen35LctRole::FullAttention);
    const auto tok = qubit::qwen35_role_topology(qubit::Qwen35LctRole::TokenizerEmbedding);
    require(ssm.family_instances == 48U && ssm.family_component_count == 672U,
            "Qwen35 recurrent family accounting changed");
    require(full.family_instances == 16U && full.family_component_count == 176U,
            "Qwen35 full-attention family accounting changed");
    require(tok.family_component_count == 3U && ssm.family_instances + full.family_instances == 64U,
            "Qwen35 trunk/topology accounting changed");
    auto first = source(qubit::Qwen35LctRole::FullAttention);
    auto second = first;
    second.context_identity = address("different-context");
    second.identity = fixture_sha(second.canonical_core_json());
    second.validate(fixture_sha);
    require(attention.cir_identity(first, fixture_sha) != attention.cir_identity(second, fixture_sha),
            "Qwen35 CIR is not bound to runtime context identity");
}

void fragment_contract() {
    auto input = source(qubit::Qwen35LctRole::FullAttention);
    const auto program = qubit::Qwen35LctCompiler::compile(input.role);
    qubit::Qwen35TranslationEvidence evidence;
    evidence.residual_identity = address("residual");
    evidence.decoder_identity = address("decoder");
    evidence.operator_receipt_root = address("operator-root");
    evidence.reconstruction_identity = address("reconstruction");
    evidence.producer_receipt_identity = address("producer");
    bool rejected = false;
    try { (void)qubit::qwen35_finalize_translation(program, input, evidence, fixture_sha); }
    catch (const qubit::QStateError&) { rejected = true; }
    require(rejected, "unverified Qwen35 translation emitted a QSA fragment");
    evidence.operator_equivalence_verified = true;
    evidence.exact_reconstruction_verified = true;
    const auto fragment = qubit::qwen35_finalize_translation(program, input, evidence, fixture_sha);
    fragment.validate(fixture_sha);
    require(fragment.source_identity == input.source_identity &&
            fragment.component_manifest_identity == input.component_manifest_identity,
            "QSA fragment lost its exact Qwen source/component binding");

    qubit::Qwen35QsaTranslationFragment vector;
    vector.role = qubit::Qwen35LctRole::FullAttention;
    vector.source_identity = "sha256:7c0c1fbf249c793d20a5b50c708da6b227ddcd775f134b9bec7c7cb66236dac0";
    vector.component_manifest_identity = "sha256:00d9ae881718982c67ac6f748a7b97eab8f9194a15129807d263c0762a11a80e";
    vector.residual_identity = "sha256:3c883b99bf17685193e8228142f9ed60782a2d8ef882aad770e6abd9312033ca";
    vector.cir_identity = "sha256:72fe2aa03a474bc315d28e0e655873b73e572acbc2719995447a141b8ffef0fd";
    vector.decoder_identity = "sha256:237fee8cba9d1eaad4cd01d1037632ee124806cdffb26ca2855c9d472e56f62e";
    vector.operator_receipt_root = "sha256:cbce1b43a7b876bf39ce5e0b49662eda6b13543f50bc51313ffcd4ea7cc99daf";
    vector.reconstruction_identity = "sha256:fbb0372cb00a68081e4dd705044d51644cbc96f208d7c426e050921805ea72a3";
    vector.operator_equivalence_verified = true;
    vector.exact_reconstruction_verified = true;
    vector.producer_receipt_identity = "sha256:25c300605c92bfc4609848e2b0dda60e7c8fb852ae11e859855b76126bb9ad9f";
    require(vector.canonical_core_json() == qsa_core, "QSA canonical JSON differs from PR163");
    vector.identity = fixture_sha(vector.canonical_core_json());
    require(vector.identity == qsa_hash, "QSA signing vector differs from PR163");
    vector.validate(fixture_sha);

    const auto wrong = qubit::Qwen35LctCompiler::compile(qubit::Qwen35LctRole::SsmRecurrent);
    rejected = false;
    try { (void)qubit::qwen35_finalize_translation(wrong, input, evidence, fixture_sha); }
    catch (const qubit::QStateError&) { rejected = true; }
    require(rejected, "cross-role Qwen35 translation did not fail closed");
}

}  // namespace

int main() {
    source_contract();
    operator_contract();
    fragment_contract();
    std::cout << "Qwen35 LCT source-bound tests passed\n";
    return 0;
}
