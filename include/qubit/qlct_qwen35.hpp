#pragma once

#include "qubit/qstate.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace qubit {

inline constexpr std::string_view Qwen35LctQwenSchema = "qelm.lct-qwen-source-fragment.v1";
inline constexpr std::string_view Qwen35LctQsaSchema = "qelm.lct-qsa-translation-fragment.v1";
inline constexpr std::string_view Qwen35LlamaCommit = "74ade52741203e5c8f81eaf06a96cb1cfe15f2a3";
inline constexpr std::string_view Qwen35SourceBlob = "6783d98ec204885caba96184fc5b9c6bd47e071b";
inline constexpr std::string_view Qwen35GraphBlob = "68c9e606c3e390c5096a99df0d5dd44df2a1996c";
inline constexpr std::string_view Qwen35DeltaNetBlob = "ad9ce77140840c2c3d631f1c27685b95bc200e94";
inline constexpr std::string_view Qwen35DonorIdentity =
    "sha256:65b753ea835627f7b511143c6ceb976525c7f21f5df8c664bc0a9c23d1c49921";
inline constexpr std::string_view Qwen35TopologyIdentity =
    "sha256:e2b88b52a3f7ecbb0aed0d88c48e1af0e212bc7cdead4ea11c858fe3d718d108";

using Qwen35Sha256 = std::string (*)(std::string_view);

enum class Qwen35LctRole : std::uint8_t { SsmRecurrent, FullAttention, TokenizerEmbedding };

struct Qwen35RoleTopology {
    int block_index;
    std::size_t source_bytes;
    std::size_t component_count;
    std::size_t family_instances;
    std::size_t family_component_count;
};

[[nodiscard]] inline constexpr std::string_view qwen35_role_name(Qwen35LctRole role) noexcept {
    switch (role) {
        case Qwen35LctRole::SsmRecurrent: return "ssm_recurrent";
        case Qwen35LctRole::FullAttention: return "full_attention";
        case Qwen35LctRole::TokenizerEmbedding: return "tokenizer_embedding";
    }
    return "unknown";
}

[[nodiscard]] inline constexpr Qwen35RoleTopology qwen35_role_topology(Qwen35LctRole role) noexcept {
    switch (role) {
        case Qwen35LctRole::SsmRecurrent: return {0, 273685376U, 14U, 48U, 672U};
        case Qwen35LctRole::FullAttention: return {3, 261859328U, 11U, 16U, 176U};
        case Qwen35LctRole::TokenizerEmbedding: return {-1, 1758126080U, 3U, 1U, 3U};
    }
    return {-1, 0U, 0U, 0U, 0U};
}

namespace qwen35_lct_detail {

[[nodiscard]] inline bool hex64(std::string_view value) noexcept {
    if (value.size() != 64U) return false;
    for (const char ch : value) {
        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) return false;
    }
    return true;
}

[[nodiscard]] inline bool address(std::string_view value) noexcept {
    return value.size() == 71U && value.substr(0U, 7U) == "sha256:" && hex64(value.substr(7U));
}

[[nodiscard]] inline std::string digest(Qwen35Sha256 sha256, std::string_view value) {
    if (sha256 == nullptr) throw QStateError("Qwen35 LCT requires a SHA-256 provider");
    std::string result = sha256(value);
    if (!hex64(result)) throw QStateError("Qwen35 LCT SHA-256 provider returned an invalid digest");
    return result;
}

[[nodiscard]] inline std::string content_address(Qwen35Sha256 sha256, std::string_view value) {
    return "sha256:" + digest(sha256, value);
}

[[nodiscard]] inline std::string json_string(std::string_view value) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(value.size() + 2U);
    out.push_back('"');
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (ch < 0x20U) {
                    out += "\\u00";
                    out.push_back(hex[(ch >> 4U) & 0x0fU]);
                    out.push_back(hex[ch & 0x0fU]);
                } else out.push_back(static_cast<char>(ch));
        }
    }
    out.push_back('"');
    return out;
}

inline void token(std::string& out, std::string_view value) {
    out += std::to_string(value.size()); out.push_back(':'); out.append(value);
}

}  // namespace qwen35_lct_detail

struct Qwen35SourceFragment {
    std::string schema{std::string(Qwen35LctQwenSchema)};
    Qwen35LctRole role{Qwen35LctRole::SsmRecurrent};
    std::string donor_identity, topology_identity, source_identity, component_manifest_identity;
    std::size_t source_bytes{0U}, component_count{0U};
    std::string context_identity, activation_receipt_identity, producer_receipt_identity, identity;

    [[nodiscard]] std::string canonical_core_json() const {
        using qwen35_lct_detail::json_string;
        return std::string{"{"} +
            "\"activation_receipt_identity\":" + json_string(activation_receipt_identity) +
            ",\"component_count\":" + std::to_string(component_count) +
            ",\"component_manifest_identity\":" + json_string(component_manifest_identity) +
            ",\"context_identity\":" + json_string(context_identity) +
            ",\"donor_identity\":" + json_string(donor_identity) +
            ",\"producer_receipt_identity\":" + json_string(producer_receipt_identity) +
            ",\"role\":" + json_string(qwen35_role_name(role)) +
            ",\"schema\":" + json_string(schema) +
            ",\"source_bytes\":" + std::to_string(source_bytes) +
            ",\"source_identity\":" + json_string(source_identity) +
            ",\"topology_identity\":" + json_string(topology_identity) + "}";
    }

    void validate(Qwen35Sha256 sha256) const {
        const auto expected = qwen35_role_topology(role);
        if (schema != Qwen35LctQwenSchema || donor_identity != Qwen35DonorIdentity ||
            topology_identity != Qwen35TopologyIdentity || source_bytes != expected.source_bytes ||
            component_count != expected.component_count) {
            throw QStateError("Qwen35 LCT source authority/topology mismatch");
        }
        using qwen35_lct_detail::address;
        if (!address(source_identity) || !address(component_manifest_identity) || !address(context_identity) ||
            !address(activation_receipt_identity) || !address(producer_receipt_identity)) {
            throw QStateError("Qwen35 LCT source fragment has an invalid content address");
        }
        if (!qwen35_lct_detail::hex64(identity) ||
            identity != qwen35_lct_detail::digest(sha256, canonical_core_json())) {
            throw QStateError("Qwen35 LCT source fragment signature mismatch");
        }
    }
};

struct Qwen35StateTransition { std::string_view state, before, update, after; };
struct Qwen35ResidualBinding { std::string_view output, left, right; };

struct Qwen35OperatorProgram {
    Qwen35LctRole role{Qwen35LctRole::SsmRecurrent};
    std::string_view source_trace{};
    std::size_t step_count{0U};
    std::vector<Qwen35StateTransition> states{};
    std::vector<Qwen35ResidualBinding> residuals{};

    [[nodiscard]] std::string canonical(const Qwen35SourceFragment& source, Qwen35Sha256 sha256) const {
        source.validate(sha256);
        if (source.role != role) throw QStateError("Qwen35 LCT CIR/source role mismatch");
        std::string out;
        out.reserve(source_trace.size() + 1024U);
        using qwen35_lct_detail::token;
        token(out, "qsa.qwen35-lct-cir.v1"); token(out, Qwen35LlamaCommit);
        token(out, Qwen35SourceBlob); token(out, Qwen35GraphBlob); token(out, Qwen35DeltaNetBlob);
        token(out, qwen35_role_name(role)); token(out, source.source_identity);
        token(out, source.component_manifest_identity); token(out, source.context_identity);
        token(out, std::to_string(step_count)); token(out, source_trace);
        token(out, std::to_string(states.size()));
        for (const auto& state : states) {
            token(out, state.state); token(out, state.before); token(out, state.update); token(out, state.after);
        }
        token(out, std::to_string(residuals.size()));
        for (const auto& residual : residuals) {
            token(out, residual.output); token(out, residual.left); token(out, residual.right);
        }
        return out;
    }

    [[nodiscard]] std::string cir_identity(const Qwen35SourceFragment& source, Qwen35Sha256 sha256) const {
        return qwen35_lct_detail::content_address(sha256, canonical(source, sha256));
    }
};

class Qwen35LctCompiler {
public:
    [[nodiscard]] static Qwen35OperatorProgram compile(Qwen35LctRole role) {
        Qwen35OperatorProgram p;
        p.role = role;
        if (role == Qwen35LctRole::FullAttention) {
            p.step_count = 25U;
            p.source_trace =
                "attn_norm=rms(input,attn_norm);QG=mm(wq,attn_norm);Q=view_first_half(QG);Q=rms(Q,q_norm);"
                "K=mm(wk,attn_norm);V=mm(wv,attn_norm);K=reshape(K);K=rms(K,k_norm);gate=contiguous_second_half(QG);"
                "V=reshape(V);Q=mrope(Q,pos,sections[4]);K=mrope(K,pos,sections[4]);"
                "scale=f_attention_scale==0?1/sqrt(n_embd_head):f_attention_scale;A=attention(Q,K,V,scale);"
                "A=A*sigmoid(gate);attention_output=mm(wo,A);attn_residual=attention_output+input;"
                "post_norm=rms(attn_residual,attn_post_norm);up=mm(ffn_up,post_norm);gate2=mm(ffn_gate,post_norm);"
                "gate2=silu(gate2);ffn=gate2*up;ffn_out=mm(ffn_down,ffn);post_ffn=ffn_out+attn_residual;l_out=build_cvec(post_ffn);";
        } else if (role == Qwen35LctRole::SsmRecurrent) {
            p.step_count = 40U;
            p.source_trace =
                "attn_norm=rms(input,attn_norm);qkv=reshape(mm(wqkv,attn_norm));z=mm(wqkv_gate,attn_norm);"
                "beta=reshape(mm(ssm_beta,attn_norm));beta=sigmoid(beta);alpha=reshape(mm(ssm_alpha,attn_norm));"
                "alpha=softplus(alpha+ssm_dt);g=reshape(alpha*ssm_a);conv_before=mctx.get_r_l(layer);ssm_before=mctx.get_s_l(layer);"
                "conv_input=build_conv_state(conv_before,qkv,ssm_conv1d);state_before=reshape(build_rs(ssm_before));"
                "conv=silu(ssm_conv(conv_input,ssm_conv1d));q=view_q(conv);k=view_k(conv);v=view_v(conv);"
                "q=l2_norm(q,eps);k=l2_norm(k,eps);if(num_k_heads!=num_v_heads&&(!fused_gdn_ar||!fused_gdn_ch))repeat_qk();"
                "output,state_after=build_recurrent_attn(q,k,v,g,beta,state_before);z=reshape(z);"
                "out=rms(output,ssm_norm)*silu(z);out=reshape(out);attention_output=reshape(mm(ssm_out,out));"
                "attn_residual=attention_output+input;post_norm=rms(attn_residual,attn_post_norm);up=mm(ffn_up,post_norm);"
                "gate2=mm(ffn_gate,post_norm);gate2=silu(gate2);ffn=gate2*up;ffn_out=mm(ffn_down,ffn);"
                "post_ffn=ffn_out+attn_residual;l_out=build_cvec(post_ffn);";
            p.states.push_back({"conv_state", "conv_before", "build_conv_state(qkv,ssm_conv1d)", "conv_after"});
            p.states.push_back({"ssm_state", "state_before", "build_recurrent_attn(q,k,v,g,beta)", "state_after"});
        } else {
            p.step_count = 4U;
            p.source_trace =
                "input=build_inp_embd(token_embd);result_norm=rms(final_hidden,output_norm);"
                "if(!embeddings_nextn_masked&&inp_out_ids)result_norm=get_rows(result_norm,inp_out_ids);"
                "logits=mm(output??token_embd,result_norm);output duplicates token_embd when absent;";
        }
        if (role != Qwen35LctRole::TokenizerEmbedding) {
            p.residuals.push_back({"attn_residual", "input", "attention_output"});
            p.residuals.push_back({"post_ffn", "attn_residual", "ffn_out"});
        }
        return p;
    }
};

struct Qwen35TranslationEvidence {
    std::string residual_identity, decoder_identity, operator_receipt_root;
    std::string reconstruction_identity, producer_receipt_identity;
    std::size_t operator_mismatch_count{0U}, reconstruction_mismatch_count{0U};
    bool operator_equivalence_verified{false}, exact_reconstruction_verified{false};

    void validate() const {
        using qwen35_lct_detail::address;
        if (!address(residual_identity) || !address(decoder_identity) || !address(operator_receipt_root) ||
            !address(reconstruction_identity) || !address(producer_receipt_identity)) {
            throw QStateError("Qwen35 LCT verification receipt has an invalid content address");
        }
        if (operator_mismatch_count != 0U || reconstruction_mismatch_count != 0U ||
            !operator_equivalence_verified || !exact_reconstruction_verified) {
            throw QStateError("Qwen35 LCT exact equivalence gates are not closed");
        }
    }
};

struct Qwen35QsaTranslationFragment {
    std::string schema{std::string(Qwen35LctQsaSchema)};
    Qwen35LctRole role{Qwen35LctRole::SsmRecurrent};
    std::string source_identity, component_manifest_identity, residual_identity, cir_identity;
    std::string decoder_identity, operator_receipt_root, reconstruction_identity;
    std::size_t reconstruction_mismatch_count{0U};
    bool operator_equivalence_verified{false}, exact_reconstruction_verified{false};
    std::string producer_receipt_identity, identity;

    [[nodiscard]] std::string canonical_core_json() const {
        using qwen35_lct_detail::json_string;
        return std::string{"{"} +
            "\"cir_identity\":" + json_string(cir_identity) +
            ",\"component_manifest_identity\":" + json_string(component_manifest_identity) +
            ",\"decoder_identity\":" + json_string(decoder_identity) +
            ",\"exact_reconstruction_verified\":" + (exact_reconstruction_verified ? "true" : "false") +
            ",\"operator_equivalence_verified\":" + (operator_equivalence_verified ? "true" : "false") +
            ",\"operator_receipt_root\":" + json_string(operator_receipt_root) +
            ",\"producer_receipt_identity\":" + json_string(producer_receipt_identity) +
            ",\"reconstruction_identity\":" + json_string(reconstruction_identity) +
            ",\"reconstruction_mismatch_count\":" + std::to_string(reconstruction_mismatch_count) +
            ",\"residual_identity\":" + json_string(residual_identity) +
            ",\"role\":" + json_string(qwen35_role_name(role)) +
            ",\"schema\":" + json_string(schema) +
            ",\"source_identity\":" + json_string(source_identity) + "}";
    }

    [[nodiscard]] std::string to_json() const {
        using qwen35_lct_detail::json_string;
        std::string core = canonical_core_json();
        core.pop_back();
        return core + ",\"identity\":" + json_string(identity) + "}";
    }

    void validate(Qwen35Sha256 sha256) const {
        using qwen35_lct_detail::address;
        if (schema != Qwen35LctQsaSchema || !address(source_identity) || !address(component_manifest_identity) ||
            !address(residual_identity) || !address(cir_identity) || !address(decoder_identity) ||
            !address(operator_receipt_root) || !address(reconstruction_identity) || !address(producer_receipt_identity) ||
            reconstruction_mismatch_count != 0U || !operator_equivalence_verified || !exact_reconstruction_verified) {
            throw QStateError("Qwen35 LCT QSA fragment does not match the frozen equivalence contract");
        }
        if (!qwen35_lct_detail::hex64(identity) ||
            identity != qwen35_lct_detail::digest(sha256, canonical_core_json())) {
            throw QStateError("Qwen35 LCT QSA fragment signature mismatch");
        }
    }
};

[[nodiscard]] inline Qwen35QsaTranslationFragment qwen35_finalize_translation(
    const Qwen35OperatorProgram& program, const Qwen35SourceFragment& source,
    const Qwen35TranslationEvidence& evidence, Qwen35Sha256 sha256) {
    source.validate(sha256); evidence.validate();
    if (program.role != source.role) throw QStateError("Qwen35 LCT operator/source role mismatch");
    Qwen35QsaTranslationFragment out;
    out.role = source.role; out.source_identity = source.source_identity;
    out.component_manifest_identity = source.component_manifest_identity;
    out.residual_identity = evidence.residual_identity; out.cir_identity = program.cir_identity(source, sha256);
    out.decoder_identity = evidence.decoder_identity; out.operator_receipt_root = evidence.operator_receipt_root;
    out.reconstruction_identity = evidence.reconstruction_identity;
    out.reconstruction_mismatch_count = evidence.reconstruction_mismatch_count;
    out.operator_equivalence_verified = evidence.operator_equivalence_verified;
    out.exact_reconstruction_verified = evidence.exact_reconstruction_verified;
    out.producer_receipt_identity = evidence.producer_receipt_identity;
    out.identity = qwen35_lct_detail::digest(sha256, out.canonical_core_json());
    out.validate(sha256);
    return out;
}

}  // namespace qubit
