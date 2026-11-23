#pragma once

#include "qubit/qpolynomial.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qubit {

enum class QpkVerificationStatus : std::uint8_t {
    ProvenExact,
    Unverified,
    Rejected,
};

enum class QpkFailureReason : std::uint8_t {
    None,
    Unsupported,
    StaleInput,
    ResourceLimit,
    MalformedCertificate,
    InvalidProof,
};

enum class QpkClaimKind : std::uint8_t {
    RationalPolynomialIdentity,
    RationalPolynomialDivision,
};

struct QpkLimits {
    std::size_t max_certificate_bytes{64U * 1024U};
    std::size_t max_source_bytes{1U * 1024U * 1024U};
    std::size_t max_polynomial_terms{129U};
    std::size_t max_rational_bytes{4096U};
    std::size_t max_variable_bytes{128U};
};

struct QpkCheckResult {
    QpkVerificationStatus status{QpkVerificationStatus::Unverified};
    QpkFailureReason reason{QpkFailureReason::None};
    QpkClaimKind claim_kind{QpkClaimKind::RationalPolynomialDivision};
    std::string claim_id{"polynomial-division-identity-and-remainder-degree-v1"};
    std::string source_id{};
    std::string theory_id{"Q[x]"};
    std::string assumptions_id{"univariate polynomials over exact rationals"};
    std::string failure_reason{};
    std::size_t checked_steps{0U};
    std::size_t certificate_bytes{0U};

    [[nodiscard]] bool accepted() const noexcept {
        return status == QpkVerificationStatus::ProvenExact;
    }
};

namespace qpk_detail {

inline constexpr std::string_view DivisionRule =
    "polynomial-division-identity-and-remainder-degree-v1";
inline constexpr std::string_view IdentityRule = "polynomial-identity-coefficientwise-v1";
inline constexpr std::string_view Assumptions = "exact-rational-univariate-polynomials";
inline constexpr std::string_view Generator = "QSA-QPK-v1";
inline constexpr std::string_view NoDependencies = "none";

class ResourceLimitError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class Sha256 {
public:
    void update(std::string_view input) {
        update(input.data(), input.size());
    }

    void update(const void* data, std::size_t size) {
        if (size > (std::numeric_limits<std::uint64_t>::max() - bit_count_) / 8U) {
            throw QMathError("QPK SHA-256 input is too large");
        }
        bit_count_ += static_cast<std::uint64_t>(size) * 8U;
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        while (size != 0U) {
            const std::size_t copied = std::min(size, block_.size() - block_size_);
            std::copy_n(bytes, copied, block_.begin() + static_cast<std::ptrdiff_t>(block_size_));
            block_size_ += copied;
            bytes += copied;
            size -= copied;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0U;
            }
        }
    }

    [[nodiscard]] std::string finish() {
        const std::uint64_t message_bits = bit_count_;
        const std::uint8_t marker = 0x80U;
        update_padding(&marker, 1U);
        const std::uint8_t zero = 0U;
        while (block_size_ != 56U) update_padding(&zero, 1U);
        std::array<std::uint8_t, 8U> length{};
        for (std::size_t i = 0U; i < length.size(); ++i) {
            length[7U - i] = static_cast<std::uint8_t>((message_bits >> (i * 8U)) & 0xFFU);
        }
        update_padding(length.data(), length.size());

        constexpr char digits[] = "0123456789abcdef";
        std::string result(64U, '0');
        for (std::size_t i = 0U; i < state_.size(); ++i) {
            for (std::size_t byte = 0U; byte < 4U; ++byte) {
                const std::uint8_t value = static_cast<std::uint8_t>(
                    (state_[i] >> ((3U - byte) * 8U)) & 0xFFU);
                const std::size_t offset = (i * 4U + byte) * 2U;
                result[offset] = digits[value >> 4U];
                result[offset + 1U] = digits[value & 0x0FU];
            }
        }
        return result;
    }

private:
    static constexpr std::array<std::uint32_t, 64U> constants_{
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    };

    void reset() noexcept {
        state_ = {
            0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
            0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
        };
        block_.fill(0U);
        block_size_ = 0U;
        bit_count_ = 0U;
    }

    void update_padding(const void* data, std::size_t size) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        while (size != 0U) {
            const std::size_t copied = std::min(size, block_.size() - block_size_);
            std::copy_n(bytes, copied, block_.begin() + static_cast<std::ptrdiff_t>(block_size_));
            block_size_ += copied;
            bytes += copied;
            size -= copied;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0U;
            }
        }
    }

    void transform(const std::uint8_t* block) noexcept {
        std::array<std::uint32_t, 64U> words{};
        for (std::size_t i = 0U; i < 16U; ++i) {
            const std::size_t offset = i * 4U;
            words[i] = (static_cast<std::uint32_t>(block[offset]) << 24U) |
                       (static_cast<std::uint32_t>(block[offset + 1U]) << 16U) |
                       (static_cast<std::uint32_t>(block[offset + 2U]) << 8U) |
                       static_cast<std::uint32_t>(block[offset + 3U]);
        }
        for (std::size_t i = 16U; i < words.size(); ++i) {
            const std::uint32_t s0 = std::rotr(words[i - 15U], 7) ^
                                     std::rotr(words[i - 15U], 18) ^
                                     (words[i - 15U] >> 3U);
            const std::uint32_t s1 = std::rotr(words[i - 2U], 17) ^
                                     std::rotr(words[i - 2U], 19) ^
                                     (words[i - 2U] >> 10U);
            words[i] = words[i - 16U] + s0 + words[i - 7U] + s1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (std::size_t i = 0U; i < words.size(); ++i) {
            const std::uint32_t sum1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
            const std::uint32_t choice = (e & f) ^ (~e & g);
            const std::uint32_t temp1 = h + sum1 + choice + constants_[i] + words[i];
            const std::uint32_t sum0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8U> state_{
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    std::array<std::uint8_t, 64U> block_{};
    std::size_t block_size_{0U};
    std::uint64_t bit_count_{0U};
};

inline std::string sha256_hex(std::string_view input) {
    Sha256 hash;
    hash.update(input);
    return hash.finish();
}

inline void append_framed(std::string& output, std::string_view value) {
    output += std::to_string(value.size());
    output.push_back(':');
    output.append(value);
    output.push_back('\n');
}

inline std::string coefficients_wire(const QPolynomial& polynomial, const QpkLimits& limits) {
    const auto& coefficients = polynomial.coefficients();
    if (coefficients.size() > limits.max_polynomial_terms) {
        throw ResourceLimitError("QPK polynomial term limit exceeded");
    }
    if (polynomial.variable().size() > limits.max_variable_bytes) {
        throw ResourceLimitError("QPK variable identity limit exceeded");
    }
    if (coefficients.empty()) return "0";

    std::string result;
    for (std::size_t index = 0U; index < coefficients.size(); ++index) {
        if (index != 0U) result.push_back(',');
        const std::string value = coefficients[index].canonical();
        if (value.size() > limits.max_rational_bytes) {
            throw ResourceLimitError("QPK rational coefficient limit exceeded");
        }
        result += value;
    }
    return result;
}

inline std::string source_digest(
    const QPolynomial& dividend,
    const QPolynomial& divisor,
    const QpkLimits& limits) {
    if (dividend.variable() != divisor.variable()) {
        throw QMathError("QPK source variables differ");
    }
    std::string canonical_source;
    append_framed(canonical_source, "QPK-SOURCE-v1");
    append_framed(canonical_source, dividend.variable());
    append_framed(canonical_source, coefficients_wire(dividend, limits));
    append_framed(canonical_source, coefficients_wire(divisor, limits));
    if (canonical_source.size() > limits.max_source_bytes) {
        throw ResourceLimitError("QPK source byte limit exceeded");
    }
    return sha256_hex(canonical_source);
}

inline std::string identity_source_digest(
    const QPolynomial& left,
    const QPolynomial& right,
    const QpkLimits& limits) {
    std::string canonical_source;
    append_framed(canonical_source, "QPK-IDENTITY-SOURCE-v1");
    append_framed(canonical_source, left.variable());
    append_framed(canonical_source, coefficients_wire(left, limits));
    append_framed(canonical_source, right.variable());
    append_framed(canonical_source, coefficients_wire(right, limits));
    if (canonical_source.size() > limits.max_source_bytes) {
        throw ResourceLimitError("QPK source byte limit exceeded");
    }
    return sha256_hex(canonical_source);
}

inline std::string hex_encode(std::string_view input) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(input.size() * 2U);
    for (const char raw : input) {
        const auto value = static_cast<unsigned char>(raw);
        result.push_back(digits[value >> 4U]);
        result.push_back(digits[value & 0x0FU]);
    }
    return result;
}

inline std::string hex_decode(std::string_view input, std::size_t max_bytes) {
    if ((input.size() % 2U) != 0U || input.size() / 2U > max_bytes) {
        if ((input.size() % 2U) == 0U) {
            throw ResourceLimitError("QPK variable identity limit exceeded");
        }
        throw QMathError("QPK variable encoding is malformed");
    }
    const auto nibble = [](char value) -> unsigned {
        if (value >= '0' && value <= '9') return static_cast<unsigned>(value - '0');
        if (value >= 'a' && value <= 'f') return static_cast<unsigned>(value - 'a') + 10U;
        throw QMathError("QPK hex encoding is not canonical lowercase");
    };
    std::string result;
    result.reserve(input.size() / 2U);
    for (std::size_t index = 0U; index < input.size(); index += 2U) {
        const unsigned high = nibble(input[index]);
        const unsigned low = nibble(input[index + 1U]);
        result.push_back(static_cast<char>((high << 4U) | low));
    }
    if (result.empty() || hex_encode(result) != input) {
        throw QMathError("QPK variable identity is not canonical");
    }
    return result;
}

inline bool valid_metadata_token(std::string_view value) noexcept {
    if (value.empty() || value.size() > 64U) return false;
    for (const char item : value) {
        const bool allowed =
            (item >= 'a' && item <= 'z') || (item >= 'A' && item <= 'Z') ||
            (item >= '0' && item <= '9') || item == '-' || item == '_' || item == '.';
        if (!allowed) return false;
    }
    return true;
}

inline bool valid_digest(std::string_view value) noexcept {
    if (value.size() != 64U) return false;
    for (const char digit : value) {
        if (!((digit >= '0' && digit <= '9') || (digit >= 'a' && digit <= 'f'))) return false;
    }
    return true;
}

inline std::vector<QRational> parse_coefficients(
    std::string_view wire,
    const QpkLimits& limits) {
    if (wire.empty()) throw QMathError("QPK coefficient list is empty");
    std::vector<QRational> result;
    std::size_t start = 0U;
    while (start <= wire.size()) {
        const std::size_t comma = wire.find(',', start);
        const std::size_t end = comma == std::string_view::npos ? wire.size() : comma;
        const std::string_view token = wire.substr(start, end - start);
        if (token.empty() || token.size() > limits.max_rational_bytes) {
            if (token.size() > limits.max_rational_bytes) {
                throw ResourceLimitError("QPK rational coefficient limit exceeded");
            }
            throw QMathError("QPK rational coefficient is empty or oversized");
        }
        if (result.size() >= limits.max_polynomial_terms) {
            throw ResourceLimitError("QPK polynomial term limit exceeded");
        }
        const QRational value = QRational::parse(token);
        if (value.canonical() != token) {
            throw QMathError("QPK rational coefficient is not canonical");
        }
        result.push_back(value);
        if (comma == std::string_view::npos) break;
        start = comma + 1U;
    }

    if (result.size() == 1U && result.front().is_zero()) {
        result.clear();
    } else if (result.back().is_zero()) {
        throw QMathError("QPK polynomial has a noncanonical trailing zero");
    }
    return result;
}

inline std::string certificate_body(
    std::string_view variable,
    std::string_view source_id,
    std::string_view quotient,
    std::string_view remainder,
    std::string_view generator = Generator) {
    std::string body;
    body += "QPK-CERT-v1\n";
    body += "claim=polynomial-division\n";
    body += "theory=Q[x]\n";
    body += "assumptions=";
    body += Assumptions;
    body += '\n';
    body += "rule=";
    body += DivisionRule;
    body += "\ngenerator=";
    body += generator;
    body += "\ndependencies=";
    body += NoDependencies;
    body += "\nvariable_hex=";
    body += hex_encode(variable);
    body += "\nsource_sha256=";
    body += source_id;
    body += "\nquotient=";
    body += quotient;
    body += "\nremainder=";
    body += remainder;
    body += '\n';
    return body;
}

inline std::string identity_certificate_body(
    std::string_view left_variable,
    std::string_view right_variable,
    std::string_view source_id,
    std::string_view generator = Generator) {
    std::string body;
    body += "QPK-CERT-v1\n";
    body += "claim=polynomial-identity\n";
    body += "theory=Q[x]\n";
    body += "assumptions=";
    body += Assumptions;
    body += '\n';
    body += "rule=";
    body += IdentityRule;
    body += "\ngenerator=";
    body += generator;
    body += "\ndependencies=";
    body += NoDependencies;
    body += "\nleft_variable_hex=";
    body += hex_encode(left_variable);
    body += "\nright_variable_hex=";
    body += hex_encode(right_variable);
    body += "\nsource_sha256=";
    body += source_id;
    body += '\n';
    return body;
}

inline std::string seal_certificate_body(std::string body) {
    const std::string digest = sha256_hex(body);
    body += "certificate_sha256=";
    body += digest;
    body += '\n';
    return body;
}

inline std::vector<std::string_view> split_lines(std::string_view input) {
    std::vector<std::string_view> lines;
    std::size_t start = 0U;
    while (start < input.size()) {
        const std::size_t newline = input.find('\n', start);
        if (newline == std::string_view::npos) {
            throw QMathError("QPK certificate must end with a newline");
        }
        lines.push_back(input.substr(start, newline - start));
        start = newline + 1U;
    }
    return lines;
}

inline std::string_view field_value(std::string_view line, std::string_view key) {
    if (!line.starts_with(key)) throw QMathError("QPK certificate fields are out of order");
    return line.substr(key.size());
}

inline QpkCheckResult rejected(
    QpkFailureReason reason,
    std::string message,
    std::size_t certificate_bytes) {
    QpkCheckResult result;
    result.status = QpkVerificationStatus::Rejected;
    result.reason = reason;
    result.failure_reason = std::move(message);
    result.certificate_bytes = certificate_bytes;
    return result;
}

inline QpkCheckResult unsupported(std::string message, std::size_t certificate_bytes) {
    QpkCheckResult result;
    result.status = QpkVerificationStatus::Unverified;
    result.reason = QpkFailureReason::Unsupported;
    result.failure_reason = std::move(message);
    result.certificate_bytes = certificate_bytes;
    return result;
}

inline QpkCheckResult identity_result(QpkCheckResult result) {
    result.claim_kind = QpkClaimKind::RationalPolynomialIdentity;
    result.claim_id = "polynomial-identity-coefficientwise-v1";
    return result;
}

}  // namespace qpk_detail

inline std::string encode_polynomial_identity_certificate(
    const QPolynomial& left,
    const QPolynomial& right,
    const QpkLimits& limits = {}) {
    const std::string source_id = qpk_detail::identity_source_digest(left, right, limits);
    const std::string body = qpk_detail::identity_certificate_body(
        left.variable(), right.variable(), source_id);
    const std::string certificate = qpk_detail::seal_certificate_body(body);
    if (certificate.size() > limits.max_certificate_bytes) {
        throw qpk_detail::ResourceLimitError("QPK certificate byte limit exceeded");
    }
    return certificate;
}

inline QpkCheckResult verify_polynomial_identity(
    const QPolynomial& trusted_left,
    const QPolynomial& trusted_right,
    std::string_view certificate,
    const QpkLimits& limits = {}) {
    const std::size_t certificate_bytes = certificate.size();
    if (certificate_bytes > limits.max_certificate_bytes) {
        return qpk_detail::identity_result(qpk_detail::rejected(
            QpkFailureReason::ResourceLimit, "QPK certificate byte limit exceeded", certificate_bytes));
    }
    if (certificate.empty() || certificate.find('\r') != std::string_view::npos ||
        certificate.find('\0') != std::string_view::npos) {
        return qpk_detail::identity_result(qpk_detail::rejected(
            QpkFailureReason::MalformedCertificate, "QPK certificate encoding is not canonical", certificate_bytes));
    }

    try {
        const auto lines = qpk_detail::split_lines(certificate);
        if (lines.size() != 11U) {
            return qpk_detail::identity_result(qpk_detail::rejected(
                QpkFailureReason::MalformedCertificate, "QPK identity certificate has the wrong field count", certificate_bytes));
        }
        if (lines[0] != "QPK-CERT-v1") {
            return qpk_detail::identity_result(qpk_detail::unsupported(
                "QPK certificate version is unsupported", certificate_bytes));
        }
        const std::string_view claim = qpk_detail::field_value(lines[1], "claim=");
        const std::string_view theory = qpk_detail::field_value(lines[2], "theory=");
        const std::string_view assumptions = qpk_detail::field_value(lines[3], "assumptions=");
        const std::string_view rule = qpk_detail::field_value(lines[4], "rule=");
        const std::string_view generator = qpk_detail::field_value(lines[5], "generator=");
        const std::string_view dependencies = qpk_detail::field_value(lines[6], "dependencies=");
        if (claim != "polynomial-identity" || theory != "Q[x]" ||
            assumptions != qpk_detail::Assumptions || rule != qpk_detail::IdentityRule ||
            dependencies != qpk_detail::NoDependencies) {
            return qpk_detail::identity_result(qpk_detail::unsupported(
                "QPK claim, theory, or rule is unsupported", certificate_bytes));
        }
        if (!qpk_detail::valid_metadata_token(generator)) {
            return qpk_detail::identity_result(qpk_detail::rejected(
                QpkFailureReason::MalformedCertificate, "QPK generator metadata is malformed", certificate_bytes));
        }

        const std::string left_variable = qpk_detail::hex_decode(
            qpk_detail::field_value(lines[7], "left_variable_hex="), limits.max_variable_bytes);
        const std::string right_variable = qpk_detail::hex_decode(
            qpk_detail::field_value(lines[8], "right_variable_hex="), limits.max_variable_bytes);
        const std::string_view source_id = qpk_detail::field_value(lines[9], "source_sha256=");
        const std::string_view certificate_id = qpk_detail::field_value(lines[10], "certificate_sha256=");
        if (!qpk_detail::valid_digest(source_id) || !qpk_detail::valid_digest(certificate_id)) {
            return qpk_detail::identity_result(qpk_detail::rejected(
                QpkFailureReason::MalformedCertificate, "QPK digest encoding is malformed", certificate_bytes));
        }

        std::size_t body_bytes = 0U;
        for (std::size_t index = 0U; index < 10U; ++index) {
            body_bytes += lines[index].size() + 1U;
        }
        const std::string_view body = certificate.substr(0U, body_bytes);
        if (qpk_detail::sha256_hex(body) != certificate_id) {
            return qpk_detail::identity_result(qpk_detail::rejected(
                QpkFailureReason::MalformedCertificate, "QPK certificate digest does not match its contents", certificate_bytes));
        }
        const std::string canonical_body = qpk_detail::identity_certificate_body(
            left_variable, right_variable, source_id, generator);
        if (body != canonical_body) {
            return qpk_detail::identity_result(qpk_detail::rejected(
                QpkFailureReason::MalformedCertificate, "QPK certificate is not in canonical form", certificate_bytes));
        }

        const std::string expected_source_id =
            qpk_detail::identity_source_digest(trusted_left, trusted_right, limits);
        if (source_id != expected_source_id ||
            left_variable != trusted_left.variable() ||
            right_variable != trusted_right.variable()) {
            QpkCheckResult result = qpk_detail::rejected(
                QpkFailureReason::StaleInput, "QPK certificate is bound to different source polynomials", certificate_bytes);
            result.source_id.assign(source_id);
            return qpk_detail::identity_result(std::move(result));
        }
        if (trusted_left.variable() != trusted_right.variable()) {
            return qpk_detail::identity_result(qpk_detail::rejected(
                QpkFailureReason::InvalidProof, "polynomials use different variable identities", certificate_bytes));
        }

        const auto& left_coefficients = trusted_left.coefficients();
        const auto& right_coefficients = trusted_right.coefficients();
        const std::size_t terms = std::max(left_coefficients.size(), right_coefficients.size());
        for (std::size_t index = 0U; index < terms; ++index) {
            const QRational left_value = index < left_coefficients.size()
                ? left_coefficients[index]
                : QRational(0);
            const QRational right_value = index < right_coefficients.size()
                ? right_coefficients[index]
                : QRational(0);
            if (left_value != right_value) {
                return qpk_detail::identity_result(qpk_detail::rejected(
                    QpkFailureReason::InvalidProof, "polynomial coefficients are not identical", certificate_bytes));
            }
        }

        QpkCheckResult result;
        result.status = QpkVerificationStatus::ProvenExact;
        result.reason = QpkFailureReason::None;
        result.source_id.assign(source_id);
        result.checked_steps = terms + 2U;
        result.certificate_bytes = certificate_bytes;
        return qpk_detail::identity_result(std::move(result));
    } catch (const qpk_detail::ResourceLimitError& error) {
        return qpk_detail::identity_result(qpk_detail::rejected(
            QpkFailureReason::ResourceLimit, error.what(), certificate_bytes));
    } catch (const std::exception& error) {
        return qpk_detail::identity_result(qpk_detail::rejected(
            QpkFailureReason::MalformedCertificate, error.what(), certificate_bytes));
    } catch (...) {
        return qpk_detail::identity_result(qpk_detail::rejected(
            QpkFailureReason::MalformedCertificate, "QPK verification failed", certificate_bytes));
    }
}

inline std::string prove_polynomial_identity(
    const QPolynomial& left,
    const QPolynomial& right,
    const QpkLimits& limits = {}) {
    const std::string certificate = encode_polynomial_identity_certificate(left, right, limits);
    const QpkCheckResult checked = verify_polynomial_identity(left, right, certificate, limits);
    if (!checked.accepted()) {
        throw QMathError("QPK rejected a polynomial identity certificate generated by QSA: " +
                         checked.failure_reason);
    }
    return certificate;
}

inline std::string encode_polynomial_division_certificate(
    const QPolynomial& dividend,
    const QPolynomial& divisor,
    const QPolynomial& quotient,
    const QPolynomial& remainder,
    const QpkLimits& limits = {}) {
    if (dividend.variable() != divisor.variable() ||
        dividend.variable() != quotient.variable() ||
        dividend.variable() != remainder.variable()) {
        throw QMathError("QPK certificate polynomials must use one variable identity");
    }
    const std::string source_id = qpk_detail::source_digest(dividend, divisor, limits);
    const std::string quotient_wire = qpk_detail::coefficients_wire(quotient, limits);
    const std::string remainder_wire = qpk_detail::coefficients_wire(remainder, limits);
    const std::string body = qpk_detail::certificate_body(
        dividend.variable(), source_id, quotient_wire, remainder_wire);
    const std::string certificate = qpk_detail::seal_certificate_body(body);
    if (certificate.size() > limits.max_certificate_bytes) {
        throw qpk_detail::ResourceLimitError("QPK certificate byte limit exceeded");
    }
    return certificate;
}

inline QpkCheckResult verify_polynomial_division(
    const QPolynomial& trusted_dividend,
    const QPolynomial& trusted_divisor,
    std::string_view certificate,
    const QpkLimits& limits = {}) {
    const std::size_t certificate_bytes = certificate.size();
    if (certificate_bytes > limits.max_certificate_bytes) {
        return qpk_detail::rejected(
            QpkFailureReason::ResourceLimit, "QPK certificate byte limit exceeded", certificate_bytes);
    }
    if (certificate.empty() || certificate.find('\r') != std::string_view::npos ||
        certificate.find('\0') != std::string_view::npos) {
        return qpk_detail::rejected(
            QpkFailureReason::MalformedCertificate, "QPK certificate encoding is not canonical", certificate_bytes);
    }

    try {
        const auto lines = qpk_detail::split_lines(certificate);
        if (lines.size() != 12U) {
            return qpk_detail::rejected(
                QpkFailureReason::MalformedCertificate, "QPK certificate has the wrong field count", certificate_bytes);
        }
        if (lines[0] != "QPK-CERT-v1") {
            return qpk_detail::unsupported("QPK certificate version is unsupported", certificate_bytes);
        }
        const std::string_view claim = qpk_detail::field_value(lines[1], "claim=");
        const std::string_view theory = qpk_detail::field_value(lines[2], "theory=");
        const std::string_view assumptions = qpk_detail::field_value(lines[3], "assumptions=");
        const std::string_view rule = qpk_detail::field_value(lines[4], "rule=");
        const std::string_view generator = qpk_detail::field_value(lines[5], "generator=");
        const std::string_view dependencies = qpk_detail::field_value(lines[6], "dependencies=");
        if (claim != "polynomial-division" || theory != "Q[x]" ||
            assumptions != qpk_detail::Assumptions || rule != qpk_detail::DivisionRule ||
            dependencies != qpk_detail::NoDependencies) {
            return qpk_detail::unsupported("QPK claim, theory, or rule is unsupported", certificate_bytes);
        }
        if (!qpk_detail::valid_metadata_token(generator)) {
            return qpk_detail::rejected(
                QpkFailureReason::MalformedCertificate, "QPK generator metadata is malformed", certificate_bytes);
        }

        const std::string variable = qpk_detail::hex_decode(
            qpk_detail::field_value(lines[7], "variable_hex="), limits.max_variable_bytes);
        const std::string_view source_id =
            qpk_detail::field_value(lines[8], "source_sha256=");
        const std::string_view quotient_wire = qpk_detail::field_value(lines[9], "quotient=");
        const std::string_view remainder_wire = qpk_detail::field_value(lines[10], "remainder=");
        const std::string_view certificate_id =
            qpk_detail::field_value(lines[11], "certificate_sha256=");
        if (!qpk_detail::valid_digest(source_id) || !qpk_detail::valid_digest(certificate_id)) {
            return qpk_detail::rejected(
                QpkFailureReason::MalformedCertificate, "QPK digest encoding is malformed", certificate_bytes);
        }

        std::size_t body_bytes = 0U;
        for (std::size_t index = 0U; index < 11U; ++index) {
            body_bytes += lines[index].size() + 1U;
        }
        const std::string_view body = certificate.substr(0U, body_bytes);
        if (qpk_detail::sha256_hex(body) != certificate_id) {
            return qpk_detail::rejected(
                QpkFailureReason::MalformedCertificate, "QPK certificate digest does not match its contents", certificate_bytes);
        }
        const std::string canonical_body = qpk_detail::certificate_body(
            variable, source_id, quotient_wire, remainder_wire, generator);
        if (body != canonical_body) {
            return qpk_detail::rejected(
                QpkFailureReason::MalformedCertificate, "QPK certificate is not in canonical form", certificate_bytes);
        }

        if (trusted_dividend.variable() != trusted_divisor.variable()) {
            return qpk_detail::rejected(
                QpkFailureReason::InvalidProof, "source polynomials use different variable identities", certificate_bytes);
        }
        const std::string expected_source_id =
            qpk_detail::source_digest(trusted_dividend, trusted_divisor, limits);
        if (source_id != expected_source_id ||
            variable != trusted_dividend.variable() ||
            variable != trusted_divisor.variable()) {
            QpkCheckResult result = qpk_detail::rejected(
                QpkFailureReason::StaleInput, "QPK certificate is bound to different source polynomials", certificate_bytes);
            result.source_id.assign(source_id);
            return result;
        }

        if (trusted_divisor.is_zero()) {
            return qpk_detail::rejected(
                QpkFailureReason::InvalidProof, "polynomial division by the zero polynomial is undefined", certificate_bytes);
        }

        const std::vector<QRational> quotient =
            qpk_detail::parse_coefficients(quotient_wire, limits);
        const std::vector<QRational> remainder =
            qpk_detail::parse_coefficients(remainder_wire, limits);

        if (!remainder.empty() &&
            remainder.size() >= trusted_divisor.coefficients().size()) {
            return qpk_detail::rejected(
                QpkFailureReason::InvalidProof, "remainder degree is not lower than divisor degree", certificate_bytes);
        }

        const auto& dividend_coefficients = trusted_dividend.coefficients();
        const auto& divisor_coefficients = trusted_divisor.coefficients();
        std::size_t product_terms = 0U;
        if (!divisor_coefficients.empty() && !quotient.empty()) {
            if (divisor_coefficients.size() >
                limits.max_polynomial_terms - (quotient.size() - 1U)) {
                return qpk_detail::rejected(
                    QpkFailureReason::ResourceLimit, "QPK product term limit exceeded", certificate_bytes);
            }
            product_terms = divisor_coefficients.size() + quotient.size() - 1U;
        }
        const std::size_t result_terms =
            std::max({product_terms, remainder.size(), dividend_coefficients.size()});
        if (result_terms > limits.max_polynomial_terms) {
            return qpk_detail::rejected(
                QpkFailureReason::ResourceLimit, "QPK verification term limit exceeded", certificate_bytes);
        }

        std::vector<QRational> reconstructed(result_terms, QRational(0));
        std::size_t checked_steps = 0U;
        for (std::size_t i = 0U; i < divisor_coefficients.size(); ++i) {
            for (std::size_t j = 0U; j < quotient.size(); ++j) {
                reconstructed[i + j] =
                    reconstructed[i + j] + divisor_coefficients[i] * quotient[j];
                ++checked_steps;
            }
        }
        for (std::size_t index = 0U; index < remainder.size(); ++index) {
            reconstructed[index] = reconstructed[index] + remainder[index];
            ++checked_steps;
        }
        for (std::size_t index = 0U; index < result_terms; ++index) {
            const QRational expected = index < dividend_coefficients.size()
                ? dividend_coefficients[index]
                : QRational(0);
            if (reconstructed[index] != expected) {
                return qpk_detail::rejected(
                    QpkFailureReason::InvalidProof, "quotient and remainder do not reconstruct the dividend",
                    certificate_bytes);
            }
            ++checked_steps;
        }

        QpkCheckResult result;
        result.status = QpkVerificationStatus::ProvenExact;
        result.reason = QpkFailureReason::None;
        result.source_id.assign(source_id);
        result.checked_steps = checked_steps + 2U;
        result.certificate_bytes = certificate_bytes;
        return result;
    } catch (const qpk_detail::ResourceLimitError& error) {
        return qpk_detail::rejected(
            QpkFailureReason::ResourceLimit, error.what(), certificate_bytes);
    } catch (const std::exception& error) {
        return qpk_detail::rejected(
            QpkFailureReason::MalformedCertificate, error.what(), certificate_bytes);
    } catch (...) {
        return qpk_detail::rejected(
            QpkFailureReason::MalformedCertificate, "QPK verification failed", certificate_bytes);
    }
}

inline std::string prove_polynomial_division(
    const QPolynomial& dividend,
    const QPolynomial& divisor,
    const QpkLimits& limits = {}) {
    const auto division = dividend.divmod(divisor);
    const std::string certificate = encode_polynomial_division_certificate(
        dividend, divisor, division.first, division.second, limits);
    const QpkCheckResult checked =
        verify_polynomial_division(dividend, divisor, certificate, limits);
    if (!checked.accepted()) {
        throw QMathError("QPK rejected a certificate generated by QSA: " + checked.failure_reason);
    }
    return certificate;
}

}  // namespace qubit

