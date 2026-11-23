#include "qubit/qpk.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

using namespace qubit;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::string resign(std::string body) {
    const std::string digest = qpk_detail::sha256_hex(body);
    body += "certificate_sha256=";
    body += digest;
    body += '\n';
    return body;
}

std::string replace_once(std::string value, const std::string& before, const std::string& after) {
    const std::size_t position = value.find(before);
    if (position == std::string::npos) throw std::runtime_error("test mutation target was not found");
    value.replace(position, before.size(), after);
    return value;
}

std::string body_without_digest(const std::string& certificate) {
    const std::size_t position = certificate.rfind("certificate_sha256=");
    if (position == std::string::npos) throw std::runtime_error("certificate digest field was not found");
    return certificate.substr(0U, position);
}

}  // namespace

int main() {
    require(qpk_detail::sha256_hex("") ==
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "QPK SHA-256 empty-input vector failed");
    require(qpk_detail::sha256_hex("abc") ==
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "QPK SHA-256 abc vector failed");
    require(qpk_detail::sha256_hex(std::string(1000U, 'a')) ==
                "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3",
            "QPK SHA-256 multi-block vector failed");

    const QPolynomial dividend("x", {QRational(-1), QRational(0), QRational(0), QRational(1)});
    const QPolynomial divisor("x", {QRational(-1), QRational(1)});
    const std::string certificate = prove_polynomial_division(dividend, divisor);
    const QpkCheckResult accepted =
        verify_polynomial_division(dividend, divisor, certificate);
    require(accepted.accepted() && accepted.reason == QpkFailureReason::None &&
                accepted.claim_id == "polynomial-division-identity-and-remainder-degree-v1" &&
                accepted.theory_id == "Q[x]" && accepted.checked_steps > 0U &&
                accepted.certificate_bytes == certificate.size(),
            "valid exact polynomial division certificate was not independently accepted");
    require(verify_polynomial_division(dividend, divisor, certificate).accepted(),
            "repeated verification was not deterministic");

    const std::string identity_certificate = prove_polynomial_identity(dividend, dividend);
    const QpkCheckResult identity =
        verify_polynomial_identity(dividend, dividend, identity_certificate);
    require(identity.accepted() &&
                identity.claim_kind == QpkClaimKind::RationalPolynomialIdentity &&
                identity.claim_id == "polynomial-identity-coefficientwise-v1",
            "valid coefficientwise polynomial identity was not proven exactly");

    const QPolynomial false_identity_right(
        "x", {QRational(0), QRational(0), QRational(0), QRational(1)});
    const std::string false_identity_certificate =
        encode_polynomial_identity_certificate(dividend, false_identity_right);
    const QpkCheckResult false_identity = verify_polynomial_identity(
        dividend, false_identity_right, false_identity_certificate);
    require(false_identity.status == QpkVerificationStatus::Rejected &&
                false_identity.reason == QpkFailureReason::InvalidProof,
            "a forged false polynomial identity was not rejected");

    const std::string body = body_without_digest(certificate);
    const std::string alternate_generator = resign(
        replace_once(body, "generator=QSA-QPK-v1\n", "generator=independent-tool-v1\n"));
    require(verify_polynomial_division(dividend, divisor, alternate_generator).accepted(),
            "untrusted generator metadata incorrectly affected proof acceptance");

    const std::string forged_quotient = resign(
        replace_once(body, "quotient=1,1,1\n", "quotient=1,1,2\n"));
    const QpkCheckResult forged =
        verify_polynomial_division(dividend, divisor, forged_quotient);
    require(forged.status == QpkVerificationStatus::Rejected &&
                forged.reason == QpkFailureReason::InvalidProof,
            "a re-signed false quotient certificate was not rejected mathematically");

    const std::string noncanonical_rational = resign(
        replace_once(body, "quotient=1,1,1\n", "quotient=01,1,1\n"));
    const QpkCheckResult malformed =
        verify_polynomial_division(dividend, divisor, noncanonical_rational);
    require(malformed.status == QpkVerificationStatus::Rejected &&
                malformed.reason == QpkFailureReason::MalformedCertificate,
            "noncanonical rational coefficient was accepted");

    const QpkCheckResult stale =
        verify_polynomial_division(
            QPolynomial("x", {QRational(-2), QRational(0), QRational(0), QRational(1)}),
            divisor,
            certificate);
    require(stale.status == QpkVerificationStatus::Rejected &&
                stale.reason == QpkFailureReason::StaleInput,
            "certificate bound to a different dividend was not rejected");

    const std::string wrong_claim = resign(
        replace_once(body, "claim=polynomial-division\n", "claim=unrestricted-proof\n"));
    const QpkCheckResult unsupported =
        verify_polynomial_division(dividend, divisor, wrong_claim);
    require(unsupported.status == QpkVerificationStatus::Unverified &&
                unsupported.reason == QpkFailureReason::Unsupported,
            "unsupported proof claim was not kept unverified");

    const QpkCheckResult truncated =
        verify_polynomial_division(dividend, divisor, certificate.substr(0U, certificate.size() - 1U));
    require(truncated.status == QpkVerificationStatus::Rejected &&
                truncated.reason == QpkFailureReason::MalformedCertificate,
            "truncated certificate did not fail closed");

    QpkLimits tiny;
    tiny.max_certificate_bytes = 16U;
    const QpkCheckResult oversized =
        verify_polynomial_division(dividend, divisor, certificate, tiny);
    require(oversized.status == QpkVerificationStatus::Rejected &&
                oversized.reason == QpkFailureReason::ResourceLimit,
            "certificate byte limit was not enforced");

    QpkLimits tiny_coefficients;
    tiny_coefficients.max_rational_bytes = 1U;
    const QpkCheckResult oversized_coefficient =
        verify_polynomial_division(dividend, divisor, certificate, tiny_coefficients);
    require(oversized_coefficient.status == QpkVerificationStatus::Rejected &&
                oversized_coefficient.reason == QpkFailureReason::ResourceLimit,
            "rational coefficient size limit was not classified and enforced");

    const QPolynomial nonzero_remainder_dividend("x", {QRational(1), QRational(0), QRational(1)});
    const QPolynomial x("x", {QRational(0), QRational(1)});
    const std::string nonzero_remainder_certificate =
        prove_polynomial_division(nonzero_remainder_dividend, x);
    require(verify_polynomial_division(
                nonzero_remainder_dividend, x, nonzero_remainder_certificate).accepted(),
            "valid nonzero-remainder division certificate was not accepted");

    const QPolynomial rational_dividend(
        "x", {QRational(1), QRational(1, 2), QRational(1)});
    const QPolynomial rational_divisor("x", {QRational(1, 2), QRational(1)});
    const std::string rational_certificate =
        prove_polynomial_division(rational_dividend, rational_divisor);
    require(verify_polynomial_division(
                rational_dividend, rational_divisor, rational_certificate).accepted(),
            "exact rational coefficient certificate was not accepted");

    const QPolynomial zero("x");
    const QPolynomial one("x", {QRational(1)});
    const std::string zero_divisor_certificate =
        encode_polynomial_division_certificate(dividend, zero, one, dividend);
    const QpkCheckResult zero_divisor =
        verify_polynomial_division(dividend, zero, zero_divisor_certificate);
    require(zero_divisor.status == QpkVerificationStatus::Rejected &&
                zero_divisor.reason == QpkFailureReason::InvalidProof,
            "division by the zero polynomial was not rejected");

    const std::string forged_remainder = resign(
        replace_once(body, "remainder=0\n", "remainder=1\n"));
    const QpkCheckResult bad_remainder =
        verify_polynomial_division(dividend, divisor, forged_remainder);
    require(bad_remainder.status == QpkVerificationStatus::Rejected &&
                bad_remainder.reason == QpkFailureReason::InvalidProof,
            "a false remainder certificate was not rejected");

    std::cout << "QSA proof-kernel polynomial-certificate tests passed\n";
}

