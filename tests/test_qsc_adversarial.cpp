#include "qubit/qstate.hpp"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "TEST FAILURE: " << message << '\n';
        std::exit(1);
    }
}

std::uint64_t fnv1a(std::span<const std::uint8_t> bytes) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::uint8_t value : bytes) {
        hash ^= value;
        hash *= 1099511628211ULL;
    }
    return hash;
}

void write_u16(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint16_t value) {
    bytes[offset] = static_cast<std::uint8_t>(value & 0xFFU);
    bytes[offset + 1U] = static_cast<std::uint8_t>((value >> 8U) & 0xFFU);
}

void write_u32(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        bytes[offset + shift / 8U] = static_cast<std::uint8_t>((value >> shift) & 0xFFU);
    }
}

void refresh_checksum(std::vector<std::uint8_t>& bytes) {
    require(bytes.size() >= 8U, "packet must contain a checksum");
    const std::uint64_t checksum = fnv1a(std::span<const std::uint8_t>(bytes).first(bytes.size() - 8U));
    for (unsigned shift = 0; shift < 64; shift += 8) {
        bytes[bytes.size() - 8U + shift / 8U] =
            static_cast<std::uint8_t>((checksum >> shift) & 0xFFU);
    }
}

void expect_rejected(const std::vector<std::uint8_t>& bytes, const std::string& label) {
    bool rejected = false;
    try {
        (void)qubit::QRegister::decode_qsc(bytes);
    } catch (const qubit::QStateError&) {
        rejected = true;
    }
    require(rejected, label);
}

} 

int main() {
    qubit::QRegister state(2);
    state.apply_h(0);
    state.apply_cnot(0, 1);
    const std::vector<std::uint8_t> valid = state.encode_qsc();
    require(qubit::QRegister::decode_qsc(valid).validate(), "valid packet must decode");

    for (std::size_t length = 0; length < valid.size(); ++length) {
        expect_rejected(
            std::vector<std::uint8_t>(valid.begin(), valid.begin() + static_cast<std::ptrdiff_t>(length)),
            "every truncated prefix must be rejected");
    }

    for (std::size_t index = 0; index < valid.size(); ++index) {
        std::vector<std::uint8_t> corrupted = valid;
        corrupted[index] ^= 0x01U;
        expect_rejected(corrupted, "single-byte corruption must be rejected");
    }

    auto mutate = [&valid](const std::string& label, const auto& function) {
        std::vector<std::uint8_t> packet = valid;
        function(packet);
        refresh_checksum(packet);
        expect_rejected(packet, label);
    };

    mutate("unsupported major version", [](auto& packet) { write_u16(packet, 8U, 2U); });
    mutate("unsupported minor version", [](auto& packet) { write_u16(packet, 10U, 0xFFFFU); });
    mutate("zero qubit count", [](auto& packet) { write_u32(packet, 12U, 0U); });
    mutate("decoder qubit safety limit", [](auto& packet) { write_u32(packet, 12U, 1'000'001U); });
    mutate("invalid component limit", [](auto& packet) { write_u32(packet, 32U, 63U); });
    mutate("zero component count", [](auto& packet) { write_u32(packet, 52U, 0U); });
    mutate("unknown component kind", [](auto& packet) { packet[56U] = 0xFFU; });
    mutate("zero member count", [](auto& packet) { write_u32(packet, 57U, 0U); });

    std::vector<std::uint8_t> trailing = valid;
    trailing.insert(trailing.end() - 8, 0xAAU);
    refresh_checksum(trailing);
    expect_rejected(trailing, "unexpected trailing payload must be rejected");

    std::cout << "QSC adversarial decoder tests passed.\n";
    return 0;
}
