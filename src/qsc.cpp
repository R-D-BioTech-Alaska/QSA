#include "qubit/qstate.hpp"
#include <array>
#include <bit>
#include <cstring>
#include <limits>

namespace qubit {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{'Q', 'S', 'C', '1', 'Q', 'B', 'T', 0};
constexpr std::uint16_t kMajor = 1;
constexpr std::uint16_t kMinor = 0;
constexpr std::uint32_t kDecodeMaxQubits = 1'000'000;

void append_u8(std::vector<std::uint8_t>& output, std::uint8_t value) {
    output.push_back(value);
}

void append_u16(std::vector<std::uint8_t>& output, std::uint16_t value) {
    for (unsigned shift = 0; shift < 16; shift += 8) {
        output.push_back(static_cast<std::uint8_t>((value >> shift) & 0xFFU));
    }
}

void append_u32(std::vector<std::uint8_t>& output, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        output.push_back(static_cast<std::uint8_t>((value >> shift) & 0xFFU));
    }
}

void append_u64(std::vector<std::uint8_t>& output, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        output.push_back(static_cast<std::uint8_t>((value >> shift) & 0xFFU));
    }
}

void append_f64(std::vector<std::uint8_t>& output, double value) {
    append_u64(output, std::bit_cast<std::uint64_t>(value));
}

[[nodiscard]] std::uint64_t fnv1a(std::span<const std::uint8_t> bytes) noexcept {
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::uint8_t value : bytes) {
        hash ^= value;
        hash *= 1099511628211ULL;
    }
    return hash;
}

class Reader {
public:
    explicit Reader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

    [[nodiscard]] std::uint8_t u8() {
        require(1);
        return bytes_[position_++];
    }

    [[nodiscard]] std::uint16_t u16() {
        std::uint32_t value = 0;
        for (unsigned shift = 0; shift < 16; shift += 8) {
            value |= static_cast<std::uint32_t>(u8()) << shift;
        }
        return static_cast<std::uint16_t>(value);
    }

    [[nodiscard]] std::uint32_t u32() {
        std::uint32_t value = 0;
        for (unsigned shift = 0; shift < 32; shift += 8) {
            value |= static_cast<std::uint32_t>(u8()) << shift;
        }
        return value;
    }

    [[nodiscard]] std::uint64_t u64() {
        std::uint64_t value = 0;
        for (unsigned shift = 0; shift < 64; shift += 8) {
            value |= static_cast<std::uint64_t>(u8()) << shift;
        }
        return value;
    }

    [[nodiscard]] double f64() {
        return std::bit_cast<double>(u64());
    }

    [[nodiscard]] std::span<const std::uint8_t> take(std::size_t count) {
        require(count);
        const auto result = bytes_.subspan(position_, count);
        position_ += count;
        return result;
    }

    [[nodiscard]] std::size_t remaining() const noexcept {
        return bytes_.size() - position_;
    }

private:
    std::span<const std::uint8_t> bytes_;
    std::size_t position_{0};

    void require(std::size_t count) const {
        if (count > bytes_.size() - position_) {
            throw QStateError("QSC data is truncated");
        }
    }
};

} 

std::vector<std::uint8_t> QStateCodec::encode(const QRegister& state) {
    std::string reason;
    if (!state.validate(&reason)) {
        throw QStateError("Cannot encode invalid QRegister: " + reason);
    }

    std::vector<std::uint8_t> output;
    output.reserve(state.estimated_bytes() + 256U);
    output.insert(output.end(), kMagic.begin(), kMagic.end());
    append_u16(output, kMajor);
    append_u16(output, kMinor);
    append_u32(output, static_cast<std::uint32_t>(state.qubit_count_));
    append_f64(output, state.config_.epsilon);
    append_f64(output, state.config_.factor_tolerance);
    append_u32(output, static_cast<std::uint32_t>(state.config_.max_component_qubits));
    append_u64(output, state.config_.max_dense_amplitudes);
    append_u64(output, static_cast<std::uint64_t>(state.config_.max_sparse_entries));
    append_u32(output, static_cast<std::uint32_t>(state.components_.size()));

    for (std::size_t component_index : state.ordered_component_indices()) {
        const StateComponent& component = state.components_[component_index];
        std::uint8_t kind = 0;
        if (!component.is_cell()) {
            kind = std::get<AmplitudeStore>(component.state).mode() == StorageMode::Sparse ? 1U : 2U;
        }
        append_u8(output, kind);
        append_u32(output, static_cast<std::uint32_t>(component.qubits.size()));
        for (QubitId qubit : component.qubits) {
            append_u32(output, qubit);
        }

        if (component.is_cell()) {
            const BlochCell& cell = std::get<BlochCell>(component.state);
            append_f64(output, cell.x);
            append_f64(output, cell.y);
            append_f64(output, cell.z);
            continue;
        }

        const AmplitudeStore& store = std::get<AmplitudeStore>(component.state);
        append_u64(output, store.dimension());
        if (store.mode() == StorageMode::Sparse) {
            const auto entries = store.entries();
            append_u64(output, static_cast<std::uint64_t>(entries.size()));
            for (const auto& [index, value] : entries) {
                append_u64(output, index);
                append_f64(output, value.re);
                append_f64(output, value.im);
            }
        } else {
            const auto dense = store.dense_copy();
            append_u64(output, static_cast<std::uint64_t>(dense.size()));
            for (const QComplex& value : dense) {
                append_f64(output, value.re);
                append_f64(output, value.im);
            }
        }
    }

    append_u64(output, fnv1a(output));
    return output;
}

QRegister QStateCodec::decode(std::span<const std::uint8_t> bytes) {
    if (bytes.size() < kMagic.size() + 8U) {
        throw QStateError("QSC data is too short");
    }
    const std::uint64_t expected_checksum = [&bytes]() {
        Reader checksum_reader(bytes.subspan(bytes.size() - 8U));
        return checksum_reader.u64();
    }();
    const std::uint64_t actual_checksum = fnv1a(bytes.first(bytes.size() - 8U));
    if (expected_checksum != actual_checksum) {
        throw QStateError("QSC checksum mismatch");
    }

    Reader reader(bytes.first(bytes.size() - 8U));
    const auto magic = reader.take(kMagic.size());
    if (!std::equal(magic.begin(), magic.end(), kMagic.begin(), kMagic.end())) {
        throw QStateError("QSC magic identifier is invalid");
    }
    const std::uint16_t major = reader.u16();
    const std::uint16_t minor = reader.u16();
    if (major != kMajor || minor > kMinor) {
        throw QStateError("Unsupported QSC version");
    }

    const std::uint32_t qubit_count = reader.u32();
    if (qubit_count == 0U || qubit_count > kDecodeMaxQubits) {
        throw QStateError("QSC qubit count exceeds the decoder safety limit");
    }
    QStateConfig config;
    config.epsilon = reader.f64();
    config.factor_tolerance = reader.f64();
    config.max_component_qubits = reader.u32();
    config.max_dense_amplitudes = reader.u64();
    const std::uint64_t max_sparse_entries = reader.u64();
    if (max_sparse_entries > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw QStateError("QSC sparse-entry limit exceeds platform capacity");
    }
    config.max_sparse_entries = static_cast<std::size_t>(max_sparse_entries);
    const std::uint32_t component_count = reader.u32();
    if (component_count == 0U || component_count > qubit_count) {
        throw QStateError("QSC component count is invalid");
    }

    QRegister state(qubit_count, config);
    state.components_.clear();
    state.component_order_.clear();
    state.components_.reserve(component_count);
    state.component_order_.reserve(component_count);
    state.next_component_order_ = 0;

    for (std::uint32_t component_index = 0; component_index < component_count; ++component_index) {
        const std::uint8_t kind = reader.u8();
        if (kind > 2U) {
            throw QStateError("QSC component has an unknown storage kind");
        }
        const std::uint32_t member_count = reader.u32();
        if (member_count == 0U || member_count > config.max_component_qubits) {
            throw QStateError("QSC component has an invalid qubit count");
        }
        std::vector<QubitId> qubits;
        qubits.reserve(member_count);
        for (std::uint32_t member = 0; member < member_count; ++member) {
            qubits.push_back(reader.u32());
        }

        StateComponent component;
        component.qubits = std::move(qubits);
        if (kind == 0U) {
            if (member_count != 1U) {
                throw QStateError("QSC geometric cell must contain one qubit");
            }
            const double cell_x = reader.f64();
            const double cell_y = reader.f64();
            const double cell_z = reader.f64();
            BlochCell cell{cell_x, cell_y, cell_z};
            cell.normalize(config.epsilon);
            component.state = cell;
        } else {
            const BasisIndex dimension = reader.u64();
            const BasisIndex expected_dimension = BasisIndex{1} << member_count;
            if (dimension != expected_dimension) {
                throw QStateError("QSC patch dimension does not match its qubit count");
            }
            const std::uint64_t value_count = reader.u64();
            if (kind == 1U) {
                if (value_count > config.max_sparse_entries || value_count > dimension) {
                    throw QStateError("QSC sparse patch exceeds configured limits");
                }
                if (value_count > reader.remaining() / 24U) {
                    throw QStateError("QSC sparse payload length is inconsistent");
                }
                std::vector<AmplitudeStore::SparseEntry> entries;
                entries.reserve(static_cast<std::size_t>(value_count));
                for (std::uint64_t entry = 0; entry < value_count; ++entry) {
                    const BasisIndex index = reader.u64();
                    const double real = reader.f64();
                    const double imag = reader.f64();
                    entries.emplace_back(index, QComplex{real, imag});
                }
                component.state = AmplitudeStore::from_entries(
                    dimension, std::move(entries), config, true);
            } else {
                if (value_count != dimension || value_count > config.max_dense_amplitudes ||
                    value_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
                    throw QStateError("QSC dense patch exceeds configured limits");
                }
                if (value_count > reader.remaining() / 16U) {
                    throw QStateError("QSC dense payload length is inconsistent");
                }
                std::vector<QComplex> dense;
                dense.reserve(static_cast<std::size_t>(value_count));
                for (std::uint64_t entry = 0; entry < value_count; ++entry) {
                    const double real = reader.f64();
                    const double imag = reader.f64();
                    dense.emplace_back(real, imag);
                }
                component.state = AmplitudeStore::from_dense(std::move(dense), config, true);
            }
        }
        state.components_.push_back(std::move(component));
        state.component_order_.push_back(
            static_cast<std::uint32_t>(state.next_component_order_++));
    }

    if (reader.remaining() != 0U) {
        throw QStateError("QSC data contains unexpected trailing bytes");
    }
    state.reindex_components();
    std::string reason;
    if (!state.validate(&reason)) {
        throw QStateError("Decoded QSC state is invalid: " + reason);
    }
    return state;
}

} 
