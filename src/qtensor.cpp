#include "qubit/qtensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace qubit {
namespace {

[[nodiscard]] bool finite(const QComplex& value) noexcept {
    return std::isfinite(value.re) && std::isfinite(value.im);
}

[[nodiscard]] std::size_t binary_entries(std::size_t variables) noexcept {
    if (variables >= std::numeric_limits<std::size_t>::digits) {
        return std::numeric_limits<std::size_t>::max();
    }
    return std::size_t{1} << variables;
}

template <typename FactorVector>
[[nodiscard]] std::vector<std::uint32_t> union_variables(
    const FactorVector& factors,
    std::span<const std::size_t> indices) {
    std::vector<std::uint32_t> variables;
    for (const std::size_t index : indices) {
        variables.insert(
            variables.end(),
            factors[index].variables.begin(),
            factors[index].variables.end());
    }
    std::sort(variables.begin(), variables.end());
    variables.erase(std::unique(variables.begin(), variables.end()), variables.end());
    return variables;
}

}  // namespace

TensorNetworkCircuit::TensorNetworkCircuit(
    std::size_t qubit_count,
    TensorNetworkConfig config)
    : qubit_count_(qubit_count), config_(config) {
    if (qubit_count_ == 0U) {
        throw QStateError("Tensor network requires at least one qubit");
    }
    if (qubit_count_ > static_cast<std::size_t>(std::numeric_limits<VariableId>::max())) {
        throw QStateError("Tensor network qubit count exceeds variable range");
    }
    if (config_.max_contraction_entries < 2U || config_.max_factors < qubit_count_) {
        throw QStateError("Tensor network resource limits are invalid");
    }

    factors_.reserve(qubit_count_);
    current_wires_.reserve(qubit_count_);
    for (std::size_t qubit = 0; qubit < qubit_count_; ++qubit) {
        const VariableId variable = next_variable_++;
        current_wires_.push_back(variable);
        Factor initial;
        initial.variables.push_back(variable);
        initial.values = {{1.0, 0.0}, {0.0, 0.0}};
        factors_.push_back(std::move(initial));
    }
}

TensorNetworkCircuit::TensorNetworkCircuit(
    std::size_t qubit_count,
    std::span<const Operation> operations,
    TensorNetworkConfig config)
    : TensorNetworkCircuit(qubit_count, config) {
    apply(operations);
}

void TensorNetworkCircuit::validate_operation(const Operation& operation) const {
    if (static_cast<std::size_t>(operation.first) >= qubit_count_) {
        throw QStateError("Tensor network operation qubit is out of range");
    }

    switch (operation.code) {
        case OperationCode::Cnot:
        case OperationCode::Cz:
        case OperationCode::Swap:
            if (static_cast<std::size_t>(operation.second) >= qubit_count_ ||
                operation.first == operation.second) {
                throw QStateError("Tensor network two-qubit operation is invalid");
            }
            break;
        case OperationCode::Rx:
        case OperationCode::Ry:
        case OperationCode::Rz:
            if (!std::isfinite(operation.parameter)) {
                throw QStateError("Tensor network rotation angle must be finite");
            }
            break;
        case OperationCode::BitFlipTrajectory:
        case OperationCode::PhaseFlipTrajectory:
        case OperationCode::DepolarizingTrajectory:
        case OperationCode::AmplitudeDampingTrajectory:
            throw QStateError("Tensor network circuit does not accept trajectory noise");
        case OperationCode::X:
        case OperationCode::Y:
        case OperationCode::Z:
        case OperationCode::H:
        case OperationCode::S:
        case OperationCode::Sdg:
        case OperationCode::T:
        case OperationCode::Tdg:
            break;
    }

    const std::size_t created =
        operation.code == OperationCode::Cnot ||
                operation.code == OperationCode::Cz ||
                operation.code == OperationCode::Swap
            ? 2U
            : 1U;
    if (static_cast<std::uint64_t>(next_variable_) + created >
        static_cast<std::uint64_t>(std::numeric_limits<VariableId>::max())) {
        throw QStateError("Tensor network variable range exhausted");
    }
    if (factors_.size() >= config_.max_factors) {
        throw QStateError("Tensor network exceeded max_factors");
    }
}

void TensorNetworkCircuit::push_factor(Factor factor) {
    if (factors_.size() >= config_.max_factors) {
        throw QStateError("Tensor network exceeded max_factors");
    }
    factors_.push_back(std::move(factor));
}

void TensorNetworkCircuit::apply_single(QubitId qubit, const QMatrix2& matrix) {
    const VariableId input = current_wires_[qubit];
    const VariableId output = next_variable_++;

    Factor factor;
    factor.variables = {input, output};
    factor.values.resize(4U);
    for (std::size_t out = 0; out < 2U; ++out) {
        for (std::size_t in = 0; in < 2U; ++in) {
            factor.values[in | (out << 1U)] = matrix(out, in);
        }
    }
    push_factor(std::move(factor));
    current_wires_[qubit] = output;
}

void TensorNetworkCircuit::apply_two(
    QubitId first,
    QubitId second,
    const QMatrix4& matrix) {
    const VariableId first_input = current_wires_[first];
    const VariableId second_input = current_wires_[second];
    const VariableId first_output = next_variable_++;
    const VariableId second_output = next_variable_++;

    Factor factor;
    factor.variables = {first_input, second_input, first_output, second_output};
    factor.values.resize(16U);
    for (std::size_t first_out = 0; first_out < 2U; ++first_out) {
        for (std::size_t second_out = 0; second_out < 2U; ++second_out) {
            const std::size_t row = (first_out << 1U) | second_out;
            for (std::size_t first_in = 0; first_in < 2U; ++first_in) {
                for (std::size_t second_in = 0; second_in < 2U; ++second_in) {
                    const std::size_t column = (first_in << 1U) | second_in;
                    const std::size_t index = first_in |
                                              (second_in << 1U) |
                                              (first_out << 2U) |
                                              (second_out << 3U);
                    factor.values[index] = matrix(row, column);
                }
            }
        }
    }
    push_factor(std::move(factor));
    current_wires_[first] = first_output;
    current_wires_[second] = second_output;
}

void TensorNetworkCircuit::apply(const Operation& operation) {
    validate_operation(operation);

    switch (operation.code) {
        case OperationCode::X:
            apply_single(operation.first, gates::x());
            break;
        case OperationCode::Y:
            apply_single(operation.first, gates::y());
            break;
        case OperationCode::Z:
            apply_single(operation.first, gates::z());
            break;
        case OperationCode::H:
            apply_single(operation.first, gates::h());
            break;
        case OperationCode::S:
            apply_single(operation.first, gates::s());
            break;
        case OperationCode::Sdg:
            apply_single(operation.first, gates::sdg());
            break;
        case OperationCode::T:
            apply_single(operation.first, gates::t());
            break;
        case OperationCode::Tdg:
            apply_single(operation.first, gates::tdg());
            break;
        case OperationCode::Rx:
            apply_single(operation.first, gates::rx(operation.parameter));
            break;
        case OperationCode::Ry:
            apply_single(operation.first, gates::ry(operation.parameter));
            break;
        case OperationCode::Rz:
            apply_single(operation.first, gates::rz(operation.parameter));
            break;
        case OperationCode::Cnot:
            apply_two(operation.first, operation.second, gates::cnot());
            break;
        case OperationCode::Cz:
            apply_two(operation.first, operation.second, gates::cz());
            break;
        case OperationCode::Swap:
            apply_two(operation.first, operation.second, gates::swap());
            break;
        case OperationCode::BitFlipTrajectory:
        case OperationCode::PhaseFlipTrajectory:
        case OperationCode::DepolarizingTrajectory:
        case OperationCode::AmplitudeDampingTrajectory:
            throw QStateError("Tensor network circuit does not accept trajectory noise");
    }
    ++operation_count_;
}

void TensorNetworkCircuit::apply(std::span<const Operation> operations) {
    std::size_t additional_factors = 0U;
    std::uint64_t additional_variables = 0U;
    for (const Operation& operation : operations) {
        validate_operation(operation);
        ++additional_factors;
        additional_variables +=
            operation.code == OperationCode::Cnot ||
                    operation.code == OperationCode::Cz ||
                    operation.code == OperationCode::Swap
                ? 2U
                : 1U;
    }
    if (additional_factors > config_.max_factors - factors_.size()) {
        throw QStateError("Tensor network exceeded max_factors");
    }
    if (static_cast<std::uint64_t>(next_variable_) + additional_variables >
        static_cast<std::uint64_t>(std::numeric_limits<VariableId>::max())) {
        throw QStateError("Tensor network variable range exhausted");
    }
    for (const Operation& operation : operations) {
        apply(operation);
    }
}

QComplex TensorNetworkCircuit::contract(
    std::vector<Factor> factors,
    TensorContractionStats* stats) const {
    TensorContractionStats local_stats;
    local_stats.source_operations = operation_count_;
    local_stats.source_factors = factors_.size();

    while (true) {
        std::vector<std::vector<std::size_t>> incidence(next_variable_);
        bool has_variable = false;
        for (std::size_t factor_index = 0; factor_index < factors.size(); ++factor_index) {
            for (const VariableId variable : factors[factor_index].variables) {
                incidence[variable].push_back(factor_index);
                has_variable = true;
            }
        }
        if (!has_variable) {
            break;
        }

        VariableId selected = 0U;
        std::vector<std::size_t> selected_bucket;
        std::vector<VariableId> selected_union;
        std::size_t selected_entries = std::numeric_limits<std::size_t>::max();
        bool found = false;

        for (std::size_t candidate = 0; candidate < incidence.size(); ++candidate) {
            if (incidence[candidate].empty()) {
                continue;
            }
            const std::vector<VariableId> variables = union_variables(factors, incidence[candidate]);
            const std::size_t entries = binary_entries(variables.size());
            if (!found || entries < selected_entries ||
                (entries == selected_entries && candidate < selected)) {
                found = true;
                selected = static_cast<VariableId>(candidate);
                selected_bucket = incidence[candidate];
                selected_union = variables;
                selected_entries = entries;
            }
        }
        if (!found) {
            throw QStateError("Tensor network contraction lost its active variables");
        }
        if (selected_entries > config_.max_contraction_entries) {
            throw QStateError("Tensor network contraction exceeded max_contraction_entries");
        }

        local_stats.peak_union_variables =
            std::max(local_stats.peak_union_variables, selected_union.size());
        local_stats.peak_contraction_entries =
            std::max(local_stats.peak_contraction_entries, selected_entries);
        ++local_stats.eliminated_variables;

        const auto selected_position_it =
            std::lower_bound(selected_union.begin(), selected_union.end(), selected);
        if (selected_position_it == selected_union.end() || *selected_position_it != selected) {
            throw QStateError("Tensor network contraction variable is missing from its bucket");
        }
        const std::size_t selected_position =
            static_cast<std::size_t>(selected_position_it - selected_union.begin());

        Factor reduced;
        reduced.variables = selected_union;
        reduced.variables.erase(reduced.variables.begin() + static_cast<std::ptrdiff_t>(selected_position));
        const std::size_t reduced_entries = binary_entries(reduced.variables.size());
        if (reduced_entries > config_.max_contraction_entries) {
            throw QStateError("Tensor network reduced factor exceeded max_contraction_entries");
        }
        reduced.values.assign(reduced_entries, {});

        std::vector<std::vector<std::size_t>> factor_positions;
        factor_positions.reserve(selected_bucket.size());
        for (const std::size_t factor_index : selected_bucket) {
            std::vector<std::size_t> positions;
            positions.reserve(factors[factor_index].variables.size());
            for (const VariableId variable : factors[factor_index].variables) {
                const auto position =
                    std::lower_bound(selected_union.begin(), selected_union.end(), variable);
                if (position == selected_union.end() || *position != variable) {
                    throw QStateError("Tensor network factor variable is missing from bucket union");
                }
                positions.push_back(static_cast<std::size_t>(position - selected_union.begin()));
            }
            factor_positions.push_back(std::move(positions));
        }

        const std::size_t lower_mask =
            selected_position == 0U ? 0U : (std::size_t{1} << selected_position) - 1U;
        for (std::size_t reduced_index = 0; reduced_index < reduced_entries; ++reduced_index) {
            QComplex sum{};
            for (std::size_t selected_bit = 0; selected_bit < 2U; ++selected_bit) {
                const std::size_t low = reduced_index & lower_mask;
                const std::size_t high = reduced_index >> selected_position;
                const std::size_t union_index = low |
                                                (selected_bit << selected_position) |
                                                (high << (selected_position + 1U));
                QComplex product{1.0, 0.0};
                for (std::size_t bucket_index = 0; bucket_index < selected_bucket.size(); ++bucket_index) {
                    const Factor& factor = factors[selected_bucket[bucket_index]];
                    std::size_t factor_index = 0U;
                    for (std::size_t local = 0; local < factor.variables.size(); ++local) {
                        const std::size_t bit =
                            (union_index >> factor_positions[bucket_index][local]) & 1U;
                        factor_index |= bit << local;
                    }
                    product *= factor.values[factor_index];
                }
                sum += product;
            }
            reduced.values[reduced_index] = sum;
        }

        std::vector<bool> removed(factors.size(), false);
        for (const std::size_t index : selected_bucket) {
            removed[index] = true;
        }
        std::vector<Factor> next;
        next.reserve(factors.size() - selected_bucket.size() + 1U);
        for (std::size_t index = 0; index < factors.size(); ++index) {
            if (!removed[index]) {
                next.push_back(std::move(factors[index]));
            }
        }
        next.push_back(std::move(reduced));
        factors = std::move(next);
    }

    QComplex result{1.0, 0.0};
    for (const Factor& factor : factors) {
        if (!factor.variables.empty() || factor.values.size() != 1U) {
            throw QStateError("Tensor network contraction did not reduce to scalars");
        }
        result *= factor.values.front();
    }
    if (stats != nullptr) {
        *stats = local_stats;
    }
    return result;
}

QComplex TensorNetworkCircuit::amplitude(
    std::span<const std::uint8_t> basis_bits,
    TensorContractionStats* stats) const {
    if (basis_bits.size() != qubit_count_) {
        throw QStateError("Tensor network basis width does not match qubit count");
    }
    if (factors_.size() + qubit_count_ > config_.max_factors) {
        throw QStateError("Tensor network query exceeded max_factors");
    }

    std::vector<Factor> factors = factors_;
    factors.reserve(factors.size() + qubit_count_);
    for (std::size_t qubit = 0; qubit < qubit_count_; ++qubit) {
        if (basis_bits[qubit] > 1U) {
            throw QStateError("Tensor network basis bits must be zero or one");
        }
        Factor pin;
        pin.variables.push_back(current_wires_[qubit]);
        pin.values = basis_bits[qubit] == 0U
            ? std::vector<QComplex>{{1.0, 0.0}, {0.0, 0.0}}
            : std::vector<QComplex>{{0.0, 0.0}, {1.0, 0.0}};
        factors.push_back(std::move(pin));
    }
    return contract(std::move(factors), stats);
}

QComplex TensorNetworkCircuit::amplitude(
    BasisIndex basis,
    TensorContractionStats* stats) const {
    if (qubit_count_ > std::numeric_limits<BasisIndex>::digits) {
        throw QStateError("Tensor network BasisIndex query is limited to 64 qubits");
    }
    if (qubit_count_ < std::numeric_limits<BasisIndex>::digits &&
        basis >= (BasisIndex{1} << qubit_count_)) {
        throw QStateError("Tensor network basis index is out of range");
    }
    std::vector<std::uint8_t> bits(qubit_count_, 0U);
    for (std::size_t qubit = 0; qubit < qubit_count_; ++qubit) {
        bits[qubit] = static_cast<std::uint8_t>((basis >> qubit) & BasisIndex{1});
    }
    return amplitude(bits, stats);
}

std::vector<QComplex> TensorNetworkCircuit::materialize(std::size_t max_qubits) const {
    if (qubit_count_ > max_qubits ||
        qubit_count_ >= std::numeric_limits<std::size_t>::digits) {
        throw QStateError("Tensor network materialization exceeds the requested width limit");
    }
    const std::size_t dimension = std::size_t{1} << qubit_count_;
    std::vector<QComplex> amplitudes(dimension);
    for (std::size_t basis = 0; basis < dimension; ++basis) {
        amplitudes[basis] = amplitude(static_cast<BasisIndex>(basis));
    }
    return amplitudes;
}

std::size_t TensorNetworkCircuit::estimated_bytes() const noexcept {
    std::size_t bytes = sizeof(*this) + current_wires_.capacity() * sizeof(VariableId) +
                        factors_.capacity() * sizeof(Factor);
    for (const Factor& factor : factors_) {
        bytes += factor.variables.capacity() * sizeof(VariableId);
        bytes += factor.values.capacity() * sizeof(QComplex);
    }
    return bytes;
}

bool TensorNetworkCircuit::validate(std::string* reason) const noexcept {
    const auto fail = [reason](const char* message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };

    if (qubit_count_ == 0U || current_wires_.size() != qubit_count_) {
        return fail("tensor network wire count is invalid");
    }
    if (config_.max_contraction_entries < 2U || factors_.size() > config_.max_factors) {
        return fail("tensor network limits are invalid");
    }
    std::vector<VariableId> wires = current_wires_;
    std::sort(wires.begin(), wires.end());
    if (std::adjacent_find(wires.begin(), wires.end()) != wires.end()) {
        return fail("tensor network output wires are not unique");
    }

    for (const Factor& factor : factors_) {
        if (factor.variables.empty()) {
            return fail("tensor network source factor has no variables");
        }
        std::vector<VariableId> variables = factor.variables;
        std::sort(variables.begin(), variables.end());
        if (std::adjacent_find(variables.begin(), variables.end()) != variables.end()) {
            return fail("tensor network factor repeats a variable");
        }
        for (const VariableId variable : factor.variables) {
            if (variable >= next_variable_) {
                return fail("tensor network factor variable is out of range");
            }
        }
        if (factor.values.size() != binary_entries(factor.variables.size())) {
            return fail("tensor network factor shape is invalid");
        }
        if (!std::all_of(factor.values.begin(), factor.values.end(), finite)) {
            return fail("tensor network factor contains a non-finite value");
        }
    }
    return true;
}

}  // namespace qubit
