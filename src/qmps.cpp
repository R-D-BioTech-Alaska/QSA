#include "qubit/qmps.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace qubit {
namespace {

[[nodiscard]] bool finite(const QComplex& value) noexcept {
    return std::isfinite(value.re) && std::isfinite(value.im);
}

[[nodiscard]] bool is_zero(const QComplex& value) noexcept {
    return value.re == 0.0 && value.im == 0.0;
}

[[nodiscard]] std::size_t count_scalars(
    const std::vector<MPSSiteTensor>& sites) noexcept {
    std::size_t count = 0U;
    for (const MPSSiteTensor& site : sites) {
        count += site.zero.size() + site.one.size();
    }
    return count;
}

[[nodiscard]] const std::vector<QComplex>& physical(
    const MPSSiteTensor& site,
    std::uint8_t bit) {
    if (bit == 0U) {
        return site.zero;
    }
    if (bit == 1U) {
        return site.one;
    }
    throw QStateError("MPS basis bits must be zero or one");
}

[[nodiscard]] QMatrix2 pauli_matrix(PauliAxis axis) {
    switch (axis) {
        case PauliAxis::I:
            return gates::identity();
        case PauliAxis::X:
            return gates::x();
        case PauliAxis::Y:
            return gates::y();
        case PauliAxis::Z:
            return gates::z();
    }
    throw QStateError("invalid Pauli axis");
}

[[nodiscard]] bool unitary(const QMatrix2& matrix, double tolerance) noexcept {
    for (const QComplex value : matrix.values) {
        if (!finite(value)) {
            return false;
        }
    }
    const double first_norm = matrix(0U, 0U).norm2() + matrix(1U, 0U).norm2();
    const double second_norm = matrix(0U, 1U).norm2() + matrix(1U, 1U).norm2();
    const QComplex cross =
        matrix(0U, 0U).conjugate() * matrix(0U, 1U) +
        matrix(1U, 0U).conjugate() * matrix(1U, 1U);
    return std::abs(first_norm - 1.0) <= tolerance &&
           std::abs(second_norm - 1.0) <= tolerance &&
           cross.magnitude() <= tolerance;
}

[[nodiscard]] QMatrix2 projector_zero() {
    QMatrix2 matrix{};
    matrix.values = {
        QComplex{1.0, 0.0}, QComplex{0.0, 0.0},
        QComplex{0.0, 0.0}, QComplex{0.0, 0.0},
    };
    return matrix;
}

[[nodiscard]] QMatrix2 projector_one() {
    QMatrix2 matrix{};
    matrix.values = {
        QComplex{0.0, 0.0}, QComplex{0.0, 0.0},
        QComplex{0.0, 0.0}, QComplex{1.0, 0.0},
    };
    return matrix;
}

void transfer_left(
    const MPSSiteTensor& site,
    const QMatrix2& operation,
    const std::vector<QComplex>& environment,
    std::vector<QComplex>& next) {
    next.assign(site.right_dimension * site.right_dimension, QComplex{});
    for (std::size_t left_bra = 0U; left_bra < site.left_dimension; ++left_bra) {
        for (std::size_t left_ket = 0U; left_ket < site.left_dimension; ++left_ket) {
            const QComplex prefix =
                environment[left_bra * site.left_dimension + left_ket];
            if (is_zero(prefix)) {
                continue;
            }
            for (std::uint8_t bra = 0U; bra < 2U; ++bra) {
                const std::vector<QComplex>& bra_tensor = physical(site, bra);
                for (std::uint8_t ket = 0U; ket < 2U; ++ket) {
                    const QComplex local = operation(bra, ket);
                    if (is_zero(local)) {
                        continue;
                    }
                    const std::vector<QComplex>& ket_tensor = physical(site, ket);
                    for (std::size_t right_bra = 0U;
                         right_bra < site.right_dimension;
                         ++right_bra) {
                        const QComplex bra_value =
                            bra_tensor[left_bra * site.right_dimension + right_bra];
                        if (is_zero(bra_value)) {
                            continue;
                        }
                        const QComplex weight = prefix * bra_value.conjugate() * local;
                        for (std::size_t right_ket = 0U;
                             right_ket < site.right_dimension;
                             ++right_ket) {
                            const QComplex ket_value =
                                ket_tensor[left_ket * site.right_dimension + right_ket];
                            if (!is_zero(ket_value)) {
                                next[right_bra * site.right_dimension + right_ket] +=
                                    weight * ket_value;
                            }
                        }
                    }
                }
            }
        }
    }
}

void transfer_right_identity(
    const MPSSiteTensor& site,
    const std::vector<QComplex>& environment,
    std::vector<QComplex>& previous) {
    previous.assign(site.left_dimension * site.left_dimension, QComplex{});
    for (std::size_t left_bra = 0U; left_bra < site.left_dimension; ++left_bra) {
        for (std::size_t left_ket = 0U; left_ket < site.left_dimension; ++left_ket) {
            QComplex value{};
            for (std::uint8_t bit = 0U; bit < 2U; ++bit) {
                const std::vector<QComplex>& tensor = physical(site, bit);
                for (std::size_t right_bra = 0U;
                     right_bra < site.right_dimension;
                     ++right_bra) {
                    const QComplex bra_value =
                        tensor[left_bra * site.right_dimension + right_bra];
                    if (is_zero(bra_value)) {
                        continue;
                    }
                    for (std::size_t right_ket = 0U;
                         right_ket < site.right_dimension;
                         ++right_ket) {
                        const QComplex ket_value =
                            tensor[left_ket * site.right_dimension + right_ket];
                        const QComplex suffix =
                            environment[right_bra * site.right_dimension + right_ket];
                        if (!is_zero(ket_value) && !is_zero(suffix)) {
                            value += bra_value.conjugate() * ket_value * suffix;
                        }
                    }
                }
            }
            previous[left_bra * site.left_dimension + left_ket] = value;
        }
    }
}

[[nodiscard]] QComplex contract_cut(
    const std::vector<QComplex>& left,
    const std::vector<QComplex>& right) {
    if (left.size() != right.size()) {
        throw QStateError("MPS environment dimensions do not match");
    }
    QComplex value{};
    for (std::size_t index = 0U; index < left.size(); ++index) {
        value += left[index] * right[index];
    }
    return value;
}

}  // namespace

MatrixProductState::MatrixProductState(
    std::vector<MPSSiteTensor> sites,
    MPSConfig config)
    : sites_(std::move(sites)), config_(config) {
    std::string reason;
    if (!validate_structure(&reason)) {
        throw QStateError("invalid matrix-product state: " + reason);
    }
    scalar_count_ = count_scalars(sites_);
    const QComplex normalization = product_expectation(std::span<const PauliAxis>{});
    if (!finite(normalization) ||
        std::abs(normalization.im) > config_.normalization_tolerance ||
        std::abs(normalization.re - 1.0) > config_.normalization_tolerance) {
        throw QStateError("invalid matrix-product state: MPS tensors are not normalized");
    }
}

MatrixProductState MatrixProductState::zero(
    std::size_t qubit_count,
    MPSConfig config) {
    if (qubit_count == 0U) {
        throw QStateError("zero MPS requires at least one qubit");
    }
    std::vector<MPSSiteTensor> sites;
    sites.reserve(qubit_count);
    for (std::size_t qubit = 0U; qubit < qubit_count; ++qubit) {
        sites.push_back({1U, 1U, {{1.0, 0.0}}, {{0.0, 0.0}}});
    }
    return MatrixProductState(std::move(sites), config);
}

MatrixProductState MatrixProductState::ghz(
    std::size_t qubit_count,
    double phase,
    MPSConfig config) {
    if (qubit_count < 2U || !std::isfinite(phase)) {
        throw QStateError("GHZ MPS requires at least two qubits and a finite phase");
    }

    const double scale = 1.0 / std::sqrt(2.0);
    const QComplex phased = QComplex::from_polar(scale, phase);
    std::vector<MPSSiteTensor> sites;
    sites.reserve(qubit_count);
    sites.push_back({
        1U,
        2U,
        {{scale, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, phased},
    });
    for (std::size_t qubit = 1U; qubit + 1U < qubit_count; ++qubit) {
        sites.push_back({
            2U,
            2U,
            {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
            {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}},
        });
    }
    sites.push_back({
        2U,
        1U,
        {{1.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {1.0, 0.0}},
    });
    return MatrixProductState(std::move(sites), config);
}

MatrixProductState MatrixProductState::w(
    std::size_t qubit_count,
    MPSConfig config) {
    if (qubit_count < 2U) {
        throw QStateError("W MPS requires at least two qubits");
    }

    const double scale = 1.0 / std::sqrt(static_cast<double>(qubit_count));
    std::vector<MPSSiteTensor> sites;
    sites.reserve(qubit_count);
    sites.push_back({
        1U,
        2U,
        {{scale, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {scale, 0.0}},
    });
    for (std::size_t qubit = 1U; qubit + 1U < qubit_count; ++qubit) {
        sites.push_back({
            2U,
            2U,
            {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}},
            {{0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        });
    }
    sites.push_back({
        2U,
        1U,
        {{0.0, 0.0}, {1.0, 0.0}},
        {{1.0, 0.0}, {0.0, 0.0}},
    });
    return MatrixProductState(std::move(sites), config);
}

MatrixProductState MatrixProductState::cluster(
    std::size_t qubit_count,
    MPSConfig config) {
    if (qubit_count < 2U) {
        throw QStateError("cluster MPS requires at least two qubits");
    }

    const double scale = 1.0 / std::sqrt(2.0);
    std::vector<MPSSiteTensor> sites;
    sites.reserve(qubit_count);
    sites.push_back({
        1U,
        2U,
        {{scale, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {scale, 0.0}},
    });
    for (std::size_t qubit = 1U; qubit + 1U < qubit_count; ++qubit) {
        sites.push_back({
            2U,
            2U,
            {{scale, 0.0}, {0.0, 0.0}, {scale, 0.0}, {0.0, 0.0}},
            {{0.0, 0.0}, {scale, 0.0}, {0.0, 0.0}, {-scale, 0.0}},
        });
    }
    sites.push_back({
        2U,
        1U,
        {{scale, 0.0}, {scale, 0.0}},
        {{scale, 0.0}, {-scale, 0.0}},
    });
    return MatrixProductState(std::move(sites), config);
}

std::size_t MatrixProductState::max_bond_dimension() const noexcept {
    std::size_t result = 1U;
    for (const MPSSiteTensor& site : sites_) {
        result = std::max(result, std::max(site.left_dimension, site.right_dimension));
    }
    return result;
}

std::size_t MatrixProductState::estimated_bytes() const noexcept {
    std::size_t bytes = sizeof(*this) + sites_.capacity() * sizeof(MPSSiteTensor);
    for (const MPSSiteTensor& site : sites_) {
        bytes += (site.zero.capacity() + site.one.capacity()) * sizeof(QComplex);
    }
    return bytes;
}

void MatrixProductState::apply_unitary(
    std::size_t qubit,
    const QMatrix2& matrix) {
    if (qubit >= sites_.size()) {
        throw QStateError("MPS single-qubit target is out of range");
    }
    const double tolerance = std::max(1e-12, config_.normalization_tolerance);
    if (!unitary(matrix, tolerance)) {
        throw QStateError("MPS single-qubit operation is not unitary");
    }

    MPSSiteTensor& site = sites_[qubit];
    std::vector<QComplex> next_zero(site.zero.size());
    std::vector<QComplex> next_one(site.one.size());
    for (std::size_t index = 0U; index < site.zero.size(); ++index) {
        next_zero[index] = matrix(0U, 0U) * site.zero[index] +
                           matrix(0U, 1U) * site.one[index];
        next_one[index] = matrix(1U, 0U) * site.zero[index] +
                          matrix(1U, 1U) * site.one[index];
    }
    site.zero.swap(next_zero);
    site.one.swap(next_one);
}

void MatrixProductState::apply_adjacent_controlled(
    std::size_t control,
    std::size_t target,
    const QMatrix2& active) {
    if (control >= sites_.size() || target >= sites_.size()) {
        throw QStateError("MPS controlled gate qubit is out of range");
    }
    if (control == target ||
        (control + 1U != target && target + 1U != control)) {
        throw QStateError("MPS controlled gates require adjacent distinct qubits");
    }

    const std::size_t left_index = std::min(control, target);
    const std::size_t right_index = left_index + 1U;
    const MPSSiteTensor& left = sites_[left_index];
    const MPSSiteTensor& right = sites_[right_index];
    const std::size_t old_bond = left.right_dimension;
    if (old_bond != right.left_dimension) {
        throw QStateError("MPS controlled gate found an inconsistent bond");
    }
    if (old_bond > std::numeric_limits<std::size_t>::max() / 2U) {
        throw QStateError("MPS controlled gate bond dimension overflows size_t");
    }
    const std::size_t next_bond = old_bond * 2U;
    if (next_bond > config_.max_bond_dimension) {
        throw QStateError("MPS controlled gate exceeds configured bond dimension");
    }

    const std::size_t old_local_scalars =
        left.zero.size() + left.one.size() + right.zero.size() + right.one.size();
    if (scalar_count_ > config_.max_scalars ||
        old_local_scalars > config_.max_scalars - scalar_count_) {
        throw QStateError("MPS controlled gate exceeds configured scalar count");
    }

    const QMatrix2 p0 = projector_zero();
    const QMatrix2 p1 = projector_one();
    const QMatrix2 identity = gates::identity();
    const std::array<QMatrix2, 2> left_ops = control == left_index
        ? std::array<QMatrix2, 2>{p0, p1}
        : std::array<QMatrix2, 2>{identity, active};
    const std::array<QMatrix2, 2> right_ops = control == left_index
        ? std::array<QMatrix2, 2>{identity, active}
        : std::array<QMatrix2, 2>{p0, p1};

    const std::size_t left_area = left.left_dimension * next_bond;
    const std::size_t right_area = next_bond * right.right_dimension;
    std::array<std::vector<QComplex>, 2> next_left{
        std::vector<QComplex>(left_area),
        std::vector<QComplex>(left_area),
    };
    std::array<std::vector<QComplex>, 2> next_right{
        std::vector<QComplex>(right_area),
        std::vector<QComplex>(right_area),
    };

    for (std::size_t branch = 0U; branch < 2U; ++branch) {
        for (std::uint8_t output = 0U; output < 2U; ++output) {
            for (std::size_t row = 0U; row < left.left_dimension; ++row) {
                for (std::size_t bond = 0U; bond < old_bond; ++bond) {
                    QComplex value{};
                    for (std::uint8_t input = 0U; input < 2U; ++input) {
                        value += left_ops[branch](output, input) *
                                 physical(left, input)[row * old_bond + bond];
                    }
                    next_left[output][row * next_bond + bond * 2U + branch] = value;
                }
            }

            for (std::size_t bond = 0U; bond < old_bond; ++bond) {
                for (std::size_t column = 0U; column < right.right_dimension; ++column) {
                    QComplex value{};
                    for (std::uint8_t input = 0U; input < 2U; ++input) {
                        value += right_ops[branch](output, input) *
                                 physical(right, input)[bond * right.right_dimension + column];
                    }
                    next_right[output][
                        (bond * 2U + branch) * right.right_dimension + column] = value;
                }
            }
        }
    }

    MPSSiteTensor& mutable_left = sites_[left_index];
    MPSSiteTensor& mutable_right = sites_[right_index];
    mutable_left.right_dimension = next_bond;
    mutable_right.left_dimension = next_bond;
    mutable_left.zero.swap(next_left[0]);
    mutable_left.one.swap(next_left[1]);
    mutable_right.zero.swap(next_right[0]);
    mutable_right.one.swap(next_right[1]);
    scalar_count_ += old_local_scalars;
}

void MatrixProductState::apply_cnot(std::size_t control, std::size_t target) {
    apply_adjacent_controlled(control, target, gates::x());
}

void MatrixProductState::apply_cz(std::size_t first, std::size_t second) {
    apply_adjacent_controlled(first, second, gates::z());
}

QComplex MatrixProductState::amplitude(std::span<const std::uint8_t> bits) const {
    if (bits.size() != sites_.size()) {
        throw QStateError("MPS basis width does not match state width");
    }

    std::vector<QComplex> left(1U, {1.0, 0.0});
    std::vector<QComplex> next;
    for (std::size_t qubit = 0U; qubit < sites_.size(); ++qubit) {
        const MPSSiteTensor& site = sites_[qubit];
        const std::vector<QComplex>& tensor = physical(site, bits[qubit]);
        next.assign(site.right_dimension, QComplex{});
        for (std::size_t row = 0U; row < site.left_dimension; ++row) {
            if (is_zero(left[row])) {
                continue;
            }
            for (std::size_t column = 0U; column < site.right_dimension; ++column) {
                const QComplex value = tensor[row * site.right_dimension + column];
                if (!is_zero(value)) {
                    next[column] += left[row] * value;
                }
            }
        }
        left.swap(next);
    }
    return left.front();
}

QComplex MatrixProductState::product_expectation(std::span<const PauliAxis> axes) const {
    if (!axes.empty() && axes.size() != sites_.size()) {
        throw QStateError("MPS Pauli width does not match state width");
    }

    std::vector<QComplex> environment(1U, {1.0, 0.0});
    std::vector<QComplex> next;
    for (std::size_t qubit = 0U; qubit < sites_.size(); ++qubit) {
        transfer_left(
            sites_[qubit],
            pauli_matrix(axes.empty() ? PauliAxis::I : axes[qubit]),
            environment,
            next);
        environment.swap(next);
    }
    return environment.front();
}

QComplex MatrixProductState::pauli_expectation(std::span<const PauliAxis> axes) const {
    if (axes.size() != sites_.size()) {
        throw QStateError("MPS Pauli width does not match state width");
    }
    return product_expectation(axes);
}

QComplex MatrixProductState::expectation(const PauliObservable& observable) const {
    if (observable.qubit_count() != sites_.size()) {
        throw QStateError("MPS observable width does not match state width");
    }
    std::string reason;
    if (!observable.validate(&reason)) {
        throw QStateError("invalid MPS Pauli observable: " + reason);
    }

    QComplex result{};
    std::vector<PauliAxis> axes(sites_.size(), PauliAxis::I);
    for (const PauliTerm& term : observable.terms()) {
        std::fill(axes.begin(), axes.end(), PauliAxis::I);
        for (const PauliFactor& factor : term.factors) {
            axes[static_cast<std::size_t>(factor.qubit)] = factor.axis;
        }
        result += term.coefficient * product_expectation(axes);
    }
    return result;
}

double MatrixProductState::norm2() const {
    return product_expectation(std::span<const PauliAxis>{}).re;
}

std::vector<QComplex> MatrixProductState::materialize() const {
    if (sites_.size() > config_.max_materialize_qubits ||
        sites_.size() >= std::numeric_limits<std::size_t>::digits) {
        throw QStateError("MPS materialization exceeds configured width");
    }

    const std::size_t dimension = std::size_t{1} << sites_.size();
    std::vector<QComplex> amplitudes(dimension);
    std::vector<std::uint8_t> bits(sites_.size(), 0U);
    for (std::size_t basis = 0U; basis < dimension; ++basis) {
        for (std::size_t qubit = 0U; qubit < sites_.size(); ++qubit) {
            bits[qubit] = static_cast<std::uint8_t>((basis >> qubit) & 1U);
        }
        amplitudes[basis] = amplitude(bits);
    }
    return amplitudes;
}

bool MatrixProductState::validate_structure(std::string* reason) const {
    const auto fail = [&](const char* message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };

    if (sites_.empty()) {
        return fail("state has no sites");
    }
    if (config_.max_bond_dimension == 0U || config_.max_scalars == 0U ||
        config_.max_materialize_qubits == 0U ||
        !std::isfinite(config_.normalization_tolerance) ||
        config_.normalization_tolerance <= 0.0) {
        return fail("invalid MPS resource configuration");
    }
    if (sites_.front().left_dimension != 1U || sites_.back().right_dimension != 1U) {
        return fail("MPS boundary bond dimensions must be one");
    }

    std::size_t total_scalars = 0U;
    for (std::size_t index = 0U; index < sites_.size(); ++index) {
        const MPSSiteTensor& site = sites_[index];
        if (site.left_dimension == 0U || site.right_dimension == 0U) {
            return fail("MPS bond dimensions must be positive");
        }
        if (site.left_dimension > config_.max_bond_dimension ||
            site.right_dimension > config_.max_bond_dimension) {
            return fail("MPS bond dimension exceeds configured limit");
        }
        if (site.left_dimension >
            std::numeric_limits<std::size_t>::max() / site.right_dimension) {
            return fail("MPS site dimension overflows size_t");
        }
        const std::size_t area = site.left_dimension * site.right_dimension;
        if (site.zero.size() != area || site.one.size() != area) {
            return fail("MPS physical tensor shape does not match bond dimensions");
        }
        if (total_scalars > config_.max_scalars ||
            area > (config_.max_scalars - total_scalars) / 2U) {
            return fail("MPS scalar count exceeds configured limit");
        }
        total_scalars += area * 2U;
        for (const QComplex value : site.zero) {
            if (!finite(value)) {
                return fail("MPS tensor contains a nonfinite value");
            }
        }
        for (const QComplex value : site.one) {
            if (!finite(value)) {
                return fail("MPS tensor contains a nonfinite value");
            }
        }
        if (index + 1U < sites_.size() &&
            site.right_dimension != sites_[index + 1U].left_dimension) {
            return fail("neighboring MPS bond dimensions do not match");
        }
    }
    return true;
}

bool MatrixProductState::validate(std::string* reason) const {
    if (!validate_structure(reason)) {
        return false;
    }
    if (count_scalars(sites_) != scalar_count_) {
        if (reason != nullptr) {
            *reason = "MPS scalar-count cache is inconsistent";
        }
        return false;
    }
    const QComplex normalization = product_expectation(std::span<const PauliAxis>{});
    if (!finite(normalization)) {
        if (reason != nullptr) {
            *reason = "MPS normalization is nonfinite";
        }
        return false;
    }
    if (std::abs(normalization.im) > config_.normalization_tolerance ||
        std::abs(normalization.re - 1.0) > config_.normalization_tolerance) {
        if (reason != nullptr) {
            *reason = "MPS tensors are not normalized";
        }
        return false;
    }
    return true;
}

MPSPauliPlan::MPSPauliPlan(
    MatrixProductState state,
    std::size_t max_environment_scalars)
    : state_(std::move(state)),
      max_environment_scalars_(max_environment_scalars) {
    if (max_environment_scalars_ == 0U) {
        throw QStateError("MPS Pauli plan environment limit must be positive");
    }

    const std::vector<MPSSiteTensor>& sites = state_.sites();
    std::size_t required = 0U;
    for (std::size_t cut = 0U; cut <= sites.size(); ++cut) {
        const std::size_t dimension =
            cut == 0U ? sites.front().left_dimension : sites[cut - 1U].right_dimension;
        if (dimension > std::numeric_limits<std::size_t>::max() / dimension) {
            throw QStateError("MPS Pauli environment dimension overflows size_t");
        }
        const std::size_t area = dimension * dimension;
        if (required > max_environment_scalars_ ||
            area > (max_environment_scalars_ - required) / 2U) {
            throw QStateError("MPS Pauli environment cache exceeds configured limit");
        }
        required += area * 2U;
    }
    environment_scalar_count_ = required;

    left_identity_.resize(sites.size() + 1U);
    right_identity_.resize(sites.size() + 1U);
    left_identity_.front() = {{1.0, 0.0}};
    std::vector<QComplex> next;
    for (std::size_t qubit = 0U; qubit < sites.size(); ++qubit) {
        transfer_left(
            sites[qubit],
            gates::identity(),
            left_identity_[qubit],
            next);
        left_identity_[qubit + 1U] = next;
    }

    right_identity_.back() = {{1.0, 0.0}};
    std::vector<QComplex> previous;
    for (std::size_t qubit = sites.size(); qubit-- > 0U;) {
        transfer_right_identity(sites[qubit], right_identity_[qubit + 1U], previous);
        right_identity_[qubit] = previous;
    }

    const QComplex cached_norm = left_identity_.back().front();
    if (!finite(cached_norm) ||
        std::abs(cached_norm.im) > state_.config().normalization_tolerance ||
        std::abs(cached_norm.re - 1.0) > state_.config().normalization_tolerance) {
        throw QStateError("MPS Pauli plan cached normalization is invalid");
    }
}

std::size_t MPSPauliPlan::estimated_bytes() const noexcept {
    std::size_t bytes = sizeof(*this) + state_.estimated_bytes();
    bytes += left_identity_.capacity() * sizeof(std::vector<QComplex>);
    bytes += right_identity_.capacity() * sizeof(std::vector<QComplex>);
    for (const std::vector<QComplex>& environment : left_identity_) {
        bytes += environment.capacity() * sizeof(QComplex);
    }
    for (const std::vector<QComplex>& environment : right_identity_) {
        bytes += environment.capacity() * sizeof(QComplex);
    }
    return bytes;
}

QComplex MPSPauliPlan::term_expectation(std::span<const PauliFactor> factors) const {
    if (factors.empty()) {
        return left_identity_.back().front();
    }

    std::size_t first = state_.qubit_count();
    std::size_t last = 0U;
    for (const PauliFactor& factor : factors) {
        const std::size_t qubit = static_cast<std::size_t>(factor.qubit);
        if (qubit >= state_.qubit_count()) {
            throw QStateError("MPS Pauli factor qubit is out of range");
        }
        first = std::min(first, qubit);
        last = std::max(last, qubit);
    }

    std::vector<PauliAxis> axes(last - first + 1U, PauliAxis::I);
    for (const PauliFactor& factor : factors) {
        axes[static_cast<std::size_t>(factor.qubit) - first] = factor.axis;
    }

    std::vector<QComplex> environment = left_identity_[first];
    std::vector<QComplex> next;
    const std::vector<MPSSiteTensor>& sites = state_.sites();
    for (std::size_t qubit = first; qubit <= last; ++qubit) {
        transfer_left(sites[qubit], pauli_matrix(axes[qubit - first]), environment, next);
        environment.swap(next);
    }
    return contract_cut(environment, right_identity_[last + 1U]);
}

QComplex MPSPauliPlan::expectation(const PauliObservable& observable) const {
    if (observable.qubit_count() != state_.qubit_count()) {
        throw QStateError("MPS Pauli plan observable width does not match state width");
    }
    std::string reason;
    if (!observable.validate(&reason)) {
        throw QStateError("invalid MPS Pauli plan observable: " + reason);
    }

    QComplex result{};
    for (const PauliTerm& term : observable.terms()) {
        result += term.coefficient * term_expectation(term.factors);
    }
    return result;
}

std::size_t required_schmidt_rank_cross_cut_bell_pairs(std::size_t pair_count) {
    if (pair_count >= std::numeric_limits<std::size_t>::digits) {
        throw QStateError("cross-cut Bell-pair Schmidt rank exceeds size_t");
    }
    return std::size_t{1} << pair_count;
}

bool bond_dimension_accepts_cross_cut_bell_pairs(
    std::size_t pair_count,
    std::size_t bond_dimension) {
    return required_schmidt_rank_cross_cut_bell_pairs(pair_count) <= bond_dimension;
}

}  // namespace qubit
