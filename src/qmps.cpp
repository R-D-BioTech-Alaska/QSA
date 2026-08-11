#include "qubit/qmps.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace qubit {
namespace {

[[nodiscard]] bool finite(const QComplex& value) noexcept {
    return std::isfinite(value.re) && std::isfinite(value.im);
}

[[nodiscard]] bool zero(const QComplex& value) noexcept {
    return value.re == 0.0 && value.im == 0.0;
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

}  // namespace

MatrixProductState::MatrixProductState(
    std::vector<MPSSiteTensor> sites,
    MPSConfig config)
    : sites_(std::move(sites)), config_(config) {
    std::string reason;
    if (!validate(&reason)) {
        throw QStateError("invalid matrix-product state: " + reason);
    }
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

std::size_t MatrixProductState::scalar_count() const noexcept {
    std::size_t count = 0U;
    for (const MPSSiteTensor& site : sites_) {
        count += site.zero.size() + site.one.size();
    }
    return count;
}

std::size_t MatrixProductState::estimated_bytes() const noexcept {
    std::size_t bytes = sizeof(*this) + sites_.capacity() * sizeof(MPSSiteTensor);
    for (const MPSSiteTensor& site : sites_) {
        bytes += (site.zero.capacity() + site.one.capacity()) * sizeof(QComplex);
    }
    return bytes;
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
            if (zero(left[row])) {
                continue;
            }
            for (std::size_t column = 0U; column < site.right_dimension; ++column) {
                const QComplex value = tensor[row * site.right_dimension + column];
                if (!zero(value)) {
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
        const MPSSiteTensor& site = sites_[qubit];
        const QMatrix2 operation = pauli_matrix(axes.empty() ? PauliAxis::I : axes[qubit]);
        next.assign(site.right_dimension * site.right_dimension, QComplex{});

        for (std::size_t left_bra = 0U; left_bra < site.left_dimension; ++left_bra) {
            for (std::size_t left_ket = 0U; left_ket < site.left_dimension; ++left_ket) {
                const QComplex prefix =
                    environment[left_bra * site.left_dimension + left_ket];
                if (zero(prefix)) {
                    continue;
                }
                for (std::uint8_t bra = 0U; bra < 2U; ++bra) {
                    const std::vector<QComplex>& bra_tensor = physical(site, bra);
                    for (std::uint8_t ket = 0U; ket < 2U; ++ket) {
                        const QComplex local = operation(bra, ket);
                        if (zero(local)) {
                            continue;
                        }
                        const std::vector<QComplex>& ket_tensor = physical(site, ket);
                        for (std::size_t right_bra = 0U;
                             right_bra < site.right_dimension;
                             ++right_bra) {
                            const QComplex bra_value =
                                bra_tensor[left_bra * site.right_dimension + right_bra];
                            if (zero(bra_value)) {
                                continue;
                            }
                            const QComplex weight = prefix * bra_value.conjugate() * local;
                            for (std::size_t right_ket = 0U;
                                 right_ket < site.right_dimension;
                                 ++right_ket) {
                                const QComplex ket_value =
                                    ket_tensor[left_ket * site.right_dimension + right_ket];
                                if (!zero(ket_value)) {
                                    next[right_bra * site.right_dimension + right_ket] +=
                                        weight * ket_value;
                                }
                            }
                        }
                    }
                }
            }
        }
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
